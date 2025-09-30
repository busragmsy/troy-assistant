# kb/ingest/process_pdfs_hf.py
import os
import re
import math
import argparse
from typing import List, Dict, Any, Tuple

import psycopg2
from psycopg2.extras import execute_values
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-large")
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LANGUAGE = os.getenv("DOC_LANGUAGE", "tr")
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "50"))  # Minimum chunk boyutu
MIN_WORD_COUNT = int(os.getenv("MIN_WORD_COUNT", "5"))  # Minimum kelime sayısı


def db_connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "pg"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "kb"),
        user=os.getenv("DB_USER", "troy"),
        password=os.getenv("DB_PASSWORD", "troy1234"),
    )

def ensure_extensions_and_schema(cur):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
      id            BIGSERIAL PRIMARY KEY,
      title         VARCHAR(500) NOT NULL,
      document_type VARCHAR(100) NOT NULL,
      file_name     VARCHAR(500) NOT NULL,
      file_path     TEXT NOT NULL UNIQUE,
      department    VARCHAR(100),
      status        VARCHAR(20) DEFAULT 'active',
      language      VARCHAR(10) DEFAULT 'tr',
      created_at    TIMESTAMP DEFAULT NOW(),
      updated_at    TIMESTAMP DEFAULT NOW()
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS document_sections (
      id            BIGSERIAL PRIMARY KEY,
      document_id   BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      section_title TEXT,
      content       TEXT NOT NULL,
      page_number   INT,
      word_count    INT,
      created_at    TIMESTAMP DEFAULT NOW()
    );
    """)

    cur.execute("""
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='document_embeddings' AND column_name='embedding'
      ) THEN
        CREATE TABLE document_embeddings (
          id          BIGSERIAL PRIMARY KEY,
          section_id  BIGINT NOT NULL REFERENCES document_sections(id) ON DELETE CASCADE,
          embedding   vector(384) NOT NULL,
          model_name  VARCHAR(100) NOT NULL,
          created_at  TIMESTAMP DEFAULT NOW()
        );
      END IF;
    END$$;
    """)
    cur.execute("""
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE indexname='idx_document_embeddings_vec'
      ) THEN
        CREATE INDEX idx_document_embeddings_vec
        ON document_embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
      END IF;
    END$$;
    """)

def ensure_document(cur, title: str, file_name: str, file_path: str,
                    doc_type: str, department: str, language: str = LANGUAGE) -> Tuple[int, bool]:
    cur.execute("SELECT id FROM documents WHERE file_path=%s", (file_path,))
    row = cur.fetchone()
    if row:
        return row[0], False
    cur.execute("""
      INSERT INTO documents (title, document_type, file_name, file_path, department, status, language)
      VALUES (%s,%s,%s,%s,%s,'active',%s)
      RETURNING id
    """, (title, doc_type, file_name, file_path, department, language))
    return cur.fetchone()[0], True

def insert_sections(cur, document_id: int, sections: List[Dict[str, Any]]) -> List[int]:
    rows = [(document_id,
             s.get("title"),
             s["text"],
             s.get("page_number"),
             s.get("word_count"))
            for s in sections]
    ids = []
    for r in rows:
        cur.execute("""
          INSERT INTO document_sections (document_id, section_title, content, page_number, word_count)
          VALUES (%s,%s,%s,%s,%s)
          RETURNING id
        """, r)
        ids.append(cur.fetchone()[0])
    return ids

def to_vec_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def insert_embeddings_batch(cur, section_ids: List[int], vectors: List[List[float]], model_name: str):
    values = [(sid, to_vec_literal(vec), model_name) for sid, vec in zip(section_ids, vectors)]
    execute_values(cur,
        """
        INSERT INTO document_embeddings (section_id, embedding, model_name)
        VALUES %s
        """,
        values,
        template="(%s, %s::vector, %s)"
    )


def clean_text(text: str) -> str:
    """Metni temizle ve normalize et"""
    if not text:
        return ""
    
    # Fazla boşlukları tek boşluğa indir
    text = re.sub(r'\s+', ' ', text)
    
    # Bozuk karakterleri temizle
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Bölünmüş kelimeleri birleştirmeye çalış (ör: "kel ime" -> "kelime")
    # Tek harf + boşluk + tek harf durumlarını birleştir
    text = re.sub(r'(?<=\w)\s(?=\w(?:\s\w){2,})', '', text)
    
    # Çoklu tire/çizgileri temizle
    text = re.sub(r'-{2,}', ' ', text)
    
    # Başta ve sonda boşlukları temizle
    text = text.strip()
    
    return text


def is_valid_chunk(text: str) -> bool:
    """Chunk'ın geçerli olup olmadığını kontrol et"""
    if not text or len(text) < MIN_CHUNK_CHARS:
        return False
    
    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        return False
    
    # Çok fazla tek karakterli "kelime" varsa muhtemelen bozuk
    single_char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
    if single_char_ratio > 0.3:
        return False
    
    # Alfabetik karakter oranı çok düşükse (sayı/sembol fazlaysa) atla
    alpha_chars = sum(1 for c in text if c.isalpha())
    if len(text) > 0 and alpha_chars / len(text) < 0.5:
        return False
    
    return True


_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')

def read_pdf_texts(pdf_path: str) -> List[Tuple[int, str]]:
    """PDF'den sayfa sayfa metin çıkar ve temizle"""
    reader = PdfReader(pdf_path)
    items = []
    
    for i, page in enumerate(reader.pages, start=1):
        try:
            # Alternatif extraction metodlarını dene
            txt = page.extract_text(extraction_mode="layout") or ""
            if not txt.strip():
                txt = page.extract_text() or ""
        except Exception as e:
            print(f"  ⚠️  Sayfa {i} okuma hatası: {e}")
            txt = ""
        
        txt = clean_text(txt)
        if txt:  # Sadece içerik varsa ekle
            items.append((i, txt))
    
    return items


def chunkify(text: str, max_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Metni anlamlı chunk'lara böl"""
    if not text:
        return []
    
    # Cümlelere böl
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    
    chunks = []
    buf = ""
    
    for s in sents:
        if not buf:
            buf = s
            continue
        
        # Mevcut buffer'a sığıyorsa ekle
        if len(buf) + 1 + len(s) <= max_chars:
            buf += " " + s
        else:
            # Buffer'ı chunk olarak kaydet
            if is_valid_chunk(buf):
                chunks.append(buf)
            
            # Overlap ile yeni buffer başlat
            if overlap > 0 and len(buf) > overlap:
                # Son overlap kadar karakteri al
                overlap_text = buf[-overlap:].strip()
                buf = overlap_text + " " + s
            else:
                buf = s
    
    # Son buffer'ı ekle
    if buf and is_valid_chunk(buf):
        chunks.append(buf)
    
    return chunks


def parse_pdf_advanced(pdf_path: str) -> List[Dict[str, Any]]:
    """PDF'i parse et ve geçerli chunk'ları döndür"""
    pages = read_pdf_texts(pdf_path)
    
    if not pages:
        return []
    
    sections = []
    
    for page_num, txt in pages:
        chunks = chunkify(txt)
        
        for chunk in chunks:
            sections.append({
                "title": f"Sayfa {page_num}",
                "text": chunk,
                "page_number": page_num,
                "word_count": len(chunk.split())
            })
    
    return sections


_model_cache: SentenceTransformer = None

def get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(EMBED_MODEL)
    return _model_cache

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=False, batch_size=32, convert_to_numpy=True)
    return [v.tolist() for v in vecs]


def process_pdf(path: str, doc_type: str, department: str) -> Tuple[int, int, int, bool]:
    """
    Returns: (doc_count, section_count, filtered_count, is_new)
    """
    title = os.path.splitext(os.path.basename(path))[0]
    sections = parse_pdf_advanced(path)
    
    original_count = len(sections)
    
    if not sections:
        print(f"  ⚠️  Boş/parse edilemedi: {os.path.basename(path)}")
        return (0, 0, 0, False)

    conn = db_connect()
    cur = conn.cursor()
    try:
        ensure_extensions_and_schema(cur)
        doc_id, is_new = ensure_document(cur, title, os.path.basename(path), path, doc_type, department, LANGUAGE)
        conn.commit()

        sec_ids = insert_sections(cur, doc_id, sections)
        conn.commit()

        total = 0
        for i in range(0, len(sec_ids), BATCH_SIZE):
            batch_ids = sec_ids[i:i+BATCH_SIZE]
            batch_txts = [sections[j]["text"] for j in range(i, min(i+BATCH_SIZE, len(sections)))]
            vecs = embed_texts(batch_txts)
            insert_embeddings_batch(cur, batch_ids, vecs, EMBED_MODEL)
            conn.commit()
            total += len(batch_ids)

        filtered = original_count - len(sections)
        status = 'new' if is_new else 'exists'
        print(f"  ✅ {os.path.basename(path)}")
        print(f"     doc_id={doc_id}, sections={total}, filtered={filtered} ({status})")
        
        return (1, total, filtered, is_new)
    finally:
        cur.close()
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Geliştirilmiş PDF ingest (metin temizleme + validasyon)")
    ap.add_argument("--dir", required=True, help="PDF klasörü")
    ap.add_argument("--doc-type", default="kullanici_kilavuzu")
    ap.add_argument("--department", default="FIP")
    args = ap.parse_args()

    pdfs = [f for f in os.listdir(args.dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("⚠️  Klasörde PDF bulunamadı.")
        return

    print(f"\n📂 {len(pdfs)} PDF işlenecek...\n")
    
    total_docs = 0
    total_secs = 0
    total_filtered = 0
    
    for f in pdfs:
        d, s, filt, _ = process_pdf(os.path.join(args.dir, f), args.doc_type, args.department)
        total_docs += d
        total_secs += s
        total_filtered += filt

    print(f"\n{'='*60}")
    print(f"📊 ÖZET:")
    print(f"   Dökümanlar: {total_docs}")
    print(f"   Kaydedilen bölümler: {total_secs}")
    print(f"   Filtrelenen (geçersiz): {total_filtered}")
    print(f"   Model: {EMBED_MODEL} (1024-dim)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()