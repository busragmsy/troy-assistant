# kb/ingest/ingest_hf_improved.py
import os
import re
import argparse
import hashlib
import time
from typing import List, Tuple, Dict, Any

import psycopg2
from psycopg2.extras import execute_values
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np

# Config
DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "kb"),
    user=os.getenv("DB_USER", "troy"),
    password=os.getenv("DB_PASSWORD", "troy1234"),
)

MODEL_NAME = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-large")
TARGET_TOKENS = int(os.getenv("TARGET_TOKENS", "400"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
OVERLAP_TOKENS = int(os.getenv("OVERLAP_TOKENS", "50"))
MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", "15"))

_model = None
_tokenizer = None
_dim = None

class Stats:
    def __init__(self):
        self.total_docs = 0
        self.total_chunks = 0
        self.total_filtered = 0
        self.total_time = 0.0
    
    def add(self, chunks, filtered, elapsed):
        self.total_docs += 1
        self.total_chunks += chunks
        self.total_filtered += filtered
        self.total_time += elapsed
    
    def print_summary(self):
        total = self.total_chunks + self.total_filtered
        eff = (self.total_chunks / total * 100) if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"OZET:")
        print(f"  Dokumanlar: {self.total_docs}")
        print(f"  Kaydedilen: {self.total_chunks}")
        print(f"  Filtrelenen: {self.total_filtered}")
        print(f"  Verimlilik: {eff:.1f}%")
        print(f"  Sure: {self.total_time:.1f}s")
        print(f"{'='*60}")

stats = Stats()

def get_model():
    global _model, _tokenizer, _dim
    if _model is None:
        print(f"Model yukleniyor: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        _model.max_seq_length = MAX_TOKENS
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _dim = _model.get_sentence_embedding_dimension()
        print(f"  Dim: {_dim}, Max tokens: {MAX_TOKENS}")
    return _model, _tokenizer, _dim

def db_connect():
    return psycopg2.connect(**DB)

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_header_block(text: str) -> str:
    """Sayfa basindaki header blogunu sil"""
    lines = text.split('\n')
    
    # Ilk 15 satira bak
    start_idx = 0
    for i, line in enumerate(lines[:15]):
        line_lower = line.lower()
        # Header keyword'u varsa atla
        if any(kw in line_lower for kw in [
            'kullanici kilavuzu', 'troy sistemi', 'dokuman no',
            'revize tarihi', 'yayin kaynagi', 'dahili', 'mudurlugu'
        ]):
            start_idx = i + 1
    
    # Header'dan sonrasini don
    return '\n'.join(lines[start_idx:]) if start_idx > 0 else text

def read_pdf(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except:
            text = ""
        
        text = clean_text(text)
        
        # Header'i sil
        text = remove_header_block(text)
        
        # Cok kisa sayfalar (icindekiler vb) atla
        if len(text.split()) < 10:
            continue
        
        pages.append((i, text))
    
    return pages

def is_chunk_noise(chunk: str) -> bool:
    """Chunk gurultu mu?"""
    words = chunk.split()
    
    # Cok kisa
    if len(words) < MIN_CHUNK_WORDS:
        return True
    
    # Header density
    chunk_lower = chunk.lower()
    noise_kw = ['kullanici kilavuzu', 'troy sistemi', 'dokuman no', 'revize tarihi']
    noise_count = sum(1 for kw in noise_kw if kw in chunk_lower)
    
    # Ilk 100 karakter cogunu header kapliyor
    if len(chunk) < 200 and noise_count >= 2:
        return True
    
    return False

def chunk_text(text: str, tokenizer, target: int, max_tok: int, overlap: int) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current = []
    current_len = 0
    
    for sent in sentences:
        tokens = tokenizer.encode(sent, add_special_tokens=False)
        sent_len = len(tokens)
        
        if sent_len > max_tok:
            if current:
                chunks.append(' '.join(current))
                current = []
                current_len = 0
            
            for i in range(0, len(tokens), target):
                piece = tokenizer.decode(tokens[i:i+target])
                chunks.append(piece)
            continue
        
        if current_len + sent_len <= target:
            current.append(sent)
            current_len += sent_len
        else:
            if current:
                chunks.append(' '.join(current))
            
            if overlap > 0 and current:
                overlap_sents = []
                overlap_len = 0
                for s in reversed(current):
                    s_len = len(tokenizer.encode(s, add_special_tokens=False))
                    if overlap_len + s_len > overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += s_len
                current = overlap_sents
                current_len = overlap_len
            else:
                current = []
                current_len = 0
            
            current.append(sent)
            current_len += sent_len
    
    if current:
        chunks.append(' '.join(current))
    
    # Gurultu chunk'lari filtrele
    return [c for c in chunks if not is_chunk_noise(c)]

def ensure_schema(cur, dim: int):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
      id BIGSERIAL PRIMARY KEY,
      title VARCHAR(500) NOT NULL,
      file_path TEXT NOT NULL UNIQUE,
      content_hash VARCHAR(64),
      status VARCHAR(20) DEFAULT 'active',
      created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS document_sections (
      id BIGSERIAL PRIMARY KEY,
      document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
      content TEXT NOT NULL,
      page_number INT
    );
    """)
    
    cur.execute("""
    SELECT 1 FROM information_schema.tables 
    WHERE table_name='document_embeddings'
    """)
    
    if not cur.fetchone():
        cur.execute(f"""
        CREATE TABLE document_embeddings (
          id BIGSERIAL PRIMARY KEY,
          section_id BIGINT REFERENCES document_sections(id) ON DELETE CASCADE,
          embedding vector({dim}) NOT NULL,
          model_name VARCHAR(100) NOT NULL
        );
        """)
    
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_embeddings_cosine
    ON document_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
    """)

def upsert_document(cur, title: str, path: str, content_hash: str) -> Tuple[int, bool]:
    cur.execute("SELECT id, content_hash FROM documents WHERE file_path=%s", (path,))
    row = cur.fetchone()
    
    if row:
        if row[1] == content_hash:
            return row[0], False
        
        cur.execute("DELETE FROM document_embeddings WHERE section_id IN (SELECT id FROM document_sections WHERE document_id=%s)", (row[0],))
        cur.execute("DELETE FROM document_sections WHERE document_id=%s", (row[0],))
        cur.execute("UPDATE documents SET content_hash=%s WHERE id=%s", (content_hash, row[0]))
        return row[0], True
    
    cur.execute("""
    INSERT INTO documents (title, file_path, content_hash, status)
    VALUES (%s, %s, %s, 'active')
    RETURNING id
    """, (title, path, content_hash))
    
    return cur.fetchone()[0], True

def insert_sections(cur, doc_id: int, chunks: List[Dict]) -> List[int]:
    ids = []
    for chunk in chunks:
        cur.execute("""
        INSERT INTO document_sections (document_id, content, page_number)
        VALUES (%s, %s, %s)
        RETURNING id
        """, (doc_id, chunk['text'], chunk['page']))
        ids.append(cur.fetchone()[0])
    return ids

def insert_embeddings(cur, section_ids: List[int], vectors: List, model: str):
    def vec_str(v):
        return '[' + ','.join(f'{x:.6f}' for x in v) + ']'
    
    values = [(sid, vec_str(vec), model) for sid, vec in zip(section_ids, vectors)]
    execute_values(cur, """
    INSERT INTO document_embeddings (section_id, embedding, model_name)
    VALUES %s
    """, values, template="(%s, %s::vector, %s)")

def process_pdf(path: str) -> Tuple[int, int]:
    start = time.time()
    
    title = os.path.splitext(os.path.basename(path))[0]
    pages = read_pdf(path)
    
    if not pages:
        print(f"  Bos: {os.path.basename(path)}")
        return 0, 0
    
    full_text = ' '.join(t for _, t in pages)
    content_hash = hashlib.sha256(full_text.encode()).hexdigest()
    
    model, tokenizer, dim = get_model()
    
    # Chunk'la
    all_chunks = []
    for page_num, text in pages:
        chunks = chunk_text(text, tokenizer, TARGET_TOKENS, MAX_TOKENS, OVERLAP_TOKENS)
        for chunk in chunks:
            all_chunks.append({'text': chunk, 'page': page_num})
    
    initial = len(all_chunks)
    
    if not all_chunks:
        print(f"  Chunk yok: {os.path.basename(path)}")
        return 0, 0
    
    # Embed
    texts = [c['text'] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    
    # Semantic dedup
    kept_chunks = []
    kept_vecs = []
    
    for chunk, vec in zip(all_chunks, embeddings):
        is_dup = False
        for existing_vec in kept_vecs[-3:]:
            sim = np.dot(vec, existing_vec)
            if sim > 0.98:
                is_dup = True
                break
        
        if not is_dup:
            kept_chunks.append(chunk)
            kept_vecs.append(vec)
    
    filtered = initial - len(kept_chunks)
    
    # DB
    conn = db_connect()
    cur = conn.cursor()
    
    try:
        ensure_schema(cur, dim)
        doc_id, changed = upsert_document(cur, title, os.path.abspath(path), content_hash)
        sec_ids = insert_sections(cur, doc_id, kept_chunks)
        insert_embeddings(cur, sec_ids, kept_vecs, MODEL_NAME)
        conn.commit()
        
        elapsed = time.time() - start
        stats.add(len(kept_chunks), filtered, elapsed)
        
        print(f"  OK: {os.path.basename(path)}")
        print(f"      chunks={len(kept_chunks)}, filtered={filtered}, {elapsed:.1f}s")
        
        return 1, len(kept_chunks)
    
    finally:
        cur.close()
        conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()
    
    get_model()
    
    pdfs = [f for f in os.listdir(args.dir) if f.lower().endswith('.pdf')]
    if not pdfs:
        print("PDF yok")
        return
    
    print(f"\n{len(pdfs)} PDF isleniyor...\n")
    
    for pdf in pdfs:
        process_pdf(os.path.join(args.dir, pdf))
    
    stats.print_summary()

if __name__ == '__main__':
    main()