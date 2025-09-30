import os, re, argparse, math, hashlib
from typing import List, Dict, Any, Tuple
import psycopg2
from psycopg2.extras import execute_values

# PDF okuyucu: öncelik PyMuPDF, yoksa pypdf
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# ==========================
# Konfig / ENV
# ==========================
DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "kb"),
    user=os.getenv("DB_USER", "troy"),
    password=os.getenv("DB_PASSWORD", "troy1234"),
)

MODEL_NAME     = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-large")  # 1024-dim
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "500"))   # 512 altı
TARGET_TOKENS  = int(os.getenv("TARGET_TOKENS", "460"))
OVERLAP_TOKENS = int(os.getenv("OVERLAP_TOKENS", "64"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", str(MAX_TOKENS)))

MIN_CHUNK_WORDS    = int(os.getenv("MIN_CHUNK_WORDS", "20"))
MIN_UNIQUE_RATIO   = float(os.getenv("MIN_UNIQUE_RATIO", "0.35"))
MIN_ALPHA_RATIO    = float(os.getenv("MIN_ALPHA_RATIO", "0.22"))  # harf/karakter oranı

DOC_TYPE_DEFAULT = os.getenv("DOC_TYPE", "kullanici_kilavuzu")
DEPARTMENT_DEFAULT = os.getenv("DEPARTMENT", "FIP")
LANGUAGE_DEFAULT = os.getenv("DOC_LANGUAGE", "tr")

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?…])\s+")

def relax_thresholds_for_doc():
    """Bu belge için eşiği adaptif gevşet (recall boost)."""
    global MIN_CHUNK_WORDS, MIN_UNIQUE_RATIO, MIN_ALPHA_RATIO
    MIN_CHUNK_WORDS = max(8, int(MIN_CHUNK_WORDS * 0.7))     # %30 düşür
    MIN_UNIQUE_RATIO = max(0.18, MIN_UNIQUE_RATIO * 0.8)     # %20 gevşet
    MIN_ALPHA_RATIO  = max(0.15, MIN_ALPHA_RATIO * 0.9)      # %10 gevşet

# ==========================
# Yardımcılar
# ==========================
def db_connect():
    return psycopg2.connect(**DB)

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def read_pdf_texts(pdf_path: str) -> List[Tuple[int, str]]:
    items = []
    if HAS_PYMUPDF:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            # block bazlı al -> satır kırıkları çok daha az, başlık belirgindir
            blocks = page.get_text("blocks") or []
            # (x0,y0,x1,y1, text, block_no, block_type)
            blocks = sorted(blocks, key=lambda b: (round(b[1],1), round(b[0],1)))
            parts = []
            for b in blocks:
                txt = (b[4] or "").strip()
                if txt:
                    parts.append(txt)
            txt = normalize_space(" \n ".join(parts))
            items.append((i, txt))
        doc.close()
    else:
        # pypdf fallback (eskisi gibi)
        reader = PdfReader(pdf_path)
        for i, p in enumerate(reader.pages, start=1):
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            txt = normalize_space(txt)
            items.append((i, txt))
    return items


def is_header_metadata(text: str) -> bool:
    """Kapak/metadata bloklarını tespit (esnek)."""
    t = text.lower()
    pats = [
        r"kullan[ıi]c[ıi] k[ıi]lavuzu", r"ilk yay[ıi]n tarihi", r"rev[ıi]ze tarihi",
        r"rev[ıi]ze no", r"dok[üu]man no", r"yay[ıi]n kayna[ğg][ıi]", r"varl[ıi]k s[ıi]n[ıi]f[ıi]",
        r"troy sistemi",
    ]
    matches = sum(1 for p in pats if re.search(p, t))
    wc = len(text.split())
    # kısa ve yoğun meta kalıpları → header
    if wc < 120 and matches >= 4:
        return True
    first_150 = text[:150].lower()
    first_matches = sum(1 for p in pats if re.search(p, first_150))
    if wc < 80 and first_matches >= 3:
        return True
    return False

# ==========================
# Model & tokenizer
# ==========================
_model = None
_tokenizer = None
_dim_cache = None

def get_model_and_tokenizer():
    global _model, _tokenizer, _dim_cache
    if _model is None:
        print(f"ℹ️  Model yükleniyor: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        _model.max_seq_length = MAX_SEQ_LENGTH
        print(f"   max_seq_length = {_model.max_seq_length}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _dim_cache = _model.get_sentence_embedding_dimension()
        print(f"ℹ️  Embedding boyutu: {_dim_cache}")
        print(f"ℹ️  Min chunk kelime: {MIN_CHUNK_WORDS}")
        print(f"ℹ️  Min benzersizlik: {MIN_UNIQUE_RATIO}")
        print(f"ℹ️  Token hedefi: {TARGET_TOKENS}, maksimum: {MAX_TOKENS}, overlap: {OVERLAP_TOKENS}")
    return _model, _tokenizer, _dim_cache

def token_len(text: str) -> int:
    _, tok, _ = get_model_and_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))

def split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Cümle odaklı, gerekirse token penceresi ile böl."""
    _, tok, _ = get_model_and_tokenizer()
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    chunks, buf = [], []
    cur = 0

    def flush():
        nonlocal chunks, buf
        if buf:
            chunks.append(" ".join(buf).strip())

    for s in sents:
        l = len(tok.encode(s, add_special_tokens=False))
        if l > max_tokens:
            flush(); buf = []; cur = 0
            ids = tok.encode(s, add_special_tokens=False)
            start = 0
            while start < len(ids):
                end = min(start + max_tokens, len(ids))
                piece = tok.decode(ids[start:end])
                chunks.append(piece.strip())
                if end == len(ids): break
                start = end - overlap_tokens if overlap_tokens>0 else end
                if start < 0: start = 0
            continue

        if cur + l <= max_tokens:
            buf.append(s); cur += l
        else:
            flush()
            # overlap: son cümlelerden token bazlı pencere
            if overlap_tokens > 0 and len(buf) > 0:
                back = []
                t = 0
                for sent in reversed(buf):
                    tl = len(tok.encode(sent, add_special_tokens=False))
                    if t + tl > overlap_tokens: break
                    back.append(sent); t += tl
                buf = list(reversed(back))
                cur = sum(len(tok.encode(x, add_special_tokens=False)) for x in buf)
            else:
                buf = []; cur = 0
            buf.append(s); cur += l
    flush()
    return [c for c in chunks if c.strip()]

def chunk_page(text: str, page_no: int) -> List[str]:
    if not text or len(text.split()) < MIN_CHUNK_WORDS:
        return []
    # header bloklarını kısa/yoğun ise ele
    if is_header_metadata(text) and token_len(text) < 120:
        return []
    return split_by_tokens(text, MAX_TOKENS, OVERLAP_TOKENS)

def to_vec_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

# ==========================
# Şema / boyut kontrol
# ==========================
def ensure_schema(cur, dim: int):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
    # documents
    cur.execute("""
    CREATE TABLE IF NOT EXISTS public.documents (
      id BIGSERIAL PRIMARY KEY,
      title VARCHAR(500) NOT NULL,
      file_path TEXT NOT NULL UNIQUE,
      document_type VARCHAR(100),
      department VARCHAR(100),
      language VARCHAR(10) DEFAULT 'tr',
      content_hash VARCHAR(64),
      status VARCHAR(20) DEFAULT 'active',
      created_at TIMESTAMP DEFAULT NOW(),
      updated_at TIMESTAMP DEFAULT NOW()
    );
    """)
    # sections
    cur.execute("""
    CREATE TABLE IF NOT EXISTS public.document_sections (
      id BIGSERIAL PRIMARY KEY,
      document_id BIGINT REFERENCES public.documents(id) ON DELETE CASCADE,
      section_title TEXT,
      content TEXT NOT NULL,
      page_number INT,
      tsv tsvector
    );
    """)
    # trigger for tsv
    cur.execute("""
    CREATE OR REPLACE FUNCTION sections_tsv_trigger() RETURNS trigger AS $$
    BEGIN
      NEW.tsv := to_tsvector('turkish', coalesce(NEW.section_title,'') || ' ' || lower(unaccent(coalesce(NEW.content,''))));
      RETURN NEW;
    END$$ LANGUAGE plpgsql;
    """)
    cur.execute("DROP TRIGGER IF EXISTS trg_sections_tsv ON public.document_sections;")
    cur.execute("""
    CREATE TRIGGER trg_sections_tsv
    BEFORE INSERT OR UPDATE ON public.document_sections
    FOR EACH ROW EXECUTE FUNCTION sections_tsv_trigger();
    """)

    # embeddings: var ise boyutu tip metninden oku
    cur.execute("""
    SELECT format_type(atttypid, atttypmod) AS typ
    FROM pg_attribute
    WHERE attrelid='public.document_embeddings'::regclass
      AND attname='embedding' AND NOT attisdropped
    """)
    row = cur.fetchone()
    if row and row[0]:
        m = re.search(r'vector\((\d+)\)', row[0])
        existing_dim = int(m.group(1)) if m else None
    else:
        existing_dim = None

    if existing_dim is None:
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS public.document_embeddings (
          id BIGSERIAL PRIMARY KEY,
          section_id BIGINT REFERENCES public.document_sections(id) ON DELETE CASCADE,
          embedding vector({dim}) NOT NULL,
          model_name VARCHAR(120) NOT NULL,
          created_at TIMESTAMP DEFAULT NOW()
        );
        """)
    elif existing_dim != dim:
        raise RuntimeError(
            f"document_embeddings.embedding dim uyumsuz: mevcut={existing_dim}, gereken={dim}. "
            f"Tabloyu boşaltıp/düşürüp {dim} ile yeniden oluşturun."
        )

    cur.execute("""
    DO $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname='idx_embeddings_vector') THEN
        CREATE INDEX idx_embeddings_vector
        ON public.document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
      END IF;
      IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname='idx_sections_tsv') THEN
        CREATE INDEX idx_sections_tsv ON public.document_sections USING gin(tsv);
      END IF;
    END$$;
    """)

# ==========================
# DB işlemleri
# ==========================
def upsert_document(cur, meta: Dict[str, Any]) -> Tuple[int, bool]:
    cur.execute("SELECT id, content_hash FROM public.documents WHERE file_path=%s", (meta["file_path"],))
    row = cur.fetchone()
    if row:
        doc_id, old_hash = row
        if old_hash == meta["content_hash"]:
            # değişiklik yok
            return doc_id, False
        # güncelle
        cur.execute("""
          UPDATE public.documents
          SET title=%s, document_type=%s, department=%s, language=%s,
              content_hash=%s, updated_at=NOW()
          WHERE id=%s
        """, (meta["title"], meta["document_type"], meta["department"], meta["language"], meta["content_hash"], doc_id))
        # eski bölümleri/embeddingleri sil
        cur.execute("DELETE FROM public.document_embeddings USING public.document_sections s WHERE document_embeddings.section_id=s.id AND s.document_id=%s", (doc_id,))
        cur.execute("DELETE FROM public.document_sections WHERE document_id=%s", (doc_id,))
        return doc_id, True
    else:
        cur.execute("""
          INSERT INTO public.documents (title, file_path, document_type, department, language, content_hash, status)
          VALUES (%s,%s,%s,%s,%s,%s,'active')
          RETURNING id
        """, (meta["title"], meta["file_path"], meta["document_type"], meta["department"], meta["language"], meta["content_hash"]))
        return cur.fetchone()[0], True

def insert_sections(cur, document_id: int, chunks: List[Dict[str, Any]]) -> List[int]:
    ids = []
    for ch in chunks:
        cur.execute("""
          INSERT INTO public.document_sections (document_id, section_title, content, page_number)
          VALUES (%s,%s,%s,%s)
          RETURNING id
        """, (document_id, ch.get("title"), ch["text"], ch.get("page_number")))
        ids.append(cur.fetchone()[0])
    return ids

def insert_embeddings(cur, section_ids: List[int], vectors: List[List[float]], model_name: str):
    values = [(sid, to_vec_literal(vec), model_name) for sid, vec in zip(section_ids, vectors)]
    execute_values(cur, """
      INSERT INTO public.document_embeddings (section_id, embedding, model_name)
      VALUES %s
    """, values, template="(%s, %s::vector, %s)")

# ==========================
# PDF işleme
# ==========================
def process_pdf(path: str, doc_type: str, department: str, language: str) -> Tuple[int,int,bool]:
    title = os.path.splitext(os.path.basename(path))[0]
    pages = read_pdf_texts(path)
    full_text = " ".join(t for _, t in pages)
    content_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

    meta = dict(
        title=title,
        file_path=os.path.abspath(path),
        document_type=doc_type,
        department=department,
        language=language,
        content_hash=content_hash,
    )

    model, _, dim = get_model_and_tokenizer()

    conn = db_connect()
    cur = conn.cursor()
    try:
        ensure_schema(cur, dim)
        conn.commit()

        doc_id, changed = upsert_document(cur, meta)
        conn.commit()

        # chunk’ları topla
        chunks: List[Dict[str,Any]] = []
        page_uniq_stats: Dict[int, float] = {}

        # ... chunks doldurulduktan hemen sonra:
        initial_total = len(chunks)

        if initial_total == 0:
            # aşırı sert olmuş → doküman için eşiği gevşet ve sayfaları yeniden işle
            relax_thresholds_for_doc()
            chunks.clear()
            page_uniq_stats.clear()
            for pno, ptxt in pages:
                pcs = chunk_page(ptxt, pno)
                # (aynı sayfa uniq hesabı…)
                uniq_vals = []
                for c in pcs:
                    words = c.split()
                    if words:
                        uniq_vals.append(len(set(words))/len(words))
                page_uniq_stats[pno] = sum(uniq_vals)/len(uniq_vals) if uniq_vals else 1.0
                for c in pcs:
                    if len(c.split()) < MIN_CHUNK_WORDS:
                        continue
                    alpha_ratio = sum(ch.isalpha() for ch in c) / max(len(c),1)
                    if alpha_ratio < MIN_ALPHA_RATIO:
                        continue
                    words = c.split()
                    uniq = len(set(words))/len(words) if words else 1.0
                    min_uniq = MIN_UNIQUE_RATIO * (0.8 if page_uniq_stats[pno] < 0.5 else 1.0)
                    if uniq < min_uniq:
                        continue
                    chunks.append(dict(title=f"Sayfa {pno}", text=c, page_number=pno))

        if len(chunks) == 0:
            print(f"  ⚠️  Tüm chunk'lar filtrelendi: {os.path.basename(path)} (relax sonrası da)")
            return (0, 0, changed)


        # embed et (normalize=True → cosine için ideal)
        texts = [c["text"] for c in chunks]
        vecs = model.encode(
            texts, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False
        )

        # sayfa içi semantik dedup (çok benzerleri at)
        import numpy as np
        DEDUP_SIM = float(os.getenv("DEDUP_SIM", "0.992"))
        DEDUP_WINDOW = int(os.getenv("DEDUP_WINDOW", "3"))

        kept_chunks, kept_vecs = [], []
        for ch, v in zip(chunks, vecs):
            v = v.astype(float)
            ok = True
            for u in kept_vecs[-max(1, DEDUP_WINDOW):]:
                sim = float((v @ u) / (np.linalg.norm(v)*np.linalg.norm(u) + 1e-8))
                if sim > DEDUP_SIM:
                    ok = False; break
            if ok:
                kept_chunks.append(ch)
                kept_vecs.append(v)
        chunks, vecs = kept_chunks, kept_vecs
        # DB’ye yaz
        sec_ids = insert_sections(cur, doc_id, chunks)
        insert_embeddings(cur, sec_ids, [v.tolist() for v in vecs], MODEL_NAME)
        conn.commit()

        print(f"  ✅ {os.path.basename(path)}")
        print(f"     doc_id={doc_id}, chunks={len(chunks)}, filtered={(len(texts)-len(chunks))} ({'new/updated' if changed else 'cached'})")
        return (1, len(chunks), changed)
    finally:
        cur.close()
        conn.close()

# ==========================
# CLI
# ==========================
def main():
    ap = argparse.ArgumentParser(description="PDF → pgvector ingest (HF, token-bazlı)")
    ap.add_argument("--dir", required=True, help="PDF klasörü")
    ap.add_argument("--doc-type", default=DOC_TYPE_DEFAULT)
    ap.add_argument("--department", default=DEPARTMENT_DEFAULT)
    ap.add_argument("--language", default=LANGUAGE_DEFAULT)
    args = ap.parse_args()

    get_model_and_tokenizer()  # loglar

    pdfs = [f for f in os.listdir(args.dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("⚠️  Klasörde PDF yok.")
        return

    total_docs = 0
    total_chunks = 0
    for f in pdfs:
        d, s, _ = process_pdf(os.path.join(args.dir, f), args.doc_type, args.department, args.language)
        total_docs += d
        total_chunks += s

    print("\n" + "="*60)
    print("📊 ÖZET:")
    print(f"   Dökümanlar: {total_docs}")
    print(f"   Kaydedilen chunk'lar: {total_chunks}")
    print("="*60)

if __name__ == "__main__":
    main()
