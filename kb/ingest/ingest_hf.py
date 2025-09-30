import os, re, argparse, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict

import psycopg2
from psycopg2.extras import execute_values
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# ======= Config =======
MODEL_NAME = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-large")  # 1024-dim
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
TARGET_TOKENS = int(os.getenv("TARGET_TOKENS", "520"))
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "800"))
OVERLAP_TOKS  = int(os.getenv("OVERLAP_TOKENS", "64"))

DB = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "kb"),
    user=os.getenv("DB_USER", "troy"),
    password=os.getenv("DB_PASSWORD", "troy1234"),
)

# ======= Utils =======
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def norm_spaces(s: str) -> str:
    # satır sonu tire birleştirme: "yan-\nlış" -> "yanlış"
    s = re.sub(r"-\n(?=\w)", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" ?\n ?", "\n", s)
    return s.strip()

def read_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    r = PdfReader(pdf_path)
    out = []
    for i, p in enumerate(r.pages, start=1):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        out.append((i, norm_spaces(t)))
    return out

@dataclass
class Chunk:
    text: str
    page: int
    title: str

def token_chunks(text: str, tokenizer, target: int, maxi: int, overlap: int) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, cur, cur_len = [], "", 0
    for para in paras:
        ids = tokenizer(para, add_special_tokens=False)["input_ids"]
        n = len(ids)
        if n > maxi:
            sents = re.split(r"(?<=[\.\!\?…])\s+", para)
            for s in sents:
                s = s.strip()
                if not s: continue
                l = len(tokenizer(s, add_special_tokens=False)["input_ids"])
                if cur_len + l > target:
                    if cur: chunks.append(cur.strip())
                    cur, cur_len = s, l
                else:
                    cur = (cur + " " + s).strip() if cur else s
                    cur_len += l
        else:
            if cur_len + n > target:
                if cur: chunks.append(cur.strip())
                cur, cur_len = para, n
            else:
                cur = (cur + "\n\n" + para).strip() if cur else para
                cur_len += n
    if cur: chunks.append(cur.strip())
    # minimum kırıntıları at
    return [c for c in chunks if len(c.split()) >= 20]

def connect():
    return psycopg2.connect(**DB)

def ensure_schema(cur, dim: int):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")

    # embedding tablosu boyut kontrol/oluşturma
    cur.execute("""
    SELECT format_type(atttypid, atttypmod) AS typ
    FROM pg_attribute
    WHERE attrelid='public.document_embeddings'::regclass
      AND attname='embedding' AND NOT attisdropped
    """)
    row = cur.fetchone()

    def parse_declared_dim(type_text: str) -> int | None:
        # "vector(1024)" -> 1024
        if not type_text:
            return None
        m = re.search(r'vector\((\d+)\)', type_text)
        return int(m.group(1)) if m else None

    existing_dim = parse_declared_dim(row[0]) if row and row[0] else None

    if existing_dim is None:
        # tablo yoksa/kolon yoksa oluştur
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
            f"Tablo boşsa: DROP TABLE public.document_embeddings; sonra ingest’i tekrar çalıştırın."
        )

    # cosine index
    cur.execute("""
    DO $$
    BEGIN
      IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname='idx_embeddings_vector') THEN
        CREATE INDEX idx_embeddings_vector
        ON public.document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
      END IF;
    END$$;""")


def upsert_document(cur, meta) -> Tuple[int, bool]:
    cur.execute("SELECT id, content_hash FROM documents WHERE file_path=%s", (meta["file_path"],))
    row = cur.fetchone()
    if row:
        if row[1] == meta["content_hash"]:
            return row[0], False
        cur.execute("""
        UPDATE documents SET title=%s, document_type=%s, department=%s, document_code=%s,
               version_number=%s, language='tr', content_hash=%s, updated_at=NOW()
        WHERE id=%s
        """, (meta["title"], meta["document_type"], meta["department"], meta["document_code"],
              meta["version_number"], meta["content_hash"], row[0]))
        return row[0], True
    cur.execute("""
    INSERT INTO documents(title, document_type, file_name, file_path, department,
                          document_code, version_number, language, content_hash)
    VALUES (%s,%s,%s,%s,%s,%s,%s,'tr',%s)
    RETURNING id
    """, (meta["title"], meta["document_type"], meta["file_name"], meta["file_path"],
          meta["department"], meta["document_code"], meta["version_number"], meta["content_hash"]))
    return cur.fetchone()[0], True

def insert_sections_and_embeddings(cur, doc_id: int, chunks: List[Chunk], vectors: List[List[float]], model_name: str):
    # sections
    sec_ids = []
    for ch in chunks:
        cur.execute("""
        INSERT INTO document_sections(document_id, section_title, content, page_number, word_count, content_hash)
        VALUES (%s,%s,%s,%s,%s,%s) RETURNING id
        """, (doc_id, ch.title, ch.text, ch.page, len(ch.text.split()), sha256(ch.text)))
        sec_ids.append(cur.fetchone()[0])
    # embeddings
    def vec_lit(v): return "[" + ",".join(f"{x:.6f}" for x in v) + "]"
    vals = [(sid, vec_lit(vec), model_name) for sid, vec in zip(sec_ids, vectors)]
    execute_values(cur,
        "INSERT INTO document_embeddings(section_id, embedding, model_name) VALUES %s",
        vals, template="(%s, %s::vector, %s)"
    )

def extract_metadata(full_text: str, file_name: str, file_path: str) -> Dict:
    code = None
    m = re.search(r"(LC\.[A-ZÇĞİÖŞÜ]{3}\.[A-ZÇĞİÖŞÜ]{2}\.\d{3})", full_text)
    if m: code = m.group(1)
    dept = (code.split(".")[1] if code else "GENEL")
    title = os.path.splitext(file_name)[0]
    ver = None
    mv = re.search(r"REV[İI]ZE NO\s*:? ?(\d+)", full_text, re.I)
    if mv: ver = mv.group(1)
    return dict(
        title=title,
        document_type=("kullanici_kilavuzu" if "KULLANICI KILAVUZU" in full_text else "genel"),
        file_name=file_name,
        file_path=file_path,
        department=dept,
        document_code=code,
        version_number=ver or "0",
        content_hash=sha256(full_text),
    )

def run_one(pdf_path: str, model, tokenizer, dim: int) -> Tuple[int, int, bool]:
    pages = read_pdf_pages(pdf_path)
    full_text = "\n\n".join(t for _, t in pages)
    meta = extract_metadata(full_text, os.path.basename(pdf_path), os.path.abspath(pdf_path))

    # sayfa başlığı tahmini + token chunking
    chunks: List[Chunk] = []
    for page, text in pages:
        head = re.findall(r"^[A-ZÇĞİÖŞÜ0-9][^\n]{0,80}$", text, flags=re.M)
        title = head[0].strip() if head else f"Sayfa {page}"
        for piece in token_chunks(text, tokenizer, TARGET_TOKENS, MAX_TOKENS, OVERLAP_TOKS):
            chunks.append(Chunk(text=piece, page=page, title=title))

    if not chunks:
        print(f"⚠️  Boş/okunamadı: {os.path.basename(pdf_path)}")
        return (0, 0, False)

    texts = [c.text for c in chunks]
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True).tolist()  # cosine için normalize

    with connect() as conn:
        with conn.cursor() as cur:
            ensure_schema(cur, dim)
            doc_id, changed = upsert_document(cur, meta)
            insert_sections_and_embeddings(cur, doc_id, chunks, vecs, MODEL_NAME)
        conn.commit()

    print(f"✅ {os.path.basename(pdf_path)} -> doc_id={doc_id}, chunks={len(chunks)} ({'new/updated' if changed else 'cached'})")
    return (1, len(chunks), changed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="PDF klasörü (örn. kb/data/docs)")
    args = ap.parse_args()

    print(f"ℹ️  model yükleniyor: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "510")) 
    print("max_seq_length =", model.max_seq_length)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dim = len(model.encode(["probe"], convert_to_numpy=True)[0])
    print(f"ℹ️  dim={dim}")

    pdfs = [f for f in os.listdir(args.dir) if f.lower().endswith(".pdf")]
    if not pdfs:
        print("⚠️  PDF bulunmadı")
        return

    docs = secs = 0
    for f in pdfs:
        d, s, _ = run_one(os.path.join(args.dir, f), model, tokenizer, dim)
        docs += d; secs += s
    print(f"\nSummary: docs={docs}, sections={secs}")

if __name__ == "__main__":
    main()
