# kb/search/smoke_vector_check.py
import os
import argparse
import psycopg2
from sentence_transformers import SentenceTransformer

def db_connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "kb"),
        user=os.getenv("DB_USER", "troy"),
        password=os.getenv("DB_PASSWORD", "troy1234"),
    )

SQL_VECTOR = """
SELECT
    d.title,
    d.document_type,
    s.section_title,
    s.page_number,
    s.content,
    (1 - (e.embedding <=> %s::vector)) AS similarity
FROM document_embeddings e
JOIN document_sections s ON e.section_id = s.id
JOIN documents d ON s.document_id = d.id
WHERE d.status = 'active'
ORDER BY e.embedding <=> %s::vector
LIMIT %s;
"""

# basit hibrit: FTS ile ilk N aday -> bu adayları vektör sırasına göre tekrar sırala
SQL_FTS_CANDIDATES = """
WITH fts AS (
    SELECT
        s.id AS section_id,
        ts_rank(s.tsv, plainto_tsquery('turkish', %s)) AS rank
    FROM document_sections s
    WHERE s.tsv @@ plainto_tsquery('turkish', %s)
    ORDER BY rank DESC
    LIMIT %s
)
SELECT
    d.title,
    d.document_type,
    s.section_title,
    s.page_number,
    s.content,
    (1 - (e.embedding <=> %s::vector)) AS similarity
FROM fts
JOIN document_sections s ON s.id = fts.section_id
JOIN document_embeddings e ON e.section_id = s.id
JOIN documents d ON d.id = s.document_id
WHERE d.status = 'active'
ORDER BY e.embedding <=> %s::vector
LIMIT %s;
"""

def run(query: str, limit: int, mode: str = "vector", fts_candidates: int = 50):
    model_name = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-large")
    print(f"ℹ️ Model yükleniyor: {model_name}\n")
    model = SentenceTransformer(model_name)
    # ingest sırasında normalize_embeddings=True kullandık; sorguyu da normalize edelim
    qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()

    conn = db_connect()
    cur = conn.cursor()

    print(f"🔍 Sorgu: {query}\n")

    if mode == "vector":
        cur.execute(SQL_VECTOR, (qvec, qvec, limit))
    elif mode == "hybrid":
        cur.execute(SQL_FTS_CANDIDATES, (query, query, fts_candidates, qvec, qvec, limit))
    else:
        raise ValueError("mode must be 'vector' or 'hybrid'")

    rows = cur.fetchall()
    if not rows:
        print("➖ Sonuç yok.")
        return

    for i, (title, dtype, sect_title, page, content, sim) in enumerate(rows, 1):
        print(f"{i}. {title}")
        if sect_title:
            print(f"   📍 {dtype} | {sect_title} | s.{page or '-'}")
        else:
            print(f"   📍 {dtype} | s.{page or '-'}")
        print(f"   💯 Skor: {sim:.3f}")
        snippet = (content or "").replace("\n", " ")
        print(f"   📝 {snippet[:300]}{'...' if len(snippet)>300 else ''}\n")

    cur.close()
    conn.close()

def main():
    ap = argparse.ArgumentParser(description="Vector/Hybrid retrieval smoke test")
    ap.add_argument("--query", required=True, help="Kullanıcı sorgusu")
    ap.add_argument("--limit", type=int, default=5, help="Döndürülecek sonuç sayısı")
    ap.add_argument("--mode", choices=["vector", "hybrid"], default="vector",
                    help="Sadece vektör veya FTS+vektör hibrit")
    ap.add_argument("--fts-candidates", type=int, default=50,
                    help="Hibrit modda FTS ile alınacak aday sayısı")
    args = ap.parse_args()
    run(args.query, args.limit, args.mode, args.fts_candidates)

if __name__ == "__main__":
    main()
