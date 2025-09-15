import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np

PG = dict(host="localhost", port=5432, dbname="kb", user="troy", password="troy")
MODEL = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")

def embed_query(q: str):
    v = MODEL.encode([f"query: {q}"], normalize_embeddings=True)[0]
    return np.asarray(v, dtype=np.float32)

def to_pgvector_str(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"

def search(menu_item: str, role: str, question: str, topk: int = 6):
    qvec = embed_query(question)
    qvec_str = to_pgvector_str(qvec)

    con = psycopg2.connect(**PG)
    with con, con.cursor() as cur:
        cur.execute("""
          SELECT title,
                 left(content, 500) AS snippet,
                 source,
                 1 - (embedding <=> %s::vector) AS score
          FROM knowledge_chunks
          WHERE is_active
            AND menu_item = %s
            AND (array_length(allowed_roles,1) IS NULL OR %s = ANY(allowed_roles))
          ORDER BY embedding <=> %s::vector
          LIMIT %s;
        """, (qvec_str, menu_item, role, qvec_str, topk))

        rows = cur.fetchall()

    for r in rows:
        print(f"\n[{r[3]:.3f}] {r[0]}  ({r[2]})\n{r[1]} ...")

    print(f"\nTop-{topk} bitti.")

if __name__ == "__main__":
    search(
        "ilk_fiyat_revize", "PricingAnalyst",
        "İlk fiyat revize ekranında indirim iptal adımları nelerdir?",
        topk=5
    )
