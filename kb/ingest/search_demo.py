import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import numpy as np

PG = dict(host="localhost", port=5432, dbname="kb", user="troy", password="troy")
MODEL = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")

def embed_query(q: str):
    v = MODEL.encode([f"query: {q}"], normalize_embeddings=True)[0]
    return np.asarray(v, dtype=np.float32).tolist()

def search(menu_item: str, role: str, question: str, topk: int = 6):
    qvec = embed_query(question)

    con = psycopg2.connect(**PG)
    register_vector(con) 
    with con, con.cursor() as cur:
        cur.execute("""
          SELECT title,
                 left(content, 500) AS snippet,
                 source,
                 1 - (embedding <=> %s) AS score
          FROM knowledge_chunks
          WHERE is_active
            AND menu_item = %s
            AND (array_length(allowed_roles,1) IS NULL OR %s = ANY(allowed_roles))
          ORDER BY embedding <=> %s
          LIMIT %s;
        """, (qvec, menu_item, role, qvec, topk))

        rows = cur.fetchall()
    con.close()

    for r in rows:
        print(f"\n[{r[3]:.3f}] {r[0]}  ({r[2]})\n{r[1]} ...")

    print(f"\nTop-{topk} bitti.")

if __name__ == "__main__":
    search(
        "ilk_fiyat_revize", "PricingAnalyst",
        "İlk fiyat revize ekranında indirim iptal adımları nelerdir?",
        topk=5
    )
