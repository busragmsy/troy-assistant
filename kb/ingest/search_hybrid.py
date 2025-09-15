# kb/ingest/search_hybrid.py
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np
import os, yaml, regex as re

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml"), "r", encoding="utf-8"))
PG = dict(host="localhost", port=5432, dbname="kb", user="troy", password="troy")

MODEL = SentenceTransformer(CFG["model"]["name"], device="cpu")

def embed_query(q: str) -> np.ndarray:
    v = MODEL.encode([f"query: {q}"], normalize_embeddings=CFG["model"]["normalize"])[0]
    return v.astype(np.float32)

def to_pgvec_str(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"

def to_tsquery_safe(q: str) -> str:
    # Çok agresif değil; boşlukları & ile birleştir
    q = re.sub(r"[^\p{L}\p{N}\s]", " ", q)  # noktalama -> boşluk
    q = re.sub(r"\s+", " & ", q.strip())
    return q or "''"

def search_hybrid(menu_item: str, role: str, question: str, k: int = 5,
                  alpha_vector: float = 0.6, alpha_bm25: float = 0.4):
    qvec = to_pgvec_str(embed_query(question))
    tsq = to_tsquery_safe(question)

    con = psycopg2.connect(**PG)
    with con, con.cursor() as cur:
        cur.execute(f"""
        WITH q AS (
          SELECT to_tsquery('turkish', %s) AS tsq
        ),
        cand AS (
          SELECT id, title_full, content, source, section, page_start, page_end, embedding,
                 ts_rank_cd(tsv, (SELECT tsq FROM q)) AS bm25
          FROM knowledge_chunks
          WHERE is_active
            AND menu_item = %s
            AND (array_length(allowed_roles,1) IS NULL OR %s = ANY(allowed_roles))
        ),
        v AS (
          SELECT id, 1 - (embedding <=> %s::vector) AS vscore
          FROM cand
        )
        SELECT c.title_full,
               left(c.content, 600) AS snippet,
               c.source,
               c.section,
               c.page_start, c.page_end,
               (%s * v.vscore + %s * c.bm25) AS score,
               v.vscore, c.bm25
        FROM cand c
        JOIN v USING (id)
        ORDER BY score DESC
        LIMIT %s;
        """, (tsq, menu_item, role, qvec, alpha_vector, alpha_bm25, k))

        rows = cur.fetchall()

    for t, snip, src, sec, p1, p2, score, vscore, bm25 in rows:
        print(f"\n[{score:.3f} | v={vscore:.3f}, bm25={bm25:.3f}] {t}  ({src} s:{p1}-{p2})\n{snip} ...")

    print(f"\nTop-{k} bitti.")

if __name__ == "__main__":
    search_hybrid(
        "ilk_fiyat_revize", "PricingAnalyst",
        "İlk fiyat revize ekranında zorunlu alanlar ve indirim iptal adımları",
        k=5, alpha_vector=0.6, alpha_bm25=0.4
    )
