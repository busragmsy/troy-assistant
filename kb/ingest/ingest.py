import os, uuid, re
import numpy as np, psycopg2
from pypdf import PdfReader

# Postgres container'ı 5433'e map'lersen portu 5433 yap
PG = dict(host="localhost", port=5432, dbname="kb", user="troy", password="troy")

def chunk_text(text, size=900, overlap=150):
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i:i+size]))
        i += max(1, size - overlap)
    return out

def fake_embed(txt:str)->np.ndarray:
    rng = np.random.default_rng(abs(hash(txt)) % (2**32))
    return rng.random(1024, dtype=np.float32)

def upsert(con, row):
    emb = fake_embed(row["content"]).tolist()
    with con.cursor() as cur:
        cur.execute("""
        INSERT INTO knowledge_chunks
          (id, doc_id, menu_item, section, title, content, embedding, allowed_roles, doc_version, source)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (id) DO NOTHING
        """,
        (str(uuid.uuid4()), row["doc_id"], row["menu_item"], row["section"],
         row["title"], row["content"], emb, row["roles"], row["version"], row["source"]))

def parse_pdf(path):
    reader = PdfReader(path)
    text = "\n".join([p.extract_text() or "" for p in reader.pages])
    text = re.sub(r"\n{2,}", "\n", text)
    return text

def main():
    pdf = "../data/LC.FIP.KL.005_0___TROY_SISTEMI_ILK_FIYAT_REVIZE_EKRANI_KULLANICI_KILAVUZU.pdf"
    assert os.path.exists(pdf), f"PDF yok: {pdf}"

    row_common = dict(
        doc_id    = "LC.FIP.KL.005",
        menu_item = "ilk_fiyat_revize",
        version   = "v0",
        roles     = ["PricingAnalyst","Admin"],
        source    = os.path.basename(pdf)
    )

    text = parse_pdf(pdf)
    parts = chunk_text(text, size=900, overlap=150)

    con = psycopg2.connect(**PG)
    for i, content in enumerate(parts):
        row = dict(**row_common)
        row["section"] = "auto"
        row["title"]   = f"İlk Fiyat Revize [p{i}]"
        row["content"] = content
        upsert(con, row)
    con.commit(); con.close()
    print(f"Indexed {len(parts)} chunks from {row_common['source']}")

if __name__ == "__main__":
    main()
