# kb/ingest/ingest.py
import os, hashlib
import yaml
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from preprocess_pdf import parse_pdf_advanced

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml"), "r", encoding="utf-8"))

PG = dict(
    host="localhost", port=5432,
    dbname="kb", user="troy", password="troy"
)

MODEL = SentenceTransformer(CFG["model"]["name"], device="cpu")

def content_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def embed_passage(x: str) -> np.ndarray:
    # e5 için passage: öneki
    v = MODEL.encode([f"passage: {x}"], normalize_embeddings=CFG["model"]["normalize"])[0]
    return v.astype(np.float32)

def inject_title(menu_item: str, section_title: str, text: str) -> str:
    return f"[MENU: {menu_item}] [BÖLÜM: {section_title}]\n{text}"

def load_doc_map():
    """config.yaml -> doc_map: { "LC.FIP.KL.005": {menu_item, roles, version, lang} }"""
    return CFG.get("doc_map", {})

def ingest_one(pdf_path: str, meta: dict):
    menu_item = meta.get("menu_item")
    allowed_roles = meta.get("roles", []) or None
    doc_version = meta.get("version", "")
    lang = meta.get("lang", "tr")

    # bölüm-odaklı parse + chunk
    rows = parse_pdf_advanced(pdf_path, lang=lang,
                              max_tokens=CFG["chunk"]["max_tokens"],
                              overlap=CFG["chunk"]["overlap"])
    # DB insert
    buf = []
    for r in rows:
        section = r["section"]
        page_start, page_end = r["page_start"], r["page_end"]
        text = r["text"].strip()

        # title injection
        title_full = f"{menu_item} · {section}"
        text_for_embed = inject_title(menu_item, section, text)
        emb = embed_passage(text_for_embed)

        buf.append((
            True,                   # is_active
            menu_item,              # menu_item
            section,                # section (kısa)
            title_full,             # title_full
            text,                   # content (ham)
            os.path.basename(pdf_path),  # source
            doc_version,            # version
            allowed_roles,          # allowed_roles (ARRAY)
            int(r["token_count"]),  # token_count
            lang,                   # lang
            int(page_start or 0),
            int(page_end or 0),
            content_hash(text_for_embed),  # content_hash (enjekte edilmiş içerikte)
            emb.tolist()            # embedding (vector)
        ))

    if not buf:
        print(f"SKIP: {os.path.basename(pdf_path)} -> 0 chunk")
        return 0

    con = psycopg2.connect(**PG)
    with con, con.cursor() as cur:
        execute_values(cur, """
            INSERT INTO knowledge_chunks
                (is_active, menu_item, section, title_full, content, source, version,
                 allowed_roles, token_count, lang, page_start, page_end, content_hash, embedding)
            VALUES %s
            ON CONFLICT (source, page_start, page_end, content_hash) DO NOTHING;
        """, buf)
    con.close()
    return len(buf)

def main():
    doc_map = load_doc_map()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    total = 0
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(data_dir, fn)
        # doc_id eşle (örn. "LC.FIP.KL.005" → dosya adında geçiyor mu?)
        doc_id = None
        for k in doc_map.keys():
            if k in fn:
                doc_id = k; break
        if not doc_id:
            print(f"SKIP (doc_id eşleşmedi): {fn}")
            continue
        count = ingest_one(pdf_path, doc_map[doc_id])
        print(f"OK: {fn} -> {count} chunks")
        total += count
    print(f"TOPLAM chunk: {total}")

if __name__ == "__main__":
    main()
