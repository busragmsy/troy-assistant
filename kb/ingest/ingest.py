# kb/ingest/ingest.py
import os, hashlib
import yaml
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from preprocess_pdf import parse_pdf_advanced

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml"), "r", encoding="utf-8"))

PG = dict(host="localhost", port=5432, dbname="kb", user="troy", password="troy")

MODEL = SentenceTransformer(CFG["model"]["name"], device="cpu")

def content_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def embed_passage(x: str) -> np.ndarray:
    # e5 için passage öneki (normalize flag'i config'ten)
    v = MODEL.encode([f"passage: {x}"], normalize_embeddings=CFG["model"]["normalize"])[0]
    return v.astype(np.float32)

def inject_title(menu_item: str, section_title: str, text: str) -> str:
    return f"[MENU: {menu_item}] [BÖLÜM: {section_title}]\n{text}"

def load_doc_map():
    """config.yaml -> doc_map: { 'LC.FIP.KL.005': {menu_item, roles, version, lang} }"""
    return CFG.get("doc_map", {})

def ingest_one(pdf_path: str, doc_id: str, meta: dict) -> int:
    menu_item     = meta.get("menu_item")
    allowed_roles = meta.get("roles", []) or []            # NOT NULL -> en az []
    doc_version   = meta.get("version", "")                # şemadaki ad doc_version
    lang          = meta.get("lang", "tr")

    rows = parse_pdf_advanced(
        pdf_path,
        lang=lang,
        max_tokens=CFG["chunk"]["max_tokens"],
        overlap=CFG["chunk"]["overlap"],
    )

    batch = []
    for r in rows:
        section = r["section"]
        page_start, page_end = int(r["page_start"] or 0), int(r["page_end"] or 0)
        text = (r["text"] or "").strip()
        if not text:
            continue

        title_full     = f"{menu_item} · {section}"
        text_for_embed = inject_title(menu_item, section, text)
        emb            = embed_passage(text_for_embed)
        hsh            = content_hash(text_for_embed)

        batch.append((
            # tablo kolon sırası ile birebir (şemaya göre)
            doc_id,                 # doc_id (NOT NULL)
            menu_item,              # menu_item
            section,                # section
            text,                   # content (ham)
            emb.tolist(),           # embedding (vector)
            allowed_roles,          # allowed_roles text[]
            doc_version,            # doc_version  <-- isim DÜZELTİLDİ
            os.path.basename(pdf_path),  # source
            page_start,             # source_page_from  (artık sayfayı da tutuyoruz)
            page_end,               # source_page_to
            True,                   # is_active
            title_full,             # title_full
            lang,                   # lang
            int(r["token_count"]),  # token_count
            page_start,             # page_start
            page_end,               # page_end
            hsh                     # content_hash
        ))

    if not batch:
        print(f"SKIP: {os.path.basename(pdf_path)} -> 0 chunk")
        return 0

    con = psycopg2.connect(**PG)
    with con, con.cursor() as cur:
        # Kolon sırasını açıkça yazarak ekleyelim (şemanla uyumlu)
        execute_values(cur, """
            INSERT INTO knowledge_chunks
                (doc_id, menu_item, section, content, embedding,
                 allowed_roles, doc_version, source, source_page_from, source_page_to,
                 is_active, title_full, lang, token_count, page_start, page_end, content_hash)
            VALUES %s
            ON CONFLICT (source, page_start, page_end, content_hash) DO NOTHING;
        """, batch)
    con.close()
    return len(batch)

def main():
    doc_map = load_doc_map()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    total = 0
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(data_dir, fn)

        # doc_id eşle (örn. 'LC.FIP.KL.005' → dosya adında geçmeli)
        doc_id = next((k for k in doc_map.keys() if k in fn), None)
        if not doc_id:
            print(f"SKIP (doc_id eşleşmedi): {fn}")
            continue

        cnt = ingest_one(pdf_path, doc_id, doc_map[doc_id])
        print(f"OK: {fn} -> {cnt} chunks")
        total += cnt

    print(f"TOPLAM chunk: {total}")

if __name__ == "__main__":
    main()
