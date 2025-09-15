import os, re, uuid, json, yaml, math
import numpy as np
import regex as regexlib
import psycopg2
from psycopg2.extras import execute_values
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml"), encoding="utf-8"))

MODEL = SentenceTransformer(CFG["model"]["name"], device="cpu")
BATCH_SIZE_EMB = int(CFG["model"].get("batch_size", 16))
NORMALIZE = bool(CFG["model"].get("normalize", True))

PG = dict(host="localhost", port=5432, dbname="kb", user="troy", password="troy")

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), CFG["ingest"]["data_dir"]))

def read_pdf_with_pages(path):
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return pages

HEADING_RE = regexlib.compile(r"^(?:\d+(?:\.\d+)*|[A-ZÇĞİÖŞÜ]{1,2}[\.\-\)]?)\s+[^\n]{3,}$", flags=regexlib.MULTILINE)

def split_by_headings(pages):
    full = "\n".join(pages)
    heads = [(m.start(), m.group(0)) for m in HEADING_RE.finditer(full)]
    if not heads:
        return [dict(text=full, p_from=1, p_to=len(pages), section="auto")]

    segments = []
    idxs = [h[0] for h in heads] + [len(full)]
    for i in range(len(heads)):
        seg = full[idxs[i]:idxs[i+1]]
        section = heads[i][1].strip()
        p_from = full[:idxs[i]].count("\f") + 1 if "\f" in full else None
        p_to   = full[:idxs[i+1]].count("\f") + 1 if "\f" in full else None
        segments.append(dict(text=seg, p_from=p_from, p_to=p_to, section=section))
    return segments

def chunk_long(text, size, overlap):
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        out.append(" ".join(toks[i:i+size]))
        i += max(1, size - overlap)
    return out

def build_chunks_from_segments(segments, cfg_chunk):
    result = []
    for seg in segments:
        parts = chunk_long(seg["text"], cfg_chunk["size"], cfg_chunk["overlap"])
        for j, p in enumerate(parts):
            if len(p.split()) < cfg_chunk["min_len"]:
                continue
            result.append({
                "section": seg.get("section") or "auto",
                "content": p,
                "p_from": seg.get("p_from"),
                "p_to": seg.get("p_to")
            })
    return result

def passage_embed(texts):
    prompts = [f"passage: {t}" for t in texts]
    embs = MODEL.encode(prompts, normalize_embeddings=NORMALIZE, batch_size=BATCH_SIZE_EMB)
    return embs.astype(np.float32)

def infer_doc_id(filename):
    base = os.path.basename(filename)
    m = re.match(r"([A-Za-z0-9\.]+)", base.replace("-", "."))
    return m.group(1).upper() if m else os.path.splitext(base)[0].upper()

def upsert_batch(rows, embs):
    assert len(rows) == len(embs)
    if not rows:
        return 0
    conn = psycopg2.connect(**PG)
    with conn:
        with conn.cursor() as cur:
            tpl = [
                (
                    str(uuid.uuid4()),
                    r["doc_id"], r["menu_item"], r.get("sub_context"),
                    r.get("section"), r.get("title"), r["content"],
                    emb.tolist(),
                    r["roles"], r["version"], r["source"],
                    r.get("p_from"), r.get("p_to")
                )
                for r, emb in zip(rows, embs)
            ]
            sql = """
                INSERT INTO knowledge_chunks
                  (id, doc_id, menu_item, sub_context, section, title, content, embedding,
                   allowed_roles, doc_version, source, source_page_from, source_page_to)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
            """
            execute_values(cur, sql, tpl, page_size=max(50, min(1000, len(tpl))))
    conn.close()
    return len(rows)

def process_file(path, cfg):
    pages = read_pdf_with_pages(path)
    pages_joined = []
    for i, p in enumerate(pages):
        pages_joined.append(p + ("\f" if i < len(pages) - 1 else ""))
    pages = pages_joined

    if CFG["ingest"].get("detect_sections", True):
        segments = split_by_headings(pages)
    else:
        segments = [dict(text="\n".join(pages), p_from=1, p_to=len(pages), section="auto")]

    chunks = build_chunks_from_segments(segments, CFG["chunk"])
    # metadata hazırla
    for i, c in enumerate(chunks):
        c["doc_id"] = cfg["doc_id"]
        c["menu_item"] = cfg["menu_item"]
        c["roles"] = cfg["roles"]
        c["version"] = "v1"
        c["source"] = os.path.basename(path)
        c["title"] = f"{c['section']} [{i}]"

    # embedding + batch insert
    inserted = 0
    for start in range(0, len(chunks), CFG["ingest"]["batch_insert"]):
        batch = chunks[start:start + CFG["ingest"]["batch_insert"]]
        embs = passage_embed([b["content"] for b in batch])
        inserted += upsert_batch(batch, embs)
    print(f"OK: {os.path.basename(path)} -> {inserted} chunks")

def main():
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    if not files:
        print(f"PDF bulunamadı: {DATA_DIR}")
        return

    for fp in files:
        doc_id = infer_doc_id(fp)
        mapping = CFG["doc_map"].get(doc_id)
        if not mapping:
            print(f"SKIP (doc_id eşleşmedi): {os.path.basename(fp)} ({doc_id})")
            continue
        cfg = dict(doc_id=doc_id, menu_item=mapping["menu_item"], roles=mapping["roles"])
        process_file(fp, cfg)

if __name__ == "__main__":
    main()
