import os
import argparse
import psycopg2
from dotenv import load_dotenv

# PDF parsing
from kb.ingest.preprocess_pdf import parse_pdf_advanced

# Embedding sağlayıcıları
import openai
import google.generativeai as genai
from sentence_transformers import SentenceTransformer


load_dotenv()


# ------------------------------
# Embedding Provider Seçimi
# ------------------------------
def get_embedding(text, provider="openai"):
    if provider == "openai":
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding

    elif provider == "gemini":
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        resp = genai.embed_content(model="models/embedding-001", content=text)
        return resp["embedding"]

    elif provider == "hf":
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text).tolist()

    else:
        raise ValueError(f"Unknown provider: {provider}")


# ------------------------------
# DB Insert Function
# ------------------------------
def insert_into_db(sections, doc_type, department, provider):
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "pg"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "kb"),
        user=os.getenv("DB_USER", "troy"),
        password=os.getenv("DB_PASSWORD", "troy1234"),
    )
    cur = conn.cursor()

    for section in sections:
        text = section["text"]
        emb = get_embedding(text, provider)

        cur.execute(
            """
            INSERT INTO documents (content, embedding, doc_type, department)
            VALUES (%s, %s, %s, %s)
            """,
            (text, emb, doc_type, department),
        )

    conn.commit()
    cur.close()
    conn.close()


# ------------------------------
# Main CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="PDF directory")
    parser.add_argument("--doc-type", required=True, help="Document type")
    parser.add_argument("--department", required=True, help="Department")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "gemini", "hf"],
        help="Embedding provider (openai | gemini | hf)",
    )
    args = parser.parse_args()

    docs = os.listdir(args.dir)
    for doc in docs:
        if not doc.lower().endswith(".pdf"):
            continue

        print(f"📄 İşleniyor: {doc}")
        path = os.path.join(args.dir, doc)
        try:
            sections = parse_pdf_advanced(path)
            insert_into_db(sections, args.doc_type, args.department, args.provider)
            print(f"✅ {doc} başarıyla işlendi.")
        except Exception as e:
            print(f"❌ {doc} -> {e}")
