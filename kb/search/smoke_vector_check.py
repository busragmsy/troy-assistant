import argparse
import os
import psycopg2
from sentence_transformers import SentenceTransformer

def get_db_config():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "kb"),
        "user": os.getenv("DB_USER", "troy"),
        "password": os.getenv("DB_PASSWORD", "troy1234")
    }

def main():
    ap = argparse.ArgumentParser(description="Vector arama smoke test")
    ap.add_argument("--query", required=True, help="Arama sorgusu (örn: 'fiyat revize nasıl yapılır')")
    ap.add_argument("--limit", type=int, default=5, help="Kaç sonuç döndürülsün")
    args = ap.parse_args()

    # Model yükle
    model_name = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-large")
    print(f"ℹ️ Model yükleniyor: {model_name}")
    model = SentenceTransformer(model_name)

    # Query vektörü oluştur
    q_vec = model.encode([args.query], normalize_embeddings=False)[0].tolist()

    conn = psycopg2.connect(**get_db_config())
    cur = conn.cursor()

    sql = f"""
    SELECT d.title, ds.section_title, ds.content,
           1 - (de.embedding <=> %s::vector) AS sim
    FROM documents d
    JOIN document_sections ds ON d.id = ds.document_id
    JOIN document_embeddings de ON ds.id = de.section_id
    WHERE d.status = 'active'
    ORDER BY sim DESC
    LIMIT {args.limit};
    """

    cur.execute(sql, (q_vec,))
    rows = cur.fetchall()

    print(f"\n🔍 Sorgu: {args.query}\n")
    for i, (title, section, content, sim) in enumerate(rows, 1):
        print(f"{i}. {title}")
        print(f"   📍 {section}")
        print(f"   💯 Skor: {sim:.3f}")
        print(f"   📝 {content[:200]}...\n")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
