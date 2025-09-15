# kb/ingest/create_ivfflat.py

import psycopg2

PG = dict(host="localhost", port=5432, dbname="kb", user="troy", password="troy")

def choose_lists(row_count: int) -> int:
    # veri büyüdükçe ivfflat lists sayısını artır
    if row_count < 20_000:   return 100
    if row_count < 100_000:  return 200
    if row_count < 300_000:  return 400
    return 800

def main():
    con = psycopg2.connect(**PG)
    with con, con.cursor() as cur:
        # aktif chunk sayısı
        cur.execute("SELECT COUNT(*) FROM knowledge_chunks WHERE is_active;")
        n = cur.fetchone()[0]
        lists = choose_lists(n)
        print(f"active rows={n}, lists={lists}")

        # varsa eski ivfflat index'i düşür
        cur.execute("DROP INDEX IF EXISTS idx_chunks_ivfflat;")

        # ivfflat (cosine) index'i oluştur
        cur.execute(f"""
            CREATE INDEX idx_chunks_ivfflat
            ON knowledge_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists});
        """)

        # istatistik güncelle
        cur.execute("ANALYZE knowledge_chunks;")

    con.close()
    print("IVFFLAT index created.")

if __name__ == "__main__":
    main()
