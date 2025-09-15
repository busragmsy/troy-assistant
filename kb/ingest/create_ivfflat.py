import psycopg2, os, yaml

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml"), encoding="utf-8"))
PG = dict(host="localhost", port=5432, dbname="kb", user="troy", password="troy")

# veri boyutuna göre lists ayarı
def choose_lists(row_count: int) -> int:
    if row_count < 20000: return 100
    if row_count < 100000: return 200
    if row_count < 300000: return 400
    return 800

def main():
    con = psycopg2.connect(**PG)
    with con, con.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM knowledge_chunks WHERE is_active;")
        n = cur.fetchone()[0]
        lists = choose_lists(n)
        print(f"active rows={n}, lists={lists}")
        # varsa düşürüp yeniden yaratmak istersen:
        cur.execute("DROP INDEX IF EXISTS idx_chunks_ivfflat;")
        cur.execute(f"""
          CREATE INDEX idx_chunks_ivfflat
          ON knowledge_chunks
          USING ivfflat (embedding vector_cosine_ops)
          WITH (lists = {lists});
        """)
        cur.execute("ANALYZE knowledge_chunks;")
    con.close()
    print("IVFFLAT index created.")

if __name__ == "__main__":
    main()
