import psycopg2
import os
from pathlib import Path

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "kb"),
    "user": os.getenv("DB_USER", "troy"),
    "password": os.getenv("DB_PASSWORD", "troy1234")
}

def run_sql_file(sql_file_path: str):
    """SQL dosyasını çalıştırır."""
    
    # SQL dosyasını oku
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # Bağlantı
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        print(f"SQL dosyası çalıştırılıyor: {sql_file_path}")
        cursor.execute(sql_content)
        print("✓ Başarıyla tamamlandı!")
        
        # Kontrol
        cursor.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename = 'rag_documents'
        """)
        
        if cursor.fetchone():
            print("✓ rag_documents tablosu oluşturuldu")
            
            # Index kontrolü
            cursor.execute("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = 'rag_documents'
            """)
            indexes = cursor.fetchall()
            print(f"✓ {len(indexes)} index oluşturuldu")
            for idx in indexes:
                print(f"  - {idx[0]}")
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        conn.rollback()
    
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Proje kök dizininden çalıştır
    project_root = Path(__file__).parent.parent.parent
    sql_file = project_root / "kb" / "schema" / "001_create_rag_tables.sql"
    
    if not sql_file.exists():
        print(f"✗ SQL dosyası bulunamadı: {sql_file}")
    else:
        run_sql_file(str(sql_file))