import psycopg2
from sentence_transformers import SentenceTransformer
from typing import List
import time

model = SentenceTransformer('intfloat/multilingual-e5-small')

DB_CONFIG = {
    "host": "localhost",
    "database": "kb",
    "user": "troy",
    "password": "troy1234"
}

def create_training_embeddings_table():
    """Training content için embedding tablosu oluştur"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Tablo var mı kontrol et
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_embeddings (
            id SERIAL PRIMARY KEY,
            training_id INTEGER REFERENCES training_content(id) ON DELETE CASCADE,
            embedding vector(384),
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(training_id)
        );
    """)
    
    # Index oluştur (hızlı arama için)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS training_embeddings_vector_idx 
        ON training_embeddings 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ training_embeddings tablosu hazır")

def generate_embeddings_for_training():
    """Training content'ler için embedding oluştur"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Henüz embedding'i olmayan training content'leri al
    cursor.execute("""
        SELECT tc.id, tc.title, tc.description, tc.step_by_step
        FROM training_content tc
        LEFT JOIN training_embeddings te ON tc.id = te.training_id
        WHERE te.id IS NULL AND tc.status = 'active'
        ORDER BY tc.id
    """)
    
    rows = cursor.fetchall()
    total = len(rows)
    
    if total == 0:
        print("✅ Tüm training content'ler zaten embed edilmiş")
        cursor.close()
        conn.close()
        return
    
    print(f"📊 {total} training content için embedding oluşturuluyor...")
    
    processed = 0
    for row in rows:
        training_id, title, description, steps = row
        
        # Metni birleştir
        text_parts = [title]
        if description:
            text_parts.append(description)
        if steps:
            text_parts.extend(steps[:5])  # İlk 5 adımı al
        
        combined_text = " ".join(text_parts)
        
        # Embedding oluştur
        try:
            embedding = model.encode(combined_text).tolist()
            
            # Veritabanına kaydet
            cursor.execute("""
                INSERT INTO training_embeddings (training_id, embedding)
                VALUES (%s, %s)
                ON CONFLICT (training_id) DO UPDATE 
                SET embedding = EXCLUDED.embedding
            """, (training_id, embedding))
            
            processed += 1
            if processed % 10 == 0:
                conn.commit()
                print(f"⏳ İşlenen: {processed}/{total}")
                
        except Exception as e:
            print(f"❌ Training ID {training_id} hata: {e}")
            continue
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ {processed}/{total} training content embed edildi")

def verify_embeddings():
    """Embedding'lerin doğru oluşturulduğunu kontrol et"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_training,
            (SELECT COUNT(*) FROM training_embeddings) as total_embeddings
        FROM training_content
        WHERE status = 'active'
    """)
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    print(f"\n📈 Özet:")
    print(f"  - Aktif training content: {result[0]}")
    print(f"  - Oluşturulan embedding: {result[1]}")
    
    if result[0] == result[1]:
        print("✅ Tüm training content'ler embed edildi!")
    else:
        print(f"⚠️  {result[0] - result[1]} training content henüz embed edilmedi")

if __name__ == "__main__":
    print("🚀 Training Content Embedding Oluşturucu")
    print("-" * 50)
    
    # 1. Tablo oluştur
    create_training_embeddings_table()
    
    # 2. Embedding'leri oluştur
    generate_embeddings_for_training()
    
    # 3. Doğrulama
    verify_embeddings()
    
    print("\n✨ İşlem tamamlandı!")