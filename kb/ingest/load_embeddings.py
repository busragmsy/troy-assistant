import json
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Türkçe destekli model (384 boyutlu)
model = SentenceTransformer('intfloat/multilingual-e5-small')

DB_CONFIG = {
    "host": "localhost",
    "database": "kb",
    "user": "troy",
    "password": "troy1234"
}

def load_embeddings(jsonl_file: str, batch_size: int = 32):
    """Lokal model ile embedding oluştur ve DB'ye yükle."""
    
    if not Path(jsonl_file).exists():
        print(f"Dosya bulunamadı: {jsonl_file}")
        return
    
    chunks = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"Toplam {len(chunks)} chunk işlenecek...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Önce tabloyu güncelle (384 boyutlu vektör için)
    cursor.execute("ALTER TABLE rag_documents ALTER COLUMN embedding TYPE vector(384);")
    conn.commit()
    
    success_count = 0
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk['content'] for chunk in batch]
        
        try:
            print(f"Embedding oluşturuluyor: {i+1}-{i+len(batch)}/{len(chunks)}")
            
            # Lokal model ile embedding
            embeddings = model.encode(texts, show_progress_bar=False)
            
            data = []
            for j, chunk in enumerate(batch):
                data.append((
                    chunk['chunk_id'],
                    chunk['file_name'],
                    chunk.get('section_title'),
                    chunk['content'],
                    chunk.get('page_start'),
                    chunk.get('page_end'),
                    chunk.get('chunk_index'),
                    chunk.get('approx_tokens'),
                    embeddings[j].tolist()
                ))
            
            execute_values(
                cursor,
                """
                INSERT INTO rag_documents 
                (chunk_id, file_name, section_title, content, page_start, 
                 page_end, chunk_index, approx_tokens, embedding)
                VALUES %s
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
                """,
                data
            )
            
            conn.commit()
            success_count += len(batch)
            print(f"✓ {success_count}/{len(chunks)} chunk yüklendi")
            
        except Exception as e:
            print(f"✗ Hata (batch {i}): {e}")
            conn.rollback()
            continue
    
    cursor.close()
    conn.close()
    
    print(f"\n✓ İşlem tamamlandı! {success_count}/{len(chunks)} chunk başarıyla yüklendi")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    jsonl_file = project_root / "temiz_rag_chunks.jsonl"
    load_embeddings(str(jsonl_file))