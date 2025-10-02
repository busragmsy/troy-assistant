from sentence_transformers import SentenceTransformer
import psycopg2

model = SentenceTransformer('intfloat/multilingual-e5-small')

DB_CONFIG = {
    "host": "localhost",
    "database": "kb",
    "user": "troy",
    "password": "troy1234"
}

def search(query: str, top_k: int = 3):
    # Query'yi vektörleştir
    query_embedding = model.encode(query).tolist()
    
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            file_name,
            section_title,
            content,
            page_start,
            1 - (embedding <=> %s::vector) as similarity
        FROM rag_documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, query_embedding, top_k))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return results

# Test
results = search("Yurt dışı fiyat revize nasıl yapılır?")
for i, (file, section, content, page, sim) in enumerate(results, 1):
    print(f"\n{i}. Sonuç (Similarity: {sim:.3f})")
    print(f"Dosya: {file}")
    print(f"Bölüm: {section}")
    print(f"Sayfa: {page}")
    print(f"İçerik: {content[:150]}...")