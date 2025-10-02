import ollama
from sentence_transformers import SentenceTransformer
import psycopg2
from typing import List, Dict

model = SentenceTransformer('intfloat/multilingual-e5-small')

DB_CONFIG = {
    "host": "localhost",
    "database": "kb",
    "user": "troy",
    "password": "troy1234"
}

def retrieve_context(query: str, top_k: int = 3) -> List[Dict]:
    """RAG için context chunk'ları getir."""
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
        WHERE 1 - (embedding <=> %s::vector) > 0.7
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, query_embedding, query_embedding, top_k))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [
        {
            "file": r[0],
            "section": r[1],
            "content": r[2],
            "page": r[3],
            "similarity": r[4]
        }
        for r in results
    ]

def chat(user_question: str) -> Dict:
    """Kullanıcı sorusuna RAG + Ollama ile cevap ver."""
    
    # 1. İlgili dökümanları bul
    contexts = retrieve_context(user_question, top_k=3)
    
    if not contexts:
        return {
            "answer": "Bu konuda dökümanlarımda bilgi bulamadım.",
            "sources": []
        }
    
    # 2. Context'i hazırla
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        context_text += f"\n[Kaynak {i}: {ctx['section']}, Sayfa {ctx['page']}]\n"
        context_text += f"{ctx['content']}\n"
    
    # 3. Prompt oluştur
    prompt = f"""Sen Troy ekranının iç süreçleri hakkında yardımcı bir asistansın.

Aşağıdaki döküman bilgilerini kullanarak kullanıcının sorusunu cevapla:

{context_text}

Kurallar:
- SADECE verilen döküman bilgilerini kullan
- Bilmediğin şeyleri uydurma
- Cevabında hangi kaynağı kullandığını belirt (Kaynak 1, Kaynak 2, vb.)
- Türkçe ve net bir dille cevapla
- Eğer dökümanlarda cevap yoksa, bunu açıkça söyle

Kullanıcının sorusu: {user_question}

Cevap:"""

    # 4. Ollama ile cevap oluştur
    response = ollama.chat(
        model='llama3.2:1b',
        messages=[{
            'role': 'user',
            'content': prompt
        }],
        options={
            'temperature': 0.3,  # Daha tutarlı cevaplar için düşük
            'num_predict': 512   # Max token
        }
    )
    
    answer = response['message']['content']
    
    # 5. Kaynakları ekle
    sources = [
        {
            "file": ctx['file'],
            "section": ctx['section'],
            "page": ctx['page'],
            "similarity": round(ctx['similarity'], 3)
        }
        for ctx in contexts
    ]
    
    return {
        "answer": answer,
        "sources": sources
    }

# Test fonksiyonu
def interactive_chat():
    """Terminal'de interaktif chat."""
    print("Troy KB Assistant (Ollama + RAG)")
    print("Çıkmak için 'exit' yazın\n")
    
    while True:
        question = input("\nSoru: ").strip()
        
        if question.lower() in ['exit', 'quit', 'çık']:
            break
        
        if not question:
            continue
        
        print("\nDüşünüyorum...\n")
        result = chat(question)
        
        print(f"Cevap: {result['answer']}\n")
        print("Kaynaklar:")
        for i, src in enumerate(result['sources'], 1):
            print(f"  {i}. {src['section']} (Sayfa {src['page']}, Similarity: {src['similarity']})")

if __name__ == "__main__":
    # Test
    result = chat("Psikolojik fiyat nedir?")
    print(result['answer'])
    print("\nKaynaklar:", result['sources'])