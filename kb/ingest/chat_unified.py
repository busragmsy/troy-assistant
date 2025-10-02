import ollama
from sentence_transformers import SentenceTransformer
import psycopg2
from typing import List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Troy KB Chatbot API")

# CORS ayarı
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('intfloat/multilingual-e5-small')

DB_CONFIG = {
    "host": "localhost",
    "database": "kb",
    "user": "troy",
    "password": "troy1234"
}

def retrieve_from_rag_documents(query_embedding: List[float], top_k: int = 2) -> List[Dict]:
    """RAG documents tablosundan arama"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            file_name,
            section_title,
            content,
            page_start,
            1 - (embedding <=> %s::vector) as similarity,
            'document' as source_type
        FROM rag_documents
        WHERE 1 - (embedding <=> %s::vector) > 0.65
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, query_embedding, query_embedding, top_k))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [
        {
            "type": "document",
            "title": r[1] or r[0],
            "content": r[2],
            "page": r[3],
            "similarity": round(r[4], 3),
            "file": r[0]
        }
        for r in results
    ]

def retrieve_from_training(query_embedding: List[float], top_k: int = 2) -> List[Dict]:
    """Training content tablosundan arama"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            tc.title,
            tc.description,
            tc.step_by_step,
            tc.tags,
            1 - (te.embedding <=> %s::vector) as similarity,
            'training' as source_type
        FROM training_content tc
        JOIN training_embeddings te ON tc.id = te.training_id
        WHERE 1 - (te.embedding <=> %s::vector) > 0.65
            AND tc.status = 'active'
        ORDER BY te.embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, query_embedding, query_embedding, top_k))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [
        {
            "type": "training",
            "title": r[0],
            "content": r[1],
            "steps": r[2] or [],
            "tags": r[3] or [],
            "similarity": round(r[4], 3)
        }
        for r in results
    ]

def retrieve_unified_context(query: str, top_k: int = 5) -> List[Dict]:
    """Her iki kaynaktan da arama yap ve birleştir"""
    query_embedding = model.encode(query).tolist()
    
    # Her iki kaynaktan da ara
    doc_results = retrieve_from_rag_documents(query_embedding, top_k=3)
    training_results = retrieve_from_training(query_embedding, top_k=3)
    
    # Birleştir ve similarity'ye göre sırala
    all_results = doc_results + training_results
    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return all_results[:top_k]

def format_context(contexts: List[Dict]) -> str:
    """Context'leri prompt için formatla"""
    context_text = ""
    
    for i, ctx in enumerate(contexts, 1):
        if ctx['type'] == 'document':
            context_text += f"\n[Kaynak {i}: Döküman - {ctx['title']}, Sayfa {ctx.get('page', 'N/A')}]\n"
            context_text += f"{ctx['content'][:800]}\n"
        
        elif ctx['type'] == 'training':
            context_text += f"\n[Kaynak {i}: Eğitim İçeriği - {ctx['title']}]\n"
            context_text += f"{ctx['content'][:800]}\n"
            
            if ctx.get('steps'):
                context_text += "Adımlar:\n"
                for j, step in enumerate(ctx['steps'][:5], 1):
                    context_text += f"  {j}. {step}\n"
    
    return context_text

def chat(user_question: str) -> Dict:
    """Birleşik RAG chatbot"""
    
    # 1. Her iki kaynaktan da ilgili içerikleri bul
    contexts = retrieve_unified_context(user_question, top_k=5)
    
    if not contexts:
        return {
            "answer": "Bu konuda dökümanlarımda ve eğitim içeriklerinde bilgi bulamadım.",
            "sources": []
        }
    
    # 2. Context'i hazırla
    context_text = format_context(contexts)
    
    # 3. Prompt oluştur
    prompt = f"""Sen Troy ekranının iç süreçleri hakkında yardımcı bir asistansın.

Aşağıdaki bilgileri kullanarak kullanıcının sorusunu cevapla:

{context_text}

Kurallar:
- SADECE verilen bilgileri kullan
- Bilmediğin şeyleri uydurma
- Cevabında hangi kaynağı kullandığını belirt (Kaynak 1, Kaynak 2, vb.)
- Eğer "Eğitim İçeriği" kaynağından adım adım bilgi varsa, bunları sıralı şekilde yaz
- Türkçe ve net bir dille cevapla
- Eğer bilgi yoksa, bunu açıkça söyle

Kullanıcının sorusu: {user_question}

Cevap:"""

    # 4. Ollama ile cevap oluştur
    response = ollama.chat(
        model='llama3.2:1b',
        messages=[{'role': 'user', 'content': prompt}],
        options={
            'temperature': 0.3,
            'num_predict': 512
        }
    )
    
    answer = response['message']['content']
    
    # 5. Kaynakları ekle
    sources = [
        {
            "type": ctx['type'],
            "title": ctx['title'],
            "similarity": ctx['similarity'],
            "page": ctx.get('page'),
            "tags": ctx.get('tags')
        }
        for ctx in contexts
    ]
    
    return {
        "answer": answer,
        "sources": sources,
        "source_count": {
            "documents": len([s for s in sources if s['type'] == 'document']),
            "training": len([s for s in sources if s['type'] == 'training'])
        }
    }

# FastAPI Endpoints
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Chatbot endpoint"""
    try:
        result = chat(request.message)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü"""
    return {"status": "ok", "model": "llama3.2:1b"}

@app.get("/stats")
async def get_stats():
    """Veritabanı istatistikleri"""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM rag_documents")
    doc_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM training_content WHERE status='active'")
    training_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM training_embeddings")
    training_embed_count = cursor.fetchone()[0]
    
    cursor.close()
    conn.close()
    
    return {
        "rag_documents": doc_count,
        "training_content": training_count,
        "training_embeddings": training_embed_count,
        "ready": training_count == training_embed_count
    }

# Terminal testi için
def interactive_chat():
    """Terminal'de test"""
    print("🏛️  Troy KB Assistant (Unified RAG)")
    print("Çıkmak için 'exit' yazın\n")
    
    while True:
        question = input("\n💬 Soru: ").strip()
        
        if question.lower() in ['exit', 'quit', 'çık']:
            break
        
        if not question:
            continue
        
        print("\n🤔 Düşünüyorum...\n")
        result = chat(question)
        
        print(f"✨ Cevap:\n{result['answer']}\n")
        print(f"📚 Kaynaklar ({result['source_count']}):")
        for i, src in enumerate(result['sources'], 1):
            type_icon = "📄" if src['type'] == 'document' else "🎓"
            print(f"  {type_icon} {i}. {src['title']} (Similarity: {src['similarity']})")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        # FastAPI modu
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Terminal test modu
        interactive_chat()