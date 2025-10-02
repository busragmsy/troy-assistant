-- RAG sistemi için gerekli extension ve tablolar
-- Çalıştırma: psql -U troy -d kb -f kb/schema/001_create_rag_tables.sql

-- pgvector extension'ı etkinleştir
CREATE EXTENSION IF NOT EXISTS vector;

-- RAG dökümanları tablosu
DROP TABLE IF EXISTS rag_documents CASCADE;

CREATE TABLE rag_documents (
    chunk_id VARCHAR(50) PRIMARY KEY,
    file_name TEXT NOT NULL,
    section_title TEXT,
    content TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    chunk_index VARCHAR(20),
    approx_tokens INTEGER,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Similarity search için HNSW index (cosine distance)
CREATE INDEX embedding_idx ON rag_documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Metadata filtreleme için B-tree indexler
CREATE INDEX idx_file_name ON rag_documents(file_name);
CREATE INDEX idx_section_title ON rag_documents(section_title);
CREATE INDEX idx_created_at ON rag_documents(created_at DESC);

-- Arama sorguları tablosu (isteğe bağlı - analytics için)
CREATE TABLE IF NOT EXISTS search_queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    user_id VARCHAR(50),
    results_count INTEGER,
    avg_similarity FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_query_created ON search_queries(created_at DESC);

-- Trigger: updated_at otomatik güncelle
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_rag_documents_updated_at 
    BEFORE UPDATE ON rag_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Yardımcı fonksiyonlar
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
RETURNS float AS $$
BEGIN
    RETURN 1 - (a <=> b);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT PARALLEL SAFE;

-- Başarılı mesajı
DO $$
BEGIN
    RAISE NOTICE 'RAG tabloları başarıyla oluşturuldu!';
END $$;