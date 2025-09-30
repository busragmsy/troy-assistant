CREATE EXTENSION IF NOT EXISTS vector; -- pgvector
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
NEW.updated_at = NOW();
RETURN NEW;
END; $$ LANGUAGE plpgsql;

-- ============ Ana tablolar ============
CREATE TABLE IF NOT EXISTS documents (
id BIGSERIAL PRIMARY KEY,
title VARCHAR(500) NOT NULL,
document_type VARCHAR(100) NOT NULL,
file_name VARCHAR(255),
file_path TEXT,
mime_type VARCHAR(100),
size_bytes BIGINT,
department VARCHAR(100),
document_code VARCHAR(50),
version_number VARCHAR(20),
publish_date DATE,
revision_date DATE,
status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active','archived','draft')),
language VARCHAR(10) DEFAULT 'tr',
checksum_sha1 CHAR(40),
source_url TEXT,
created_at TIMESTAMP DEFAULT NOW(),
updated_at TIMESTAMP DEFAULT NOW()
);


CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_code_version
ON documents (document_code, COALESCE(version_number,''));


CREATE TABLE IF NOT EXISTS document_sections (
id BIGSERIAL PRIMARY KEY,
document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
section_title VARCHAR(500),
section_number VARCHAR(20),
content TEXT NOT NULL,
page_number INTEGER,
word_count INTEGER,
created_at TIMESTAMP DEFAULT NOW(),
updated_at TIMESTAMP DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS document_embeddings (
id BIGSERIAL PRIMARY KEY,
section_id BIGINT REFERENCES document_sections(id) ON DELETE CASCADE,
embedding vector(1536), -- text-embedding-3-small
model_name VARCHAR(100) DEFAULT 'text-embedding-3-small',
created_at TIMESTAMP DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS document_keywords (
id BIGSERIAL PRIMARY KEY,
document_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
keyword VARCHAR(100) NOT NULL,
weight REAL DEFAULT 1.0,
created_at TIMESTAMP DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS categories (
id BIGSERIAL PRIMARY KEY,
name VARCHAR(200) NOT NULL,
parent_id BIGINT REFERENCES categories(id),
description TEXT,
created_at TIMESTAMP DEFAULT NOW()
);

-- ============ Index’ler ============
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_department ON documents(department);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_code ON documents(document_code);


CREATE INDEX IF NOT EXISTS idx_sections_document ON document_sections(document_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_section ON document_embeddings(section_id);
CREATE INDEX IF NOT EXISTS idx_keywords_document ON document_keywords(document_id);
CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON document_keywords(keyword);


CREATE INDEX IF NOT EXISTS idx_embeddings_vector
ON document_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);


CREATE INDEX IF NOT EXISTS idx_sections_content_fts
ON document_sections USING gin (to_tsvector('turkish', unaccent(content)));


CREATE INDEX IF NOT EXISTS idx_documents_title_fts
ON documents USING gin (to_tsvector('turkish', unaccent(title)));


CREATE INDEX IF NOT EXISTS idx_training_active
ON training_content (topic_category, difficulty_level)
WHERE status = 'active';


-- ============ Trigger’lar ============
CREATE TRIGGER trg_documents_updated
BEFORE UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION set_updated_at();


CREATE TRIGGER trg_sections_updated
BEFORE UPDATE ON document_sections
FOR EACH ROW EXECUTE FUNCTION set_updated_at();


CREATE TRIGGER trg_doc_usage_updated
BEFORE UPDATE ON document_usage
FOR EACH ROW EXECUTE FUNCTION set_updated_at();


CREATE TRIGGER trg_training_updated
BEFORE UPDATE ON training_content
FOR EACH ROW EXECUTE FUNCTION set_updated_at();


CREATE TRIGGER trg_user_training_updated
BEFORE UPDATE ON user_training_progress
FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ============ Kategori başlangıç verisi ============
INSERT INTO categories (name, description) VALUES
('Fiyat ve İndirim', 'Fiyatlandırma süreçleri ve indirim yönetimi'),
('Yurt Dışı İşlemleri', 'Uluslararası operasyonlar ve süreçler'),
('Troy Sistemi', 'Troy yazılım sistemi kullanımı'),
('Kullanıcı Kılavuzları', 'Sistem kullanım kılavuzları'),
('İş Süreçleri', 'Operasyonel iş süreçleri ve talimatlar'),
('Yazılım Geliştirme', 'Agile, Scrum ve yazılım süreçleri')
ON CONFLICT DO NOTHING;


INSERT INTO categories (name, parent_id, description)
SELECT 'Psikolojik Fiyat', id, 'Psikolojik fiyatlandırma yönetimi' FROM categories WHERE name='Fiyat ve İndirim'
UNION ALL
SELECT 'Devir Ürünler', id, 'Sezon geçiş ürün fiyatları' FROM categories WHERE name='Fiyat ve İndirim'
UNION ALL
SELECT 'Corporate Ülkeler', id, 'Şirket kontrolündeki ülke operasyonları' FROM categories WHERE name='Yurt Dışı İşlemleri'
UNION ALL
SELECT 'Franchise Ülkeler', id, 'Franchise ülke operasyonları' FROM categories WHERE name='Yurt Dışı İşlemleri'
ON CONFLICT DO NOTHING;

