-- pgvector ve unaccent
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Ana doküman tablosu
CREATE TABLE IF NOT EXISTS documents (
  id              BIGSERIAL PRIMARY KEY,
  title           VARCHAR(500) NOT NULL,
  document_type   VARCHAR(100) NOT NULL,
  file_name       VARCHAR(255),
  file_path       TEXT UNIQUE,
  department      VARCHAR(100),
  document_code   VARCHAR(50),
  version_number  VARCHAR(20),
  publish_date    DATE,
  status          VARCHAR(20) DEFAULT 'active',
  language        VARCHAR(10) DEFAULT 'tr',
  content_hash    CHAR(64),              -- aynı dosya değişmiş mi?
  created_at      TIMESTAMP DEFAULT NOW(),
  updated_at      TIMESTAMP DEFAULT NOW()
);

-- Bölümler (chunk'lar)
CREATE TABLE IF NOT EXISTS document_sections (
  id            BIGSERIAL PRIMARY KEY,
  document_id   BIGINT REFERENCES documents(id) ON DELETE CASCADE,
  section_title TEXT,
  content       TEXT NOT NULL,
  content_norm  TEXT,                    -- lower + unaccent
  page_number   INT,
  word_count    INT,
  content_hash  CHAR(64),
  created_at    TIMESTAMP DEFAULT NOW(),
  tsv           tsvector
);

-- tsvector trigger (Türkçe)
CREATE OR REPLACE FUNCTION sections_tsv_trigger() RETURNS trigger AS $$
BEGIN
  NEW.content_norm := lower(unaccent(coalesce(NEW.content,'')));
  NEW.tsv := to_tsvector('turkish', coalesce(NEW.section_title,'') || ' ' || NEW.content_norm);
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sections_tsv ON document_sections;
CREATE TRIGGER trg_sections_tsv
BEFORE INSERT OR UPDATE ON document_sections
FOR EACH ROW EXECUTE FUNCTION sections_tsv_trigger();

-- FTS index
CREATE INDEX IF NOT EXISTS idx_sections_tsv ON document_sections USING gin(tsv);

-- Embedding tablosu: boyutu ingest belirleyecek (ilk çalıştırmada oluşturulacak)
-- Sadece yardımcı indexler:
CREATE INDEX IF NOT EXISTS idx_sections_doc ON document_sections(document_id);
