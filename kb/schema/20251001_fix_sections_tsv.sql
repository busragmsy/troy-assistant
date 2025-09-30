-- kb/schema/20251001_fix_sections_tsv.sql

CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS vector;

-- Eski tabloda eksik kolonlar varsa ekle
ALTER TABLE document_sections
  ADD COLUMN IF NOT EXISTS content_norm TEXT,
  ADD COLUMN IF NOT EXISTS tsv tsvector;

-- Trigger fonksiyonunu güncelle
CREATE OR REPLACE FUNCTION sections_tsv_trigger() RETURNS trigger AS $$
BEGIN
  NEW.content_norm := lower(unaccent(coalesce(NEW.content,'')));
  NEW.tsv := to_tsvector('turkish', coalesce(NEW.section_title,'') || ' ' || NEW.content_norm);
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- Eski trigger’ı düşürüp yeniden oluştur
DROP TRIGGER IF EXISTS trg_sections_tsv ON document_sections;
CREATE TRIGGER trg_sections_tsv
BEFORE INSERT OR UPDATE ON document_sections
FOR EACH ROW EXECUTE FUNCTION sections_tsv_trigger();

-- Mevcut satırları geriye dönük doldur (backfill)
UPDATE document_sections
SET
  content_norm = lower(unaccent(coalesce(content,''))),
  tsv = to_tsvector('turkish', coalesce(section_title,'') || ' ' || lower(unaccent(coalesce(content,''))))
WHERE tsv IS NULL OR content_norm IS NULL;

-- GIN index
CREATE INDEX IF NOT EXISTS idx_sections_tsv ON document_sections USING gin(tsv);

-- Vektör index (embedding tablosu varsa cosine için)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.tables
             WHERE table_name='document_embeddings') THEN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname='idx_embeddings_vector') THEN
      CREATE INDEX idx_embeddings_vector
      ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    END IF;
  END IF;
END$$;
