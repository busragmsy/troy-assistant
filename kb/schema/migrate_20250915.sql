CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

ALTER TABLE knowledge_chunks
  ADD COLUMN IF NOT EXISTS page_start INT,
  ADD COLUMN IF NOT EXISTS page_end   INT,
  ADD COLUMN IF NOT EXISTS lang       TEXT,
  ADD COLUMN IF NOT EXISTS token_count INT,
  ADD COLUMN IF NOT EXISTS content_hash TEXT,
  ADD COLUMN IF NOT EXISTS title_full TEXT;

ALTER TABLE knowledge_chunks
  ADD COLUMN IF NOT EXISTS tsv tsvector
  GENERATED ALWAYS AS (
    setweight(to_tsvector('turkish', coalesce(title_full,'')), 'A') ||
    setweight(to_tsvector('turkish', unaccent(coalesce(content,''))), 'B')
  ) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON knowledge_chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_menu ON knowledge_chunks (menu_item);
CREATE INDEX IF NOT EXISTS idx_chunks_active ON knowledge_chunks (is_active);
CREATE INDEX IF NOT EXISTS idx_chunks_roles ON knowledge_chunks USING GIN (allowed_roles);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON knowledge_chunks (section);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON knowledge_chunks (content_hash);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM   pg_constraint
    WHERE  conname = 'uq_chunks_src_page_hash'
  ) THEN
    ALTER TABLE knowledge_chunks
      ADD CONSTRAINT uq_chunks_src_page_hash UNIQUE (source, page_start, page_end, content_hash);
  END IF;
END$$;

ALTER TABLE error_knowledge
  ADD COLUMN IF NOT EXISTS menu_item TEXT;

CREATE INDEX IF NOT EXISTS idx_error_menu ON error_knowledge (menu_item);
CREATE INDEX IF NOT EXISTS idx_error_code ON error_knowledge (error_code);
