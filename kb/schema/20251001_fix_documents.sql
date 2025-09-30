-- Eksik kolonları tamamla (mevcut tabloyu bozmaz)
ALTER TABLE public.documents
  ADD COLUMN IF NOT EXISTS document_code   VARCHAR(50),
  ADD COLUMN IF NOT EXISTS version_number  VARCHAR(20),
  ADD COLUMN IF NOT EXISTS publish_date    DATE,
  ADD COLUMN IF NOT EXISTS status          VARCHAR(20) DEFAULT 'active',
  ADD COLUMN IF NOT EXISTS language        VARCHAR(10) DEFAULT 'tr',
  ADD COLUMN IF NOT EXISTS content_hash    CHAR(64),
  ADD COLUMN IF NOT EXISTS updated_at      TIMESTAMP DEFAULT NOW();

-- document_sections tarafı (zaten çoğunu ekledik ama tamamlayalım)
ALTER TABLE public.document_sections
  ADD COLUMN IF NOT EXISTS content_norm    TEXT,
  ADD COLUMN IF NOT EXISTS content_hash    CHAR(64),
  ADD COLUMN IF NOT EXISTS tsv             tsvector;

-- Trigger fonksiyonu ve tetikleyici (FTS için)
CREATE OR REPLACE FUNCTION sections_tsv_trigger() RETURNS trigger AS $$
BEGIN
  NEW.content_norm := lower(unaccent(coalesce(NEW.content,'')));
  NEW.tsv := to_tsvector('turkish', coalesce(NEW.section_title,'') || ' ' || NEW.content_norm);
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sections_tsv ON public.document_sections;
CREATE TRIGGER trg_sections_tsv
BEFORE INSERT OR UPDATE ON public.document_sections
FOR EACH ROW EXECUTE FUNCTION sections_tsv_trigger();

-- Var olan satırlar için geriye dönük doldurma
UPDATE public.document_sections
SET
  content_norm = lower(unaccent(coalesce(content,''))),
  tsv = to_tsvector('turkish', coalesce(section_title,'') || ' ' || lower(unaccent(coalesce(content,''))))
WHERE (content_norm IS NULL OR tsv IS NULL);

-- FTS index (varsa dokunmaz)
CREATE INDEX IF NOT EXISTS idx_sections_tsv ON public.document_sections USING gin(tsv);
