-- Önce varsa eski tsv kolonunu temizle
ALTER TABLE knowledge_chunks DROP COLUMN IF EXISTS tsv;

-- Yeni tsv kolonunu ekle (boş)
ALTER TABLE knowledge_chunks ADD COLUMN tsv tsvector;

-- Mevcut satırları doldur
UPDATE knowledge_chunks
SET tsv =
    setweight(to_tsvector('turkish', coalesce(title_full,'')), 'A') ||
    setweight(to_tsvector('turkish', unaccent(coalesce(content,''))), 'B');

-- Trigger fonksiyonu
CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
  NEW.tsv :=
    setweight(to_tsvector('turkish', coalesce(NEW.title_full,'')), 'A') ||
    setweight(to_tsvector('turkish', unaccent(coalesce(NEW.content,''))), 'B');
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- Eski trigger varsa sil
DROP TRIGGER IF EXISTS trg_chunks_tsv ON knowledge_chunks;

-- Yeni trigger oluştur
CREATE TRIGGER trg_chunks_tsv
BEFORE INSERT OR UPDATE ON knowledge_chunks
FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();

