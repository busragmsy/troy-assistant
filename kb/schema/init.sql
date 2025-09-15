CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Bilgi parçası tablosu (RAG)
CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id             UUID PRIMARY KEY,
  doc_id         TEXT NOT NULL,          -- LC.FIP.KL.005
  menu_item      TEXT NOT NULL,          -- 'ilk_fiyat_revize' vb.
  sub_context    TEXT,                   -- opsiyonel: islem/rapor alt bağlamı
  section        TEXT,                   -- başlık/alt başlık
  title          TEXT,
  content        TEXT NOT NULL,
  embedding      VECTOR(1024) NOT NULL,  -- e5-large (normalize edilmiş)
  allowed_roles  TEXT[] NOT NULL,        -- ["PricingAnalyst","Admin"]
  doc_version    TEXT,                   -- v1, v2...
  source         TEXT,                   -- dosya adı
  source_page_from INT,                  -- sayfa aralığı (isteğe bağlı)
  source_page_to   INT,
  is_active      BOOLEAN DEFAULT TRUE,
  valid_from     TIMESTAMP DEFAULT now(),
  valid_to       TIMESTAMP,
  updated_at     TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS error_knowledge (
  id           UUID PRIMARY KEY,
  error_code   TEXT,                     -- örn: ERR1234
  pattern      TEXT,                     -- LIKE/regex için kalıp
  solution     TEXT,                     -- kısaca çözüm adımları
  severity     TEXT,                     -- info|warn|critical
  menu_item    TEXT,
  allowed_roles TEXT[],
  source       TEXT,
  updated_at   TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_menu    ON knowledge_chunks(menu_item);
CREATE INDEX IF NOT EXISTS idx_chunks_roles   ON knowledge_chunks USING GIN (allowed_roles);
CREATE INDEX IF NOT EXISTS idx_chunks_docid   ON knowledge_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_active  ON knowledge_chunks(is_active);

-- Büyük veri için ANN (Approximate) ivfflat cosine index (VERİ YÜKLENDİKTEN SONRA OLUŞTUR!)
-- Başlangıçta ÇALIŞTIRMA; yükleme bitince aşağıdakini çalıştır:
-- CREATE INDEX idx_chunks_ivfflat
--   ON knowledge_chunks
--   USING ivfflat (embedding vector_cosine_ops)
--   WITH (lists = 100);


CREATE INDEX IF NOT EXISTS idx_chunks_trgm ON knowledge_chunks
  USING GIN (content gin_trgm_ops);