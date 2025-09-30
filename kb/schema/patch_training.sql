-- Patch: training_content ve ilişkili tablolar + FTS indeksleri
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- updated_at tetikleyicisi
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = NOW(); RETURN NEW;
END; $$ LANGUAGE plpgsql;

-- Eğitim içerikleri
CREATE TABLE IF NOT EXISTS training_content (
    id                          BIGSERIAL PRIMARY KEY,
    title                       VARCHAR(500) NOT NULL,
    description                 TEXT,
    content_type                VARCHAR(50),
    topic_category              VARCHAR(100),
    difficulty_level            VARCHAR(20) DEFAULT 'beginner',
    estimated_duration_minutes  INTEGER,
    video_url                   TEXT,
    thumbnail_url               TEXT,
    step_by_step                TEXT[],
    related_screens             TEXT[],
    prerequisites               TEXT[],
    tags                        TEXT[],
    monthly_theme               VARCHAR(100),
    email_campaign_id           VARCHAR(50),
    status                      VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active','archived','draft')),
    created_at                  TIMESTAMP DEFAULT NOW(),
    updated_at                  TIMESTAMP DEFAULT NOW()
);

-- Kullanıcı eğitim ilerlemesi
CREATE TABLE IF NOT EXISTS user_training_progress (
    id                    BIGSERIAL PRIMARY KEY,
    user_id               VARCHAR(100) NOT NULL,
    training_content_id   BIGINT REFERENCES training_content(id) ON DELETE CASCADE,
    status                VARCHAR(20) DEFAULT 'not_started' CHECK (status IN ('not_started','in_progress','completed')),
    completion_percentage INTEGER DEFAULT 0 CHECK (completion_percentage BETWEEN 0 AND 100),
    started_at            TIMESTAMP,
    completed_at          TIMESTAMP,
    time_spent_minutes    INTEGER DEFAULT 0,
    notes                 TEXT,
    rating                INTEGER CHECK (rating BETWEEN 1 AND 5),
    created_at            TIMESTAMP DEFAULT NOW(),
    updated_at            TIMESTAMP DEFAULT NOW()
);

-- Eğitim <-> doküman ilişkisi
CREATE TABLE IF NOT EXISTS training_document_relations (
    training_content_id BIGINT REFERENCES training_content(id) ON DELETE CASCADE,
    document_id         BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    relation_type       VARCHAR(50),
    PRIMARY KEY (training_content_id, document_id)
);

-- Analitik (opsiyonel ama faydalı)
CREATE TABLE IF NOT EXISTS search_queries (
    id               BIGSERIAL PRIMARY KEY,
    query_text       TEXT NOT NULL,
    user_id          VARCHAR(100),
    result_count     INTEGER,
    search_method    VARCHAR(50),
    response_time_ms INTEGER,
    created_at       TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_usage (
    id               BIGSERIAL PRIMARY KEY,
    document_id      BIGINT REFERENCES documents(id) ON DELETE CASCADE,
    access_count     INTEGER DEFAULT 0,
    last_accessed    TIMESTAMP,
    average_rating   REAL,
    total_ratings    INTEGER DEFAULT 0,
    created_at       TIMESTAMP DEFAULT NOW(),
    updated_at       TIMESTAMP DEFAULT NOW()
);

-- FTS indeksleri (doküman metin araması için)
CREATE INDEX IF NOT EXISTS idx_sections_content_fts
  ON document_sections USING gin (to_tsvector('turkish', unaccent(content)));
CREATE INDEX IF NOT EXISTS idx_documents_title_fts
  ON documents USING gin (to_tsvector('turkish', unaccent(title)));

-- training_content için birkaç yardımcı indeks
CREATE INDEX IF NOT EXISTS idx_training_active
  ON training_content (topic_category, difficulty_level)
WHERE status = 'active';

-- tetikleyiciler
DROP TRIGGER IF EXISTS trg_training_updated ON training_content;
CREATE TRIGGER trg_training_updated
BEFORE UPDATE ON training_content
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_user_training_updated ON user_training_progress;
CREATE TRIGGER trg_user_training_updated
BEFORE UPDATE ON user_training_progress
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_doc_usage_updated ON document_usage;
CREATE TRIGGER trg_doc_usage_updated
BEFORE UPDATE ON document_usage
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
