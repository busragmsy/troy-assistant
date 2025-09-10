CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id UUID PRIMARY KEY,
  doc_id TEXT,
  menu_item TEXT,
  section TEXT,
  title TEXT,
  content TEXT,
  embedding VECTOR(1024),
  allowed_roles TEXT[],
  doc_version TEXT,
  source TEXT,
  updated_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_menu ON knowledge_chunks(menu_item);
CREATE INDEX IF NOT EXISTS idx_chunks_roles ON knowledge_chunks USING GIN(allowed_roles);
