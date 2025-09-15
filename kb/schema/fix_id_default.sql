-- UUID üretmek için
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- id kolonu auto-generate olsun
ALTER TABLE knowledge_chunks
  ALTER COLUMN id SET DEFAULT gen_random_uuid();
