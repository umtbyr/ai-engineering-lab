CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,

    content TEXT,
    storage_key TEXT,

    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'indexed', 'failed')),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY,

    document_id TEXT NOT NULL
        REFERENCES documents(id) ON DELETE CASCADE,

    parent_section_id TEXT
        REFERENCES sections(id) ON DELETE CASCADE,

    heading_path TEXT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,

    document_id TEXT NOT NULL
        REFERENCES documents(id) ON DELETE CASCADE,

    section_id TEXT NOT NULL
        REFERENCES sections(id) ON DELETE CASCADE,

    content TEXT NOT NULL,
    heading_path TEXT[] NOT NULL,

    source_lines JSONB NOT NULL DEFAULT '[]'::jsonb,

    previous_chunk_id TEXT,
    next_chunk_id TEXT,

    embedding VECTOR(384) NOT NULL
);

ALTER TABLE chunks
ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
GENERATED ALWAYS AS (
    to_tsvector('simple', content)
) STORED;


CREATE INDEX IF NOT EXISTS idx_sections_document_id
ON sections(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_section_id
ON chunks(section_id);

CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
ON chunks USING GIN (search_vector);