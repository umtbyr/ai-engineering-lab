import hashlib
from dataclasses import replace

import numpy as np
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer

from app.rag.parser import parse_markdown
from app.rag.chunking import build_sections, chunk_sections


def ingest_markdown(
    source: str,
    content: str,
    model: SentenceTransformer,
    pool: ConnectionPool,
) -> dict:

    source = source.strip()

    # 1. Document identity and content hash
    document_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # 2. Check whether the document changed
    with pool.connection() as conn:
        existing = conn.execute(
            """
            SELECT content_hash, status
            FROM documents
            WHERE source = %s
            """,
            (source,),
        ).fetchone()

        if (
            existing
            and existing["content_hash"] == content_hash
            and existing["status"] == "indexed"
        ):
            return {
                "document_id": document_id,
                "status": "unchanged",
                "chunks_indexed": 0,
            }

    # 3. Parse and chunk
    parsed = parse_markdown(content, source=source)

    document = replace(
        parsed,
        document_id=document_id,
    )

    sections = build_sections(document)
    chunks = chunk_sections(sections, max_chars=500)

    if not chunks:
        raise ValueError("Document has no searchable content")

    # 4. Generate embeddings
    vectors = model.encode_document(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    if vectors.ndim != 2 or vectors.shape[1] != 384:
        raise ValueError("Expected 384-dimensional embeddings")

    # 5. Save everything inside one transaction
    with pool.connection() as conn:

        conn.execute(
            """
            INSERT INTO documents (
                id, source, content_hash, content, status
            )
            VALUES (%s, %s, %s, %s, 'indexed')

            ON CONFLICT (source)
            DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                content = EXCLUDED.content,
                status = 'indexed',
                updated_at = NOW()
            """,
            (document_id, source, content_hash, content),
        )

        # Remove outdated processed data
        conn.execute(
            "DELETE FROM chunks WHERE document_id = %s",
            (document_id,),
        )

        conn.execute(
            "DELETE FROM sections WHERE document_id = %s",
            (document_id,),
        )

        # Insert sections (parents before children)
        for section in sections:
            conn.execute(
                """
                INSERT INTO sections (
                    id, document_id, parent_section_id, heading_path
                )
                VALUES (%s, %s, %s, %s)
                """,
                (
                    section.id,
                    document_id,
                    section.parent_section_id,
                    list(section.heading_path),
                ),
            )

        # Insert chunks and embeddings
        for chunk, vector in zip(chunks, vectors, strict=True):
            conn.execute(
                """
                INSERT INTO chunks (
                    id, document_id, section_id,
                    content, heading_path, source_lines,
                    previous_chunk_id, next_chunk_id, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chunk.id,
                    document_id,
                    chunk.section_id,
                    chunk.text,
                    list(chunk.heading_path),
                    Jsonb([list(pair) for pair in chunk.source_lines]),
                    chunk.previous_chunk_id,
                    chunk.next_chunk_id,
                    np.asarray(vector, dtype=np.float32),
                ),
            )

    return {
        "document_id": document_id,
        "status": "indexed",
        "chunks_indexed": len(chunks),
    }
