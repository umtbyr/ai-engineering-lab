import numpy as np
from sentence_transformers import SentenceTransformer
from psycopg_pool import ConnectionPool
import re
from app.rag.models import Chunk
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from typing import Protocol
from app.rag.models import Chunk


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    source: str | None = None
    score_type: str = "cosine_similarity"


class RetrieverProtocol(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchHit]: ...


class PostgresChunkSearcher:
    def __init__(
        self,
        model: SentenceTransformer,
        pool: ConnectionPool,
    ):
        self.model = model
        self.pool = pool

    def keyword_candidates(
        self,
        query: str,
        limit: int = 20,
    ) -> list[str]:

        if limit < 1:
            raise ValueError("limit must be positive")

        # Extract search terms from the user question
        terms = re.findall(r"\w+", query, flags=re.UNICODE)

        if not terms:
            return []

        search_text = " OR ".join(terms)

        with self.pool.connection() as conn:
            rows = conn.execute(
                """
    SELECT c.id
    FROM chunks AS c
    JOIN documents AS d
        ON d.id = c.document_id
    WHERE d.status = 'indexed'
      AND c.search_vector
          @@ websearch_to_tsquery('simple', %s)
    ORDER BY
        ts_rank_cd(
            c.search_vector,
            websearch_to_tsquery('simple', %s)
        ) DESC,
        c.id
    LIMIT %s
    """,
                (search_text, search_text, limit),
            ).fetchall()

        return [row["id"] for row in rows]

    def semantic_candidates(
        self,
        query: str,
        limit: int = 20,
    ) -> list[str]:

        if limit < 1:
            raise ValueError("limit must be positive")

        query_vector = self.model.encode_query(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        with self.pool.connection() as conn:
            rows = conn.execute(
                """
            SELECT c.id
            FROM chunks AS c
            JOIN documents AS d
                ON d.id = c.document_id
            WHERE d.status = 'indexed'
            ORDER BY c.embedding <=> %s
            LIMIT %s
                """,
                (query_vector, limit),
            ).fetchall()

        return [row["id"] for row in rows]

    def fetch_ranked_hits(
        self,
        ranked: list[tuple[str, float]],
    ) -> list[SearchHit]:

        if not ranked:
            return []

        ids = [chunk_id for chunk_id, _ in ranked]

        with self.pool.connection() as conn:
            rows = conn.execute(
                """
            SELECT
                c.id,
                c.section_id,
                c.content,
                c.heading_path,
                c.source_lines,
                c.previous_chunk_id,
                c.next_chunk_id,
                d.source
            FROM chunks AS c
            JOIN documents AS d
                ON d.id = c.document_id
            WHERE c.id = ANY(%s)
              AND d.status = 'indexed'
            """,
                (ids,),
            ).fetchall()

            rows_by_id = {row["id"]: row for row in rows}

        hits = []

        for chunk_id, score in ranked:
            row = rows_by_id.get(chunk_id)

            if row is None:
                continue

            chunk = Chunk(
                id=row["id"],
                section_id=row["section_id"],
                strategy="section_aware",
                text=row["content"],
                heading_path=tuple(row["heading_path"]),
                source_lines=tuple(tuple(pair) for pair in row["source_lines"]),
                previous_chunk_id=row["previous_chunk_id"],
                next_chunk_id=row["next_chunk_id"],
            )

            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=score,
                    source=row["source"],
                    score_type="rrf",
                )
            )

        return hits


class HybridRetriever:
    def __init__(
        self,
        searcher: PostgresChunkSearcher,
        candidate_k: int = 20,
    ):
        self.searcher = searcher
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchHit]:

        if top_k < 1:
            raise ValueError("top_k must be positive")

        candidate_k = max(self.candidate_k, top_k)

        semantic_ids = self.searcher.semantic_candidates(
            query,
            limit=candidate_k,
        )

        keyword_ids = self.searcher.keyword_candidates(
            query,
            limit=candidate_k,
        )

        ranked = reciprocal_rank_fusion(
            semantic_ids,
            keyword_ids,
        )

        return self.searcher.fetch_ranked_hits(ranked[:top_k])


def reciprocal_rank_fusion(
    semantic_ids: list[str],
    keyword_ids: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:

    scores: dict[str, float] = {}

    for ranking in (semantic_ids, keyword_ids):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
