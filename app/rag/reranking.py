from sentence_transformers import CrossEncoder

from app.rag.pg_retrieval import RetrieverProtocol, SearchHit


class RerankingRetriever:
    def __init__(
        self,
        retriever: RetrieverProtocol,
        reranker: CrossEncoder,
        candidate_k: int = 10,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchHit]:

        if top_k < 1:
            raise ValueError("top_k must be positive")

        # 1. Get candidates from hybrid retrieval
        candidates = self.retriever.search(
            query,
            top_k=max(self.candidate_k, top_k),
        )

        if not candidates:
            return []

        # 2. Prepare question-chunk pairs
        pairs = [(query, hit.chunk.text) for hit in candidates]

        # 3. Score every pair with Cross-Encoder
        scores = self.reranker.predict(
            pairs,
            batch_size=4,
            show_progress_bar=False,
        )

        # 4. Sort by reranker score
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )

        # 5. Return the best results
        return [
            SearchHit(
                chunk=hit.chunk,
                source=hit.source,
                score=float(score),
                score_type="cross_encoder_logit",
            )
            for hit, score in ranked[:top_k]
        ]
