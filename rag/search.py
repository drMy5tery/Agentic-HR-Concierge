"""``search_policy`` retrieval: local embeddings + brute-force cosine top-k.

No vector database. Chunks are embedded once with a local ``sentence-transformers``
model (default ``all-MiniLM-L6-v2``); a query is embedded the same way and scored
against every chunk by cosine similarity (a single NumPy matrix-vector product,
since the embeddings are L2-normalised). Only chunks at or above the relevance
threshold are returned, so an off-topic or unsupported question yields an empty
result — which the agent treats as 'uncovered' and escalates rather than guessing.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

import config
from rag.ingest import load_chunks


class PolicyIndex:
    """An in-memory index of policy chunks with their normalised embeddings."""

    def __init__(self, chunks: list[dict[str, str]], embeddings: np.ndarray, model: Any):
        self.chunks = chunks
        self.embeddings = embeddings  # shape (n_chunks, dim), L2-normalised
        self.model = model

    def search(self, query: str, top_k: int, threshold: float) -> list[dict[str, Any]]:
        """Return up to ``top_k`` chunks scoring >= ``threshold``, best first."""
        if not query or not query.strip() or self.embeddings.shape[0] == 0:
            return []
        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ np.asarray(query_vec, dtype=self.embeddings.dtype)
        order = np.argsort(-scores)[:top_k]
        results: list[dict[str, Any]] = []
        for idx in order:
            score = float(scores[idx])
            if score < threshold:
                continue
            results.append({**self.chunks[int(idx)], "score": round(score, 4)})
        return results


def build_index(policies_dir: Optional[str] = None, model_name: Optional[str] = None) -> PolicyIndex:
    """Load chunks and embed them with the local model (downloaded once on first use)."""
    from sentence_transformers import SentenceTransformer

    chunks = load_chunks(policies_dir)
    model = SentenceTransformer(model_name or config.EMBED_MODEL)
    if chunks:
        embeddings = model.encode(
            [c["text"] for c in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
    else:
        embeddings = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    return PolicyIndex(chunks, embeddings, model)


def make_searcher(
    index: PolicyIndex,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Callable[[str], dict[str, Any]]:
    """Return a ``search_policy(query) -> {ok, query, chunks}`` function for the registry."""
    top_k = top_k or config.RETRIEVAL_TOP_K
    threshold = config.RETRIEVAL_THRESHOLD if threshold is None else threshold

    def search_policy(query: str) -> dict[str, Any]:
        hits = index.search(query or "", top_k, threshold)
        return {"ok": True, "query": query, "chunks": hits}

    return search_policy
