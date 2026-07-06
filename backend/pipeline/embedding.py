"""Vector indexing and retrieval over document chunks (Chroma + MiniLM).

The SentenceTransformer model and the Chroma persistent client are heavy
imports (torch, onnx, ...), so importing this module MUST stay cheap: both
are lazy-loaded on first use and cached in the module-level ``_model`` /
``_client`` variables.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from backend import config

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Lazy singletons — populated on first call, never at import time.
_model: SentenceTransformer | None = None
_client: ClientAPI | None = None


def get_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer, loading it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _get_client() -> ClientAPI:
    """Return the cached Chroma persistent client, creating it on first call."""
    global _client
    if _client is None:
        import chromadb

        logger.info("Opening Chroma persistent store at %s", config.CHROMA_PATH)
        _client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    return _client


def get_collection() -> Collection:
    """Return the ConstructIQ document collection (cosine space)."""
    return _get_client().get_or_create_collection(
        config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def index_chunks(chunks: list[dict], project_id: str, source_type: str = "tender") -> int:
    """Embed and upsert chunks into the collection; returns count indexed.

    Upsert keyed on ``chunk_id`` means re-ingesting the same document does not
    create duplicate vectors.
    """
    if not chunks:
        logger.warning("index_chunks called with no chunks (project_id=%s)", project_id)
        return 0

    ids: list[str] = [str(chunk["chunk_id"]) for chunk in chunks]
    texts: list[str] = [str(chunk["text"]) for chunk in chunks]
    metadatas: list[dict[str, Any]] = [
        {
            "project_id": project_id,
            "page_number": int(chunk.get("page_number", 0)),
            "source_file": str(chunk.get("source_file", "")),
            "chunk_type": str(chunk.get("chunk_type", "text")),
            "source_type": source_type,
        }
        for chunk in chunks
    ]

    embeddings = get_model().encode(texts, show_progress_bar=False)
    get_collection().upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    logger.info(
        "Indexed %d chunks for project %s (source_type=%s)",
        len(ids), project_id, source_type,
    )
    return len(ids)


def retrieve(query: str, project_id: str, top_k: int = 5) -> dict:
    """Semantic search scoped to one project.

    Returns ``{"results": [...], "low_confidence": bool}`` where
    ``low_confidence`` is True when there are no results at all, or when the
    best (smallest) cosine distance exceeds ``config.MIN_CONFIDENCE_DISTANCE``.
    """
    query_embedding = get_model().encode([query], show_progress_bar=False)
    raw = get_collection().query(
        query_embeddings=query_embedding.tolist(),
        n_results=max(top_k, 1),
        where={"project_id": project_id},
        include=["documents", "metadatas", "distances"],
    )

    documents = (raw.get("documents") or [[]])[0] or []
    metadatas = (raw.get("metadatas") or [[]])[0] or []
    distances = (raw.get("distances") or [[]])[0] or []

    results: list[dict[str, Any]] = []
    for text, meta, distance in zip(documents, metadatas, distances):
        meta = meta or {}
        results.append(
            {
                "text": text,
                "page_number": int(meta.get("page_number", 0)),
                "source_file": str(meta.get("source_file", "")),
                "chunk_type": str(meta.get("chunk_type", "text")),
                "source_type": str(meta.get("source_type", "tender")),
                "distance": float(distance),
            }
        )

    if not results:
        logger.info("retrieve: no indexed chunks for project %s", project_id)
        return {"results": [], "low_confidence": True}

    best_distance = min(result["distance"] for result in results)
    low_confidence = best_distance > config.MIN_CONFIDENCE_DISTANCE
    if low_confidence:
        logger.info(
            "retrieve: low confidence for project %s (best distance %.4f > %.4f)",
            project_id, best_distance, config.MIN_CONFIDENCE_DISTANCE,
        )
    return {"results": results, "low_confidence": low_confidence}
