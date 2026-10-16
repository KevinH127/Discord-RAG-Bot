"""Embeddings served by a local Ollama instance (model and settings in .env)."""

from __future__ import annotations

import ollama

from RAG_ENGINE.config import EMBED_DIM, EMBED_DOCUMENT_PREFIX, EMBED_MODEL, EMBED_QUERY_PREFIX

BATCH_SIZE = 32


def _embed(texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        response = ollama.embed(model=EMBED_MODEL, input=texts[i : i + BATCH_SIZE])
        embeddings.extend(response.embeddings)

    for embedding in embeddings:
        if len(embedding) != EMBED_DIM:
            raise ValueError(f"{EMBED_MODEL} returned {len(embedding)} dims, expected {EMBED_DIM}")
    return embeddings


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed([EMBED_DOCUMENT_PREFIX + t for t in texts])


def embed_query(text: str) -> list[float]:
    return _embed([EMBED_QUERY_PREFIX + text])[0]
