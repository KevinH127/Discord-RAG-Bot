"""Postgres/pgvector persistence for chunks."""

from __future__ import annotations

from collections.abc import Sequence

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from RAG_ENGINE.config import EMBED_DIM

from .chunker import Chunk

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    source_file  TEXT        NOT NULL,
    doc_title    TEXT        NOT NULL,
    question     TEXT        NOT NULL,
    content      TEXT        NOT NULL,
    metadata     JSONB       NOT NULL DEFAULT '{{}}',
    content_hash TEXT        NOT NULL,
    embedding    vector({EMBED_DIM}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_file, question)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
"""


def connect(database_url: str) -> psycopg.Connection:
    """Open an autocommit connection; group atomic work with `conn.transaction()`."""
    conn = psycopg.connect(database_url, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def existing_hashes(conn: psycopg.Connection, source_file: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT question, content_hash FROM chunks WHERE source_file = %s", (source_file,)
    ).fetchall()
    return dict(rows)


def upsert_chunks(
    conn: psycopg.Connection, rows: Sequence[tuple[Chunk, str, list[float]]]
) -> None:
    """Insert or update (chunk, content_hash, embedding) rows."""
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks
                (source_file, doc_title, question, content, metadata, content_hash, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_file, question) DO UPDATE SET
                doc_title    = EXCLUDED.doc_title,
                content      = EXCLUDED.content,
                metadata     = EXCLUDED.metadata,
                content_hash = EXCLUDED.content_hash,
                embedding    = EXCLUDED.embedding,
                updated_at   = now()
            """,
            [
                (
                    chunk.source_file,
                    chunk.doc_title,
                    chunk.question,
                    chunk.text,
                    Jsonb(chunk.metadata),
                    content_hash,
                    Vector(embedding),
                )
                for chunk, content_hash, embedding in rows
            ],
        )


def delete_missing_questions(
    conn: psycopg.Connection, source_file: str, keep_questions: list[str]
) -> int:
    """Remove chunks whose `## ` question was deleted or renamed in the file."""
    cur = conn.execute(
        "DELETE FROM chunks WHERE source_file = %s AND NOT (question = ANY(%s))",
        (source_file, keep_questions),
    )
    return cur.rowcount


def delete_missing_files(conn: psycopg.Connection, keep_files: list[str]) -> int:
    """Remove chunks from files that no longer exist in the knowledge base."""
    if not keep_files:
        raise ValueError("refusing to delete every chunk: no source files given")
    cur = conn.execute("DELETE FROM chunks WHERE NOT (source_file = ANY(%s))", (keep_files,))
    return cur.rowcount
