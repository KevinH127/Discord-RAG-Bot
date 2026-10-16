"""Ingest knowledge-base markdown into pgvector.

Run from the repo root:

    python -m RAG_ENGINE.ingestion.ingest              # incremental sync
    python -m RAG_ENGINE.ingestion.ingest --dry-run    # print chunks only
    python -m RAG_ENGINE.ingestion.ingest --force      # re-embed everything

Re-running is safe: only chunks whose text changed are re-embedded, and chunks
for deleted questions or files are removed, so the table mirrors the folder.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import ollama

from RAG_ENGINE.config import DATABASE_URL, EMBED_DOCUMENT_PREFIX, EMBED_MODEL

from .chunker import Chunk, chunk_file
from .embedder import embed_documents
from .store import (
    connect,
    delete_missing_files,
    delete_missing_questions,
    ensure_schema,
    existing_hashes,
    upsert_chunks,
)

log = logging.getLogger("ingest")


def content_hash(chunk: Chunk) -> str:
    # Hash the model and the exact text sent to it, so changing either in .env
    # invalidates the stored embeddings and they are re-embedded on the next run.
    return hashlib.sha256(f"{EMBED_MODEL}\n{EMBED_DOCUMENT_PREFIX}{chunk.text}".encode()).hexdigest()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, default=Path("KNOWLEDGE_BASE"), help="folder of .md files")
    parser.add_argument("--dry-run", action="store_true", help="chunk and print, skip embedding and the database")
    parser.add_argument("--force", action="store_true", help="re-embed every chunk even if unchanged")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # ollama logs every request
    args = parse_args(argv)

    files = sorted(args.path.glob("*.md"))
    if not files:
        log.error("No .md files found in %s", args.path)
        return 1
    chunks_by_file = {f.name: chunk_file(f) for f in files}

    if args.dry_run:
        for chunks in chunks_by_file.values():
            for chunk in chunks:
                print(f"--- {chunk.source_file} {chunk.metadata}\n{chunk.text}\n")
        total = sum(len(c) for c in chunks_by_file.values())
        log.info("%d chunks from %d files (dry run, nothing written)", total, len(files))
        return 0

    if not DATABASE_URL:
        log.error("DATABASE_URL is not set (see .env.example)")
        return 1

    embedded = removed = 0
    try:
        with connect(DATABASE_URL) as conn:
            ensure_schema(conn)
            for source_file, chunks in chunks_by_file.items():
                hashes = {c.question: content_hash(c) for c in chunks}
                stored = {} if args.force else existing_hashes(conn, source_file)
                stale = [c for c in chunks if stored.get(c.question) != hashes[c.question]]

                # Embed before opening the transaction so it stays short.
                embeddings = embed_documents([c.text for c in stale]) if stale else []
                with conn.transaction():
                    upsert_chunks(
                        conn,
                        [(c, hashes[c.question], e) for c, e in zip(stale, embeddings, strict=True)],
                    )
                    dropped = delete_missing_questions(conn, source_file, list(hashes))

                embedded += len(stale)
                removed += dropped
                log.info(
                    "%-42s %3d chunks, %3d embedded, %d removed",
                    source_file, len(chunks), len(stale), dropped,
                )

            removed += delete_missing_files(conn, list(chunks_by_file))
            total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    except ConnectionError:
        log.error("Could not reach Ollama. Is it running? (`ollama serve`)")
        return 1
    except ollama.ResponseError as e:
        log.error("Ollama error: %s (try `ollama pull %s`)", e.error, EMBED_MODEL)
        return 1

    log.info("Done: %d embedded, %d removed, %d chunks in database", embedded, removed, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
