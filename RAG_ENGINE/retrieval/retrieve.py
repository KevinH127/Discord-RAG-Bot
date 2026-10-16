"""Find the knowledge-base chunks most relevant to a user question.

Run from the repo root:

    python -m RAG_ENGINE.retrieval.retrieve "how do i pay the fee?"
    python -m RAG_ENGINE.retrieval.retrieve                # interactive
    python -m RAG_ENGINE.retrieval.retrieve "..." --json   # machine-readable

Uses pure vector (cosine) search. Hybrid vector + keyword search was tried
and ranked worse on this FAQ: the chunks are short and share vocabulary
("ACO fee", "profile", "Canada"), so keyword matches mostly add noise.

With chat history, the question is first rewritten into a standalone one
(LLM call 1, verifier.py) and only the rewrite is searched.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import ollama
import psycopg
from pgvector import Vector

from RAG_ENGINE.config import DATABASE_URL, EMBED_MODEL
from RAG_ENGINE.ingestion.embedder import embed_query
from RAG_ENGINE.ingestion.store import connect

from .verifier import ChatTurn, verify_question

DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.55

NO_RELEVANT_CHUNKS = "No relevant chunks found for this question."

log = logging.getLogger("retrieve")


@dataclass(frozen=True)
class RetrievedChunk:
    source_file: str
    doc_title: str
    question: str
    content: str  # "Doc Title > Question\n\nAnswer", ready to hand to the LLM
    similarity: float  # cosine similarity, 1.0 = identical direction
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    question: str  # what the member typed
    verified_question: str  # standalone rewrite (same as `question` when there's no history)
    chunks: list[RetrievedChunk]

    @property
    def has_context(self) -> bool:
        """False means nothing relevant was found: the LLM should not guess."""
        return bool(self.chunks)

    @property
    def context(self) -> str:
        """The retrieved text to hand to the LLM, or an explicit "nothing found" message."""
        if not self.has_context:
            return NO_RELEVANT_CHUNKS
        # Numbered and separated so the LLM can tell where each chunk starts and
        # ends (each chunk already contains a blank line between question and answer).
        return "\n\n---\n\n".join(
            f"[{i}] Source: {chunk.source_file}\n{chunk.content}"
            for i, chunk in enumerate(self.chunks, 1)
        )


def retrieve(
    conn: psycopg.Connection,
    question: str,
    history: Sequence[ChatTurn] = (),
    *,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> RetrievalResult:
    question = question.strip()
    verified = verify_question(question, history)
    chunks = _search(conn, verified, top_k, min_similarity)
    return RetrievalResult(question=question, verified_question=verified, chunks=chunks)


def _search(conn: psycopg.Connection, text: str, top_k: int, min_similarity: float) -> list[RetrievedChunk]:
    embedding = Vector(embed_query(text))
    rows = conn.execute(
        """
        SELECT source_file, doc_title, question, content, metadata,
               1 - (embedding <=> %(embedding)s) AS similarity
        FROM chunks
        ORDER BY embedding <=> %(embedding)s
        LIMIT %(top_k)s
        """,
        {"embedding": embedding, "top_k": top_k},
    ).fetchall()

    return [
        RetrievedChunk(
            source_file=source_file,
            doc_title=doc_title,
            question=chunk_question,
            content=content,
            similarity=float(similarity),
            metadata=metadata,
        )
        for source_file, doc_title, chunk_question, content, metadata, similarity in rows
        if similarity >= min_similarity
    ]


def print_result(result: RetrievalResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps({**asdict(result), "context": result.context}, ensure_ascii=False, indent=2))
        return

    print(f"\nQuestion: {result.question}")
    if result.verified_question != result.question:
        print(f"Rewritten: {result.verified_question}")
    print()
    if not result.has_context:
        print(f"{result.context}\n")
        return
    for i, chunk in enumerate(result.chunks, 1):
        print(f"[{i}] {chunk.similarity:.3f}  {chunk.source_file}")
        print(f"{chunk.content}\n")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="omit to ask questions interactively")
    parser.add_argument("-k", "--top-k", type=int, default=DEFAULT_TOP_K, help="max chunks to return")
    parser.add_argument(
        "--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY,
        help="drop chunks scoring below this cosine similarity",
    )
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # ollama logs every request
    args = parse_args(argv)

    if not DATABASE_URL:
        log.error("DATABASE_URL is not set (see .env.example)")
        return 1

    try:
        with connect(DATABASE_URL) as conn:
            if args.question:
                print_result(retrieve(conn, args.question, top_k=args.top_k, min_similarity=args.min_similarity), args.json)
                return 0

            print("Ask a question (blank line or Ctrl-D to quit).")
            while True:
                try:
                    question = input("> ").strip()
                except EOFError:
                    break
                if not question:
                    break
                print_result(retrieve(conn, question, top_k=args.top_k, min_similarity=args.min_similarity), args.json)
    except ConnectionError:
        log.error("Could not reach Ollama. Is it running? (`ollama serve`)")
        return 1
    except ollama.ResponseError as e:
        log.error("Ollama error: %s (try `ollama pull %s`)", e.error, EMBED_MODEL)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
