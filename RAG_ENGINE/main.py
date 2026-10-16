"""The RAG engine: one `RAGEngine` per user session.

Each session keeps its own chat history for as long as the session lasts, so
follow-ups like "what about walmart?" are understood. Every message runs the
full pipeline:

    retrieval   LLM call 1 rewrites the message using the history, then vector search
    augment     builds the JSON prompt (history + message + rewrite + context)
    generate    LLM call 2 classifies the intent and writes the reply

Usage from code (e.g. the Discord bot):

    with connect(DATABASE_URL) as conn:              # one connection, shared by sessions
        sessions: dict[str, RAGEngine] = {}
        engine = sessions.setdefault(user_id, RAGEngine(user_id, conn=conn))
        reply = engine.ask("how much is the aco fee for walmart?")
        reply.intent, reply.message

Chat with one session in the terminal (from the repo root):

    python -m RAG_ENGINE.main
"""

from __future__ import annotations

import logging
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import ollama
import psycopg

from RAG_ENGINE.config import DATABASE_URL
from RAG_ENGINE.ingestion.store import connect
from RAG_ENGINE.retrieval.retrieve import RetrievedChunk, retrieve
from RAG_ENGINE.retrieval.verifier import ChatTurn
from RAG_ENGINE.augmentation.augment import build_prompt
from RAG_ENGINE.generation.generate import generate
from RAG_ENGINE.generation.schemas import Intent

# Eastern time: EST in winter, EDT in summer (switches automatically).
EASTERN = ZoneInfo("America/Toronto")

log = logging.getLogger("engine")


@dataclass(frozen=True)
class EngineReply:
    user_message: str  # what the member sent
    verified_user_message: str  # standalone rewrite that was searched (same as user_message on a first message)
    reason: str  # the model's one-line explanation of the intent (for logs/debugging)
    intent: Intent
    message: str  # the reply to send in Discord
    contact_admin: bool  # True when support should follow up with the member (handoff, or an uncovered follow-up)
    sources: list[RetrievedChunk]  # knowledge-base entries the reply was based on


class RAGEngine:
    """One user's conversation session with the RAG engine."""

    def __init__(
        self,
        session_id: str | None = None,
        *,
        conn: psycopg.Connection    | None = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        # Every turn is kept and sent with each message for the life of the session.
        # Each turn adds roughly 50-300 tokens to both LLM prompts; helpers/context_window.py
        # logs a warning as a prompt nears LLM_NUM_CTX or gets cut by Ollama.
        self.history: list[ChatTurn] = []
        self.created_at = self.last_active = datetime.now(EASTERN)
        # A shared connection is left open on close(); one this session opened itself is closed.
        self._conn = conn
        self._owns_conn = conn is None

    @property
    def conn(self) -> psycopg.Connection:
        if self._conn is None:
            if not DATABASE_URL:
                raise RuntimeError("DATABASE_URL is not set (see .env.example)")
            self._conn = connect(DATABASE_URL)
        return self._conn

    def ask(self, user_message: str) -> EngineReply:
        """Reply to one member message in the context of this session's history."""
        user_message = user_message.strip()
        if not user_message:
            raise ValueError("user_message is empty")

        result = retrieve(self.conn, user_message, self.history)
        output = generate(build_prompt(result, self.history))

        self.history.append(ChatTurn(user_message=user_message, bot_message=output.message))
        self.last_active = datetime.now(EASTERN)

        return EngineReply(
            user_message=user_message,
            verified_user_message=result.verified_question,
            reason=output.reason,
            intent=output.intent,
            message=output.message,
            contact_admin=output.contact_admin,
            sources=result.chunks,
        )

    def close(self) -> None:
        if self._owns_conn and self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> RAGEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # ollama logs every request

    print("New session. Type a message; blank line or Ctrl-D quits.")
    try:
        with RAGEngine() as engine:
            while True:
                try:
                    message = input("\nyou> ").strip()
                except EOFError:
                    break
                if not message:
                    break

                reply = engine.ask(message)
                if reply.verified_user_message != reply.user_message:
                    print(f"  (searched as: {reply.verified_user_message})")
                print(f"  (reason: {reply.reason})")
                print(f"bot [{reply.intent}{', contact admin' if reply.contact_admin else ''}]> {reply.message}")
    except psycopg.OperationalError:
        log.error("Could not reach Postgres. Is it running? (`docker compose up -d`)")
        return 1
    except ConnectionError:
        log.error("Could not reach Ollama. Is it running? (`ollama serve`)")
        return 1
    except ollama.ResponseError as e:
        log.error("Ollama error: %s", e.error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
