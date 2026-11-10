"""HTTP API for the RAG engine, so the Discord ticket bot (Node.js) can ask it questions.

The ticket bot keeps the one Discord connection and all the ticket rules (sanitizing,
staff takeover, answer cap, escalation pings, transcripts); this process only turns a
question into a reply.

The API is stateless. Each request carries the conversation so far, which the bot
re-reads from the Discord channel every time, so restarting either process loses
nothing and two tickets never share a session.

    POST /ask
        {"message": "do i need prime?",
         "history": [{"user_message": "...", "bot_message": "..."}],   # oldest first
         "session_id": "1234"}                                        # optional, for logs
    ->  {"intent": "answer", "message": "...", "contact_admin": false, "reason": "...",
         "verified_user_message": "...", "model": "gemma4:e2b", "sources": [...]}

    GET /health  ->  {"ok": true, "model": "gemma4:e2b"}

Errors come back as {"error": "..."}: 400 bad request, 404 unknown path, 413 body too
large, 502 Ollama rejected the request, 503 Postgres or Ollama unreachable, 500 other.

Run from the repo root (Postgres and Ollama must be up):

    python -m RAG_ENGINE.api                  # 127.0.0.1:8765, or API_HOST / API_PORT
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import sys
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import ollama
import psycopg
from pydantic import BaseModel, Field, ValidationError

from RAG_ENGINE.config import API_HOST, API_PORT, DATABASE_URL, LLM_MODEL
from RAG_ENGINE.ingestion.store import connect
from RAG_ENGINE.main import EngineReply, RAGEngine
from RAG_ENGINE.retrieval.verifier import ChatTurn

log = logging.getLogger("api")

MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGE_CHARS = 4000  # Discord's own limit for a Nitro message
MAX_HISTORY_TURNS = 20


class Turn(BaseModel):
    user_message: str = Field(max_length=MAX_MESSAGE_CHARS)
    bot_message: str = Field(max_length=MAX_MESSAGE_CHARS)


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    session_id: str | None = Field(default=None, max_length=200)


AskFn = Callable[[AskRequest], EngineReply]


class SharedConnection:
    """One Postgres connection shared by every request, reopened if it drops.

    psycopg connections are thread-safe (queries are serialized), and a search is a
    few milliseconds next to seconds of LLM time, so one connection is plenty.
    """

    def __init__(self, database_url: str, connect_fn: Callable[[str], psycopg.Connection] = connect) -> None:
        self._url = database_url
        self._connect = connect_fn
        self._conn: psycopg.Connection | None = None
        self._lock = threading.Lock()

    def get(self) -> psycopg.Connection:
        with self._lock:
            if self._conn is None or self._conn.closed:
                self._conn = self._connect(self._url)
            return self._conn

    def reset(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:  # already broken; we only want it gone
                    pass
            self._conn = None


def make_ask(db: SharedConnection) -> AskFn:
    """Run one request through a fresh session seeded with the caller's history."""

    def run(req: AskRequest) -> EngineReply:
        history = [ChatTurn(user_message=t.user_message, bot_message=t.bot_message) for t in req.history]
        engine = RAGEngine(req.session_id, conn=db.get(), history=history)
        return engine.ask(req.message)

    def ask(req: AskRequest) -> EngineReply:
        try:
            return run(req)
        except psycopg.OperationalError:
            # Postgres restarted, or the connection timed out while idle. Retry once on
            # a new connection; a second failure means it is really down.
            log.warning("database connection lost, reconnecting")
            db.reset()
            return run(req)

    return ask


def reply_to_json(reply: EngineReply) -> dict[str, Any]:
    return {
        "intent": reply.intent,
        "message": reply.message,
        "contact_admin": reply.contact_admin,
        "reason": reply.reason,
        "verified_user_message": reply.verified_user_message,
        "model": LLM_MODEL,
        "sources": [
            {"source_file": c.source_file, "question": c.question, "similarity": round(c.similarity, 4)}
            for c in reply.sources
        ],
    }


def make_handler(ask: AskFn) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "RAGEngine/1"

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(HTTPStatus.OK, {"ok": True, "model": LLM_MODEL})
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/ask":
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length > MAX_BODY_BYTES:
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": f"body over {MAX_BODY_BYTES} bytes"})
                return
            if length <= 0:
                self._send(HTTPStatus.BAD_REQUEST, {"error": "missing or invalid Content-Length"})
                return

            try:
                req = AskRequest.model_validate_json(self.rfile.read(length))
            except ValidationError as exc:
                details = [f"{'.'.join(map(str, e['loc'])) or 'body'}: {e['msg']}" for e in exc.errors()]
                self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid request", "details": details})
                return

            try:
                reply = ask(req)
            except psycopg.OperationalError:
                log.error("Postgres unreachable. Is it running? (`docker compose up -d`)")
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "database unavailable"})
                return
            except ConnectionError:
                log.error("Ollama unreachable. Is it running? (`ollama serve`)")
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "ollama unavailable"})
                return
            except ollama.ResponseError as exc:
                log.error("Ollama error: %s", exc.error)
                self._send(HTTPStatus.BAD_GATEWAY, {"error": f"ollama error: {exc.error}"})
                return
            except ValueError as exc:  # e.g. a message that is only whitespace
                self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except Exception:
                log.exception("unexpected error answering session %s", req.session_id)
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal error"})
                return

            log.info(
                "session %s: %s%s",
                req.session_id or "-",
                reply.intent,
                " (contact admin)" if reply.contact_admin else "",
            )
            self._send(HTTPStatus.OK, reply_to_json(reply))

        def _send(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            log.debug("%s %s", self.address_string(), format % args)

    return Handler


def make_server(host: str, port: int, ask: AskFn) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(ask))
    server.daemon_threads = True
    return server


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the RAG engine over HTTP for the Discord bot.")
    parser.add_argument("--host", default=API_HOST, help=f"address to listen on (default {API_HOST})")
    parser.add_argument("--port", type=int, default=API_PORT, help=f"port (default {API_PORT})")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # ollama logs every request

    if not DATABASE_URL:
        log.error("DATABASE_URL is not set (see .env.example)")
        return 1
    if not _is_loopback(args.host):
        log.warning("listening on %s: the API has no authentication, anyone who can reach it can use it", args.host)

    db = SharedConnection(DATABASE_URL)
    try:
        db.get()  # fail now, not on the first ticket
    except psycopg.OperationalError:
        log.error("Could not reach Postgres. Is it running? (`docker compose up -d`)")
        return 1

    server = make_server(args.host, args.port, make_ask(db))
    log.info("RAG engine API on http://%s:%d (model %s)", args.host, args.port, LLM_MODEL)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        db.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
