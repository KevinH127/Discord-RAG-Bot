"""Tests for RAG_ENGINE/api.py. No Postgres or Ollama needed: the engine is faked.

Run from the repo root:

    .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
import unittest
from typing import Any
from unittest import mock

import ollama
import psycopg

from RAG_ENGINE import api
from RAG_ENGINE.main import EngineReply
from RAG_ENGINE.retrieval.retrieve import RetrievedChunk
from RAG_ENGINE.retrieval.verifier import ChatTurn

logging.getLogger("api").setLevel(logging.CRITICAL)


def make_reply(**overrides: Any) -> EngineReply:
    fields: dict[str, Any] = {
        "user_message": "how much is the fee for walmart?",
        "verified_user_message": "How much is the ACO fee for Walmart?",
        "reason": "Covered by the fees FAQ.",
        "intent": "answer",
        "message": "The Walmart ACO fee is $5.",
        "contact_admin": False,
        "sources": [
            RetrievedChunk(
                source_file="04-fees-and-payment.md",
                doc_title="ACO Fees and Payment",
                question="How much is the ACO fee?",
                content="...",
                similarity=0.812345,
            )
        ],
    }
    fields.update(overrides)
    return EngineReply(**fields)


class ServerTestCase(unittest.TestCase):
    """Runs a real HTTP server on a free port with a swappable `ask`."""

    def setUp(self) -> None:
        self.requests: list[api.AskRequest] = []
        self.ask_impl = lambda req: make_reply()

        def ask(req: api.AskRequest) -> EngineReply:
            self.requests.append(req)
            return self.ask_impl(req)

        self.server = api.make_server("127.0.0.1", 0, ask)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def request(
        self, method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None
    ) -> tuple[int, dict[str, Any]]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            res = conn.getresponse()
            return res.status, json.loads(res.read() or b"{}")
        finally:
            conn.close()

    def post_ask(self, payload: Any) -> tuple[int, dict[str, Any]]:
        data = json.dumps(payload).encode()
        return self.request("POST", "/ask", data, {"Content-Type": "application/json"})


class HealthTests(ServerTestCase):
    def test_health_reports_the_model(self) -> None:
        status, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["model"], api.LLM_MODEL)

    def test_unknown_paths_are_404(self) -> None:
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/nope", b"{}")[0], 404)


class AskTests(ServerTestCase):
    def test_answer_round_trip(self) -> None:
        status, body = self.post_ask({"message": "how much is the fee for walmart?"})
        self.assertEqual(status, 200)
        self.assertEqual(body["intent"], "answer")
        self.assertEqual(body["message"], "The Walmart ACO fee is $5.")
        self.assertFalse(body["contact_admin"])
        self.assertEqual(body["model"], api.LLM_MODEL)
        self.assertEqual(
            body["sources"],
            [{"source_file": "04-fees-and-payment.md", "question": "How much is the ACO fee?", "similarity": 0.8123}],
        )

    def test_history_and_session_reach_the_engine(self) -> None:
        self.post_ask(
            {
                "message": "walmart",
                "history": [{"user_message": "how much is the fee?", "bot_message": "Which retailer?"}],
                "session_id": "0273",
            }
        )
        [req] = self.requests
        self.assertEqual(req.message, "walmart")
        self.assertEqual(req.session_id, "0273")
        self.assertEqual(req.history[0].bot_message, "Which retailer?")

    def test_contact_admin_is_passed_through(self) -> None:
        self.ask_impl = lambda req: make_reply(intent="handoff", contact_admin=True)
        status, body = self.post_ask({"message": "can you check my order?"})
        self.assertEqual(status, 200)
        self.assertEqual(body["intent"], "handoff")
        self.assertTrue(body["contact_admin"])

    def test_invalid_json_is_400(self) -> None:
        status, body = self.request("POST", "/ask", b"not json")
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid request")

    def test_missing_message_is_400(self) -> None:
        status, body = self.post_ask({"history": []})
        self.assertEqual(status, 400)
        self.assertTrue(any("message" in d for d in body["details"]))

    def test_empty_message_is_400(self) -> None:
        self.assertEqual(self.post_ask({"message": ""})[0], 400)

    def test_too_many_history_turns_is_400(self) -> None:
        turns = [{"user_message": "q", "bot_message": "a"}] * (api.MAX_HISTORY_TURNS + 1)
        self.assertEqual(self.post_ask({"message": "q", "history": turns})[0], 400)

    def test_oversized_body_is_413(self) -> None:
        status, _ = self.request("POST", "/ask", b"{}", {"Content-Length": str(api.MAX_BODY_BYTES + 1)})
        self.assertEqual(status, 413)
        self.assertEqual(self.requests, [])

    def test_whitespace_message_is_400(self) -> None:
        def raise_value_error(req: api.AskRequest) -> EngineReply:
            raise ValueError("user_message is empty")

        self.ask_impl = raise_value_error
        status, body = self.post_ask({"message": "   "})
        self.assertEqual(status, 400)
        self.assertIn("empty", body["error"])


class AskErrorTests(ServerTestCase):
    def assert_error(self, exc: BaseException, status: int, text: str) -> None:
        def raise_it(req: api.AskRequest) -> EngineReply:
            raise exc

        self.ask_impl = raise_it
        got_status, body = self.post_ask({"message": "q"})
        self.assertEqual(got_status, status)
        self.assertIn(text, body["error"])

    def test_database_down_is_503(self) -> None:
        self.assert_error(psycopg.OperationalError("connection refused"), 503, "database")

    def test_ollama_down_is_503(self) -> None:
        self.assert_error(ConnectionError("refused"), 503, "ollama")

    def test_ollama_rejection_is_502(self) -> None:
        self.assert_error(ollama.ResponseError("model not found"), 502, "model not found")

    def test_unexpected_error_is_500_without_details(self) -> None:
        self.assert_error(RuntimeError("secret stack detail"), 500, "internal error")


class FakeConn:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class SharedConnectionTests(unittest.TestCase):
    def test_reuses_one_connection(self) -> None:
        opened: list[FakeConn] = []
        db = api.SharedConnection("url", lambda url: opened.append(FakeConn()) or opened[-1])
        self.assertIs(db.get(), db.get())
        self.assertEqual(len(opened), 1)

    def test_reopens_after_reset_or_close(self) -> None:
        opened: list[FakeConn] = []
        db = api.SharedConnection("url", lambda url: opened.append(FakeConn()) or opened[-1])
        first = db.get()
        db.reset()
        self.assertTrue(first.closed)
        second = db.get()
        self.assertIsNot(first, second)
        second.closed = True  # dropped behind our back
        self.assertIsNot(db.get(), second)
        self.assertEqual(len(opened), 3)


class MakeAskTests(unittest.TestCase):
    def test_seeds_the_session_with_the_request_history(self) -> None:
        db = api.SharedConnection("url", lambda url: FakeConn())
        req = api.AskRequest(
            message="walmart",
            history=[api.Turn(user_message="how much is the fee?", bot_message="Which retailer?")],
            session_id="0273",
        )
        with mock.patch.object(api, "RAGEngine") as engine_cls:
            engine_cls.return_value.ask.return_value = make_reply()
            api.make_ask(db)(req)

        args, kwargs = engine_cls.call_args
        self.assertEqual(args, ("0273",))
        self.assertEqual(kwargs["history"], [ChatTurn("how much is the fee?", "Which retailer?")])
        engine_cls.return_value.ask.assert_called_once_with("walmart")

    def test_retries_once_on_a_dropped_connection(self) -> None:
        opened: list[FakeConn] = []
        db = api.SharedConnection("url", lambda url: opened.append(FakeConn()) or opened[-1])
        req = api.AskRequest(message="q")
        with mock.patch.object(api, "RAGEngine") as engine_cls:
            engine_cls.return_value.ask.side_effect = [psycopg.OperationalError("gone"), make_reply()]
            reply = api.make_ask(db)(req)

        self.assertEqual(reply.intent, "answer")
        self.assertEqual(len(opened), 2)  # reconnected for the retry
        self.assertTrue(opened[0].closed)

    def test_gives_up_after_the_retry(self) -> None:
        db = api.SharedConnection("url", lambda url: FakeConn())
        with mock.patch.object(api, "RAGEngine") as engine_cls:
            engine_cls.return_value.ask.side_effect = psycopg.OperationalError("down")
            with self.assertRaises(psycopg.OperationalError):
                api.make_ask(db)(api.AskRequest(message="q"))
        self.assertEqual(engine_cls.return_value.ask.call_count, 2)


class LoopbackTests(unittest.TestCase):
    def test_loopback_detection(self) -> None:
        for host in ("127.0.0.1", "::1", "localhost"):
            self.assertTrue(api._is_loopback(host), host)
        for host in ("0.0.0.0", "192.168.1.5", "example.com"):
            self.assertFalse(api._is_loopback(host), host)


if __name__ == "__main__":
    unittest.main()
