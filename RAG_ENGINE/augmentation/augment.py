"""Combine a user question and its retrieved chunks into the prompt for LLM call 2.

No LLM calls happen here: this only builds the messages that generation sends.

Run from the repo root to preview the exact prompt the LLM would receive:

    python -m RAG_ENGINE.augmentation.augment "do i need prime"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from RAG_ENGINE.config import DATABASE_URL
from RAG_ENGINE.ingestion.store import connect
from RAG_ENGINE.retrieval.retrieve import RetrievalResult, retrieve
from RAG_ENGINE.retrieval.verifier import ChatTurn

from .prompts import SYSTEM_PROMPT

log = logging.getLogger("augment")


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str

    def to_messages(self) -> list[dict[str, str]]:
        """Chat-message format used by Ollama and OpenAI-style APIs.

        (Anthropic's API takes `system` as a separate parameter: pass
        `prompt.system` there and `[{"role": "user", "content": prompt.user}]`.)
        """
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def build_payload(result: RetrievalResult, history: Sequence[ChatTurn] = ()) -> dict[str, Any]:
    """The JSON object sent as the user message (INPUT STRUCTURE in SYSTEM_PROMPT).

    `history` is the earlier turns, oldest first. The caller decides how many to
    keep so the prompt stays within LLM_NUM_CTX.
    """
    return {
        "chat_history": [{"user_message": turn.user_message, "bot_message": turn.bot_message} for turn in history],
        "user_message": result.question,  # as the member typed it
        "verified_user_message": result.verified_question,  # standalone rewrite from retrieval (LLM call 1)
        "context": [
            {
                "topic": chunk.doc_title,
                "question": chunk.question,
                # `content` is "Title > Question\n\nAnswer"; the first two are sent as their own fields.
                "answer": chunk.content.removeprefix(f"{chunk.doc_title} > {chunk.question}\n\n"),
            }
            for chunk in result.chunks
        ],
    }


def build_prompt(result: RetrievalResult, history: Sequence[ChatTurn] = ()) -> Prompt:
    # JSON serialization escapes quotes and newlines, so a member's question can't
    # break out of its "user_message" string and pose as context or instructions.
    user = json.dumps(build_payload(result, history), ensure_ascii=False, indent=2)
    return Prompt(system=SYSTEM_PROMPT, user=user)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # ollama logs every request

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    args = parser.parse_args(argv)

    if not DATABASE_URL:
        log.error("DATABASE_URL is not set (see .env.example)")
        return 1

    with connect(DATABASE_URL) as conn:
        prompt = build_prompt(retrieve(conn, args.question))

    print("===== SYSTEM =====")
    print(prompt.system)
    print("\n===== USER =====")
    print(prompt.user)
    return 0


if __name__ == "__main__":
    sys.exit(main())
