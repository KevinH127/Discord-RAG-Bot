"""LLM call 2: send the augmented prompt to the LLM and get back a structured answer.

Run from the repo root to go question -> retrieve -> augment -> generate:

    python -m RAG_ENGINE.generation.generate "do i need prime"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import ollama
import psycopg
from pydantic import ValidationError

from RAG_ENGINE.augmentation.augment import Prompt, build_prompt
from RAG_ENGINE.config import DATABASE_URL, LLM_MODEL, LLM_OPTIONS, LLM_THINK
from RAG_ENGINE.helpers.context_window import check_prompt_size
from RAG_ENGINE.ingestion.store import connect
from RAG_ENGINE.retrieval.retrieve import retrieve

from .schemas import Answer

log = logging.getLogger("generate")


def generate(prompt: Prompt) -> Answer:
    # Called even when retrieval found nothing, so the model can tell an off-topic
    # message from a Trevona question the knowledge base doesn't cover.
    response = ollama.chat(
        model=LLM_MODEL,
        messages=prompt.to_messages(),
        format=Answer.model_json_schema(),
        options=LLM_OPTIONS,
        think=LLM_THINK,
    )
    check_prompt_size(response, "Answer (call 2)")
    raw = response.message.content
    try:
        output = Answer.model_validate_json(raw)
    except ValidationError as e:
        raise ValueError(f"{LLM_MODEL} returned output that doesn't match the schema: {raw!r}") from e

    # A handoff always needs the team, even if the model forgot to set the flag.
    if output.intent == "handoff" and not output.contact_admin:
        output = output.model_copy(update={"contact_admin": True})
    return output


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # ollama logs every request

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    args = parser.parse_args(argv)

    if not DATABASE_URL:
        log.error("DATABASE_URL is not set (see .env.example)")
        return 1

    try:
        with connect(DATABASE_URL) as conn:
            result = retrieve(conn, args.question)
        answer = generate(build_prompt(result))
    except psycopg.OperationalError:
        log.error("Could not reach Postgres. Is it running? (`docker compose up -d`)")
        return 1
    except ConnectionError:
        log.error("Could not reach Ollama. Is it running? (`ollama serve`)")
        return 1
    except ollama.ResponseError as e:
        log.error("Ollama error: %s (try `ollama pull %s`)", e.error, LLM_MODEL)
        return 1

    output = {"verified_user_message": result.verified_question, **answer.model_dump()}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
