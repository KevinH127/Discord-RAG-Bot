"""LLM call 1: rewrite a follow-up question into a standalone one before retrieval.

"what about walmart?" can't be searched on its own; with the chat history it
becomes "What is the ACO fee for Walmart Canada?", which can.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import ollama
from pydantic import BaseModel, Field, ValidationError

from RAG_ENGINE.config import LLM_OPTIONS, VERIFIER_MODEL, VERIFIER_THINK
from RAG_ENGINE.helpers.context_window import check_prompt_size

from .prompts import VERIFIER_PROMPT

log = logging.getLogger("verifier")


@dataclass(frozen=True)
class ChatTurn:
    """One earlier exchange in the conversation."""

    user_message: str  # what the member sent
    bot_message: str  # the reply they were sent


class VerifiedQuestion(BaseModel):
    verified_user_message: str = Field(description="The member's question rewritten as a standalone question.")


def verify_question(question: str, history: Sequence[ChatTurn] = ()) -> str:
    """Return `question` rewritten as a standalone question, using `history` for follow-ups.

    The first message of a conversation has nothing to resolve, so it is returned
    unchanged without an LLM call. If the rewrite fails, the original question is
    returned so the turn still goes through.
    """
    question = question.strip()
    if not history:
        return question

    payload = {
        "chat_history": [{"user_message": turn.user_message, "bot_message": turn.bot_message} for turn in history],
        "user_message": question,
    }
    try:
        response = ollama.chat(
            model=VERIFIER_MODEL,
            messages=[
                {"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            format=VerifiedQuestion.model_json_schema(),
            options=LLM_OPTIONS,
            think=VERIFIER_THINK,
        )
        check_prompt_size(response, "Question rewrite (call 1)")
        verified = VerifiedQuestion.model_validate_json(response.message.content).verified_user_message.strip()
    except (ValidationError, ollama.ResponseError) as e:
        log.warning("Question rewrite failed, searching with the original question: %s", e)
        return question
    return verified or question
