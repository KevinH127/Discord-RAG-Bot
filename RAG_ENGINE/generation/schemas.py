"""Structured output schema for LLM answers.

Ollama constrains the model's output to match this schema's JSON Schema, and
the response is validated back into this model. Add or change fields here;
the `description`s are included in the schema and guide what the model writes.
Also update OUTPUT STRUCTURE in augmentation/prompts.py, which is written by hand.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["answer", "follow-up", "handoff", "off-topic"]


class Answer(BaseModel):
    # Field order matters: the model writes fields in this order, so it reasons
    # briefly, then classifies, then answers. (The question was already rewritten
    # by LLM call 1 in retrieval/verifier.py.)
    reason: str = Field(description="One sentence: which classification check decided the intent, and why.")
    intent: Intent = Field(
        description="answer: answerable from the context. follow-up: Trevona-related but more "
        "information is needed. handoff: a request for the team to do something. "
        "off-topic: not about Trevona ACO."
    )
    message: str = Field(description="The reply to send to the member in Discord.")
    # Set after the message. Required (not defaulted) so the model always writes it;
    # the prompt tells it to use false unless support needs to follow up.
    contact_admin: bool = Field(
        description="true if intent is handoff, or a follow-up whose message says support will be "
        "with them shortly; otherwise false."
    )
