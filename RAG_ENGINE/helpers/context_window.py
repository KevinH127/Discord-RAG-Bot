"""Detect when an LLM prompt is close to, or over, the context window (LLM_NUM_CTX).

Ollama doesn't fail on an oversized prompt. Measured on Ollama 0.35:

- up to LLM_NUM_CTX tokens, the whole prompt is kept and `prompt_eval_count`
  is its true size;
- past that, Ollama keeps the first `num_keep` (default 4) tokens plus the
  newest half of the window, silently dropping the middle, and reports that
  cut size: 4 + (8192 - 4) // 2 = 4098 for an 8192 window.

The start of our prompts is the system prompt, so a cut prompt means the model
answers without most of its rules. A cut prompt also *looks* small, which is
why it is matched by its exact size rather than by a threshold.
"""

from __future__ import annotations

import logging

import ollama

from RAG_ENGINE.config import LLM_NUM_CTX

# Warn once a prompt uses this much of the window, while there's still room.
WARN_RATIO = 0.75
_OLLAMA_NUM_KEEP = 4
_TRUNCATED_SIZE = _OLLAMA_NUM_KEEP + (LLM_NUM_CTX - _OLLAMA_NUM_KEEP) // 2

log = logging.getLogger("context_window")


def check_prompt_size(response: ollama.ChatResponse, label: str) -> None:
    """Log a warning if the prompt behind `response` was near or over LLM_NUM_CTX."""
    used = response.prompt_eval_count
    if not used:
        return
    if used == _TRUNCATED_SIZE:
        log.warning(
            "%s prompt was likely cut by Ollama to %d tokens (LLM_NUM_CTX=%d): the system prompt "
            "may have been dropped. Raise LLM_NUM_CTX or shorten the session.",
            label, used, LLM_NUM_CTX,
        )
    elif used >= WARN_RATIO * LLM_NUM_CTX:
        log.warning(
            "%s prompt used %d of %d tokens (%.0f%%); it will be cut once it exceeds LLM_NUM_CTX.",
            label, used, LLM_NUM_CTX, 100 * used / LLM_NUM_CTX,
        )
