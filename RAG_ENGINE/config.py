"""Settings loaded from `.env` in the repo root (see `.env.example`).

Loaded once, on import, so every module sees the same values regardless of
which command is run or the current working directory. Real environment
variables take precedence over `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _require(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise RuntimeError(f"{name} is not set. Add it to .env (see .env.example).")
    return value


# Optional here so commands that don't touch the database (e.g. --dry-run) still run.
DATABASE_URL = os.environ.get("DATABASE_URL")

# Embeddings. EMBED_DIM must match the model's output and the `vector(N)`
# column; changing the model means migrating that column and re-ingesting.
EMBED_MODEL = _require("EMBED_MODEL")
EMBED_DIM = int(_require("EMBED_DIM"))
# Task prefixes some models (like nomic-embed-text) were trained with. Set both
# to "" for models that don't use them.
EMBED_DOCUMENT_PREFIX = os.environ.get("EMBED_DOCUMENT_PREFIX", "")
EMBED_QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "")

# Answer generation.
LLM_MODEL = _require("LLM_MODEL")
LLM_TEMPERATURE = float(_require("LLM_TEMPERATURE"))
LLM_NUM_CTX = int(_require("LLM_NUM_CTX"))


def _think(value: str) -> bool | str:
    """"true"/"false", or a level ("low"/"medium"/"high") for models that support levels."""
    value = value.strip().lower()
    if value in ("true", "false"):
        return value == "true"
    if value in ("low", "medium", "high"):
        return value
    raise RuntimeError(f"LLM_THINK must be true, false, low, medium or high, not {value!r}")


# Reasoning before answering, for thinking models like qwen3. "false" is safe for
# every model; "true" or a level makes Ollama reject models that can't think.
LLM_THINK = _think(os.environ.get("LLM_THINK", "false"))

# Question rewriting (LLM call 1, retrieval/verifier.py) can use its own model.
# Defaults to LLM_MODEL. Its thinking switch is separate so a non-thinking
# rewrite model keeps working when LLM_THINK is on.
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL") or LLM_MODEL
VERIFIER_THINK = _think(os.environ.get("VERIFIER_THINK", "false"))
# Shared by both LLM calls: question rewriting (retrieval) and answering (generation).
LLM_OPTIONS = {"temperature": LLM_TEMPERATURE, "num_ctx": LLM_NUM_CTX}
