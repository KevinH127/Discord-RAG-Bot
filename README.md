# Discord-RAG-Bot

A retrieval-augmented support assistant for the Trevona ACO Discord server. It answers
member questions from a Markdown knowledge base, keeps per-user conversation sessions,
classifies each message's intent, and flags when the admin team needs to step in.
Everything runs locally: Postgres + pgvector for search, Ollama for embeddings and the LLM.

## How it works

Each member message goes through four stages:

```
user message + session history
  │
  ├─ retrieval/      LLM call 1 rewrites a follow-up ("what about walmart?") into a
  │                  standalone question using the history, then vector-searches the
  │                  knowledge base for the 5 closest FAQ entries
  │
  ├─ augmentation/   builds the JSON prompt: history, original message, rewrite, context
  │                  (no LLM call here; the system prompt and examples live in prompts.py)
  │
  ├─ generation/     LLM call 2 returns structured output:
  │                  { reason, intent, message, contact_admin }
  │
  └─ main.py         RAGEngine: one instance per user session; records each turn
```

The `intent` is one of:

| Intent | When | Reply |
|---|---|---|
| `answer` | a knowledge-base entry answers the question | the answer, from the context only |
| `follow-up` | a detail is missing (e.g. which retailer), or the docs don't cover it | a clarifying question, or "support will be with you shortly" |
| `handoff` | the member asks the team to do or check something on their account | "support will be with you shortly" |
| `off-topic` | the message isn't about Trevona ACO | a short refusal |

`contact_admin` is `true` for every handoff and for follow-ups that promise support will
reach out, so the bot knows when to notify the team.

Ingestion is a separate step that splits the knowledge base into one chunk per FAQ
question, embeds each chunk and stores it in pgvector.

## Requirements

- Python 3.12+ (developed on 3.13)
- Docker (for Postgres + pgvector)
- [Ollama](https://ollama.com) running locally
- The knowledge base: Markdown files in `KNOWLEDGE_BASE/` (gitignored, not in this repo)

## Setup

1. **Install the Python dependencies**

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Configure.** Copy the example settings and set a real database password (use the same
   password in `POSTGRES_PASSWORD` and `DATABASE_URL`).

   ```bash
   cp .env.example .env
   ```

3. **Start Postgres + pgvector**

   ```bash
   docker compose up -d
   ```

4. **Pull the models** named in `.env`

   ```bash
   ollama pull nomic-embed-text
   ollama pull gemma4:e2b
   ```

5. **Add the knowledge base.** Put the Markdown files in `KNOWLEDGE_BASE/`. Each file needs
   one `# Title` and one `## Question?` heading per FAQ entry, with the answer below it:

   ```markdown
   # ACO Fees and Payment

   ## How do I pay the ACO fee?
   Pay the ACO fee by e-transfer through your ACO fee ticket.
   ```

   Files named `NN-retailer-<name>.md` get `retailer` metadata automatically.

6. **Ingest**

   ```bash
   .venv/bin/python -m RAG_ENGINE.ingestion.ingest
   ```

   Re-run it whenever the knowledge base changes: only changed chunks are re-embedded, and
   chunks for deleted questions or files are removed. `--dry-run` prints the chunks without
   touching the database; `--force` re-embeds everything.

## Usage

**Chat with one session in the terminal** (shows the rewrite, reason and intent):

```bash
.venv/bin/python -m RAG_ENGINE.main
```

**From code** (e.g. the Discord bot): keep one `RAGEngine` per user and share a database
connection between them.

```python
from RAG_ENGINE.config import DATABASE_URL
from RAG_ENGINE.ingestion.store import connect
from RAG_ENGINE.main import RAGEngine

with connect(DATABASE_URL) as conn:
    sessions: dict[str, RAGEngine] = {}
    engine = sessions.setdefault(user_id, RAGEngine(user_id, conn=conn))
    reply = engine.ask("how much is the aco fee for walmart?")
    reply.message        # text to send in Discord
    reply.intent         # "answer" | "follow-up" | "handoff" | "off-topic"
    reply.contact_admin  # True when the team should follow up
```

`ask()` blocks for a few seconds, so call it with `await asyncio.to_thread(engine.ask, msg)`
from discord.py, and defer slash-command replies (Discord expects a response within 3 s).
Sessions live in memory; `last_active` (Eastern time) lets the bot expire idle ones.

**Inspect a single stage** (one question, no session history):

```bash
.venv/bin/python -m RAG_ENGINE.retrieval.retrieve "do i need prime"      # chunks found
.venv/bin/python -m RAG_ENGINE.augmentation.augment "do i need prime"    # exact prompt
.venv/bin/python -m RAG_ENGINE.generation.generate "do i need prime"     # full output
```

## Configuration

All settings come from `.env` (see `.env.example`). Real environment variables override it.

| Setting | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | | Postgres connection string |
| `EMBED_MODEL`, `EMBED_DIM` | `nomic-embed-text`, `768` | embedding model and its vector size |
| `EMBED_DOCUMENT_PREFIX`, `EMBED_QUERY_PREFIX` | nomic's `search_document: ` / `search_query: ` | task prefixes; `""` for models without them |
| `LLM_MODEL` | `gemma4:e2b` | model for classification and replies (call 2) |
| `VERIFIER_MODEL` | `LLM_MODEL` | model for rewriting follow-ups (call 1) |
| `LLM_TEMPERATURE` | `0` | 0 gives the same reply for the same input |
| `LLM_NUM_CTX` | `16384` | context window in tokens, shared by both calls |
| `LLM_THINK`, `VERIFIER_THINK` | `false` | thinking for models that support it; `true` errors on models that don't |

Changing `EMBED_MODEL` to one with a different dimension also requires dropping the
`chunks` table and re-ingesting. If a prompt approaches `LLM_NUM_CTX` (very long sessions),
a warning is logged; past the limit Ollama silently drops the start of the prompt.

## Project layout

```
RAG_ENGINE/
  config.py               settings loaded from .env
  main.py                 RAGEngine (one per session) + terminal chat
  ingestion/              chunker, embedder, pgvector store, ingest CLI
  retrieval/              question rewrite (LLM call 1), vector search
  augmentation/           system prompt, examples, JSON payload
  generation/             answer call (LLM call 2), output schema
  helpers/context_window  warns when prompts near or exceed the context window
  db/init/                SQL run when the Postgres volume is first created
docker-compose.yml        Postgres 17 + pgvector
```
