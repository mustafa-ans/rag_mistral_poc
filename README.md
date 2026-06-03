# RAG FAQ Chatbot

A retrieval-augmented generation (RAG) chatbot that answers customer questions **strictly from a curated FAQ knowledge base**. It pairs fully local semantic search with [Mistral Large](https://mistral.ai/) for answer generation, and is deliberately built to refuse questions it cannot ground in the FAQ rather than hallucinate.

Vectors and FAQ content live in **PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension** — retrieval (top-k + a similarity threshold) runs inside the database. The embedding model itself stays local and free.

The sample knowledge base covers support topics for *Novagear*, a fictional industrial-hardware vendor (ordering, shipping, returns, warranties, certifications, firmware, mounting, integration, and similar).

## How it works

```
question
   │
   ▼
[ local embedding ]      all-MiniLM-L6-v2  (384-dim, runs on CPU, no API call)
   │
   ▼
[ pgvector search ]      cosine similarity in Postgres: top-k above a threshold
   │
   ├─ no match above threshold ──▶  "Sorry, I don't know the answer to that question…"
   │
   ▼
[ Mistral Large ]        answer generated ONLY from the retrieved FAQ entries
   │
   ▼
grounded answer
```

1. **Embed** – Every FAQ question is embedded locally with `sentence-transformers/all-MiniLM-L6-v2`. Vectors are L2-normalized and stored in the `faq` table as `vector(384)` columns.
2. **Retrieve** – The user's question is embedded the same way, then matched in Postgres with pgvector's cosine-distance operator (`<=>`). The top *k* rows above a minimum-similarity threshold are returned. If nothing clears the threshold, the bot short-circuits and returns a fixed "I don't know" — no LLM call is made.
3. **Generate** – The retrieved FAQ entries are passed to Mistral Large as context, under a system prompt that instructs the model to answer **only** from that context. This keeps answers grounded and makes out-of-scope or adversarial questions (e.g. requests for unrelated personal data) safely deflected.

## Features

- **Grounded answers only.** A strict system prompt plus a retrieval threshold means the bot declines anything not supported by the FAQ instead of guessing.
- **Local, free embeddings.** Semantic search embeddings run on-device via Sentence Transformers — no embedding API costs, and FAQ/query text never leaves the machine for retrieval.
- **Postgres + pgvector storage.** FAQ rows and their embeddings live in one indexed table; similarity search is a SQL query, not an in-memory scan. A `reingest` command refreshes the table from the JSON on demand.
- **Two run modes.** An interactive chat loop, and a batch evaluation mode that scores answers against labeled metrics.
- **Provider-swappable generation.** Generation goes through Mistral's OpenAI-compatible Chat Completions API, so swapping the model is a one-line change.

## Tech stack

| Concern | Choice |
|---|---|
| Answer generation | Mistral Large (`mistral-large-latest`) via the OpenAI-compatible API |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local) |
| Vector storage + search | PostgreSQL + pgvector (`vector(384)`, HNSW cosine index) |
| DB driver | `psycopg` (v3) with the `pgvector` adapter |
| Config | `.env` via `python-dotenv` |
| Language | Python 3.9+ |

## Project structure

```
.
├── main.py                       # Entry point: chat + evaluation modes; bootstraps schema + ingest
├── db.py                         # Postgres + pgvector layer: connection, schema, ingest, search
├── local_embeddings_faq.py       # Local embedding model (all-MiniLM-L6-v2)
├── requirements.txt              # Python dependencies
├── .env                          # MISTRAL_API_KEY + DATABASE_URL (not committed)
├── knowledge_base/
│   └── faq_jso_data.json         # FAQ corpus / seed data (112 rows → 101 unique questions)
└── evaluation/
    ├── baseline_25/              # Original 25-question evaluation
    │   ├── evaluation_questions.txt
    │   ├── evaluation_results.csv
    │   └── rag_evaluation_bar_chart.png
    └── stress_75/                # 75-question adversarial stress test
        ├── evaluation_questions.txt
        ├── gold_reference.md     # Expected FAQ + key facts per question (labeling key)
        └── evaluation_results.csv  # Created when you run evaluation mode
```

The repo root holds only Python and config; data and evaluation artifacts live in `knowledge_base/` and `evaluation/`.

## Setup

### Prerequisites

- **Python 3.9+**
- **PostgreSQL** (any recent version) running locally
- A **Mistral API key** — from the [Mistral console](https://console.mistral.ai/)

### 1. Install the pgvector extension

pgvector is a **compiled Postgres extension and is not bundled with PostgreSQL** — you install it once into your Postgres installation.

- **Windows:** follow the "Installation Notes – Windows" in the [pgvector README](https://github.com/pgvector/pgvector#windows). In short: open the *x64 Native Tools Command Prompt for VS*, then

  ```bat
  set "PGROOT=C:\Program Files\PostgreSQL\17"
  git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
  cd pgvector
  nmake /F Makefile.win
  nmake /F Makefile.win install
  ```

  (Adjust `PGROOT` to your Postgres version. Some users prefer a prebuilt binary or the StackBuilder package instead of compiling.)

- **macOS / Linux:** `brew install pgvector`, or `sudo apt install postgresql-XX-pgvector` (match `XX` to your Postgres major version), or build from source per the README.

### 2. Create the database

Create the database the app expects (default name `rag_poc_db`) and enable the extension. Using `psql`:

```sql
CREATE DATABASE rag_poc_db;
\c rag_poc_db
CREATE EXTENSION vector;
```

> `main.py` will also run `CREATE EXTENSION IF NOT EXISTS vector` on startup, so the manual `CREATE EXTENSION` is only needed if your app DB user lacks the privilege to create extensions. The **database itself** must exist before you run the app.

### 3. Configure `.env`

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_api_key_here
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_poc_db
```

Both are git-ignored. Adjust the user, password, host, and port in `DATABASE_URL` to match your Postgres.

### 4. Install dependencies and run

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

On the **first run**, `main.py` connects, creates the `faq` table + HNSW index if missing, and — because the table is empty — embeds every FAQ question locally and loads all rows into Postgres. (The MiniLM model downloads once from Hugging Face, ~90 MB; retrieval is offline thereafter.) Expect it to report the row count, then drop you into the mode picker.

> **Note on `setup_db.py`:** an optional helper script that automates database creation + ingest is kept outside the repo. You don't need it — the manual steps above plus `main.py`'s first-run bootstrap accomplish the same thing.

## Usage

```bash
python main.py
```

If the `faq` table already has data, you'll be asked whether to re-ingest (answer `n` to reuse what's there), then to pick a mode:

**1) Interactive chat** — ask questions in a loop. Commands:

- `reingest` — re-read `faq_jso_data.json` and refresh the table (re-embeds)
- `clear` — clear the screen
- `quit` / `exit` — leave

Example:

```
Ask a question: How do I open an account?
Answer:
To create a Novagear account, go to the Novagear Customer Portal and click 'Sign Up'…

Ask a question: What's the capital of France?
Answer:
Sorry, I don't know the answer to that question based on the available FAQ.
```

**2) Evaluation mode** — runs a question set (default: `evaluation/stress_75/evaluation_questions.txt`; you can enter a different path, e.g. `evaluation/baseline_25/evaluation_questions.txt`), shows the retrieved context and the model's answer for each, and prompts you to label three metrics:

- **retrieval_success** — was a sufficient FAQ entry retrieved?
- **answer_correct** — is the answer correct and aligned with the FAQ?
- **no_hallucination_on_oos** — for out-of-scope questions, did it correctly decline instead of inventing facts?

Labels are appended to an `evaluation_results.csv` written **next to the question set** (e.g. `evaluation/stress_75/evaluation_results.csv`). The stress set ships with a `gold_reference.md` answer key that tells you the expected FAQ and key facts for each question, so labeling is fast and consistent — and lets you compute retrieval recall@k objectively.

## Updating the knowledge base

Edit `knowledge_base/faq_jso_data.json` (each row is `{"row_idx": N, "row": {"question": "...", "answer": "..."}}`), then either type `reingest` inside chat mode or restart the app and answer `y` to the re-ingest prompt. Ingestion upserts on the question text, so existing rows are refreshed and new ones added.

## Configuration

Retrieval behavior is controlled by arguments in `main.py` / `db.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `k` | 4 | Number of FAQ entries retrieved as context |
| `min_sim` | 0.55 | Minimum cosine similarity (`1 - (embedding <=> query)`) to count as a match |
| `max_tokens` | 250 | Max tokens in the generated answer |
| `temperature` | 0.2 | Low, for factual/grounded answers |

Lowering `min_sim` retrieves more loosely (higher recall, more risk of irrelevant context); raising it makes the bot stricter about declining.

## Design notes

- **Why local embeddings + a hosted LLM?** Retrieval is high-volume and latency-sensitive, so it runs on-device for free. Generation benefits from a strong model, so only the final, already-grounded step calls an API.
- **Why Postgres + pgvector?** A single source of truth for FAQ data and vectors, durable ACID writes, an indexable similarity search, and room to grow (metadata filtering, larger corpora) — without changing the embedding model or pipeline. At ~100 rows it isn't faster than an in-memory scan; the benefit is architectural, and the design scales as the corpus grows.
- **Why the "I don't know" fallback?** For a support assistant, a confidently wrong answer is worse than no answer. The threshold + system prompt together make refusal the default for anything outside the FAQ.

## Future plans

Roughly in priority order:

1. **A measured evaluation harness** — a labeled gold set (question → correct FAQ) with automated retrieval metrics (recall@k, precision@k) and answer-faithfulness scoring (e.g. RAGAS), wired into CI so changes are provable rather than anecdotal.
2. **Stronger retrieval** — a larger local embedder (BGE/E5 family) and embedding question + answer together, then **cross-encoder reranking** of a wider candidate set for higher precision.
3. **Reliability** — exponential backoff/retries on the Mistral call instead of surfacing transient API errors as answers.
4. **Conversation memory** — multi-turn support with history-aware query rewriting for follow-up questions.
5. **Scale** — pgvector already supports HNSW/IVFFlat indexes; tune them (and add metadata filtering) as the corpus grows well beyond a few hundred entries.
