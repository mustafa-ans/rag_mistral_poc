# RAG FAQ Chatbot

A retrieval-augmented generation (RAG) chatbot that answers customer questions only from a
curated FAQ. The semantic search runs locally and [Mistral Large](https://mistral.ai/) writes
the answer. If a question isn't covered by the FAQ, the bot says it doesn't know instead of
making something up.

The FAQ rows and their embeddings are stored in PostgreSQL with the
[pgvector](https://github.com/pgvector/pgvector) extension, so the search (top-k plus a
similarity threshold) happens inside the database. The embedding model itself stays on your
machine.

The sample FAQ is for *Novagear*, a made-up industrial-hardware company (ordering, shipping,
returns, warranties, certifications, firmware, mounting, integration, and so on).

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
   ├─ nothing clears the threshold ─▶  "Sorry, I don't know..."
   │
   ▼
[ Mistral Large ]        answer written only from the retrieved FAQ entries
   │
   ▼
grounded answer
```

1. Embed. Each FAQ question is embedded locally with `sentence-transformers/all-MiniLM-L6-v2`. The vectors are L2-normalised and stored in the `faq` table as `vector(384)`.
2. Retrieve. The user's question is embedded the same way and matched in Postgres with pgvector's cosine-distance operator (`<=>`). We keep the top *k* rows above a minimum similarity. If nothing clears the threshold the bot returns a fixed "I don't know" and never calls the model.
3. Generate. The retrieved entries go to Mistral Large as context, with a system prompt telling it to answer only from that context. That's what keeps the answers grounded and makes the bot decline out-of-scope or adversarial questions.

## Features

- Answers stay inside the FAQ. The system prompt plus the similarity threshold mean the bot declines anything the FAQ doesn't cover instead of guessing.
- Local embeddings. The semantic search runs on your machine through Sentence Transformers, so there's no embedding API cost and the FAQ and query text don't leave the machine for retrieval.
- Postgres + pgvector. The rows and their embeddings sit in one indexed table and the search is a SQL query. A `reingest` command reloads the table from the JSON after you change it.
- Two run modes. An interactive chat loop and an evaluation mode that scores answers against labelled metrics.
- Easy to swap the model. Generation goes through Mistral's OpenAI-compatible API, so pointing it at a different model is a one-line change.

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
├── main.py                       # Entry point: connect, set up the schema/data, start a mode
├── rag.py                        # RAG pipeline: retrieve (db) + generate (Mistral)
├── cli.py                        # Chat + evaluation/retrieval-only modes
├── db.py                         # Postgres + pgvector: connection, schema, ingest, search
├── local_embeddings_faq.py       # Local embedding model (all-MiniLM-L6-v2)
├── requirements.txt              # Python dependencies
├── .env                          # MISTRAL_API_KEY + DATABASE_URL (not committed)
├── knowledge_base/
│   └── faq_jso_data.json         # The FAQ (112 rows, 101 unique questions)
└── evaluation/
    ├── score_eval.py             # Scores a run against gold.json (recall@k, etc.)
    ├── baseline_25/              # The original 25-question evaluation
    │   ├── evaluation_questions.txt
    │   ├── evaluation_results.csv
    │   └── rag_evaluation_bar_chart.png
    └── stress_75/                # 75-question adversarial stress test
        ├── evaluation_questions.txt
        ├── gold_reference.md     # Answer key for a human (expected FAQ + key facts)
        ├── gold.json             # Answer key for score_eval.py
        ├── retrieval_log.jsonl   # What was retrieved per question (written each run)
        ├── evaluation_results.csv  # Your manual labels (written in labeling mode)
        └── stress75_scorecard.png  # The metrics chart (written by score_eval.py)
```

The repo root has only Python and config. The data and evaluation files live under
`knowledge_base/` and `evaluation/`.

## Setup

### Prerequisites

- Python 3.9+
- PostgreSQL (any recent version) running locally
- A Mistral API key from the [Mistral console](https://console.mistral.ai/)

### 1. Install the pgvector extension

pgvector is a compiled Postgres extension and doesn't ship with PostgreSQL, so you install it
once into your Postgres.

- Windows: follow "Installation Notes - Windows" in the [pgvector README](https://github.com/pgvector/pgvector#windows). In short, open the *x64 Native Tools Command Prompt for VS* and run:

  ```bat
  set "PGROOT=C:\Program Files\PostgreSQL\17"
  git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
  cd pgvector
  nmake /F Makefile.win
  nmake /F Makefile.win install
  ```

  (Set `PGROOT` to your Postgres version. A prebuilt binary or the StackBuilder package works too if you'd rather not compile.)

- macOS / Linux: `brew install pgvector`, or `sudo apt install postgresql-XX-pgvector` (match `XX` to your Postgres major version), or build from source.

### 2. Create the database

Create the database the app expects (default name `rag_poc_db`) and enable the extension. With
`psql`:

```sql
CREATE DATABASE rag_poc_db;
\c rag_poc_db
CREATE EXTENSION vector;
```

`main.py` also runs `CREATE EXTENSION IF NOT EXISTS vector` on startup, so the manual
`CREATE EXTENSION` is only needed if your DB user can't create extensions. The database itself
has to exist before you run the app.

### 3. Configure `.env`

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_api_key_here
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_poc_db
```

Both are git-ignored. Change the user, password, host, and port in `DATABASE_URL` to match your
Postgres.

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

On the first run `main.py` connects, creates the `faq` table and HNSW index if they're missing,
and (because the table is empty) embeds every FAQ question locally and loads the rows into
Postgres. The MiniLM model downloads once from Hugging Face (~90 MB) and is cached on disk. You
may see a one-time "set a HF_TOKEN for higher rate limits" message during that download; it's
harmless (a version check, not an upload of your data). Once the model is cached, the app sets
`HF_HUB_OFFLINE` on its own, so every later run is offline and quiet. After it reports the row
count it drops you into the mode picker.

If you kept `setup_db.py` (it's a local helper, not in the repo), running it does the database
creation and the load in one step. You don't need it; the manual steps above plus the first-run
bootstrap do the same thing.

## Usage

```bash
python main.py
```

If the `faq` table already has data you'll be asked whether to re-ingest (answer `n` to keep
what's there), then to pick a mode:

1) Interactive chat. Ask questions in a loop. Commands:

- `reingest` re-reads `knowledge_base/faq_jso_data.json` and refreshes the table
- `clear` clears the screen
- `quit` / `exit` leaves

Example:

```
Ask a question: How do I open an account?
Answer:
To create a Novagear account, go to the Novagear Customer Portal and click 'Sign Up'…

Ask a question: What's the capital of France?
Answer:
Sorry, I don't know the answer to that question based on the available FAQ.
```

2) Evaluation mode. Runs a question set (default `evaluation/stress_75/evaluation_questions.txt`;
you can type a different path, e.g. `evaluation/baseline_25/evaluation_questions.txt`), shows the
retrieved context and the answer for each, and asks you to label three things:

- retrieval_success: was a good-enough FAQ entry retrieved?
- answer_correct: is the answer right and in line with the FAQ?
- no_hallucination_on_oos: for out-of-scope questions, did it decline instead of inventing facts?

Labels are appended to an `evaluation_results.csv` next to the question set. The stress set comes
with `gold_reference.md`, which tells you the expected FAQ and key facts per question, so labeling
is quicker and more consistent.

3) Retrieval-only mode. Same as evaluation mode but with no labeling prompts. It runs every
question and writes `retrieval_log.jsonl` (the retrieved FAQs per question). Use it when you just
want the retrieval numbers fast; you can do the labeling separately later.

## Scoring a stress-test run

Manual labels tell you whether the answers were good; they don't measure retrieval on their own.
So every evaluation run also writes a retrieval log, `retrieval_log.jsonl`, next to the question
set, recording which FAQs were retrieved (with scores) and whether the bot refused. The
`stress_75` set comes with `gold.json`, which lists the FAQ that should be retrieved for each
question.

After a run (mode 2 or mode 3), score the log against the key:

```bash
python evaluation/score_eval.py            # defaults to evaluation/stress_75
python evaluation/score_eval.py --no-chart # skip the PNG
```

It prints the numbers and saves a chart, `stress75_scorecard.png`, in the same folder: your three
labelled metrics (retrieval success, answer correctness, no-hallucination on OOS) plus one
retrieval recall@k bar (needs `matplotlib`, which is in `requirements.txt`). The terminal output
also breaks retrieval down per category if you want it; the chart is the short version. It
reports:

- In-scope recall@k: for each in-scope question, did an expected FAQ show up in the top-k? Overall and per category (paraphrase, jargon, hard_negative, multi_part, typo, rambling, ambiguous, negation).
- Multi-part questions: how often both expected FAQs were retrieved vs at least one.
- Out-of-scope and adversarial refusal: how often the bot declined to retrieve anything. (Whether the answer itself safely declined when something was retrieved is the human-label call, the `no_hallucination_on_oos` column.)
- If `evaluation_results.csv` is there, it also prints your manual-label totals.

How matching works: each expected FAQ is identified by a key phrase from its question, and it
counts as retrieved if that phrase shows up in any retrieved FAQ. The `hard_negative` category
(near-duplicate stock and return entries) is the one most likely to be weak, so `score_eval.py`
reports it on its own.

## Updating the knowledge base

Edit `knowledge_base/faq_jso_data.json` (each row is
`{"row_idx": N, "row": {"question": "...", "answer": "..."}}`), then either type `reingest` in
chat mode or restart and answer `y` to the re-ingest prompt. Ingestion upserts on the question
text, so existing rows are refreshed and new ones are added.

## Configuration

Retrieval behaviour is set by arguments in `rag.py` / `db.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `k` | 4 | How many FAQ entries to retrieve as context |
| `min_sim` | 0.55 | Minimum cosine similarity (`1 - (embedding <=> query)`) to count as a match |
| `max_tokens` | 250 | Max tokens in the answer |
| `temperature` | 0.2 | Low, to keep answers factual |

Lowering `min_sim` retrieves more loosely (higher recall, more risk of irrelevant context);
raising it makes the bot stricter about declining.

## Design notes

- Why local embeddings with a hosted LLM? Retrieval runs often and needs to be quick, so it stays on the machine with no API cost. The answer benefits from a strong model, so only that last step (which is already grounded in the retrieved FAQ) calls an API.
- Why Postgres + pgvector? One place for the FAQ data and the vectors, proper writes, an indexable search, and room to add metadata filtering or grow the corpus later, without changing the model or the rest of the pipeline. At ~100 rows it isn't faster than an in-memory scan; the point is the structure, which holds up as the corpus grows.
- Why the "I don't know" fallback? For a support bot a confidently wrong answer is worse than no answer, so the threshold plus the system prompt make declining the default for anything outside the FAQ.

## Future plans

Roughly in order:

1. A proper evaluation harness: a labelled gold set (question to correct FAQ) with automatic retrieval metrics (recall@k, precision@k) and answer-faithfulness scoring (e.g. RAGAS), run in CI so changes are measured rather than guessed at.
2. Better retrieval: a larger local embedder (BGE/E5 family), embedding the question and answer together, and a cross-encoder reranker over a wider candidate set.
3. Reliability: retries with backoff on the Mistral call instead of returning a transient API error as the answer.
4. Conversation memory: multi-turn support, rewriting a follow-up into a standalone question before retrieval.
5. Scale: tune the pgvector HNSW/IVFFlat index and add metadata filtering once the corpus is well past a few hundred entries.
