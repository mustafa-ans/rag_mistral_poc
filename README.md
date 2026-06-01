# RAG FAQ Chatbot

A retrieval-augmented generation (RAG) chatbot that answers customer questions **strictly from a curated FAQ knowledge base**. It pairs fast, fully local semantic search with [Mistral Large](https://mistral.ai/) for answer generation, and is deliberately built to refuse questions it cannot ground in the FAQ rather than hallucinate.

The sample knowledge base covers support topics for *Novagear*, a fictional industrial-hardware vendor (ordering, shipping, returns, warranties, certifications, firmware, mounting, integration, and similar).

## How it works

```
question
   │
   ▼
[ local embedding ]      all-MiniLM-L6-v2  (384-dim, runs on CPU, no API call)
   │
   ▼
[ similarity search ]    cosine similarity over the FAQ corpus, top-k above a threshold
   │
   ├─ no match above threshold ──▶  "Sorry, I don't know the answer to that question…"
   │
   ▼
[ Mistral Large ]        answer generated ONLY from the retrieved FAQ entries
   │
   ▼
grounded answer
```

1. **Embed** – Every FAQ question is embedded locally with the `sentence-transformers/all-MiniLM-L6-v2` model. Vectors are L2-normalized and cached to disk so the model only runs once.
2. **Retrieve** – The user's question is embedded the same way, then matched against the corpus by cosine similarity. The top *k* entries above a minimum-similarity threshold are selected. If nothing clears the threshold, the bot short-circuits and returns a fixed "I don't know" response — no LLM call is made.
3. **Generate** – The retrieved FAQ entries are passed to Mistral Large as context, under a system prompt that instructs the model to answer **only** from that context. This keeps answers grounded and makes out-of-scope or adversarial questions (e.g. requests for unrelated personal data) safely deflected.

## Features

- **Grounded answers only.** A strict system prompt plus a retrieval threshold means the bot declines anything not supported by the FAQ instead of guessing.
- **Local, free embeddings.** Semantic search runs on-device via Sentence Transformers — no embedding API costs and no data leaves the machine for retrieval.
- **Cached vectors.** Embeddings are pickled after the first build; subsequent runs start instantly. A `rebuild` command regenerates them on demand.
- **Two run modes.** An interactive chat loop, and a batch evaluation mode that scores answers against labeled metrics.
- **Provider-swappable generation.** Generation goes through Mistral's OpenAI-compatible Chat Completions API, so swapping the model is a one-line change.

## Tech stack

| Concern | Choice |
|---|---|
| Answer generation | Mistral Large (`mistral-large-latest`) via the OpenAI-compatible API |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local) |
| Similarity | Cosine similarity over normalized vectors (NumPy) |
| Config | `.env` via `python-dotenv` |
| Language | Python 3.9+ |

## Project structure

```
.
├── main.py                  # Entry point: interactive chat + evaluation modes
├── faq_data.py              # Loads the FAQ JSON into a question→answer map
├── faq_jso_data.json        # The FAQ knowledge base (~111 Q&A entries)
├── local_embeddings_faq.py  # Local embedding model + on-disk vector cache
├── retrieval_faq.py         # Top-k cosine-similarity retrieval
├── evaluation_questions.txt # Prompts used in evaluation mode
├── evaluation_results.csv   # Labeled evaluation output
├── requirements.txt         # Python dependencies
└── .env                     # MISTRAL_API_KEY (not committed)
```

## Setup

**1. Clone and create a virtual environment**

```bash
git clone <your-repo-url>
cd rag_perplexity
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

> The first run downloads the `all-MiniLM-L6-v2` model from Hugging Face (~90 MB) and caches it locally. This requires an internet connection once; afterwards retrieval works offline.

**3. Add your Mistral API key**

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get a key from the [Mistral console](https://console.mistral.ai/). The `.env` file is git-ignored.

## Usage

```bash
python main.py
```

On startup you'll be asked whether to rebuild embeddings (answer `n` on normal runs — the cache is reused), then to pick a mode:

**1) Interactive chat** — ask questions in a loop. Commands:

- `rebuild` — regenerate the embedding cache
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

**2) Evaluation mode** — runs a list of questions from `evaluation_questions.txt`, shows the retrieved context and the model's answer for each, and prompts you to label three metrics:

- **retrieval_success** — was a sufficient FAQ entry retrieved?
- **answer_correct** — is the answer correct and aligned with the FAQ?
- **no_hallucination_on_oos** — for out-of-scope questions, did it correctly decline instead of inventing facts?

Labels are appended to `evaluation_results.csv` for later analysis.

## Configuration

Retrieval behavior is controlled by arguments in `main.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `k` | 4 | Number of FAQ entries retrieved as context |
| `min_sim` | 0.55 | Minimum cosine similarity to count as a match |
| `max_tokens` | 250 | Max tokens in the generated answer |
| `temperature` | 0.2 | Low, for factual/grounded answers |

Lowering `min_sim` retrieves more loosely (higher recall, more risk of irrelevant context); raising it makes the bot stricter about declining.

## Design notes

- **Why local embeddings + a hosted LLM?** Retrieval is high-volume and latency-sensitive, so it runs on-device for free. Generation benefits from a strong model, so only the final, already-grounded step calls an API.
- **Why the "I don't know" fallback?** For a support assistant, a confidently wrong answer is worse than no answer. The threshold + system prompt together make refusal the default for anything outside the FAQ.
- **Swapping models.** Because generation uses the OpenAI-compatible Chat Completions shape, pointing it at a different Mistral model (or another compatible provider) is just a change to the base URL and model name in `main.py`.
