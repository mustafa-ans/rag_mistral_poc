# db.py
# Postgres + pgvector storage and retrieval layer.
#
# Replaces the old pickle embedding cache. The FAQ data AND its 384-dim
# embeddings live in a single `faq` table; retrieval (top-k + similarity
# threshold) runs inside Postgres via pgvector's cosine-distance operator.
#
# The embedding model is still local (sentence-transformers, all-MiniLM-L6-v2);
# only WHERE the vectors are stored and searched has changed. The embedder is
# imported lazily so that simply connecting to the DB does not load the model.

import os
import json

import psycopg
from pgvector.psycopg import register_vector

EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension


# --- connection -------------------------------------------------------------

def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env, e.g.\n"
            "  DATABASE_URL=postgresql://postgres:PASSWORD@localhost:5432/rag_poc_db"
        )
    return url


def connect(url: str = None) -> psycopg.Connection:
    """Open a connection and register the pgvector type adapters on it."""
    conn = psycopg.connect(url or get_database_url())
    register_vector(conn)  # lets us pass/receive numpy arrays as `vector`
    return conn


# --- schema -----------------------------------------------------------------

_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS faq (
    id         SERIAL PRIMARY KEY,
    row_idx    INTEGER,
    question   TEXT NOT NULL UNIQUE,
    answer     TEXT NOT NULL,
    embedding  vector({EMBED_DIM}) NOT NULL
);
"""

# HNSW index for cosine distance. Embeddings are L2-normalised, so cosine is the
# right metric. At ~100 rows the index is optional (a scan is instant), but it is
# best practice and makes the design scale without code changes.
_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS faq_embedding_hnsw
    ON faq USING hnsw (embedding vector_cosine_ops);
"""


def init_schema(conn: psycopg.Connection) -> None:
    """Enable pgvector and create the table + index. Idempotent."""
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except psycopg.Error as e:
            conn.rollback()
            raise RuntimeError(
                "Failed to enable the pgvector extension. Install pgvector for your "
                "Postgres and ensure your DB user may CREATE EXTENSION "
                "(see pgvector_migration.md).\nOriginal error: " + str(e)
            )
        cur.execute(_TABLE_SQL)
        cur.execute(_INDEX_SQL)
    conn.commit()


# --- ingestion --------------------------------------------------------------

def _load_rows(json_path: str = None):
    """Read the FAQ JSON and return de-duplicated (row_idx, question, answer)
    tuples. De-dup is on the question text (later rows win), matching the old
    dict behaviour."""
    base = os.path.dirname(__file__)
    path = json_path or os.path.join(base, "faq_jso_data.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen = {}  # question -> (row_idx, question, answer); preserves last occurrence
    for item in data:
        row = item.get("row", {})
        q = row.get("question")
        a = row.get("answer", "")
        if not q:
            continue
        seen[q] = (item.get("row_idx"), q, a)
    return list(seen.values())


def faq_count(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM faq;")
        return cur.fetchone()[0]


def ingest_faq(conn: psycopg.Connection, json_path: str = None) -> int:
    """(Re)load the FAQ JSON into the `faq` table, embedding each question
    locally. Idempotent: upserts on the unique question, refreshing the answer
    and embedding. Returns the number of rows ingested."""
    from local_embeddings_faq import batch_embeddings  # lazy: avoid model load on import

    rows = _load_rows(json_path)
    questions = [q for (_idx, q, _a) in rows]
    embeddings = batch_embeddings(questions)  # 384-dim, L2-normalised

    with conn.cursor() as cur:
        for (row_idx, q, a), emb in zip(rows, embeddings):
            cur.execute(
                """
                INSERT INTO faq (row_idx, question, answer, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (question) DO UPDATE
                    SET row_idx   = EXCLUDED.row_idx,
                        answer    = EXCLUDED.answer,
                        embedding = EXCLUDED.embedding;
                """,
                (row_idx, q, a, emb),
            )
    conn.commit()
    return len(rows)


# --- retrieval --------------------------------------------------------------

def search(conn: psycopg.Connection, query: str, k: int = 4, min_sim: float = 0.55):
    """Embed the query locally and return (context_block, hits) — the same shape
    the old retrieve_faq_context returned — but the top-k and similarity threshold
    run in Postgres.

    pgvector's `<=>` is cosine DISTANCE; similarity = 1 - distance. We keep rows
    with similarity >= min_sim and order by nearest first.
    """
    from local_embeddings_faq import embed_query_local  # lazy

    q_emb = embed_query_local(query)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT question, answer, 1 - (embedding <=> %s) AS similarity
            FROM faq
            WHERE 1 - (embedding <=> %s) >= %s
            ORDER BY embedding <=> %s
            LIMIT %s;
            """,
            (q_emb, q_emb, min_sim, q_emb, k),
        )
        results = cur.fetchall()

    if not results:
        return "", []

    hits = [(question, float(sim)) for (question, _answer, sim) in results]
    blocks = [f"Q: {question}\nA: {answer}"
              for (question, answer, _sim) in results if answer]
    context_block = "\n\n---\n\n".join(blocks)
    return context_block, hits
