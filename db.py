# db.py
# Postgres + pgvector storage and search. The FAQ rows and their 384-dim embeddings sit in
# one faq table, and the top-k search with the similarity threshold runs in Postgres using
# pgvector's cosine-distance operator. The faq table is created once by hand (see the README).

import os
import json

import psycopg
from pgvector.psycopg import register_vector

from local_embeddings_faq import batch_embeddings, embed_query_local


def connect():
    # open a connection and register pgvector so we can pass numpy arrays as vectors
    conn = psycopg.connect(os.environ["DATABASE_URL"])
    register_vector(conn)
    return conn


def faq_count(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM faq;")
        return cur.fetchone()[0]


def load_rows(json_path=None):
    # read the FAQ JSON into (row_idx, question, answer) tuples; duplicate questions are
    # collapsed, keeping the last one
    base = os.path.dirname(__file__)
    path = json_path or os.path.join(base, "knowledge_base", "faq_jso_data.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    seen = {}
    for item in data:
        row = item.get("row", {})
        q = row.get("question")
        a = row.get("answer", "")
        if not q:
            continue
        seen[q] = (item.get("row_idx"), q, a)
    return list(seen.values())


def ingest_faq(conn, json_path=None):
    # load the FAQ into the table, embedding each question; re-running updates the answer and
    # embedding for a question that's already there. returns the row count
    rows = load_rows(json_path)
    questions = [q for (_idx, q, _a) in rows]
    embeddings = batch_embeddings(questions)

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


def search(conn, query, k=4, min_sim=0.55):
    # embed the query and return (context_block, hits). the top-k and the threshold are done
    # in Postgres. <=> is cosine distance, so similarity is 1 - distance; we keep rows at or
    # above min_sim and order them nearest first
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
    blocks = [f"Q: {question}\nA: {answer}" for (question, answer, _sim) in results if answer]
    context_block = "\n\n---\n\n".join(blocks)
    return context_block, hits
