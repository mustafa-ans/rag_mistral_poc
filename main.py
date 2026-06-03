# main.py
# Entry point. Connect to Postgres, load the FAQ if the table is empty, then start a mode.
# The work lives in rag.py (pipeline), cli.py (modes), db.py (storage), and
# local_embeddings_faq.py (the embedding model). The faq table is created once by hand,
# see the README for the SQL.
from dotenv import load_dotenv

import db
from cli import run_interactive_chat, run_evaluation

load_dotenv()


if __name__ == "__main__":
    conn = db.connect()

    # if the table is empty, load it from the JSON; otherwise ask before reloading
    if db.faq_count(conn) == 0:
        print("FAQ table is empty, loading from the JSON...")
        n = db.ingest_faq(conn)
        print(f"Loaded {n} rows.")
    else:
        choice = input("Re-ingest FAQ data into Postgres? (y/n): ").strip().lower()
        if choice == "y":
            n = db.ingest_faq(conn)
            print(f"Re-ingested {n} rows.")

    print("\nSelect mode:")
    print("1) Interactive chat")
    print("2) Evaluation mode (batch questions -> labels + retrieval log)")
    print("3) Retrieval-only (no labeling; just write retrieval_log.jsonl for scoring)")
    mode = input("Choose 1, 2 or 3 (default 1): ").strip()
    if mode == "2":
        run_evaluation(conn)
    elif mode == "3":
        run_evaluation(conn, label=False)
    else:
        run_interactive_chat(conn)

    conn.close()
