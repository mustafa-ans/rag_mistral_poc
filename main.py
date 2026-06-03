from dotenv import load_dotenv

import db
from cli import run_interactive_chat, run_evaluation

load_dotenv()


if __name__ == "__main__":
    try:
        conn = db.connect()
    except Exception as e:
        print(f"Could not connect to the database: {e}")
        raise SystemExit(1)

    # Create the table and index if they aren't there yet (safe to run every time).
    try:
        db.init_schema(conn)
    except RuntimeError as e:
        print(e)
        raise SystemExit(1)

    # If the table is empty, load it from the JSON. Otherwise ask before reloading.
    if db.faq_count(conn) == 0:
        print("FAQ table is empty; ingesting from faq_jso_data.json ...")
        n = db.ingest_faq(conn)
        print(f"Ingested {n} rows.")
    else:
        choice = input("Re-ingest FAQ data into Postgres? (y/n): ").strip().lower()
        if choice == "y":
            n = db.ingest_faq(conn)
            print(f"Re-ingested {n} rows.")

    # Pick a mode.
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
