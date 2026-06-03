import os
import csv
import json
import contextlib
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

import db

load_dotenv()

# Mistral exposes an OpenAI-compatible API, so we reuse the openai SDK
# pointed at Mistral's base URL. MISTRAL_API_KEY is read from .env.
client = OpenAI(
    api_key=os.environ["MISTRAL_API_KEY"],
    base_url="https://api.mistral.ai/v1",
)

MODEL = "mistral-large-latest"

SYSTEM_PROMPT = (
    "You are a helpful assistant for our fictional company.\n"
    "Answer the user's question ONLY using the provided FAQ context.\n"
    "If the answer is not clearly in the context, say:\n"
    "\"Sorry, I don't know the answer to that question based on the available FAQ.\""
)


def ask_mistral_rag(
    question: str,
    conn,
    max_tokens: int = 250,
    k: int = 4,
    min_sim: float = 0.55,
    show_debug: bool = False,
    return_debug: bool = False,
):
    """
    RAG wrapper. Retrieval runs in Postgres via pgvector (db.search).
    If return_debug=True, also returns retrieved context and hits.
    Return:
      - if return_debug=False: answer (str)
      - if return_debug=True: (answer, context_block, hits)
    """
    context_block, hits = db.search(conn, question, k=k, min_sim=min_sim)

    if not context_block:
        answer = "Sorry, I don't know the answer to that question based on the available FAQ."
        if return_debug:
            return answer, "", []
        return answer

    faq_context_system = "Relevant FAQ entries:\n\n" + context_block

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": faq_context_system},
        {"role": "user", "content": question},
    ]

    if show_debug:
        print("\n--- Full prompt (debug) ---\n")
        for m in messages:
            print(f"{m['role'].upper()}: {m['content']}\n")
        print("--- End prompt ---\n")

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,  # low temp for factual, FAQ-grounded answers
        )
        answer = completion.choices[0].message.content
    except OpenAIError as e:
        answer = f"Error from API: {e}"

    if return_debug:
        return answer, context_block, hits
    return answer


def run_interactive_chat(conn):
    """
    Interactive loop: ask questions manually.
    """
    while True:
        q = input("\nAsk a question (or 'reingest' / 'quit' / 'clear'): ").strip()
        cmd = q.lower()

        if cmd == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("RAG FAQ Chatbot (Mistral + pgvector)")
            print("Type your question, or 'reingest' / 'quit' / 'clear'.")
            continue

        if cmd in {"quit", "exit"}:
            break

        if cmd in {"reingest", "rebuild"}:
            # Re-read the JSON from disk and refresh the faq table (re-embeds).
            print("Re-ingesting FAQ data into Postgres...")
            n = db.ingest_faq(conn)
            print(f"Done. {n} FAQ rows in the database.")
            continue

        ans = ask_mistral_rag(q, conn)
        print(f"\nAnswer:\n{ans}\n")


def run_evaluation(conn, label: bool = True):
    """
    Evaluation mode:
    - Reads questions from a text file (one per line).
    - For each question, calls the RAG chatbot.
    - Shows retrieved context and answer.
    - Asks the human to label 3 metrics:
        retrieval_success (y/n)
        answer_correct (y/n)
        no_hallucination_on_oos (y/n)
    - Saves results to evaluation_results.csv.
    """
    questions_file = input(
        "Enter path to questions file (default: evaluation/stress_75/evaluation_questions.txt): "
    ).strip()
    if questions_file == "":
        questions_file = "evaluation/stress_75/evaluation_questions.txt"

    if not os.path.exists(questions_file):
        print(f"Questions file not found at {questions_file}")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    if not questions:
        print("No questions found in the file.")
        return

    print(f"\nLoaded {len(questions)} questions for evaluation.\n")

    # Write results next to the questions file (e.g. evaluation/stress_75/evaluation_results.csv)
    csv_path = os.path.join(os.path.dirname(questions_file) or ".", "evaluation_results.csv")
    write_header = not os.path.exists(csv_path)

    # Per-question retrieval log (fresh each run) for automatic scoring against gold.json
    log_path = os.path.join(os.path.dirname(questions_file) or ".", "retrieval_log.jsonl")

    with contextlib.ExitStack() as stack:
        # The retrieval log is always written; the labels CSV only in labeling mode.
        logf = stack.enter_context(open(log_path, "w", encoding="utf-8"))
        writer = None
        if label:
            csvfile = stack.enter_context(
                open(csv_path, "a", newline="", encoding="utf-8")
            )
            writer = csv.writer(csvfile)
            if write_header:
                writer.writerow(
                    [
                        "question",
                        "answer",
                        "retrieval_success",
                        "answer_correct",
                        "no_hallucination_on_oos",
                    ]
                )

        for idx, q in enumerate(questions, start=1):
            if label:
                print("\n" + "=" * 80)
                print(f"Question {idx}/{len(questions)}:")
                print(q)

            ans, context_block, hits = ask_mistral_rag(
                q,
                conn,
                max_tokens=250,
                k=4,
                min_sim=0.55,
                show_debug=False,
                return_debug=True,
            )

            # Log this question's retrieval (for automatic recall@k scoring via score_eval.py)
            logf.write(json.dumps({
                "idx": idx,
                "question": q,
                "refused": not bool(context_block),
                "retrieved": [{"faq": h[0], "score": round(float(h[1]), 4)} for h in hits],
                "answer": ans.replace("\n", " ").strip(),
            }, ensure_ascii=False) + "\n")
            logf.flush()

            # Retrieval-only mode: log it, print a one-line status, skip labeling.
            if not label:
                top = hits[0][0] if hits else "(none)"
                status = "refused" if not context_block else f"top match -> {top}"
                print(f"[{idx}/{len(questions)}] {status}")
                continue

            print("\n--- Retrieved FAQ context (top-k) ---\n")
            print(context_block if context_block else "[No context found]")

            print("\n--- Model answer ---\n")
            print(ans)

            # Now ask you to label metrics (y/n)
            def ask_yn(prompt: str) -> int:
                while True:
                    val = input(prompt + " (y/n): ").strip().lower()
                    if val in {"y", "yes"}:
                        return 1
                    if val in {"n", "no"}:
                        return 0
                    print("Please answer with 'y' or 'n'.")

            print("\nPlease label this example:")
            retrieval_success = ask_yn(
                "1) Was at least one retrieved FAQ entry sufficient to answer the question?"
            )
            answer_correct = ask_yn(
                "2) Is the model's answer correct and aligned with the FAQ?"
            )
            no_hallucination_on_oos = ask_yn(
                "3) For out-of-scope questions, did the model avoid hallucinating and respond with 'don't know' (or equivalent)? "
                "For in-scope questions that were answered, treat this as 1 if it did NOT invent unsupported facts."
            )

            writer.writerow(
                [
                    q,
                    ans.replace("\n", " ").strip(),
                    retrieval_success,
                    answer_correct,
                    no_hallucination_on_oos,
                ]
            )

            print("Labels saved.")

    print(f"\nDone. Retrieval log written to {log_path}.")
    if label:
        print(f"Labels saved to {csv_path}.")
    print("Score it with:  python evaluation/score_eval.py\n")


if __name__ == "__main__":
    # Connect to Postgres (DATABASE_URL from .env). Run `python setup_db.py` first.
    try:
        conn = db.connect()
    except Exception as e:
        print(f"Could not connect to the database: {e}")
        print("Run `python setup_db.py` first, and check DATABASE_URL in your .env.")
        raise SystemExit(1)

    # Make sure the schema exists (idempotent), then ensure the table has data.
    try:
        db.init_schema(conn)
    except RuntimeError as e:
        print(e)
        raise SystemExit(1)

    if db.faq_count(conn) == 0:
        print("FAQ table is empty; ingesting from faq_jso_data.json ...")
        n = db.ingest_faq(conn)
        print(f"Ingested {n} rows.")
    else:
        choice = input("Re-ingest FAQ data into Postgres? (y/n): ").strip().lower()
        if choice == "y":
            n = db.ingest_faq(conn)
            print(f"Re-ingested {n} rows.")

    # Choose mode: interactive chat or evaluation
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
