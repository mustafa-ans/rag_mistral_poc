# cli.py
# The two interactive modes: the chat loop and the evaluation harness. Both use the
# RAG pipeline (rag.ask_mistral_rag) and the database (db).
import os
import csv
import json
import contextlib

import db
from rag import ask_mistral_rag


def run_interactive_chat(conn):
    """Ask questions one at a time from the terminal."""
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
            # Re-read the JSON and refresh the table. Use this after editing the FAQ.
            print("Re-ingesting FAQ data into Postgres...")
            n = db.ingest_faq(conn)
            print(f"Done. {n} FAQ rows in the database.")
            continue

        ans = ask_mistral_rag(q, conn)
        print(f"\nAnswer:\n{ans}\n")


def run_evaluation(conn, label: bool = True):
    """
    Run a set of questions through the pipeline.

    We always write a retrieval log (one line per question) that score_eval.py reads.
    With label=True we also show the answer and ask for the three yes/no labels, and
    save them to evaluation_results.csv. With label=False we skip the prompts and only
    write the log, which is handy when you just want the retrieval numbers.
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

    # Keep the results and the log next to the questions file.
    csv_path = os.path.join(os.path.dirname(questions_file) or ".", "evaluation_results.csv")
    write_header = not os.path.exists(csv_path)
    log_path = os.path.join(os.path.dirname(questions_file) or ".", "retrieval_log.jsonl")

    with contextlib.ExitStack() as stack:
        # The log is always written. The CSV is only opened when we're labeling.
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

            # Write this question's retrieval to the log so score_eval.py can score it.
            logf.write(json.dumps({
                "idx": idx,
                "question": q,
                "refused": not bool(context_block),
                "retrieved": [{"faq": h[0], "score": round(float(h[1]), 4)} for h in hits],
                "answer": ans.replace("\n", " ").strip(),
            }, ensure_ascii=False) + "\n")
            logf.flush()

            # Retrieval-only: we already logged it, so print one line and move on.
            if not label:
                top = hits[0][0] if hits else "(none)"
                status = "refused" if not context_block else f"top match -> {top}"
                print(f"[{idx}/{len(questions)}] {status}")
                continue

            print("\n--- Retrieved FAQ context (top-k) ---\n")
            print(context_block if context_block else "[No context found]")

            print("\n--- Model answer ---\n")
            print(ans)

            # Ask for the three labels (y/n).
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
