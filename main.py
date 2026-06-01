import os
import csv
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from faq_data import faq_database
from local_embeddings_faq import get_or_build_faq_embeddings
from retrieval_faq import retrieve_faq_context

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
    vector_db: dict,
    max_tokens: int = 250,
    k: int = 4,
    min_sim: float = 0.55,
    show_debug: bool = False,
    return_debug: bool = False,
):
    """
    RAG wrapper. If return_debug=True, also returns retrieved context and hits.
    Return:
      - if return_debug=False: answer (str)
      - if return_debug=True: (answer, context_block, hits)
    """
    context_block, hits = retrieve_faq_context(question, vector_db, k=k, min_sim=min_sim)

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


def run_interactive_chat(vector_db):
    """
    Original interactive loop: ask questions manually.
    """
    while True:
        q = input("\nAsk a question (or 'rebuild' / 'quit' / 'clear'): ").strip()
        cmd = q.lower()

        if cmd == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("RAG FAQ Chatbot (Mistral-powered)")
            print("Type your question, or 'rebuild' / 'quit' / 'clear'.")
            continue

        if cmd in {"quit", "exit"}:
            break

        if cmd == "rebuild":
            print("Rebuilding embeddings...")
            vector_db = get_or_build_faq_embeddings(force_rebuild=True)
            print("Done.")
            continue

        ans = ask_mistral_rag(q, vector_db)
        print(f"\nAnswer:\n{ans}\n")


def run_evaluation(vector_db):
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
        "Enter path to questions file (default: evaluation_questions.txt): "
    ).strip()
    if questions_file == "":
        questions_file = "evaluation_questions.txt"

    if not os.path.exists(questions_file):
        print(f"Questions file not found at {questions_file}")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    if not questions:
        print("No questions found in the file.")
        return

    print(f"\nLoaded {len(questions)} questions for evaluation.\n")

    csv_path = "evaluation_results.csv"
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
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
            print("\n" + "=" * 80)
            print(f"Question {idx}/{len(questions)}:")
            print(q)

            ans, context_block, hits = ask_mistral_rag(
                q,
                vector_db,
                max_tokens=250,
                k=4,
                min_sim=0.55,
                show_debug=False,
                return_debug=True,
            )

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

    print(f"\nEvaluation completed. Results saved to {csv_path}.\n")


if __name__ == "__main__":
    # Ask once at startup
    choice = input("Rebuild FAQ embeddings? (y/n): ").strip().lower()
    if choice == "":
        choice = "n"
    force = choice == "y"

    vector_db = get_or_build_faq_embeddings(force_rebuild=force)

    # Choose mode: interactive chat or evaluation
    print("\nSelect mode:")
    print("1) Interactive chat")
    print("2) Evaluation mode (batch questions -> CSV metrics)")
    mode = input("Choose 1 or 2 (default 1): ").strip()
    if mode == "2":
        run_evaluation(vector_db)
    else:
        run_interactive_chat(vector_db)
