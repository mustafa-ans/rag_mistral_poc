# rag.py
# The RAG pipeline. For a question we pull the most relevant FAQ entries from the database
# (db.search) and ask Mistral to answer using only those entries. We talk to Mistral through
# the OpenAI SDK because its API is OpenAI-compatible.
import os

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

import db

load_dotenv()  # load .env so MISTRAL_API_KEY is set before we build the client

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


def ask_mistral_rag(question, conn, max_tokens=250, k=4, min_sim=0.55,
                    show_debug=False, return_debug=False):
    # Answer a question with retrieval-augmented generation. Retrieval runs in Postgres
    # (db.search). With return_debug=True we also return the retrieved context and the hits,
    # which the evaluation mode needs.
    context_block, hits = db.search(conn, question, k=k, min_sim=min_sim)

    if not context_block:
        # Nothing cleared the threshold, so we don't call the model and just say we don't know.
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
            temperature=0.2,  # keep it low so answers stay factual and close to the FAQ
        )
        answer = completion.choices[0].message.content
    except OpenAIError as e:
        answer = f"Error from API: {e}"

    if return_debug:
        return answer, context_block, hits
    return answer
