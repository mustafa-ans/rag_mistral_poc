# faq_data.py
import json
import os

def load_faq_database():
    """
    Load faq_jso_data.json from the same directory and return:
    - faq_qa: dict[question] -> answer
    - corpus_texts: list[str] where each item is 'Q: ...\nA: ...'
    """
    base = os.path.dirname(__file__)
    path = os.path.join(base, "faq_jso_data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAQ JSON not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    faq_qa = {}
    corpus_texts = []
    for item in data:
        row = item.get("row", {})
        q = row.get("question")
        a = row.get("answer", "")
        if not q:
            continue
        faq_qa[q] = a
        corpus_texts.append(f"Q: {q}\nA: {a}")

    return faq_qa, corpus_texts

# Initialize module-level variables so other modules can `from faq_data import faq_database`
try:
    faq_database, corpus_texts = load_faq_database()
except FileNotFoundError:
    # fall back to empty structures to avoid import-time crashes during testing
    faq_database = {}
    corpus_texts = []
