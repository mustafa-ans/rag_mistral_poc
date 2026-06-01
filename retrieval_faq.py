# retrieval_faq.py
import numpy as np
from faq_data import faq_database
from local_embeddings_faq import embed_query_local

def find_top_k_similar_faq(query_embedding: np.ndarray, vector_db: dict, k: int = 5, min_sim: float = 0.5):
    keys = list(vector_db.keys())
    mat = np.vstack([vector_db[k] for k in keys])
    sims = mat.dot(query_embedding)
    valid_idx = np.where(sims >= min_sim)[0]
    if valid_idx.size == 0:
        return []
    k_eff = min(k, valid_idx.size)
    top_unsorted = valid_idx[np.argpartition(sims[valid_idx], -k_eff)[-k_eff:]]
    top_sorted = top_unsorted[np.argsort(sims[top_unsorted])[::-1]]
    return [(keys[i], float(sims[i])) for i in top_sorted]

def retrieve_faq_context(query: str, vector_db: dict, k: int = 5, min_sim: float = 0.5):
    q_emb = embed_query_local(query)
    top_hits = find_top_k_similar_faq(q_emb, vector_db, k=k, min_sim=min_sim)
    if not top_hits:
        return "", []
    blocks = []
    for q_text, score in top_hits:
        a_text = faq_database.get(q_text, "")
        if a_text:
            blocks.append(f"Q: {q_text}\nA: {a_text}")
    context_block = "\n\n---\n\n".join(blocks)
    return context_block, top_hits
