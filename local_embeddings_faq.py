# local_embeddings_faq.py
import os

# --- Silence Hugging Face / sentence-transformers startup noise --------------
# These must be set BEFORE importing sentence_transformers / huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")     # no download bars
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # no symlink warning
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
import pickle
import logging

import numpy as np

# Quiet the "unauthenticated requests to the HF Hub" log line and friends.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass  # older huggingface_hub: env vars above already cover this

from sentence_transformers import SentenceTransformer

from faq_data import faq_database

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ~384-dim, fast
EMBEDDING_CACHE = os.path.join("data", "faq_embeddings_local.pkl")
BATCH_SIZE = 100

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > eps else vec

def save_cache(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_cache(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def batch_embeddings(texts):
    model = get_model()
    # model.encode already batches internally; you can still chunk if you want. 
    embs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    # normalize_embeddings=True already L2 normalizes; l2_normalize here would be redundant. 
    return embs

def get_or_build_faq_embeddings(cache_file=EMBEDDING_CACHE, force_rebuild: bool = False):
    corpus_keys = list(faq_database.keys())
    cache = None
    if not force_rebuild:
        cache = load_cache(cache_file)

    if cache and set(cache.keys()) == set(corpus_keys):
        # Use cached embeddings
        return {k: np.array(v, dtype=np.float32) for k, v in cache.items()}

    print(f"Building local embeddings for {len(corpus_keys)} FAQ questions...")
    embs = batch_embeddings(corpus_keys)
    result = {corpus_keys[i]: embs[i] for i in range(len(corpus_keys))}
    save_cache(cache_file, {k: v.tolist() for k, v in result.items()})
    return result

def embed_query_local(query: str) -> np.ndarray:
    model = get_model()
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32)
