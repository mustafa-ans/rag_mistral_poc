# local_embeddings_faq.py
# Local sentence-embedding model (all-MiniLM-L6-v2, 384-dim).
# Storage/retrieval of the vectors now lives in db.py (Postgres + pgvector);
# this module's only job is turning text into normalised vectors.
import os

# --- Silence Hugging Face / sentence-transformers startup noise --------------
# These must be set BEFORE importing sentence_transformers / huggingface_hub.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")     # no download bars
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # no symlink warning
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # ~384-dim, fast
BATCH_SIZE = 100

_model = None


def get_model():
    """Lazy singleton: load the model once, on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def batch_embeddings(texts):
    """Embed a list of texts -> (N, 384) float32 array, L2-normalised."""
    model = get_model()
    # We apply L2 normalization to both stored FAQ embeddings and the user query embedding so each vector has unit length.
    # That removes magnitude from the comparison, and then the dot product (pgvector cosine) becomes equivalent to cosine similarity.
    embs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embs


def embed_query_local(query: str) -> np.ndarray:
    """Embed a single query the same way as the corpus (normalised float32)."""
    model = get_model()
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32)
