import os
import logging
import numpy as np

# Quiet down the Hugging Face
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")     
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
_hf_home = os.environ.get("HF_HOME")
_cache_root = (Path(_hf_home) / "hub") if _hf_home else (Path.home() / ".cache" / "huggingface" / "hub")
if (_cache_root / "models--sentence-transformers--all-MiniLM-L6-v2").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except Exception:
    pass  

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 100

_model = None


def get_model():
    # load the model once and reuse it
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def batch_embeddings(texts):
    # embed a list of texts into an (N, 384) float32 array, L2-normalised
    model = get_model()
    # We apply L2 normalization to both stored FAQ embeddings and the user query embedding so each vector has unit length.
    # That removes magnitude from the comparison, and then the dot product (pgvector cosine) becomes equivalent to cosine similarity.
    embs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embs


def embed_query_local(query: str) -> np.ndarray:
    # embed a single query the same way as the corpus (normalised float32)
    model = get_model()
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype(np.float32)
