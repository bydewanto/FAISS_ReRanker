import numpy as np
from sklearn.preprocessing import normalize

def normalize_embeddings(embeddings):
    """Normalize vectors for cosine similarity."""
    embeddings = embeddings.astype("float32")
    return normalize(embeddings)
