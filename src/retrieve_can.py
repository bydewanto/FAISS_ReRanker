import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def retrieve(query, faiss_index_path, questions, top_k=3, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([f"query: {query}"], normalize_embeddings=True)
    query_embedding = query_embedding.astype("float32")
    query_embedding = normalize(query_embedding)

    index = faiss.read_index(faiss_index_path)
    D, I = index.search(query_embedding, k=top_k)

    results = [(questions[idx], D[0][i]) for i, idx in enumerate(I[0])]
    return results
