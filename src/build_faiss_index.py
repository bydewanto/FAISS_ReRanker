import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils import normalize_embeddings

def build_index(csv_path="data/corpus_all.csv", model_name="all-MiniLM-L6-v2", save_path="faiss_index.bin"):
    df = pd.read_csv(csv_path)
    questions = df["Pertanyaan"].tolist()
    answers = df["Jawaban"].tolist()

    model = SentenceTransformer(model_name)
    question_embeddings = model.encode(questions, show_progress_bar=True, convert_to_numpy=True)
    question_embeddings = normalize_embeddings(question_embeddings)

    dim = question_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(question_embeddings)
    faiss.write_index(index, save_path)

    print(f"✅ FAISS index built and saved to {save_path}")
    return index, questions, answers
