# !pip install sentence-transformers faiss-cpu transformers
# !pip install accelerate bitsandbytes

# !pip install bitsandbytes
# !pip install accelerate
# !pip install transformers
# !pip install sentencepiece

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

df = pd.read_csv("/kaggle/input/cp-all/corpus_all.csv")  # Pastikan ada kolom 'Pertanyaan' dan 'Jawaban'
questions = df["Pertanyaan"].tolist()
answers = df["Jawaban"].tolist()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Cepat dan ringan

question_embeddings = embed_model.encode(questions, show_progress_bar=True)

dimension = question_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(np.array(question_embeddings))

from sklearn.preprocessing import normalize

question_embeddings = embed_model.encode(questions, show_progress_bar=True, convert_to_numpy=True)
question_embeddings = question_embeddings.astype('float32')
question_embeddings = normalize(question_embeddings)

dimension = question_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)  # pakai Inner Product (cosine similarity setelah normalisasi)
faiss_index.add(question_embeddings)

query = "apa itu semester antara?"
query_embedding = embed_model.encode([f"query: {query}"], normalize_embeddings=True)
query_embedding = query_embedding.astype('float32')
query_embedding = normalize(query_embedding)

D, I = faiss_index.search(query_embedding, k=3)
for idx in I[0]:
    print(f"Pertanyaan: {questions[idx]}")
    print(f"Jawaban: {answers[idx]}")
    print("-" * 50)
    
    
from huggingface_hub import login
login(token="def_token")

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import numpy as np

model_name = "GajahTerbang/llama3-8B-4BitIndoAlpacaFineTuned"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
)
streamer = TextStreamer(tokenizer)

from sentence_transformers import CrossEncoder

retrieved_candidates = [(questions[idx], answers[idx]) for idx in I[0]]

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
rerank_inputs = [(query, q) for q, a in retrieved_candidates]
scores = reranker.predict(rerank_inputs)
reranked = sorted(zip(scores, retrieved_candidates), key=lambda x: x[0], reverse=True)

# Display reranked results
print("=== Reranked Results ===")
for score, (q, a) in reranked:
    print(f"[Score: {score:.4f}]")
    print(f"Pertanyaan: {q}")
    print(f"Jawaban: {a}")
    print("-" * 50)
    
from sentence_transformers import CrossEncoder

# --- Config ---
alpha = 0.01  # Weight for length penalty/boost; tune this value
length_mode = 'boost'  # 'boost' or 'penalize' based on your task

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

retrieved_candidates = [(questions[idx], answers[idx]) for idx in I[0]]

rerank_inputs = [(query, q) for q, a in retrieved_candidates]

scores = reranker.predict(rerank_inputs)

adjusted_scores = []
for (score, (q, a)) in zip(scores, retrieved_candidates):
    word_length = len(q.split())
    
    # Normalize or boost based on length
    if length_mode == 'boost':
        adjusted_score = score + alpha * word_length
    else:  # 'penalize'
        adjusted_score = score - alpha * word_length

    adjusted_scores.append((adjusted_score, word_length, q, a))

# --- Rerank by adjusted score ---
reranked = sorted(adjusted_scores, key=lambda x: x[0], reverse=True)

# --- Display reranked results ---
print("=== Reranked Results with Length Adjustment ===")
for score, length, q, a in reranked:
    print(f"[Adjusted Score: {score:.4f}] (Length: {length} words)")
    print(f"Pertanyaan: {q}")
    print(f"Jawaban: {a}")
    print("-" * 50)
    
# Prompt Alpaca-style
alpaca_prompt = (
    "Instruksi:\n{0}\n\n"
    "Masukan:\n{1}\n\n"
    "Jawaban:\n"
)

# Gunakan jawaban Top-1 untuk prompt ke LLaMA
top1_answer = reranked[I[0][0]]

# Format prompt ke LLaMA
formatted_prompt = alpaca_prompt.format(
    "Tolong jawab pertanyaan ini dengan sopan dan jelas:\n" + query,
    top1_answer
)

# Tokenisasi dan generate jawaban dengan LLaMA
inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = tokenizer([formatted_prompt], return_tensors="pt", padding=True, truncation=True)
inputs = {key: val.to(device) for key, val in inputs.items()}

print("\n=== Jawaban Natural dari LLaMA ===")
_ = model.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=250,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    repetition_penalty=1.1
)