#!/bin/bash
# Full pipeline automation

echo "=== Step 1: Build FAISS index ==="
python src/build_faiss_index.py

echo "=== Step 2: Retrieve candidates ==="
python - <<'PYCODE'
from src.build_faiss_index import build_index
from src.retrieve_candidates import retrieve
from src.rerank_candidates import rerank
from src.generate_answer import generate_answer

index, questions, answers = build_index()
query = "apa itu semester antara?"

retrieved = retrieve(query, "faiss_index.bin", questions)
candidates = [(q, answers[questions.index(q)]) for q, _ in retrieved]
reranked = rerank(query, candidates)
best_answer = reranked[0][2]
generate_answer(query, best_answer)
PYCODE
