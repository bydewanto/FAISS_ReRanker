from sentence_transformers import CrossEncoder

def rerank(query, candidates, alpha=0.01, length_mode="boost"):
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_inputs = [(query, q) for q, a in candidates]
    scores = reranker.predict(rerank_inputs)

    adjusted = []
    for score, (q, a) in zip(scores, candidates):
        length = len(q.split())
        if length_mode == "boost":
            adj_score = score + alpha * length
        else:
            adj_score = score - alpha * length
        adjusted.append((adj_score, q, a))

    adjusted.sort(key=lambda x: x[0], reverse=True)
    return adjusted
