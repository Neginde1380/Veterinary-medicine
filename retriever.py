# retriever.py

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def load_documents(documents_path):
    with open(documents_path, "r", encoding="utf-8") as f:
        return json.load(f)

def search_faiss_index(query, model, index, documents, k=1):
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)

    # Collect results
    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "score": float(distances[0][rank]),
            "content": documents[idx]
        })
        

    return results
