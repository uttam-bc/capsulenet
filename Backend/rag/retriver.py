from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("data/docs.json") as f:
    docs = json.load(f)

texts = [d["content"] for d in docs]

embeddings = model.encode(texts)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings))


def search(query):
    query_lower = query.lower()

    # 🔹 Step 1: Keyword match
    for doc in docs:
        for kw in doc.get("keywords", []):
            if kw in query_lower:
                return doc

    # 🔹 Step 2: Semantic search
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=1)

    return docs[I[0][0]]