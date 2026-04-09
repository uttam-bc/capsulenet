from fastapi import FastAPI
from rag.retriever import search

app = FastAPI()

@app.get("/ask")
def ask(query: str):
    doc = search(query)

    bullets = doc["content"].split(". ")

    formatted = f"⚠️ {doc['title']}\n\n"

    for b in bullets:
        if b.strip():
            formatted += f"• {b.strip()}\n"

    return {"answer": formatted}