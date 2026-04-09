"""
Data ingestion pipeline for the Offline Internet Capsule.
Reads knowledge.json, generates embeddings, and stores in SQLite + FAISS index.
"""

import json
import sqlite3
import struct
import os
import sys
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWLEDGE_JSON = os.path.join(DATA_DIR, "knowledge.json")
DB_PATH = os.path.join(DATA_DIR, "knowledge.db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "knowledge.faiss")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_knowledge() -> list[dict]:
    """Load knowledge chunks from JSON file."""
    with open(KNOWLEDGE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} knowledge chunks from {KNOWLEDGE_JSON}")
    return data


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def create_database(documents: list[dict], embeddings: np.ndarray):
    """Create SQLite database with document data and embeddings."""
    # Remove existing database
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            keywords TEXT NOT NULL,
            embedding BLOB,
            version INTEGER DEFAULT 1,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create manifests table (for future sync)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS manifests (
            capsule_id TEXT PRIMARY KEY,
            version TEXT NOT NULL,
            checksum TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert documents
    for i, doc in enumerate(documents):
        embedding_blob = embeddings[i].tobytes()
        cursor.execute(
            """INSERT INTO documents (id, title, category, content, keywords, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                doc["id"],
                doc["title"],
                doc["category"],
                doc["content"],
                json.dumps(doc["keywords"]),
                embedding_blob,
            ),
        )

    # Insert initial manifest
    cursor.execute(
        """INSERT INTO manifests (capsule_id, version, checksum)
           VALUES (?, ?, ?)""",
        ("base-capsule", "1.0.0", "initial"),
    )

    conn.commit()
    conn.close()
    print(f"Created database with {len(documents)} documents at {DB_PATH}")


def create_faiss_index(embeddings: np.ndarray):
    """Create and save FAISS index for vector similarity search."""
    import faiss

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine sim for normalized vectors)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Created FAISS index with {index.ntotal} vectors (dim={dimension}) at {FAISS_INDEX_PATH}")


def main():
    """Run the full ingestion pipeline."""
    print("=" * 60)
    print("Offline Internet Capsule — Data Ingestion Pipeline")
    print("=" * 60)

    # Load knowledge
    documents = load_knowledge()

    # Generate search texts combining title, content, and keywords
    search_texts = []
    for doc in documents:
        text = f"{doc['title']}. {doc['content']} Keywords: {', '.join(doc['keywords'])}"
        search_texts.append(text)

    # Generate embeddings
    embeddings = generate_embeddings(search_texts)

    # Create database
    create_database(documents, embeddings)

    # Create FAISS index
    create_faiss_index(embeddings)

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  Database: {DB_PATH}")
    print(f"  FAISS Index: {FAISS_INDEX_PATH}")
    print(f"  Documents: {len(documents)}")
    print(f"  Embedding Dim: {embeddings.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
