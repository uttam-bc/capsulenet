"""
Hybrid search engine for the Offline Internet Capsule.
Combines BM25 keyword search with FAISS vector similarity search.
"""

import json
import sqlite3
import os
import numpy as np
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "knowledge.db")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "knowledge.faiss")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class SearchResult:
    """A single search result with relevance scoring."""
    id: str
    title: str
    category: str
    content: str
    keywords: list[str]
    score: float  # Combined relevance score (0-1)
    bm25_score: float = 0.0
    faiss_score: float = 0.0


class HybridSearchEngine:
    """
    Hybrid search engine combining BM25 (keyword) and FAISS (vector) search.
    Weights: 0.4 BM25 + 0.6 FAISS by default.
    """

    def __init__(self, bm25_weight: float = 0.4, faiss_weight: float = 0.6):
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        self.documents: list[dict] = []
        self.bm25 = None
        self.faiss_index = None
        self.embedding_model = None
        self._loaded = False

    def load(self):
        """Load all components: documents from SQLite, BM25 index, FAISS index, and embedding model."""
        if self._loaded:
            return

        self._load_documents()
        self._build_bm25_index()
        self._load_faiss_index()
        self._load_embedding_model()
        self._loaded = True

    def _load_documents(self):
        """Load documents from SQLite database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, category, content, keywords FROM documents")
        rows = cursor.fetchall()
        conn.close()

        self.documents = []
        for row in rows:
            self.documents.append({
                "id": row[0],
                "title": row[1],
                "category": row[2],
                "content": row[3],
                "keywords": json.loads(row[4]),
            })
        print(f"Loaded {len(self.documents)} documents from database")

    def _build_bm25_index(self):
        """Build BM25 index from document content."""
        from rank_bm25 import BM25Okapi

        tokenized_corpus = []
        for doc in self.documents:
            # Combine title, content, and keywords for BM25
            text = f"{doc['title']} {doc['content']} {' '.join(doc['keywords'])}"
            tokens = text.lower().split()
            tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(tokenized_corpus)
        print("Built BM25 index")

    def _load_faiss_index(self):
        """Load FAISS index from disk."""
        import faiss

        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")

    def _load_embedding_model(self):
        """Load sentence-transformers embedding model."""
        from sentence_transformers import SentenceTransformer

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")

    def search(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining BM25 and FAISS results.

        Args:
            query: User's search query
            category: Optional category filter
            top_k: Number of results to return

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        if not self._loaded:
            self.load()

        # Get candidate indices (apply category filter first)
        candidate_indices = self._get_candidate_indices(category)
        if not candidate_indices:
            return []

        # BM25 search
        bm25_scores = self._bm25_search(query)

        # FAISS search
        faiss_scores = self._faiss_search(query)

        # Combine scores
        results = []
        for idx in candidate_indices:
            bm25_s = bm25_scores.get(idx, 0.0)
            faiss_s = faiss_scores.get(idx, 0.0)

            combined = self.bm25_weight * bm25_s + self.faiss_weight * faiss_s

            doc = self.documents[idx]
            results.append(SearchResult(
                id=doc["id"],
                title=doc["title"],
                category=doc["category"],
                content=doc["content"],
                keywords=doc["keywords"],
                score=combined,
                bm25_score=bm25_s,
                faiss_score=faiss_s,
            ))

        # Sort by combined score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _get_candidate_indices(self, category: str | None) -> list[int]:
        """Get document indices that match the category filter."""
        if category is None:
            return list(range(len(self.documents)))
        return [
            i for i, doc in enumerate(self.documents)
            if doc["category"] == category.lower()
        ]

    def _bm25_search(self, query: str) -> dict[int, float]:
        """Perform BM25 search and return normalized scores per document index."""
        tokens = query.lower().split()
        raw_scores = self.bm25.get_scores(tokens)

        # Normalize to 0-1
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        return {i: score / max_score for i, score in enumerate(raw_scores) if score > 0}

    def _faiss_search(self, query: str) -> dict[int, float]:
        """Perform FAISS similarity search and return scores per document index."""
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )
        query_vector = np.array(query_embedding, dtype=np.float32)

        # Search (return all for merging)
        k = min(len(self.documents), self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(query_vector, k)

        # Convert cosine similarity to 0-1 scores
        scores = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                # Cosine similarity is already in [-1, 1], normalize to [0, 1]
                scores[int(idx)] = max(0.0, float((dist + 1) / 2))

        return scores

    def get_categories(self) -> list[str]:
        """Return all unique categories."""
        if not self._loaded:
            self.load()
        categories = sorted(set(doc["category"] for doc in self.documents))
        return categories

    def get_document_count(self) -> int:
        """Return total number of documents."""
        if not self._loaded:
            self.load()
        return len(self.documents)

    def get_document_by_id(self, doc_id: str) -> dict | None:
        """Retrieve a specific document by ID."""
        if not self._loaded:
            self.load()
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None
