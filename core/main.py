"""
FastAPI application for the Offline Internet Capsule.
Provides REST endpoints for search, categories, emergency access, and health checks.
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
    CategoryResponse,
    EmergencyCard,
    Source,
)
from core.search import HybridSearchEngine
from core.formatter import format_response, format_emergency_cards

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Search engine (initialized on startup)
search_engine = HybridSearchEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load search engine on startup."""
    print("Loading search engine...")
    start = time.time()
    search_engine.load()
    elapsed = time.time() - start
    print(f"Search engine loaded in {elapsed:.2f}s ({search_engine.get_document_count()} documents)")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Offline Internet Capsule",
    description="A portable, self-contained layer of the internet that works offline.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Endpoints ────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="offline",
        version="1.0.0",
        documents_loaded=search_engine.get_document_count(),
    )


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Ask a question and get a structured answer from the offline knowledge base.
    Combines keyword (BM25) and semantic (FAISS) search for best results.
    """
    start = time.time()

    # Perform hybrid search
    results = search_engine.search(
        query=request.query,
        category=request.category,
        top_k=5,
    )

    # Format response
    formatted = format_response(request.query, results)

    elapsed = time.time() - start
    print(f"Query '{request.query}' answered in {elapsed*1000:.0f}ms (confidence: {formatted['confidence']})")

    return QueryResponse(
        answer=formatted["answer"],
        sources=[Source(**s) for s in formatted["sources"]],
        confidence=formatted["confidence"],
        category=formatted["category"],
        offline=True,
    )


@app.get("/categories", response_model=CategoryResponse)
async def categories():
    """Get all available knowledge categories."""
    return CategoryResponse(categories=search_engine.get_categories())


@app.get("/emergency", response_model=list[EmergencyCard])
async def emergency():
    """Get pre-built emergency quick-access cards."""
    cards = format_emergency_cards()
    return [EmergencyCard(**card) for card in cards]


# ── Frontend Serving ─────────────────────────────────────

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend not found")


# Serve static files from frontend directory if it exists
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
