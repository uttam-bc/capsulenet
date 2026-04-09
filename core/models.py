"""Pydantic models for the Offline Internet Capsule API."""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Request model for the /ask endpoint."""
    query: str = Field(..., min_length=1, max_length=500, description="User's question")
    category: Optional[str] = Field(None, description="Optional category filter")


class Source(BaseModel):
    """A source document referenced in the answer."""
    title: str
    category: str
    relevance: float = Field(..., ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str
    sources: list[Source]
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: Optional[str] = None
    offline: bool = True


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: str = "offline"
    version: str = "1.0.0"
    documents_loaded: int = 0


class CategoryResponse(BaseModel):
    """Response model for the /categories endpoint."""
    categories: list[str]


class EmergencyCard(BaseModel):
    """An emergency quick-access card."""
    id: str
    title: str
    icon: str
    category: str
    steps: list[str]
    warning: Optional[str] = None
