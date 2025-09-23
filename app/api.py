from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import shutil
import time

from .main import (
    get_llm, semantic_cache_lookup, semantic_cache_store,
    store_chat_history, get_chat_history, search_documents,
    answer_question
)

app = FastAPI(
    title="RAG Combined API",
    description="API untuk RAG dengan semantic cache dan chat memory, menggunakan vector store yang dikelola n8n."
)

llm = get_llm()

class QARequest(BaseModel):
    question: str
    k: Optional[int] = 3
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Filter metadata/tag, misal: {'role': 'finance'} atau {'source': 'file.pdf'}.")
    distance_threshold: Optional[float] = Field(default=None, description="Threshold kemiripan vektor (0-1, lebih kecil = lebih mirip)")
    text_filter: Optional[str] = Field(default=None, description="Filter text hybrid, misal: hanya chunk yang mengandung kata tertentu.")

class QAResponse(BaseModel):
    answer: str
    context: List[str]
    cached: bool
    latency_ms: float
    cache_latency_ms: Optional[float] = None
    search_latency_ms: Optional[float] = None
    num_candidates: Optional[int] = None
    search_type: str
    filters_applied: Optional[Dict[str, Any]] = None
    distance_threshold: Optional[float] = None

class ChatRequest(BaseModel):
    user_id: str
    message: str
    k: Optional[int] = 3
    filters: Optional[Dict[str, Any]] = None
    distance_threshold: Optional[float] = None
    text_filter: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    chat_history: List[dict]
    cached: bool
    latency_ms: float
    cache_latency_ms: Optional[float] = None
    search_latency_ms: Optional[float] = None
    num_candidates: Optional[int] = None
    search_type: str
    filters_applied: Optional[Dict[str, Any]] = None
    distance_threshold: Optional[float] = None

@app.post("/ask", response_model=QAResponse)
def ask_question(req: QARequest):
    """Jawab pertanyaan berbasis dokumen dengan semantic caching dan vector search."""
    t0 = time.perf_counter()
    
    # Get answer using RAG pipeline
    result = answer_question(
        query=req.question,
        k=req.k if req.k is not None else 3,
        filters=req.filters,
        distance_threshold=req.distance_threshold,
        text_filter=req.text_filter
    )
    
    latency_ms = (time.perf_counter() - t0) * 1000
    
    return QAResponse(
        answer=result["answer"],
        context=result["context"],
        cached=result["cached"],
        latency_ms=latency_ms,
        cache_latency_ms=None,  # Could be added if needed
        search_latency_ms=None,  # Could be added if needed
        num_candidates=len(result["context"]),
        search_type="similarity" if not req.text_filter else "hybrid",
        filters_applied=req.filters,
        distance_threshold=req.distance_threshold
    )

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Multi-turn chat dengan memory per user dan semantic caching."""
    t0 = time.perf_counter()
    
    # Get chat history
    chat_history = get_chat_history(request.user_id)
    
    # Add user message to history
    chat_history.append({"role": "user", "content": request.message})
    
    # Get last few messages for context
    last_context = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in chat_history[-6:]
    ])
    
    # Get answer using RAG pipeline
    result = answer_question(
        query=request.message,
        k=request.k if request.k is not None else 3,
        filters=request.filters,
        distance_threshold=request.distance_threshold,
        text_filter=request.text_filter
    )
    
    # Add assistant response to history
    chat_history.append({"role": "assistant", "content": result["answer"]})
    
    # Store updated chat history
    store_chat_history(request.user_id, chat_history)
    
    latency_ms = (time.perf_counter() - t0) * 1000
    
    return ChatResponse(
        answer=result["answer"],
        chat_history=chat_history,
        cached=result["cached"],
        latency_ms=latency_ms,
        cache_latency_ms=None,  # Could be added if needed
        search_latency_ms=None,  # Could be added if needed
        num_candidates=len(result["context"]),
        search_type="similarity" if not request.text_filter else "hybrid",
        filters_applied=request.filters,
        distance_threshold=request.distance_threshold
    )

@app.get("/", include_in_schema=False)
def root():
    return {
        "msg": "RAG API with semantic cache and chat memory. See /docs for documentation.",
        "features": [
            "Vector search from n8n-managed store",
            "Semantic caching for similar questions",
            "Multi-turn chat with memory",
            "Hybrid search and filtering"
        ]
    }