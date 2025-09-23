"""
RAG Combined App - Main Entry Point

Features:
- Vector store retrieval from existing Redis index
- Semantic caching
- Chat memory
"""

import os
from typing import List, Optional, Dict, Any

# LangChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.redis import Redis as RedisVectorStore
from langchain_community.llms import OpenAI

# Redis vector store and caching
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.llmcache import SemanticCache

import os
from typing import List, Optional, Dict, Any

import numpy as np

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.llmcache import SemanticCache
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag, Text

# FastAPI (untuk API, diimplementasikan di file terpisah)
# from fastapi import FastAPI

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://10.100.34.246:12345")
OPENAI_API_KEY = ""
INDEX_NAME = "talent-pool"  # Index name used by n8n
REDIS_PREFIX = "doc:talent-pool:"  # Key prefix used by n8n

# Initialize embedding model with OpenAI (to match n8n's vector dimensions)
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"  # 1536 dimensions
)

# Initialize vector store connection to existing index
vector_store = RedisVectorStore(
    redis_url=REDIS_URL,
    index_name=INDEX_NAME,
    key_prefix=REDIS_PREFIX,
    embedding=embeddings
)

# Initialize semantic cache
hf = HFTextVectorizer(
    model="sentence-transformers/all-MiniLM-L6-v2",
    cache=EmbeddingsCache(
        name="embedcache",
        ttl=600,
        redis_url=REDIS_URL,
    )
)

# Fungsi load dokumen PDF dan split (gunakan PyPDFLoader dari langchain_community jika perlu)
def search_documents(
    query: str,
    k: int = 3,
    filters: Optional[Dict[str, Any]] = None,
    distance_threshold: Optional[float] = None,
    text_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search documents in the existing Redis vector store.
    
    Args:
        query: Search query
        k: Number of results to return
        filters: Metadata filters (e.g., {"file_type": "pdf"})
        distance_threshold: Maximum vector distance for similarity
        text_filter: Text to filter in content
    """
    # Use LangChain's Redis vector store for retrieval
    search_kwargs = {}
    if filters:
        search_kwargs["filter"] = filters
    if text_filter:
        search_kwargs["text_filter"] = text_filter
        
    docs = vector_store.similarity_search_with_score(
        query,
        k=k * 5 if distance_threshold else k,  # Get more if we need to filter by distance
        **search_kwargs
    )
    
    # Filter by distance threshold if specified
    if distance_threshold:
        docs = [(doc, score) for doc, score in docs if score <= distance_threshold]
        docs = docs[:k]
    
    # Format results
    results = []
    for doc, score in docs:
        result = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "vector_distance": score
        }
        results.append(result)
        
    return results

def get_llm():
    """Get LLM instance (OpenAI)."""
    return OpenAI(openai_api_key=OPENAI_API_KEY)

# Semantic cache (redisvl)
llmcache = SemanticCache(
    name="llmcache",
    vectorizer=hf,
    redis_url=REDIS_URL,
    ttl=120,
    distance_threshold=0.2
)

def semantic_cache_lookup(query: str) -> Optional[str]:
    """Cek semantic cache untuk jawaban yang mirip."""
    query_vector = hf.embed(query)
    # Pastikan query_vector adalah list[float], bukan bytes
    if isinstance(query_vector, bytes):
        return None
    result = llmcache.check(vector=query_vector)
    if result:
        return result[0]['response']
    return None

def semantic_cache_store(query: str, response: str):
    """Simpan jawaban ke semantic cache."""
    query_vector = hf.embed(query)
    if isinstance(query_vector, bytes):
        return
    llmcache.store(query, response, query_vector)

# Chat memory (sederhana, bisa dikembangkan)
def store_chat_history(user: str, messages: List[dict]):
    """Simpan riwayat chat ke Redis (dummy, sync). Untuk API, gunakan async."""
    # NOTE: Untuk FastAPI, gunakan async dan AsyncSearchIndex
    idx = SearchIndex.from_dict({
        "index": {"name": "chat_history", "prefix": "chat"},
        "fields": [{"name": "user", "type": "tag"}, {"name": "message", "type": "text"}]
    }, redis_url=REDIS_URL)
    for msg in messages:
        idx.load([{"user": user, "message": msg["content"]}], id_field=None)

def get_chat_history(user_id: str, limit: int = 10) -> List[dict]:
    """Retrieve chat history for a user from Redis."""
    from redis import Redis
    
    client = Redis.from_url(REDIS_URL)
    try:
        # Get the latest messages for the user
        messages = []
        pattern = f"chat:{user_id}:*"
        for key in client.scan_iter(match=pattern):
            msg = client.hgetall(key)
            if msg:
                messages.append({
                    "content": msg.get(b"message", b"").decode("utf-8"),
                    "timestamp": float(msg.get(b"timestamp", 0))
                })
        
        # Sort by timestamp and return latest messages
        messages.sort(key=lambda x: x["timestamp"], reverse=True)
        return messages[:limit]
    finally:
        client.close()

def store_chat_history(user_id: str, messages: List[dict]):
    """Store chat history in Redis."""
    from redis import Redis
    import time
    
    client = Redis.from_url(REDIS_URL)
    try:
        current_time = time.time()
        for i, msg in enumerate(messages):
            key = f"chat:{user_id}:{current_time + i}"
            client.hset(key, mapping={
                "message": msg["content"],
                "timestamp": str(current_time + i)
            })
            # Set TTL for chat messages (e.g., 30 days)
            client.expire(key, 60 * 60 * 24 * 30)
    finally:
        client.close()

def answer_question(
    query: str,
    k: int = 3,
    filters: Optional[Dict[str, Any]] = None,
    distance_threshold: Optional[float] = None,
    text_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Answer a question using the RAG pipeline with semantic caching.
    """
    # Check semantic cache first
    cached = semantic_cache_lookup(query)
    if cached:
        return {
            "answer": cached,
            "cached": True,
            "context": []
        }
    
    # Search for relevant documents
    docs = search_documents(
        query,
        k=k,
        filters=filters,
        distance_threshold=distance_threshold,
        text_filter=text_filter
    )
    
    # Extract content for context
    context = [doc["content"] for doc in docs]
    
    # Generate answer using LLM
    llm = get_llm()
    prompt = f"""Use the following context to answer the question.
Context:
{chr(10).join(context)}

Question: {query}
Answer:"""
    
    answer = llm(prompt)
    
    # Store in semantic cache
    semantic_cache_store(query, answer)
    
    return {
        "answer": answer,
        "cached": False,
        "context": context,
        "metadata": [doc.get("metadata", {}) for doc in docs]
    }

if __name__ == "__main__":
    # Example usage
    result = answer_question(
        "What are the candidate's technical skills?",
        k=3,
        filters={"mime_type": "application/pdf"}
    )
    print("Answer:", result["answer"])
    print("Cached:", result["cached"])
    print("Number of context documents:", len(result["context"]))