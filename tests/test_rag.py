import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import answer_question, search_documents, semantic_cache_lookup, semantic_cache_store

def test_search():
    """Test vector search functionality"""
    print("\n=== Testing Document Search ===")
    results = search_documents(
        query="What programming languages and frameworks?",
        k=3,
        filters={"mime_type": "application/pdf"}
    )
    
    print(f"Found {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result['content'][:200]}...")
        print(f"Distance: {result['vector_distance']}")
        print(f"Metadata: {result['metadata']}")

def test_qa():
    """Test question answering with RAG"""
    print("\n=== Testing Question Answering ===")
    result = answer_question(
        query="What are the candidate's technical skills and experience?",
        k=3,
        filters={"mime_type": "application/pdf"}
    )
    
    print("\nQuestion: What are the candidate's technical skills and experience?")
    print(f"Answer: {result['answer']}")
    print(f"Cached: {result['cached']}")
    print(f"Number of context documents: {len(result['context'])}")

def test_semantic_cache():
    """Test semantic caching"""
    print("\n=== Testing Semantic Cache ===")
    
    # First query - should miss cache
    question = "What is the candidate's background in machine learning?"
    print("\nFirst attempt (should miss cache):")
    result = answer_question(query=question)
    print(f"Answer: {result['answer']}")
    print(f"Cached: {result['cached']}")
    
    # Similar question - should hit cache
    similar_question = "Tell me about the candidate's ML experience"
    print("\nSimilar question (should hit cache):")
    result = answer_question(query=similar_question)
    print(f"Answer: {result['answer']}")
    print(f"Cached: {result['cached']}")

if __name__ == "__main__":
    # Run all tests
    test_search()
    test_qa()
    test_semantic_cache()