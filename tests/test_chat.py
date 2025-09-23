import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import store_chat_history, get_chat_history

def test_chat_history():
    """Test chat history storage and retrieval"""
    print("\n=== Testing Chat History ===")
    
    # Test user
    user_id = "test_user_1"
    
    # Sample messages
    messages = [
        {"role": "user", "content": "What are your technical skills?"},
        {"role": "assistant", "content": "Based on the CV, I have experience in Python, JavaScript, and Machine Learning."},
        {"role": "user", "content": "Tell me more about your ML experience"},
        {"role": "assistant", "content": "I have worked on several ML projects using TensorFlow and PyTorch..."}
    ]
    
    # Store messages
    print("\nStoring chat history...")
    store_chat_history(user_id, messages)
    
    # Retrieve messages
    print("\nRetrieving chat history...")
    retrieved_messages = get_chat_history(user_id, limit=5)
    
    print(f"\nRetrieved {len(retrieved_messages)} messages:")
    for msg in retrieved_messages:
        print(f"Message: {msg['content']}")
        print(f"Timestamp: {msg['timestamp']}")
        print("---")

if __name__ == "__main__":
    test_chat_history()