#!/usr/bin/env python3
"""
Simple test cases to verify SmartChunker functionality.
"""

from toksum import SmartChunker

def test_basic_functionality():
    """Test basic chunking functionality."""
    print("Testing SmartChunker basic functionality...")
    
    # Test 1: Simple sentence chunking
    print("\n1. Testing sentence chunking:")
    chunker = SmartChunker("gpt-4", max_tokens=25)
    text = "Hello world. This is a test. How are you today?"
    chunks = chunker.chunk_by_sentences(text)
    
    print(f"   Input: {text}")
    print(f"   Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: '{chunk}'")
    
    # Test 2: Paragraph chunking
    print("\n2. Testing paragraph chunking:")
    chunker = SmartChunker("gpt-4", max_tokens=40)
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunker.chunk_by_paragraphs(text)
    
    print(f"   Input paragraphs: 3")
    print(f"   Output chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: '{chunk.replace(chr(10), ' | ')}'")
    
    # Test 3: Python code chunking
    print("\n3. Testing Python code chunking:")
    chunker = SmartChunker("gpt-4", max_tokens=60)
    code = '''def hello():
    print("Hello")

def goodbye():
    print("Goodbye")'''
    
    chunks = chunker.chunk_code(code, "python")
    
    print(f"   Input: 2 functions")
    print(f"   Output chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        first_line = chunk.split('\n')[0]
        print(f"   Chunk {i+1}: starts with '{first_line}'")

def test_edge_cases():
    """Test edge cases."""
    print("\n\nTesting edge cases...")
    
    chunker = SmartChunker("gpt-4", max_tokens=50)
    
    # Test empty input
    print("\n1. Empty input:")
    result = chunker.chunk_by_sentences("")
    print(f"   Empty string result: {result}")
    
    # Test single sentence
    print("\n2. Single sentence:")
    result = chunker.chunk_by_sentences("Just one sentence.")
    print(f"   Single sentence result: {result}")
    
    # Test very long sentence (should still work)
    print("\n3. Long sentence:")
    long_sentence = "This is a very long sentence that might exceed token limits but should still be handled gracefully by the chunker."
    result = chunker.chunk_by_sentences(long_sentence)
    print(f"   Long sentence chunks: {len(result)}")

def test_different_models():
    """Test with different model names."""
    print("\n\nTesting different models...")
    
    models = ["gpt-4", "claude-3-opus", "gemini-pro"]
    text = "Test sentence one. Test sentence two. Test sentence three."
    
    for model in models:
        try:
            chunker = SmartChunker(model, max_tokens=30)
            chunks = chunker.chunk_by_sentences(text)
            print(f"   {model}: {len(chunks)} chunks")
        except Exception as e:
            print(f"   {model}: Error - {e}")

if __name__ == "__main__":
    print("Simple SmartChunker Test Suite")
    print("=" * 40)
    
    test_basic_functionality()
    test_edge_cases()
    test_different_models()
    
    print("\n" + "=" * 40)
    print("Test suite completed!")