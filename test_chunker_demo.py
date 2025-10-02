#!/usr/bin/env python3
"""
Demo script to show how SmartChunker works with different types of text.
"""

from toksum import SmartChunker

def demo_sentence_chunking():
    """Demonstrate sentence-based chunking."""
    print("=== Sentence Chunking Demo ===")
    
    chunker = SmartChunker("gpt-4", max_tokens=30)
    
    text = """
    This is the first sentence. This is the second sentence that might be longer. 
    This is the third sentence. This is the fourth sentence which could also be quite long. 
    This is the fifth and final sentence.
    """
    
    chunks = chunker.chunk_by_sentences(text)
    
    print(f"Original text: {text.strip()}")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Content: {chunk}")
        print(f"  Token count: {chunker.model} - estimated")

def demo_paragraph_chunking():
    """Demonstrate paragraph-based chunking."""
    print("\n=== Paragraph Chunking Demo ===")
    
    chunker = SmartChunker("gpt-4", max_tokens=50)
    
    text = """
    This is the first paragraph. It contains multiple sentences that talk about one topic.
    
    This is the second paragraph. It discusses a different topic and has its own focus.
    
    This is the third paragraph. It's also about something different and stands alone.
    
    This is the fourth paragraph. It concludes our discussion with final thoughts.
    """
    
    chunks = chunker.chunk_by_paragraphs(text)
    
    print(f"Original text has {len(text.split('\\n\\n'))} paragraphs")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  Content: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
        print(f"  Length: {len(chunk)} characters")

def demo_python_code_chunking():
    """Demonstrate Python code chunking."""
    print("\n=== Python Code Chunking Demo ===")
    
    chunker = SmartChunker("gpt-4", max_tokens=100)
    
    code = '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n-1)

class MathUtils:
    """Utility class for mathematical operations."""
    
    def __init__(self):
        self.pi = 3.14159
    
    def circle_area(self, radius):
        """Calculate area of a circle."""
        return self.pi * radius * radius
    
    def square_area(self, side):
        """Calculate area of a square."""
        return side * side

def main():
    """Main function to demonstrate usage."""
    print("Fibonacci of 10:", calculate_fibonacci(10))
    print("Factorial of 5:", factorial(5))
    
    math_utils = MathUtils()
    print("Circle area (r=5):", math_utils.circle_area(5))
    print("Square area (s=4):", math_utils.square_area(4))

if __name__ == "__main__":
    main()
    '''
    
    chunks = chunker.chunk_code(code, "python")
    
    print(f"Original code has multiple functions and a class")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        lines = chunk.split('\n')
        first_line = next((line.strip() for line in lines if line.strip()), "")
        print(f"  Starts with: {first_line}")
        print(f"  Lines: {len(lines)}")
        print(f"  Characters: {len(chunk)}")

def demo_mixed_content():
    """Demonstrate chunking with mixed content."""
    print("\n=== Mixed Content Demo ===")
    
    chunker = SmartChunker("claude-3-opus", max_tokens=75)
    
    # Mix of documentation and code
    content = '''
# Data Processing Module

This module provides utilities for processing data efficiently.

def process_data(data):
    """Process input data and return cleaned results."""
    cleaned = []
    for item in data:
        if item is not None:
            cleaned.append(str(item).strip())
    return cleaned

## Usage Examples

Here are some examples of how to use the data processing functions:

```python
data = [1, 2, None, "  hello  ", 4]
result = process_data(data)
print(result)  # ['1', '2', 'hello', '4']
```

The function handles None values and strips whitespace automatically.
    '''
    
    # Treat as general text (paragraph chunking)
    chunks = chunker.chunk_by_paragraphs(content)
    
    print(f"Mixed content (documentation + code)")
    print(f"Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        preview = chunk.replace('\n', ' ')[:80]
        print(f"  Preview: {preview}{'...' if len(chunk) > 80 else ''}")

def demo_token_limits():
    """Demonstrate how different token limits affect chunking."""
    print("\n=== Token Limit Comparison ===")
    
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
    
    limits = [20, 40, 80]
    
    for limit in limits:
        chunker = SmartChunker("gpt-4", max_tokens=limit)
        chunks = chunker.chunk_by_sentences(text)
        
        print(f"\nMax tokens: {limit}")
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}: {chunk}")

if __name__ == "__main__":
    print("SmartChunker Demonstration")
    print("=" * 50)
    
    demo_sentence_chunking()
    demo_paragraph_chunking()
    demo_python_code_chunking()
    demo_mixed_content()
    demo_token_limits()
    
    print("\n" + "=" * 50)
    print("Demo completed! The SmartChunker intelligently splits text while")
    print("respecting semantic boundaries and token limits.")