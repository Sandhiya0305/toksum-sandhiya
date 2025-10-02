"""
Example usage of the SmartChunker class for intelligent text chunking.
"""

from toksum import SmartChunker, count_tokens


# Sample long text for demonstration
long_text = """
Natural language processing (NLP) is a subfield of artificial intelligence, computer science, and linguistics concerned with the interaction between computers and human language. In practice, this means enabling computers to understand, interpret, and respond to human language in a meaningful way.

NLP draws from various fields including computational linguistics, machine learning, and natural language understanding. It has applications in machine translation, sentiment analysis, question answering, and chatbots.

The field has seen significant advancements with deep learning models like transformers and large language models such as GPT and BERT.
"""

# Sample code for chunking
sample_code = """
class SmartChunker:
    def __init__(self, model, max_tokens):
        self.model = model
        self.max_tokens = max_tokens
    
    def chunk_by_sentences(self, text):
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            test_chunk = current_chunk + ("" if not current_chunk else " ") + sentence
            if count_tokens(test_chunk, self.model) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def calculate_factorial(n):
    if n == 0:
        return 1
    return n * calculate_factorial(n-1)

# Usage
if __name__ == "__main__":
    chunker = SmartChunker("gpt-4", 100)
    sentences = chunker.chunk_by_sentences(long_text)
    print("Chunks by sentences:", sentences)
    
    paragraphs = chunker.chunk_by_paragraphs(long_text)
    print("Chunks by paragraphs:", paragraphs)
    
    code_chunks = chunker.chunk_code(sample_code, "python")
    print("Code chunks:", code_chunks)
"""

if __name__ == "__main__":
    # Create chunker instance
    chunker = SmartChunker("gpt-4", max_tokens=50)
    
    # Chunk by sentences
    sentence_chunks = chunker.chunk_by_sentences(long_text)
    print(f"Number of sentence chunks: {len(sentence_chunks)}")
    for i, chunk in enumerate(sentence_chunks, 1):
        tokens = count_tokens(chunk, "gpt-4")
        print(f"Chunk {i} ({tokens} tokens): {chunk[:50]}...")
    
    # Chunk by paragraphs
    paragraph_chunks = chunker.chunk_by_paragraphs(long_text)
    print(f"\nNumber of paragraph chunks: {len(paragraph_chunks)}")
    for i, chunk in enumerate(paragraph_chunks, 1):
        tokens = count_tokens(chunk, "gpt-4")
        print(f"Chunk {i} ({tokens} tokens): {chunk[:50]}...")
    
    # Chunk code (note: the sample_code above is for illustration; use it in practice)
    code_chunks = chunker.chunk_code(sample_code, "python")
    print(f"\nNumber of code chunks: {len(code_chunks)}")
    for i, chunk in enumerate(code_chunks, 1):
        tokens = count_tokens(chunk, "gpt-4")
        print(f"Code Chunk {i} ({tokens} tokens):\n{chunk}\n")
