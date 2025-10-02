"""
Simple text chunking utilities.
"""

from typing import List
import re

from .core import count_tokens


class SimpleChunker:
    """
    Simple chunker for text based on token limits.
    """
    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
    
    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        current_chunk: List[str] = []
        for word in words:
            test_chunk = ' '.join(current_chunk + [word])
            if count_tokens(test_chunk, self.model) <= self.max_tokens:
                current_chunk.append(word)
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks
