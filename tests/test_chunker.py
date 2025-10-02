"""
Tests for the SmartChunker class in toksum.
"""

import pytest
from unittest.mock import patch, Mock
from typing import List

from toksum import SmartChunker


class TestSmartChunker:
    """Test cases for SmartChunker class."""
    
    def test_init_valid(self):
        """Test initialization with valid parameters."""
        chunker = SmartChunker("gpt-4", max_tokens=100)
        assert chunker.model == "gpt-4"
        assert chunker.max_tokens == 100
    
    def test_init_invalid_max_tokens(self):
        """Test initialization with invalid max_tokens."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            SmartChunker("gpt-4", max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            SmartChunker("gpt-4", max_tokens=-10)
    
    @patch('toksum.chunker.count_tokens')
    def test_chunk_by_sentences_short_text(self, mock_count_tokens):
        """Test sentence chunking for short text that fits in one chunk."""
        mock_count_tokens.return_value = 20  # Always under limit
        chunker = SmartChunker("gpt-4", max_tokens=50)
        text = "This is a short text. It fits in one chunk."
        chunks = chunker.chunk_by_sentences(text)
        assert len(chunks) == 1
        assert "This is a short text." in chunks[0]
        assert mock_count_tokens.call_count > 0  # Multiple calls for incremental checks
    
    @patch('toksum.chunker.count_tokens')
    def test_chunk_by_sentences_long_text(self, mock_count_tokens):
        """Test sentence chunking for long text that spans multiple chunks."""
        calls = [0]
        def side_effect(text, model):
            calls[0] += 1
            if calls[0] == 1:  # First sentence
                return 10
            if calls[0] == 2:  # First + second
                return 25  # Under 50
            if calls[0] == 3:  # First+second + third
                return 60  # Over 50
            if calls[0] == 4:  # Third
                return 35  # Under
            if calls[0] == 5:  # Third + fourth
                return 45  # Under
            return 10  # Fourth alone
        
        mock_count_tokens.side_effect = side_effect
        
        chunker = SmartChunker("gpt-4", max_tokens=50)
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = chunker.chunk_by_sentences(text)
        assert len(chunks) == 2  # First two, then third + fourth
        assert "Sentence one. Sentence two." in chunks[0]
        assert "Sentence three. Sentence four." in chunks[1]
        assert mock_count_tokens.call_count >= 5
    
    @patch('toksum.chunker.count_tokens')
    def test_chunk_by_sentences_empty_text(self, mock_count_tokens):
        """Test sentence chunking for empty text."""
        chunker = SmartChunker("gpt-4", max_tokens=50)
        chunks = chunker.chunk_by_sentences("")
        assert chunks == []
        mock_count_tokens.assert_not_called()
    
    @patch('toksum.chunker.count_tokens')
    def test_chunk_by_paragraphs_short_text(self, mock_count_tokens):
        """Test paragraph chunking for short text."""
        mock_count_tokens.return_value = 30
        chunker = SmartChunker("gpt-4", max_tokens=50)
        text = "First paragraph.\n\nSecond paragraph."
        chunks = chunker.chunk_by_paragraphs(text)
        assert len(chunks) == 1  # Combined since under limit
        assert "\n\n" in chunks[0]
        assert mock_count_tokens.call_count > 0
    
    @patch('toksum.chunker.count_tokens')
    def test_chunk_by_paragraphs_long_text(self, mock_count_tokens):
        """Test paragraph chunking for text exceeding limit."""
        calls = [0]
        def side_effect(text, model):
            calls[0] += 1
            if calls[0] == 1:  # First para
                return 25
            if calls[0] == 2:  # First + second
                return 45  # Under
            if calls[0] == 3:  # First+second + third
                return 80  # Over
            if calls[0] == 4:  # Third
                return 35  # Under
            if calls[0] == 5:  # Third + fourth
                return 55  # Over
            return 20  # Fourth
        
        mock_count_tokens.side_effect = side_effect
        
        chunker = SmartChunker("gpt-4", max_tokens=50)
        text = "Para1\n\nPara2\n\nPara3\n\nPara4"
        chunks = chunker.chunk_by_paragraphs(text)
        assert len(chunks) == 3  # Para1+Para2, Para3, Para4
        assert "Para1\n\nPara2" in chunks[0]
        assert mock_count_tokens.call_count >= 5
    
    @patch('toksum.chunker.count_tokens')
    def test_chunk_by_paragraphs_empty(self, mock_count_tokens):
        """Test paragraph chunking for empty text."""
        chunker = SmartChunker("gpt-4", max_tokens=50)
        chunks = chunker.chunk_by_paragraphs("")
        assert chunks == []
        mock_count_tokens.assert_not_called()
    
    @patch('toksum.chunker.count_tokens')
    def test_chunk_code_python_short(self, mock_count_tokens):
        """Test code chunking for short Python code."""
        mock_count_tokens.return_value = 40
        chunker = SmartChunker("gpt-4", max_tokens=50)
        code = """
def hello():
    print('Hello, world!')
"""
        chunks = chunker.chunk_code(code, "python")
        assert len(chunks) == 1
        assert "def hello():" in chunks[0]
        assert mock_count_tokens.call_count > 0
    
    @patch('toksum.chunker.count_tokens')
    def test_chunk_code_python_long(self, mock_count_tokens):
        """Test code chunking for long Python splitting between functions."""
        calls = [0]
        def side_effect(text, model):
            calls[0] += 1
            if "def fibonacci" in text:
                return 30  # First def under
            if "class Calculator" in text:
                return 20  # Class under
            if len(text) > 50:  # Body parts
                return 25  # Fits
            return 10
        
        mock_count_tokens.side_effect = side_effect
        
        chunker = SmartChunker("gpt-4", max_tokens=50)
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
"""
        chunks = chunker.chunk_code(code, "python")
        assert len(chunks) == 2  # One for fibonacci, one for class
        assert "def fibonacci" in chunks[0]
        assert "class Calculator" in chunks[1]
        assert mock_count_tokens.call_count > 0
    
    @patch('toksum.chunker.count_tokens')
    def test_chunk_code_non_python_fallback(self, mock_count_tokens):
        """Test code chunking falls back to paragraphs for non-Python."""
        mock_count_tokens.return_value = 30
        chunker = SmartChunker("gpt-4", max_tokens=50)
        code = "function hello() { console.log('Hi'); }"
        chunks = chunker.chunk_code(code, "javascript")
        assert len(chunks) == 1  # Fallback to paragraphs
        assert code in chunks[0]
        assert mock_count_tokens.call_count > 0
    
    @patch('toksum.chunker.count_tokens')
    def test_chunk_code_empty(self, mock_count_tokens):
        """Test code chunking for empty code."""
        chunker = SmartChunker("gpt-4", max_tokens=50)
        chunks = chunker.chunk_code("", "python")
        assert chunks == []
        mock_count_tokens.assert_not_called()
