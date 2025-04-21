from typing import List, Dict, Any
import tiktoken
from sentence_transformers import CrossEncoder
import numpy as np
import atexit
import threading
from llama_index.core.settings import Settings

class LongContextHandler:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, model_name: str = "gpt-3.5-turbo"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LongContextHandler, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        if self._initialized:
            return
            
        self.model_name = model_name
        self._reranker = None
        self._initialized = True
        atexit.register(self.cleanup)
        
    @property
    def reranker(self):
        """Lazy initialization of reranker."""
        if self._reranker is None:
            with self._lock:
                if self._reranker is None:
                    self._reranker = CrossEncoder('BAAI/bge-reranker-base')
        return self._reranker
        
    def cleanup(self):
        """Cleanup resources."""
        if self._reranker:
            # Delete the model and clear CUDA cache if using GPU
            try:
                import torch
                del self._reranker
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
        
    def get_max_tokens(self) -> int:
        """Get max token limit based on model."""
        model_limits = {
            "gpt-3.5-turbo": 16000,  # Conservative limit
            "gpt-4": 128000,
            "mistralai/Mixtral-8x7B-Instruct-v0.1": 24000,
            "deepseek/deepseek-r1:free": 30000
        }
        return model_limits.get(self.model_name, 16000)  # Default to conservative limit
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using appropriate tokenizer."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Error counting tokens: {str(e)}")
            # Fallback to approximate count
            return len(text.split()) * 1.3
            
    def chunk_text(self, text: str, chunk_size: int = 1024, overlap: int = 20) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            start = end - overlap
            
        return chunks
        
    def rerank_chunks(self, query: str, chunks: List[str], top_k: int = 3) -> List[str]:
        """Rerank chunks based on relevance to query."""
        if not chunks:
            return []
            
        # Create pairs of (query, chunk) for each chunk
        pairs = [[query, chunk] for chunk in chunks]
        
        try:
            # Get relevance scores using the reranker
            scores = self.reranker.predict(pairs)
            
            # Sort chunks by score and get top_k
            chunk_scores = list(zip(chunks, scores))
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [chunk for chunk, _ in chunk_scores[:top_k]]
        except Exception as e:
            print(f"Error in reranking: {str(e)}")
            return chunks[:top_k]  # Fallback to first top_k chunks
        
    def process_documents(self, query: str, documents: List[str], max_output_tokens: int = 1000) -> str:
        """Process documents with chunking and reranking."""
        max_input_tokens = self.get_max_tokens() - max_output_tokens - 500  # Reserve tokens for system message
        
        # Combine all documents
        combined_text = " ".join(documents)
        
        # Calculate appropriate chunk size
        chunk_size = min(1024, max_input_tokens // 2)  # Ensure chunks fit within limits
        overlap = min(50, chunk_size // 10)  # 10% overlap by default
        
        # Chunk the combined text
        chunks = self.chunk_text(combined_text, chunk_size=chunk_size, overlap=overlap)
        
        # Rerank chunks
        ranked_chunks = self.rerank_chunks(query, chunks, top_k=3)
        
        # Combine ranked chunks while respecting token limit
        final_text = ""
        current_tokens = 0
        
        for chunk in ranked_chunks:
            chunk_tokens = self.count_tokens(chunk)
            if current_tokens + chunk_tokens > max_input_tokens:
                break
            final_text += chunk + "\n\n"
            current_tokens += chunk_tokens
            
        return final_text 