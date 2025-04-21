from typing import List, Dict, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryType
from llama_index.core.indices.base import BaseIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SimilarityPostprocessor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining BM25, vector, and keyword search."""
    
    def __init__(
        self,
        index: BaseIndex,
        top_k: int = 10,
        weights: Dict[str, float] = {'bm25': 0.4, 'vector': 0.4, 'keyword': 0.2}
    ):
        """Initialize the hybrid retriever."""
        self.index = index
        self.top_k = top_k
        self.weights = weights
        
        # Normalize weights if they don't sum to 1
        total = sum(weights.values())
        if total != 1.0:
            self.weights = {k: v/total for k, v in weights.items()}
    
    def _retrieve(self, query: QueryType, **kwargs) -> List[NodeWithScore]:
        """Retrieve nodes using hybrid search strategy."""
        try:
            # Get results from each retriever
            bm25_nodes = []
            vector_nodes = []
            keyword_nodes = []
            
            if self.weights['bm25'] > 0:
                bm25_nodes = self.index.as_retriever(
                    similarity_top_k=self.top_k
                ).retrieve(query)
            
            if self.weights['vector'] > 0:
                vector_nodes = self.index.as_retriever(
                    similarity_top_k=self.top_k,
                    mode="embedding"
                ).retrieve(query)
            
            if self.weights['keyword'] > 0:
                keyword_nodes = self.index.as_retriever(
                    similarity_top_k=self.top_k,
                    mode="keyword"
                ).retrieve(query)
            
            # Combine and normalize scores
            node_scores = {}
            
            # Process BM25 results
            for node in bm25_nodes:
                if node.node.node_id not in node_scores:
                    node_scores[node.node.node_id] = {
                        'node': node.node,
                        'score': 0.0
                    }
                node_scores[node.node.node_id]['score'] += node.score * self.weights['bm25']
            
            # Process vector results
            for node in vector_nodes:
                if node.node.node_id not in node_scores:
                    node_scores[node.node.node_id] = {
                        'node': node.node,
                        'score': 0.0
                    }
                node_scores[node.node.node_id]['score'] += node.score * self.weights['vector']
            
            # Process keyword results
            for node in keyword_nodes:
                if node.node.node_id not in node_scores:
                    node_scores[node.node.node_id] = {
                        'node': node.node,
                        'score': 0.0
                    }
                node_scores[node.node.node_id]['score'] += node.score * self.weights['keyword']
            
            # Convert to NodeWithScore list and sort by score
            results = [
                NodeWithScore(
                    node=info['node'],
                    score=info['score']
                )
                for info in node_scores.values()
            ]
            
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Return top k results
            return results[:self.top_k]
            
        except Exception as e:
            print(f"Error in hybrid retrieval: {str(e)}")
            return [] 