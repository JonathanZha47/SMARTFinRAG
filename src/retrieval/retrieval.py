import logging
from typing import Optional, Dict, List

# LlamaIndex Core Imports
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    QueryFusionRetriever,
    AutoMergingRetriever,
)
from llama_index.core.schema import Node # Needed for BM25/Node access

# Specific Retriever Imports
from llama_index.retrievers.bm25 import BM25Retriever

# Configuration Loader
from src.config.config_loader import ConfigLoader

from nltk.stem import SnowballStemmer

logger = logging.getLogger(__name__)

class RetrieverFactory:
    """Factory class to create different LlamaIndex retrievers."""

    def __init__(self, config_loader: ConfigLoader):
        """Initializes the factory with configurations."""
        self.rag_config = config_loader.get_rag_config()
        logger.info("RetrieverFactory initialized.")

    def _get_nodes_from_index(self, index: VectorStoreIndex) -> List[Node]:
        """Safely retrieves all nodes from the index's docstore."""
        nodes: List[Node] = []
        if hasattr(index, 'docstore') and index.docstore:
             try:
                 nodes = list(index.docstore.docs.values())
                 if nodes:
                      logger.info(f"Retrieved {len(nodes)} nodes from index docstore.")
                 else:
                      logger.warning("Index docstore exists but contains no nodes.")
             except Exception as e:
                  logger.warning(f"Could not retrieve nodes from docstore: {e}. BM25/Fusion might fail.", exc_info=True)
        else:
             logger.warning("Index does not have a 'docstore' attribute or it's None. BM25/Fusion will likely fail.")
        return nodes

    def create_retriever(
        self,
        index: VectorStoreIndex,
        retriever_type: str,
        top_k: int
    ) -> Optional[BaseRetriever]:
        """
        Creates and returns a specific LlamaIndex retriever instance.

        Args:
            index: The VectorStoreIndex object to retrieve from.
            retriever_type: The type of retriever requested (e.g., "Vector", "BM25").
            top_k: The default number of top results to retrieve.

        Returns:
            An instance of a LlamaIndex BaseRetriever, or None if creation fails.
        """
        if index is None:
            logger.error("Cannot create retriever: Index object is None.")
            return None

        retriever_type_lower = retriever_type.lower().replace("-", "").replace(" ", "")
        logger.info(f"Attempting to create retriever of type: '{retriever_type}' (Normalized: '{retriever_type_lower}') with top_k={top_k}")

        # --- Base Vector Retriever (common component) ---
        try:
            vector_retriever = index.as_retriever(
                similarity_top_k=top_k
            )
        except Exception as e:
             logger.error(f"Failed to create base VectorIndexRetriever: {e}", exc_info=True)
             return None # Cannot proceed if base retriever fails

        # --- Specific Retriever Logic ---
        if retriever_type_lower == 'vector':
            logger.info("Using standard VectorIndexRetriever.")
            return vector_retriever

        elif retriever_type_lower == 'bm25':
            # Get nodes specifically for BM25
            all_nodes = self._get_nodes_from_index(index)
            if not all_nodes:
                logger.error("Cannot create BM25Retriever: No nodes found or accessible in the index's docstore.")
                return None
            try:
                logger.info(f"Creating BM25Retriever with {len(all_nodes)} nodes.")
                # Optional: Stemmer requires nltk data
                # stemmer = Stemmer.Stemmer("english")
                return BM25Retriever.from_defaults(
                    nodes=all_nodes,
                    similarity_top_k=top_k
                    # stemmer=stemmer,
                    # language="english"
                )
            except ImportError:
                logger.error("BM25Retriever requires 'rank_bm25' package. Install it (`pip install rank_bm25`).")
                return None
            except Exception as e:
                 logger.error(f"Error creating BM25Retriever: {e}", exc_info=True)
                 return None

        elif retriever_type_lower == 'hybridfusion':
            # Get nodes specifically for BM25
            all_nodes = self._get_nodes_from_index(index)
            if not all_nodes:
                logger.error("Cannot create Hybrid Fusion Retriever: No nodes found or accessible for BM25 part.")
                return None
            try:
                logger.info(f"Creating BM25 retriever part for fusion with {len(all_nodes)} nodes.")
                bm25_retriever_part = BM25Retriever.from_defaults(
                    nodes=all_nodes,
                    similarity_top_k=top_k # Use same top_k or configure separately?
                )

                logger.info("Creating QueryFusionRetriever (Vector + BM25).")

                return QueryFusionRetriever(
                    retrievers=[vector_retriever, bm25_retriever_part],
                    similarity_top_k=top_k, # Final top_k after fusion
                    use_async=False,
                    retriever_weights=[0.6, 0.4] # Get from config if needed
                )
            except ImportError:
                 logger.error("Hybrid Fusion requires 'rank_bm25'. Install it (`pip install rank_bm25`).")
                 return None
            except Exception as e:
                 logger.error(f"Error creating QueryFusionRetriever: {e}", exc_info=True)
                 return None

        elif retriever_type_lower == 'automerging':
            if not hasattr(index, 'storage_context') or index.storage_context is None:
                 logger.error("Cannot create AutoMergingRetriever: Index missing a valid storage_context.")
                 return None

            logger.info("Creating AutoMergingRetriever.")
            # AutoMergingRetriever wraps the vector_retriever
            return AutoMergingRetriever(
                vector_retriever=vector_retriever,
                storage_context=index.storage_context,
                verbose=True
            )

        else:
            logger.warning(f"Unsupported retriever type specified: '{retriever_type}'. Falling back to default Vector retriever.")
            return vector_retriever
