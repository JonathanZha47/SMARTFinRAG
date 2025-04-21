import logging
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.evaluation.benchmarks import BeirEvaluator as LlamaBeirEvaluator
# Import specific embedding types you might use
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from src.config.config_loader import ConfigLoader
# Removed: from src.indexing.index_manager import IndexManager
# Import your HybridRetriever if you use it for BEIR hybrid eval
from src.retrieval.hybrid_retriever import HybridRetriever


logger = logging.getLogger(__name__)

class BeirEvaluatorWrapper:
    """
    A wrapper class to handle BEIR benchmark evaluations using LlamaIndex,
    allowing dynamic selection of embedding models and retrievers.
    """
    def __init__(self, config_loader: ConfigLoader):
        """
        Initializes the BEIR evaluator wrapper. Only stores references.
        Models and retrievers are created dynamically per evaluation run.

        Args:
            config_loader: Loads project configuration (might be needed for future settings).
        """
        self.config_loader = config_loader
        self.beir_evaluator = LlamaBeirEvaluator()
        logger.info("BeirEvaluatorWrapper initialized (without IndexManager dependency).")


    def _get_embedding_model(self, embed_model_name: str):
        """Helper to instantiate embedding model based on name."""
        logger.info(f"Instantiating embedding model: {embed_model_name}")
        if embed_model_name == "openai":
            # Assumes OPENAI_API_KEY is set in environment
            return OpenAIEmbedding()
        elif embed_model_name == "bert-base-uncased":
             # Example using HuggingFaceEmbedding
             return HuggingFaceEmbedding(model_name="bert-base-uncased")
        elif embed_model_name: # Treat other non-empty strings as HuggingFace model names
             # Assumes HUGGINGFACE_API_KEY might be needed depending on model
             return HuggingFaceEmbedding(model_name=embed_model_name)
        else:
            logger.error("No valid embedding model name provided.")
            raise ValueError("Embedding model name is required for BEIR evaluation.")


    def run_evaluation(self,
                       datasets: List[str],
                       metrics_k_values: List[int],
                       embed_model_name: str,
                       retriever_type: str,
                       retriever_weights: Optional[Dict[str, float]] = None
                       ) -> Dict[str, Any]:
        """
        Runs the BEIR evaluation for specified datasets and k values,
        using the dynamically selected embedding model and retriever.

        Args:
            datasets: A list of BEIR dataset names.
            metrics_k_values: A list of k values for ranking metrics.
            embed_model_name: Identifier for the embedding model.
            retriever_type: Type of retriever ("vector", "hybrid").
            retriever_weights: Optional weights for hybrid retriever.

        Returns:
            A dictionary containing the evaluation results. Returns an empty dict on failure.
        """
        try:
            embed_model = self._get_embedding_model(embed_model_name)
        except Exception as e:
            logger.error(f"Failed to instantiate embedding model '{embed_model_name}': {e}")
            return {}

        logger.info(f"Starting BEIR evaluation: datasets={datasets}, k={metrics_k_values}, embed='{embed_model_name}', retriever='{retriever_type}'")

        # Determine the maximum k needed for retrieval based on metrics
        similarity_top_k = max(metrics_k_values) if metrics_k_values else 10 # Default k=10 if none specified

        def create_dynamic_beir_retriever(documents: List[Document]):
            """
            Nested function to create a retriever dynamically for BEIR evaluation.
            Uses the embedding model and retriever type selected for this run.
            """
            nonlocal embed_model, retriever_type, retriever_weights, similarity_top_k # Ensure closure uses the right vars

            logger.info(f"Building temporary index for BEIR dataset ({len(documents)} docs) using {embed_model_name}...")
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model,
                show_progress=True
            )
            logger.info("Temporary index built.")

            logger.info(f"Creating retriever of type: {retriever_type} with k={similarity_top_k}")
            if retriever_type == "vector":
                return index.as_retriever(similarity_top_k=similarity_top_k)
            elif retriever_type == "hybrid":
                if not retriever_weights:
                    logger.warning("Hybrid retriever selected but no weights provided. Using default weights.")
                    # Use default weights if none provided, get from config or define here
                    retriever_weights = {'bm25': 0.4, 'vector': 0.4, 'keyword': 0.2}

                # Ensure your HybridRetriever class can be initialized this way
                # It needs the index (for vector search) and potentially access
                # to nodes/docstore for BM25/keyword parts if not handled internally.
                # This assumes HybridRetriever is designed to work with the created index.
                logger.info(f"Using HybridRetriever with weights: {retriever_weights}")
                return HybridRetriever(
                    index=index,
                    weights=retriever_weights,
                    top_k=similarity_top_k # Pass top_k here too
                )
            # Add elif for "bm25" if implemented
            # elif retriever_type == "bm25":
            #     # Requires nodes, potentially different initialization
            #     logger.warning("BM25 retriever for BEIR not fully implemented in this example.")
            #     return index.as_retriever(similarity_top_k=similarity_top_k) # Fallback to vector?
            else:
                logger.error(f"Unsupported retriever type for BEIR: {retriever_type}. Falling back to vector.")
                return index.as_retriever(similarity_top_k=similarity_top_k)

        try:
            # Run the evaluation using the dynamic retriever creation function
            results = self.beir_evaluator.run(
                create_dynamic_beir_retriever,
                datasets=datasets,
                metrics_k_values=metrics_k_values
            )
            logger.info("BEIR evaluation completed successfully.")
            return results
        except Exception as e:
            logger.error(f"BEIR evaluation run failed: {e}", exc_info=True)
            return {} 