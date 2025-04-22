from typing import Dict, List, Optional, Tuple
from llama_index.core import VectorStoreIndex, Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from src.config.config_loader import ConfigLoader
from src.querying.query_preprocessor import QueryPreprocessor
from src.querying.problem_decomposer import ProblemDecomposer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.prompting.prompt_manager import PromptManager
from src.retrieval.retrieval import RetrieverFactory
from src.utils.long_context_handler import LongContextHandler
from llm_providers import get_llm_provider_from_toml
import logging
import traceback
from src.indexing.index_manager import IndexManager
from src.storing.storage_manager import StorageManager
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.data_structs import Node
from llama_index.core.prompts import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

class QueryEngine:
    """Manages the query processing pipeline."""
    
    def __init__(self, llm, config_loader: ConfigLoader, openrouter_api_key: str):
        self.config_loader = config_loader
        self.query_config = config_loader.get_query_config()
        self.index_manager = IndexManager(config_loader)
        self.storage_manager = StorageManager(config_loader)
        self.prompt_manager = PromptManager(config_loader)
        self.query_preprocessor = QueryPreprocessor(config_loader, openrouter_api_key)
        self.context_handler = LongContextHandler(config_loader)
        self.problem_decomposer = ProblemDecomposer(openrouter_api_key)
        self.retriever_factory = RetrieverFactory(config_loader)
        self.llm = llm
        

        # --- Index Initialization (Adapted from previous step) ---
        self.index: Optional[VectorStoreIndex] = self.storage_manager.load_index()

        if self.index:
            logger.info("QueryEngine: Stored index loaded successfully.")
        else:
            logger.warning("QueryEngine: No stored index found or failed to load. Initializing a new empty index.")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_manager.persist_dir)
                self.index = VectorStoreIndex.from_documents(
                    [],
                    storage_context=storage_context,
                    show_progress=False
                )
                logger.info(f"QueryEngine: Initialized new empty VectorStoreIndex associated with: {self.storage_manager.persist_dir}")
            except Exception as e:
                logger.error(f"QueryEngine: Failed to initialize a new empty index: {e}", exc_info=True)
                self.index = None
        # --- End Index Initialization ---
            
    # Method to add processed nodes and persist (Keep from previous step)
    def add_nodes_to_index(self, nodes: List[Node]) -> bool:
        """Inserts processed nodes into the index and persists the changes."""
        if self.index is None:
            logger.error("Cannot add nodes: Index is not initialized.")
            return False
        if not nodes:
            logger.warning("No nodes provided to add to the index.")
            return True # Nothing to do, technically successful

        try:
            logger.info(f"Inserting {len(nodes)} nodes into the index.")
            self.index.insert_nodes(nodes, show_progress=True)
            logger.info(f"Successfully inserted {len(nodes)} nodes.")

            logger.info("Persisting updated index...")
            save_successful = self.storage_manager.save_index(self.index)
            if save_successful:
                logger.info("Successfully persisted the updated index.")
                return True
            else:
                logger.error("Failed to persist the updated index after insertion.")
                return False
        except Exception as e:
            logger.error(f"Error inserting nodes or persisting index: {e}", exc_info=True)
            return False

    def process_query(self, query: str, index: Optional[BaseIndex], provider_name: str,
                      model_name: str, retriever_type: str, force_rag: bool, prompt_type: Optional[str], top_k: int,
                      persona: Optional[str]) -> Tuple[str, bool, Dict, Optional[List[NodeWithScore]]]:
        """Process a user query through the RAG pipeline."""
        analysis_results = {}
        try:
            logger.info(f"Processing query: '{query}' with Provider: {provider_name}, Model: {model_name}, Retriever: {retriever_type}")
            analysis_results['original_query'] = query

            if not self.llm:
                 return f"Error: Could not initialize LLM provider '{provider_name}'"
            logger.info("LLM set for query.")

            # --- 1. Preprocess Query ---
            processed_query, analysis = self.query_preprocessor.preprocess_query(query)
            analysis_results.update(analysis)
            analysis_results['enhanced_query'] = processed_query
            logger.info(f"Query preprocessed. Enhanced Query: '{processed_query}'")
            
            # --- 2. Decompose Query (Optional) ---
            try:
                decomposition = self.problem_decomposer.decompose_query(processed_query)
                analysis['decomposition'] = decomposition
                # Log only sub-problem queries for brevity if available
                sub_queries = [sp.get('query', 'N/A') for sp in decomposition.get('sub_problems', [])]
                logging.info(f"Query decomposition successful: {sub_queries}")
            except Exception as e:
                logging.error(f"Error during query decomposition: {str(e)}")
                analysis['decomposition'] = {'error': str(e)} # Store error in analysis
            
            # --- 3. Decide on RAG vs Direct Query ---
            active_index = self.index # Always use the managed index
            analysis_needs_retrieval = analysis.get('needs_retrieval', False)
            use_rag = force_rag or analysis_needs_retrieval
            analysis_results['rag_decision_forced'] = force_rag
            logger.info(f"Force RAG: {force_rag}, Analysis needs retrieval: {analysis_needs_retrieval}, Use RAG: {use_rag}")


            # Check index usability for RAG
            if use_rag and active_index is None:
                logger.warning("RAG needed but index is not available. Fallback Direct.")
                use_rag = False; analysis_results['rag_status'] = "RAG Needed - No Index (Fallback Direct)"

            # --- 4. Execute RAG or Direct Query ---
            if use_rag:
                analysis_results['rag_status'] = f"Attempting RAG ({retriever_type})"
                logger.info(f"Executing RAG pipeline with retriever: {retriever_type}...")
                try:
                    # --- Get Retriever using the Factory ---
                    retriever = self.retriever_factory.create_retriever(
                        index=active_index, # Pass the managed index
                        retriever_type=retriever_type,
                        top_k=top_k
                    )
                    if retriever:
                        # --- Get Custom Prompt Template ---
                        custom_prompt_str = self.prompt_manager.get_rag_prompt_template(
                             prompt_type=prompt_type, persona=persona, query=processed_query
                        )
                        custom_llama_template = PromptTemplate(custom_prompt_str)
                        logger.debug(f"Using custom RAG prompt template (start): {custom_prompt_str[:100]}...")

                        # --- Configure Response Synthesizer ---
                        response_synthesizer = get_response_synthesizer(
                            llm=self.llm,
                            response_mode="refine",
                            text_qa_template=custom_llama_template
                            # callback_manager=self.callback_manager,
                        )

                        # --- Assemble Query Engine ---
                        engine = RetrieverQueryEngine(
                            retriever=retriever,
                            response_synthesizer=response_synthesizer
                            # callback_manager=self.callback_manager,
                        )
                        logger.info(f"Assembled RetrieverQueryEngine with {retriever_type} retriever.")

                        # --- Execute Query ---
                        response_obj = engine.query(processed_query)
                        response_text = response_obj.response

                        response_context = response_obj.source_nodes

                        # --- Update Analysis ---
                        analysis_results['rag_status'] = f"Used RAG ({retriever_type})"
                        if hasattr(response_obj, 'source_nodes'):
                             context_nodes = response_obj.source_nodes
                             analysis_results['llm_context'] = "\n---\n".join([node.get_content() for node in context_nodes])
                             analysis_results['retrieved_nodes_count'] = len(context_nodes)
                        else: analysis_results['llm_context'] = ""; analysis_results['retrieved_nodes_count'] = 0
                        logger.info(f"RAG query successful. Retrieved {analysis_results['retrieved_nodes_count']} nodes.")
                        # if use RAG, then return response_text, if_success, analysis_results, response_context
                        return response_text, True, analysis_results, response_context
                    else:
                         logger.error(f"Failed to create retriever '{retriever_type}'. Fallback Direct.")
                         analysis_results['rag_status'] = f"RAG Attempted ({retriever_type}) - Retriever Failed (Fallback Direct)"
                         # Fall through

                except Exception as e:
                    logger.error(f"Error during RAG pipeline execution: {e}", exc_info=True)
                    analysis_results['rag_status'] = f"RAG Attempted ({retriever_type}) - Failed ({type(e).__name__}) (Fallback Direct)"
                    # Fall through

            # --- Execute Direct Query ---
            if 'rag_status' not in analysis_results or "Fallback Direct" in analysis_results['rag_status']:
                if 'rag_status' not in analysis_results: analysis_results['rag_status'] = "Direct Query"
                logger.info("Executing direct query to LLM...")
                try:
                    direct_prompt = self.prompt_manager.get_direct_prompt(
                        query=processed_query, prompt_type=prompt_type, persona=persona
                    )
                    logger.debug(f"Using direct prompt (start): {direct_prompt[:100]}...")

                    if self.llm and hasattr(self.llm, 'complete'):
                        # Pass necessary kwargs if your generate_response expects them (like temperature)
                        # Temperature is likely already set during LLM init, but check provider implementation
                        response_text = self.llm.complete(direct_prompt) # Add kwargs if needed
                    else:
                        logger.error("LLM object is None or does not have 'complete' method.")
                        raise NotImplementedError("LLM interaction method not available.")

                    # Basic check if the response indicates an error returned by the provider
                    if "Error:" in response_text:
                         logger.error(f"Direct LLM query failed with provider error: {response_text}")
                         return response_text, False, analysis_results, None # Return error state

                    logger.info("Direct LLM query successful.")
                    # if direct query, then we do not have response_context, use empty string instead
                    return response_text, True, analysis_results, None
                except Exception as e:
                    logger.error(f"Error during direct LLM query execution: {e}", exc_info=True)
                    return f"Error processing direct query: {e}", False, analysis_results, None

        except Exception as e:
             logger.error(f"Unhandled error in process_query: {e}", exc_info=True)
             analysis_results['error'] = f"Unhandled error: {str(e)}"
             return f"An unexpected error occurred: {str(e)}", False, analysis_results, None
    
    
    def _check_if_needs_retrieval(self, query: str, analysis: Dict) -> bool:
        """Determine if query needs retrieval based on analysis."""
        # Simplified check: if analysis contains certain keys indicating complexity or need for specific data
        retrieval_indicators = [
            'requires_calculation',
            'is_comparison',
            'time_sensitive',
            'requires_specific_info'
        ]
        
        needs_retrieval = any(analysis.get(key, False) for key in retrieval_indicators)
        
        # Also consider if companies are mentioned
        if len(analysis.get('companies_mentioned', [])) > 0:
            needs_retrieval = True
            
        logging.debug(f"Checking retrieval need for query: '{query}'. Analysis: {analysis}. Needs retrieval: {needs_retrieval}")
        return needs_retrieval 