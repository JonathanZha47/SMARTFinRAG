from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
    RetrieverEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from src.config.config_loader import ConfigLoader
from src.evaluation.beir_evaluator import BeirEvaluatorWrapper
import plotly.graph_objects as go
import plotly.express as px
import os
import logging
from src.storing.storage_manager import StorageManager
from src.evaluation.question_generator import QuestionGenerator
from llama_index.core.llms import LLM
from llama_index.core.base.response.schema import Response
import numpy as np
from src.querying.query_engine import QueryEngine
from src.retrieval.retrieval import RetrieverFactory


logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Orchestrates RAG system evaluation using different benchmark types,
    including retriever and response quality evaluations.
    Acts as a universal interface.
    """

    def __init__(self,
                 config_loader: ConfigLoader,
                 storage_manager: StorageManager
                 ):
        """
        Initializes the RAG evaluator orchestrator.

        Args:
            config_loader: Loads project configuration.
            storage_manager: Manages index storage. Needed for QA generation.
        """
        self.config_loader = config_loader
        self.storage_manager = storage_manager

        # Initialize specific evaluators
        self.beir_evaluator_wrapper = BeirEvaluatorWrapper(config_loader)
        self.question_generator = QuestionGenerator(storage_manager)

        # Define evaluation metrics
        self.custom_retriever_metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
        self.custom_response_metrics = ["faithfulness", "relevancy"] # Define response metrics

    # --- Methods for Custom QA Pair Evaluation ---

    def generate_qa_pairs(self,
                           judge_llm: LLM,
                           num_questions_per_chunk: int = 2
                           ) -> Optional[EmbeddingQAFinetuneDataset]:
        """Generate QA pairs from document nodes using the specified judge LLM."""
        logger.info(f"Delegating QA pair generation to QuestionGenerator...")

        if judge_llm is None:
            logger.error("Cannot generate QA pairs without a valid judge LLM.")
            return None

        try:
            qa_dataset = self.question_generator.generate(
                judge_llm=judge_llm,
                num_questions_per_chunk=num_questions_per_chunk
            )
            return qa_dataset
        except Exception as e:
            logger.error(f"Error calling QuestionGenerator: {e}", exc_info=True)
            return None

    async def evaluate_retriever_custom_qa(self, retriever, qa_dataset_path: str = "data/evaluation/qa_pairs.json") -> Tuple[pd.DataFrame, Dict]:
        """Evaluate retriever performance using generated QA pairs (Custom QA Method)."""
        logger.info(f"Evaluating retriever using custom QA dataset: {qa_dataset_path}")
        try:
            # Load existing QA dataset
            if not os.path.exists(qa_dataset_path):
                 logger.error(f"QA dataset not found at {qa_dataset_path}")
                 return pd.DataFrame(), {}
            qa_dataset = EmbeddingQAFinetuneDataset.from_json(qa_dataset_path)
            logger.info(f"Loaded {len(qa_dataset.queries)} QA pairs.")
            # logger.debug(f"QA dataset sample: {list(qa_dataset.queries.items())[:2]}") # Optional: log a sample

            # Initialize retriever evaluator
            retriever_evaluator = RetrieverEvaluator.from_metric_names(
                self.custom_retriever_metrics,
                retriever=retriever
            )

            # Evaluate on entire dataset
            logger.info("Starting dataset evaluation with RetrieverEvaluator...")
            retriever_eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
            logger.info(f"Completed evaluation for {len(retriever_eval_results)} queries.")

            # Process results
            processed_results = [] # Renamed for clarity
            for eval_result in retriever_eval_results:
                metric_dict = eval_result.metric_vals_dict.copy() # Get metrics

                # --- Add the query text ---
                if hasattr(eval_result, 'query') and eval_result.query:
                    metric_dict['query'] = eval_result.query # Add query to the dict
                else:
                    # Fallback if query attribute is missing for some reason
                    query_id = getattr(eval_result, 'query_id', None) # Try to get ID
                    metric_dict['query'] = qa_dataset.queries.get(query_id, "Query not found")
                    if metric_dict['query'] == "Query not found":
                         logger.warning(f"Query string not found in evaluation result object or QA dataset for query_id: {query_id}")
                # --- End query addition ---

                processed_results.append(metric_dict) # Append the combined dict

            # Create detailed DataFrame from the processed results
            retriever_results_df = pd.DataFrame(processed_results)

            if retriever_results_df.empty:
                 logger.warning("Evaluation resulted in an empty DataFrame.")
                 return retriever_results_df, {}

            # --- Reorder columns to put 'query' first ---
            if 'query' in retriever_results_df.columns:
                 cols = ['query'] + [col for col in retriever_results_df.columns if col != 'query']
                 retriever_results_df = retriever_results_df[cols]
            # --- End reordering ---

            # Calculate aggregate metrics (ensure required columns exist)
            agg_metrics = {}
            required_cols = ['hit_rate', 'mrr', 'precision', 'recall', 'ap', 'ndcg']
            for col in required_cols:
                if col in retriever_results_df.columns:
                    # Use .astype(float, errors='ignore') or similar if types might be mixed
                    numeric_col = pd.to_numeric(retriever_results_df[col], errors='coerce')
                    agg_metrics[f'avg_{col}'] = numeric_col.mean()
                else:
                    logger.warning(f"Metric column '{col}' not found in results DataFrame for aggregation.")
                    agg_metrics[f'avg_{col}'] = None # Or 0 or NaN

            agg_metrics['total_queries'] = len(retriever_results_df)
            logger.info(f"Aggregate metrics: {agg_metrics}")

            return retriever_results_df, agg_metrics

        except FileNotFoundError:
             logger.error(f"QA dataset file not found at: {qa_dataset_path}")
             return pd.DataFrame(), {}
        except KeyError as ke:
            logger.error(f"KeyError during evaluation processing, likely missing metric: {ke}", exc_info=True)
            # Return partial results if possible, or empty
            return pd.DataFrame(processed_results) if 'processed_results' in locals() else pd.DataFrame(), {}
        except Exception as e:
            logger.error(f"Error evaluating retriever with custom QA: {e}", exc_info=True)
            return pd.DataFrame(), {}

    # --- NEW Method for Response Quality Evaluation ---
    def evaluate_response_quality(self,
                                 query: str, # path to latest query in data/queries.txt
                                 retrieved_contexts: str,
                                 response_str: str,
                                 judge_llm: LLM,
                                 faithfulness_threshold: float = 0.7,
                                 relevancy_threshold: float = 0.7
                                 ) -> Tuple[pd.DataFrame, Dict]:
        """
        Evaluates faithfulness and relevancy for a *single given query, response, and context*.

        Args:
            query: The user query string.
            response_str: The generated response string.
            retrieved_contexts: A list of context strings used for the response.
            judge_llm: The LLM instance to use for judging.

        Returns:
            A tuple containing:
            - A pandas DataFrame with the evaluation results for the single query (or None on error).
            - A dictionary with metrics for the single result (or None on error).
        """
        logger.info(f"Starting response quality evaluation for query: '{query[:50]}...'")

        # --- Input Validation ---
        if not query:
            logger.error("evaluate_response_quality requires a query string.")
            return None, {"error": "Query string is required"}
        if not response_str:
            logger.warning("evaluate_response_quality called with an empty response string.")
            # Decide how to handle: error or evaluate with empty response? Let's error for now.
            return None, {"error": "Empty response string provided"}
        if not judge_llm:
            logger.error("evaluate_response_quality requires a Judge LLM instance.")
            return None, {"error": "Judge LLM not provided"}
        # Allow empty context, but log it
        if not retrieved_contexts:
             logger.warning("evaluate_response_quality called with empty retrieved_contexts list. Faithfulness might be affected.")
             retrieved_contexts = [] # Ensure it's an empty list

        # --- Initialize Evaluators ---
        try:
            faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
            relevancy_evaluator = RelevancyEvaluator(llm=judge_llm)
            logger.info("Faithfulness and Relevancy evaluators initialized.")
        except Exception as e:
             logger.error(f"Failed to initialize response quality evaluators: {e}", exc_info=True)
             return None, {"error": f"Evaluator initialization failed: {e}"}

        # --- Prepare Data Structure ---
        eval_data = {
            'query': query,
            'response': response_str,
            'faithfulness': np.nan,
            'relevancy': np.nan,
            'is_faithful': None,
            'is_relevant': None,
            'contexts': retrieved_contexts, # Store context for inspection
            'error': None
        }

        # --- Perform Evaluations ---
        try:
            # 1. Evaluate Faithfulness
            faith_error = None
            try:
                 logger.debug("Evaluating Faithfulness...")
                 # Pass contexts directly to the evaluate method
                 faithfulness_result = faithfulness_evaluator.evaluate(
                     query=query,
                     response=response_str,
                     contexts=retrieved_contexts # Pass context strings directly
                 )
                 eval_data['is_faithful'] = faithfulness_result.passing
                 eval_data['faithfulness'] = faithfulness_result.score
                 logger.info(f"Faithfulness evaluation complete. Score: {eval_data['faithfulness']}, Passing: {eval_data['is_faithful']}")
            except Exception as fe:
                 logger.error(f"Faithfulness evaluation failed: {fe}", exc_info=True)
                 faith_error = f"Faithfulness Error: {fe}"


            # 2. Evaluate Relevancy
            relevancy_error = None
            try:
                 logger.debug("Evaluating Relevancy...")
                 relevancy_result = relevancy_evaluator.evaluate(
                     query=query,
                     response=response_str,
                     contexts=retrieved_contexts # Context might be less critical here but pass anyway
                 )
                 eval_data['is_relevant'] = relevancy_result.passing
                 eval_data['relevancy'] = relevancy_result.score
                 logger.info(f"Relevancy evaluation complete. Score: {eval_data['relevancy']}, Passing: {eval_data['is_relevant']}")
            except Exception as re:
                 logger.error(f"Relevancy evaluation failed: {re}", exc_info=True)
                 relevancy_error = f"Relevancy Error: {re}"

            # Combine errors if they occurred
            errors = [e for e in [faith_error, relevancy_error] if e is not None]
            if errors:
                 eval_data['error'] = "; ".join(errors)


        except Exception as e:
            # Catch broader errors during the process
            logger.error(f"Unexpected error during response evaluation processing: {e}", exc_info=True)
            eval_data['error'] = f"Processing Error: {e}"

        # --- Format Results ---
        # Create DataFrame for the single result
        try:
            results_df = pd.DataFrame([eval_data]) # List containing the single dict
        except Exception as df_e:
             logger.error(f"Failed to create DataFrame from evaluation data: {df_e}", exc_info=True)
             return None, {"error": f"DataFrame creation failed: {df_e}"}


        # Calculate "aggregate" metrics (just the values from the single run)
        aggregate_metrics = {
            'faithfulness_score': eval_data['faithfulness'],
            'relevancy_score': eval_data['relevancy'],
            'is_faithful': eval_data['is_faithful'],
            'is_relevant': eval_data['is_relevant'],
            'error': eval_data['error']
        }
        logger.info(f"Single response quality metrics: {aggregate_metrics}")

        return results_df, aggregate_metrics

    # --- Method for BEIR Evaluation ---

    def evaluate_beir(self,
                      datasets: List[str],
                      metrics_k_values: List[int],
                      embed_model_name: str,
                      retriever_type: str,
                      retriever_weights: Optional[Dict[str, float]] = None
                      ) -> Dict[str, Any]:
        """
        Runs the BEIR evaluation using the BeirEvaluatorWrapper with dynamic settings.

        Args:
            datasets: A list of BEIR dataset names (e.g., ["nfcorpus", "scifact"]).
            metrics_k_values: A list of k values for ranking metrics (e.g., [3, 10, 30]).
            embed_model_name: Identifier for the embedding model to use (e.g., "openai", "BAAI/bge-small-en").
            retriever_type: Type of retriever to use ("vector", "hybrid").
            retriever_weights: Optional dictionary of weights if retriever_type is "hybrid".

        Returns:
            A dictionary containing the evaluation results.
        """
        logger.info(f"Delegating BEIR evaluation with dynamic settings: embed='{embed_model_name}', retriever='{retriever_type}'")
        # Pass the new parameters to the wrapper's run_evaluation method
        return self.beir_evaluator_wrapper.run_evaluation(
            datasets=datasets,
            metrics_k_values=metrics_k_values,
            embed_model_name=embed_model_name,
            retriever_type=retriever_type,
            retriever_weights=retriever_weights
        )

    # --- Common Helper Methods (like plotting) ---

    def generate_evaluation_plots(self, results_df: pd.DataFrame, metrics_list: Optional[List[str]] = None) -> Dict:
        """Generate visualization plots for evaluation results (primarily for Custom QA)."""
        # Use default custom QA metrics if none provided
        metrics_to_plot = metrics_list if metrics_list else self.custom_retriever_metrics
        plots = {}
        logger.info("Generating evaluation plots...")

        if results_df.empty:
             logger.warning("Cannot generate plots from empty DataFrame.")
             return {}

        try:
            # Check if provided metrics exist in the DataFrame
            valid_metrics = [m for m in metrics_to_plot if m in results_df.columns]
            if not valid_metrics:
                 logger.warning(f"None of the specified metrics {metrics_to_plot} found in results DataFrame.")
                 return {}

            # Score Distribution Plot
            fig_scores = go.Figure()
            for metric in valid_metrics:
                fig_scores.add_trace(go.Box(y=results_df[metric], name=metric.upper()))
            fig_scores.update_layout(
                title='Distribution of Retrieval Metrics',
                yaxis_title='Score',
                showlegend=True
            )
            plots['score_distribution'] = fig_scores

            # Metric Correlation Plot (only if more than one metric)
            if len(valid_metrics) > 1:
                fig_correlation = px.scatter_matrix(
                    results_df[valid_metrics],
                    title='Correlation between Retrieval Metrics'
                )
                plots['metric_correlation'] = fig_correlation

            # Performance Trend
            fig_performance = go.Figure()
            for metric in valid_metrics:
                fig_performance.add_trace(go.Scatter(
                    y=results_df[metric],
                    name=metric.upper(),
                    mode='lines+markers'
                ))
            fig_performance.update_layout(
                title='Performance Across Queries',
                xaxis_title='Query Index',
                yaxis_title='Score'
            )
            plots['performance_trend'] = fig_performance

            logger.info("Successfully generated plots.")
            return plots

        except Exception as e:
            logger.error(f"Error generating evaluation plots: {e}", exc_info=True)
            return {} 