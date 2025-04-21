import streamlit as st
import pandas as pd
from src.evaluation.rag_evaluator import RAGEvaluator
from src.config.config_loader import ConfigLoader
from src.storing.storage_manager import StorageManager
from src.querying.query_engine import QueryEngine
from src.retrieval.retrieval import RetrieverFactory
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
import json # For pretty printing BEIR results
import os
import traceback # Import traceback for detailed error printing
import logging # Import logging
from src.evaluation.judge_llm_provider import get_judge_llm_from_toml # Import judge LLM provider
from typing import Dict, Any, List # Import Dict, Any, List for type hinting
import asyncio
import numpy as np # Import numpy for NaN checks
# --- Import the new state manager ---
import src.evaluation.evaluation_state_manager as eval_state

# Ensure logger is configured (it inherits config from root logger in app.py)
logger = logging.getLogger(__name__) # Initialize logger for this page

# Function to display detailed results (ensure it exists or define it)
def display_detailed_results(df):
    st.dataframe(df)

# Function to display metrics (ensure it exists or define it)
def display_metrics(metrics_dict):
    st.write("### Aggregate Metrics")
    if not metrics_dict:
        st.warning("No metrics data available.")
        return
    # Filter out keys that shouldn't be displayed directly as metrics if needed
    display_keys = [k for k in metrics_dict if k not in ['error']] # Example filter

    cols = st.columns(len(display_keys))
    i = 0
    for key in display_keys:
        value = metrics_dict[key]
        label = key.replace('_', ' ').title()
        # Handle specific metric display formats
        if isinstance(value, bool):
             cols[i].metric(label=label, value="✅ Yes" if value else "❌ No")
        elif isinstance(value, float):
             cols[i].metric(label=label, value=f"{value:.4f}" if not pd.isna(value) else "N/A")
        elif pd.isna(value): # Catch other NaN types
            cols[i].metric(label=label, value="N/A")
        else: # Handle integers, strings, etc.
            cols[i].metric(label=label, value=value)
        i += 1
    # Optionally display errors separately
    if 'error' in metrics_dict and metrics_dict['error']:
         st.error(f"Errors occurred during evaluation: {metrics_dict['error']}")

def display_beir_results(results: dict):
    """Display BEIR evaluation results."""
    st.write("### BEIR Evaluation Results")
    if not results:
        st.warning("No BEIR results to display.")
        return

    for dataset, metrics in results.items():
        st.subheader(f"Dataset: {dataset}")
        # Use json.dumps for pretty printing the dictionary of metrics
        st.json(metrics)
        # Optionally convert to DataFrame for better table display
        try:
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index').transpose()
            st.dataframe(metrics_df)
        except Exception as e:
            st.error(f"Could not display metrics for {dataset} as DataFrame: {e}")

def evaluation_sidebar_settings(eval_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create sidebar for evaluation settings and return them as a dictionary."""
    settings_dict = {}

    st.sidebar.title("⚙️ Evaluation Settings")
    st.sidebar.subheader("Evaluation Type")
    eval_type = st.sidebar.selectbox(
        "Choose Benchmark",
        ["Custom QA Pairs Generation Retrieval Quality", "Current Query Response Quality", "BEIR Benchmarks"],
        key="eval_type_select"
    )
    settings_dict["eval_type"] = eval_type
    logger.debug(f"Selected eval_type: {eval_type}")

    # --- Judge LLM Selection (Common Setting) ---
    st.sidebar.subheader("Judge LLM Settings")
    # Load options from the evaluation config loaded earlier
    judge_llm_config = eval_config.get('judge_llm_providers', {})
    judge_llm_providers = list(judge_llm_config.keys())

    if not judge_llm_providers:
        st.sidebar.warning("No Judge LLM providers configured in config.yaml -> evaluation -> judge_llm_providers")
        # Set defaults or handle error
        judge_provider = None
        judge_model = None
    else:
        judge_provider = st.sidebar.selectbox(
            "Select Judge LLM Provider",
            options=judge_llm_providers,
            index=0, # Or find index of a default if needed
            key="judge_provider"
        )
        judge_models_options = judge_llm_config.get(judge_provider, [])
        if not judge_models_options:
             st.sidebar.warning(f"No models configured for Judge LLM provider '{judge_provider}'")
             judge_model = None
        else:
            judge_model = st.sidebar.selectbox(
                "Select Judge Model",
                options=judge_models_options,
                index=0, # Or find index of a default model
                key="judge_model"
            )

    # Store judge LLM selection in the dictionary
    settings_dict["judge_provider"] = judge_provider
    settings_dict["judge_model"] = judge_model

    # --- Evaluation Specific Settings (in Sidebar) ---
    if eval_type == "Custom QA Pairs Generation Retrieval Quality":
        st.sidebar.subheader("Custom QA Generation Settings")
        questions_per_chunk = st.sidebar.slider("Questions per Document Chunk", 1, 5, 2, key="qa_questions")
        st.sidebar.subheader("Retrieval Settings (for Custom QA)")
        top_k = st.sidebar.slider("Top K Results", 1, 20, 10, key="qa_topk")

        retriever_type = st.sidebar.selectbox(
        "Select Retriever Type",
        options=["Vector", "BM25", "Custom Hybrid Fusion", "Auto-Merging"], # Add more if implemented
        index=0, # Default to Vector
        help="Choose the method for retrieving relevant documents."
        )
        if retriever_type == "Custom Hybrid Fusion":
            # --- Constrained Weight Input for Custom QA ---
            st.sidebar.write("Hybrid Search Weights (Sum = 1)")
            # Use number inputs or sliders for the first two, calculate the third
            col1, col2 = st.sidebar.columns(2)
            with col1:
                bm25_weight_qa = st.slider("BM25 Weight", 0.0, 1.0, 0.4, 0.01, key="qa_bm25_w") # Finer step
            with col2:
                # Ensure vector weight doesn't make sum exceed 1
                max_vector_weight = 1.0 - bm25_weight_qa
                vector_weight_qa = st.slider("Vector Weight", 0.0, max_vector_weight, min(0.4, max_vector_weight), 0.01, key="qa_vector_w")

            # Calculate keyword weight
            keyword_weight_qa = round(1.0 - bm25_weight_qa - vector_weight_qa, 2) # Use round to avoid floating point issues

            # Display the calculated keyword weight (read-only)
            st.sidebar.metric("Keyword Weight (calculated)", f"{keyword_weight_qa:.2f}")
            # Check if calculation resulted in a negative value (shouldn't happen with sliders)
            if keyword_weight_qa < 0:
                st.sidebar.error("Weights configuration error (sum > 1)")
                # Reset to default or handle error
                weights = {'bm25': 0.4, 'vector': 0.4, 'keyword': 0.2}
            else:
                weights = {
                    'bm25': bm25_weight_qa,
                    'vector': vector_weight_qa,
                    'keyword': keyword_weight_qa
                }
                settings_dict["custom_qa_weights"] = weights
        # --- End Constrained Weight Input ---

        settings_dict["custom_qa_top_k"] = top_k
        settings_dict["custom_qa_retriever_type"] = retriever_type
        settings_dict["custom_qa_questions_per_chunk"] = questions_per_chunk
    elif eval_type == "Current Query Response Quality":
        st.sidebar.subheader("All the LLM and RAG settings from the main page will be used for this evaluation")
    elif eval_type == "BEIR Benchmarks":
        st.sidebar.subheader("BEIR Settings")
        beir_embedding_model_name = st.sidebar.selectbox(
            "Select Embedding Model (BEIR)", options=["BAAI/bge-small-en-v1.5", "openai", "bert-base-uncased"], index=0, key="beir_embed_model",
            help="Choose the embedding model to use for indexing BEIR datasets."
        )
        beir_retriever_type = st.sidebar.selectbox(
            "Select Retriever Type (BEIR)", options=["vector", "hybrid"], index=0, key="beir_retriever_type",
            help="Choose the retrieval strategy for BEIR evaluation."
        )

        weights = {'bm25': 0.4, 'vector': 0.4, 'keyword': 0.2} # Default weights for BEIR Hybrid
        if beir_retriever_type == "hybrid":
            # --- Constrained Weight Input for BEIR ---
            st.sidebar.write("Hybrid Search Weights (BEIR - Sum = 1)")
            col1_beir, col2_beir = st.sidebar.columns(2)
            with col1_beir:
                 bm25_weight_beir = st.slider("BM25 Weight", 0.0, 1.0, 0.4, 0.01, key="beir_bm25_w")
            with col2_beir:
                 max_vector_weight_beir = 1.0 - bm25_weight_beir
                 vector_weight_beir = st.slider("Vector Weight", 0.0, max_vector_weight_beir, min(0.4, max_vector_weight_beir), 0.01, key="beir_vector_w")

            keyword_weight_beir = round(1.0 - bm25_weight_beir - vector_weight_beir, 2)
            st.sidebar.caption("Keyword Weight (calculated)", f"{keyword_weight_beir:.2f}")

            if keyword_weight_beir < 0:
                st.sidebar.error("Weights configuration error (sum > 1)")
                weights = {'bm25': 0.4, 'vector': 0.4, 'keyword': 0.2} # Fallback
            else:
                 weights = {
                     'bm25': bm25_weight_beir,
                     'vector': vector_weight_beir,
                     'keyword': keyword_weight_beir
                 }
            # --- End Constrained Weight Input ---

        beir_datasets = st.sidebar.multiselect(
            "Select BEIR Datasets", options=["nfcorpus", "scifact", "arguana", "climate-fever", "fiqa", "quora", "webis-touche2020", "trec-covid", "nq"],
            default=["nfcorpus"], key="beir_datasets"
        )
        beir_k_values_str = st.sidebar.text_input(
            "Metrics K Values (comma-separated)", value="3,10,30", key="beir_k"
        )
        try:
            beir_k_values = [int(k.strip()) for k in beir_k_values_str.split(',') if k.strip()]
        except ValueError:
            st.sidebar.error("Invalid K values. Please enter comma-separated integers.")
            beir_k_values = []

        settings_dict["beir_embedding_model_name"] = beir_embedding_model_name
        settings_dict["beir_retriever_type"] = beir_retriever_type
        settings_dict["beir_weights"] = weights # Store the calculated weights
        settings_dict["beir_datasets_selected"] = beir_datasets
        settings_dict["beir_k_values_list"] = beir_k_values

    # Add Clear History button conditionally based on selected eval_type
    eval_type = settings_dict.get("eval_type") # Get the selected type
    if eval_type == "Current Query Response Quality":
        st.sidebar.markdown("---")
        st.sidebar.caption("Clear accumulated response quality results for this session.")
        if st.sidebar.button("Clear Response Quality History", key="clear_resp_qual_hist"):
            eval_state.clear_response_quality_history()
            st.sidebar.success("History Cleared!")
            # Optional: Rerun immediately if the display needs instant update
            st.rerun() # Use st.rerun instead of experimental_rerun

    logger.debug("Sidebar settings processed.")
    return settings_dict

def render_evaluation_page():
    logger.debug("Starting evaluation page render")
    st.title("📊 RAG System Evaluation")
    st.write("Evaluate the performance of the retrieval system using various benchmarks.")

    try:
        # Initialize components if not in session state (or load if needed)
        if 'config_loader' not in st.session_state:
            st.error("Configuration not loaded. Please go back to the main app.")
            return
        config_loader = st.session_state.config_loader

        if 'storage_manager' not in st.session_state:
            st.error("Storage Manager not initialized. Please go back to the main app.")
            return
        storage_manager = st.session_state.storage_manager

        # Instantiate RetrieverFactory (needs config_loader)
        retriever_factory = RetrieverFactory(config_loader)
        logger.info("Initializing RAGEvaluator...")
        evaluator = RAGEvaluator(
                config_loader,
                storage_manager
            )

        # Load evaluation config once
        eval_config = config_loader.get_evaluation_config()

        # Get settings from the dedicated sidebar function, passing the eval config
        settings = evaluation_sidebar_settings(eval_config)
        eval_type = settings["eval_type"]

        # Get API keys from session state
        openai_key = st.session_state.get("openai_key")
        huggingface_key = st.session_state.get("huggingface_key")
        openrouter_key = st.session_state.get("openrouter_key")

        # """
        # Below is another way for eval_type selection
        #   # --- Evaluation Type Selection ---
        # eval_types = [
        #     "BEIR Dataset Evaluation",
        #     "Custom QA Pairs Generation", 
        #     "Retriever (Custom QA)",
        #     "Response Quality",
        #     "Response Quality (Detailed)" # Placeholder for future detailed view
        # ]
        # eval_type = st.selectbox("Select Evaluation Type", eval_types)

        # tabs = st.tabs([f"📈 {t}" for t in eval_types])
        # current_tab = eval_types.index(eval_type)

        # # --- BEIR Dataset Eval Tab ---
        # # ... (keep existing BEIR code) ...
        # # current_tab += 1 # Increment if BEIR tab exists

        # # --- Custom QA Generation Tab ---
        # # ... (keep existing Custom QA Generation code) ...
        # # current_tab += 1 # Increment if QA Gen tab exists
        # """

        # --- Main Evaluation Section ---
        st.subheader(f"Evaluation: {eval_type}")
        logger.info(f"Starting evaluation section: {eval_type}")

        index = storage_manager.load_index()
        if index is None:
            logger.error("Failed to load index. Please ensure it's built correctly.")
            return
        else:
            logger.info(f"Index loaded successfully")

        # --- Custom QA Evaluation Workflow ---
        if eval_type == "Custom QA Pairs Generation Retrieval Quality":
            st.write("You should generate a new QA pairs dataset first based on the documents uploaded in the main page, and then evaluate the retrieval performance of the RAG system using the generated QA pairs.")
            generate_tab, evaluate_tab = st.tabs(["Generate QA Pairs", "Evaluate Retrieval (Custom QA)"])

            # Retrieve settings from the returned dictionary
            questions_per_chunk_ss = settings.get('custom_qa_questions_per_chunk', 2)
            judge_provider_ss = settings.get('judge_provider')
            judge_model_ss = settings.get('judge_model')
            top_k_ss = settings.get('custom_qa_top_k', 10)
            weights_ss = settings.get('custom_qa_weights', {'bm25': 0.4, 'vector': 0.4, 'keyword': 0.2})
            retriever_type_ss = settings.get('custom_qa_retriever_type', "Vector")

            with generate_tab:
                st.write("### Generate Question-Answer Pairs")
                st.info(f"This will generate question-context pairs using the **{judge_model_ss}** model via **{judge_provider_ss}**.")
                if st.button("Generate New QA Pairs", key="gen_qa"):
                    logger.debug("Generate QA button clicked.")
                    if not judge_provider_ss or not judge_model_ss:
                         st.error("Judge LLM provider or model not selected in sidebar.")
                         logger.error("Judge LLM provider/model missing in settings.")
                    else:
                         with st.spinner(f"Initializing Judge LLM ({judge_model_ss}) and generating QA pairs..."):
                            try:
                                judge_llm = get_judge_llm_from_toml(judge_provider_ss, judge_model_ss, openai_api_key=openai_key, huggingface_api_key=huggingface_key, openrouter_api_key=openrouter_key)
                                if judge_llm is None:
                                     st.error(f"Failed to initialize Judge LLM: {judge_provider_ss}/{judge_model_ss}")
                                     logger.error("Failed to get judge LLM instance.")
                                else:
                                     logger.info(f"Calling generate_qa_pairs with judge LLM {judge_llm.metadata.model_name} and {questions_per_chunk_ss} questions per chunk.")
                                     qa_dataset = evaluator.generate_qa_pairs(
                                         judge_llm=judge_llm,
                                         num_questions_per_chunk=questions_per_chunk_ss
                                     )
                                     if qa_dataset:
                                         st.success(f"✅ Successfully generated {len(qa_dataset.queries)} QA pairs! Saved to `data/evaluation/qa_pairs.json`")
                                         st.session_state.qa_dataset = qa_dataset
                                         logger.info("QA dataset generated and saved.")
                                     else:
                                         st.error("❌ Failed to generate QA pairs. Check the logs for details.")
                                         logger.error("evaluator.generate_qa_pairs returned None.")
                            except Exception as e:
                                 st.error(f"❌ Error during QA pair generation: {e}")
                                 logger.error("Exception during QA pair generation.", exc_info=True)

            with evaluate_tab:
                st.write("### Evaluate Retrieval Performance (Custom QA)")
                qa_file = "data/evaluation/qa_pairs.json"
                qa_exists = os.path.exists(qa_file)

                if not qa_exists:
                     st.warning("No QA dataset found (`data/evaluation/qa_pairs.json`). Please generate one first in the 'Generate QA Pairs' tab.")

                if st.button("Run Evaluation (Custom QA)", key="run_eval_custom", disabled=not qa_exists):
                    logger.debug("Run Custom QA Eval button clicked.")
                    with st.spinner("Evaluating retrieval performance using custom QA pairs..."):
                        try:
                            logger.info("Loading index for Custom QA evaluation...")
                            index = storage_manager.load_index()
                            if index is None:
                                st.error("❌ Failed to load index. Please ensure it's built correctly.")
                                logger.error("Failed to load index for Custom QA eval.")
                                return

                            logger.info(f"Initializing Retriever with retriever_type={retriever_type_ss} and top_k={top_k_ss}")
                            retriever = retriever_factory.create_retriever(
                                index=index,
                                retriever_type=retriever_type_ss,
                                top_k=top_k_ss
                            )

                            if retriever is None:
                                st.error("❌ Failed to create retriever. Please check the logs for details.")
                                logger.error("Failed to create retriever for Custom QA eval.")
                                return

                            logger.info("Running evaluator.evaluate_retriever_custom_qa...")
                            results_df, metrics = asyncio.run(evaluator.evaluate_retriever_custom_qa(
                                retriever,
                                qa_dataset_path=qa_file
                            ))

                            if not results_df.empty:
                                st.session_state.eval_results = {'metrics': metrics, 'results_df': results_df}
                                st.success("✅ Evaluation complete!")
                                logger.info("Custom QA evaluation successful.")
                                display_metrics(metrics)
                                display_detailed_results(results_df)
                            else:
                                st.error("❌ Evaluation failed or produced no results. Check logs.")
                                logger.warning("Custom QA evaluation returned empty results.")
                        except Exception as e:
                            st.error(f"❌ Evaluation failed: {str(e)}")
                            logger.error("Exception during Custom QA evaluation.", exc_info=True)

            logger.debug("Rendered Custom QA UI.")

        # --- BEIR Evaluation Workflow ---
        elif eval_type == "BEIR Benchmarks":
            # Retrieve settings from the returned dictionary
            beir_embedding_model_name_ss = settings.get('beir_embedding_model_name', "BAAI/bge-small-en-v1.5")
            beir_retriever_type_ss = settings.get('beir_retriever_type', "vector")
            beir_weights_ss = settings.get('beir_weights', {'bm25': 0.4, 'vector': 0.4, 'keyword': 0.2})
            beir_datasets_ss = settings.get('beir_datasets_selected', [])
            beir_k_values_ss = settings.get('beir_k_values_list', [])

            st.write("### Evaluate using BEIR Benchmarks")
            st.info(f"This will evaluate using the **{beir_embedding_model_name_ss}** embedding model and the **{beir_retriever_type_ss}** retriever strategy.")

            if not beir_datasets_ss:
                 st.warning("Please select at least one BEIR dataset from the sidebar.")
            if not beir_k_values_ss:
                 st.warning("Please enter valid K values in the sidebar.")

            if st.button("Run Evaluation (BEIR)", key="run_eval_beir", disabled=not beir_datasets_ss or not beir_k_values_ss):
                logger.debug("Run BEIR Eval button clicked.")
                with st.spinner(f"Running BEIR evaluation for datasets: {', '.join(beir_datasets_ss)}... This might take a long time."):
                     try:
                         logger.info(f"Running evaluator.evaluate_beir with datasets={beir_datasets_ss}, k_values={beir_k_values_ss}, embed_model='{beir_embedding_model_name_ss}', retriever='{beir_retriever_type_ss}', weights={beir_weights_ss if beir_retriever_type_ss == 'hybrid' else 'N/A'}")

                         beir_results = evaluator.evaluate_beir(
                             datasets=beir_datasets_ss,
                             metrics_k_values=beir_k_values_ss,
                             embed_model_name=beir_embedding_model_name_ss,
                             retriever_type=beir_retriever_type_ss,
                             retriever_weights=beir_weights_ss if beir_retriever_type_ss == 'hybrid' else None
                         )

                         if beir_results:
                             st.session_state.beir_eval_results = beir_results
                             st.success("✅ BEIR Evaluation complete!")
                             logger.info("BEIR evaluation successful.")
                             display_beir_results(beir_results)
                         else:
                             st.error("❌ BEIR Evaluation failed or returned no results. Check logs for details.")
                             logger.warning("BEIR evaluation returned empty results.")
                     except Exception as e:
                         st.error(f"❌ BEIR Evaluation failed: {str(e)}")
                         logger.error("Exception during BEIR evaluation.", exc_info=True)

            logger.debug("Rendered BEIR UI.")
            
        elif eval_type == "Current Query Response Quality":
            st.write("You should generate a new query first based on the documents uploaded in the main page, and then evaluate the response quality of the RAG system using the generated query.")
            st.write("This will use the same settings as the main page.")
            # Retrieve necessary data from session state
            last_query = st.session_state.get("query")
            last_response = st.session_state.get("response")
            last_context_nodes = st.session_state.get("context") # This should be List[NodeWithScore] or similar

            # Get Judge LLM settings from sidebar
            judge_provider_ss = settings.get('judge_provider')
            judge_model_ss = settings.get('judge_model')

            can_evaluate = True
            if not last_query:
                st.warning("⚠️ No query found in session state. Please run a query on the main page first.")
                can_evaluate = False
            if not last_response:
                st.warning("⚠️ No response found in session state. Please run a query on the main page first.")
                can_evaluate = False
            # Context is optional for relevancy, but required for faithfulness
            if not last_context_nodes:
                st.info("ℹ️ No retrieved context found in session state (RAG might not have been used). Faithfulness check cannot be performed accurately.")
                # Allow evaluation, but expect faithfulness to fail/be NaN
                last_context_nodes = [] # Ensure it's an empty list

            if not judge_provider_ss or not judge_model_ss:
                 st.warning("⚠️ Judge LLM provider or model not selected in the sidebar.")
                 can_evaluate = False

            # Display the last interaction details
            st.markdown("---")
            st.write("**Last Interaction Details:**")
            st.text_input("Query:", value=last_query or "N/A", disabled=True)
            st.text_area("Response:", value=last_response or "N/A", height=150, disabled=True)

            # Prepare context strings needed for evaluation
            context_texts_for_eval: List[str] = []
            with st.expander("Retrieved Context Snippets Used"):
                if last_context_nodes:
                    for i, node in enumerate(last_context_nodes):
                        try:
                            # Extract text content using get_content()
                            text = node.get_content() if hasattr(node, 'get_content') else getattr(node, 'text', 'Error: Cannot get content')
                            context_texts_for_eval.append(text)
                            st.markdown(f"**Snippet {i+1} (Score: {node.score:.3f}):**" if hasattr(node, 'score') else f"**Snippet {i+1}:**")
                            st.text_area(f"snippet_{i}", value=text, height=100, disabled=True, label_visibility="collapsed")
                        except Exception as e:
                             st.error(f"Could not display/process context snippet {i+1}: {e}")
                             logger.warning(f"Error processing node {i+1} for display/eval", exc_info=True)
                else:
                    st.write("No context snippets available.")
            st.markdown("---")

            # --- Evaluation Button and Logic ---
            if st.button("Evaluate and Accumulate Response Quality", key="eval_accumulate_response", disabled=not can_evaluate):
                 logger.debug("Evaluate and Accumulate Response Quality button clicked.")
                 with st.spinner(f"Evaluating response quality using Judge LLM: {judge_provider_ss}/{judge_model_ss}..."):
                     try:
                        # 1. Initialize Judge LLM
                        logger.info(f"Initializing Judge LLM: {judge_provider_ss}/{judge_model_ss}")
                        judge_llm = get_judge_llm_from_toml(judge_provider_ss, judge_model_ss, openai_api_key=openai_key, huggingface_api_key=huggingface_key, openrouter_api_key=openrouter_key)
                        if judge_llm is None:
                            st.error(f"Failed to initialize Judge LLM: {judge_provider_ss}/{judge_model_ss}. Check API keys and configuration.")
                            return # Stop execution here

                        # 2. Call the evaluation function (already have context strings)
                        logger.info(f"Calling evaluate_response_quality with query, response, {len(context_texts_for_eval)} context strings.")
                        new_results_df, metrics = evaluator.evaluate_response_quality(
                            query=last_query,
                            response_str=last_response,
                            retrieved_contexts=context_texts_for_eval, # Pass list of strings
                            judge_llm=judge_llm
                        )

                        # 3. Accumulate results
                        if new_results_df is not None:
                            # Append the new DataFrame to the one in session state
                            eval_state.append_response_quality_result(new_results_df)
                            st.success("✅ Response Quality Evaluated and Added to History!")
                            logger.info("Response quality evaluation successful and appended.")
                            # Rerun to update the display immediately after appending
                            st.rerun()
                        else:
                             # Display error from metrics dict if available
                             error_msg = metrics.get("error", "Evaluation function returned None.") if metrics else "Evaluation function returned None."
                             st.error(f"❌ Evaluation failed: {error_msg}")
                             logger.error(f"Response quality evaluation failed. Error: {error_msg}")

                     except Exception as e:
                         st.error(f"❌ Evaluation failed with unexpected error: {str(e)}")
                         logger.error("Exception during Current Response Quality evaluation.", exc_info=True)
                         st.exception(e) # Show detailed traceback in Streamlit UI

            # --- Display Accumulated Results ---
            st.markdown("---")
            st.subheader("Cumulative Response Quality Results")
            cumulative_df = eval_state.get_response_quality_df()

            if not cumulative_df.empty:
                 # Calculate aggregate metrics from the cumulative DataFrame
                 cumulative_metrics = eval_state.calculate_cumulative_response_metrics(cumulative_df)
                 display_metrics(cumulative_metrics) # Display aggregated metrics
                 st.write("### Detailed History")
                 display_detailed_results(cumulative_df) # Display the full accumulated DataFrame
            else:
                 st.info("No response quality evaluations run in this session yet, or history has been cleared.")

            logger.debug("Rendered Current Query Response Quality UI.")
        # Display cached results if available
        if eval_type == "Custom QA Pairs" and 'eval_results' in st.session_state:
            logger.debug("Displaying cached Custom QA results.")
            st.subheader("Cached Custom QA Results")
            display_metrics(st.session_state.eval_results['metrics'])
            plots = evaluator.generate_evaluation_plots(st.session_state.eval_results['results_df'])
            display_detailed_results(st.session_state.eval_results['results_df']) # Display details too

        if eval_type == "BEIR Benchmarks" and 'beir_eval_results' in st.session_state:
            logger.debug("Displaying cached BEIR results.")
            st.subheader("Cached BEIR Results")
            display_beir_results(st.session_state.beir_eval_results)

        logger.debug("Reached end of evaluation page render.")

    except Exception as e:
        st.error(f"An error occurred on the Evaluation page: {e}")
        st.error("Check the console/terminal for a detailed traceback.")
        st.exception(e)
        logger.error("Error rendering evaluation page", exc_info=True)

# --- IMPORTANT ---
# Ensure the script execution calls the main rendering function
# If this file is run directly (it shouldn't be in multipage), it does nothing.
# Streamlit runs the file top-to-bottom when the page is selected.
render_evaluation_page()

