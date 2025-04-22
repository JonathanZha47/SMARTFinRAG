import streamlit as st
import os
import logging
import tempfile
from src.config.config_loader import ConfigLoader
from src.data_ingestion.document_loader import DocumentLoader
from src.indexing.index_manager import IndexManager
from src.storing.storage_manager import StorageManager
from src.querying.query_engine import QueryEngine
from src.prompting.prompt_manager import PromptManager
from typing import Dict, List

# Added LlamaIndex core imports for pipeline
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
from llm_providers import get_llm_provider_from_toml
from llama_index.core.query_engine import RetrieverQueryEngine
# --- Configure Logging --- 
# Set root logger level - Note: Streamlit might interfere slightly, 
# but this sets the baseline for non-Streamlit handlers and our own loggers.
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Get a logger for this specific module (app.py)
logger = logging.getLogger(__name__)

# Silence overly verbose library loggers by setting their level higher
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING) # httpx uses httpcore
# You might want to adjust llama_index level too if it becomes noisy
# logging.getLogger("llama_index").setLevel(logging.WARNING) 

logger.info("Logging configured. OpenAI and HTTPx loggers set to WARNING level.")
# --- End Logging Config ---

# --- Load API Keys from st.secrets ---
# logger.info("Environment variables loaded.") # No longer loading .env

# Use st.secrets (which reads .streamlit/secrets.toml locally or secrets from deployment)
# Use .get() for robustness in case a key is missing in the file/secrets manager
openai_key = st.secrets.get("OPENAI_API_KEY")
huggingface_key = st.secrets.get("HUGGINGFACE_API_KEY")
openrouter_key = st.secrets.get("OPENROUTER_API_KEY")

st.session_state["openai_key"] = openai_key
st.session_state["huggingface_key"] = huggingface_key
st.session_state["openrouter_key"] = openrouter_key

# Check specifically if OpenAI key exists as it's needed for embeddings by default
if not openai_key:
    # Error message in the app UI
    st.error("OpenAI API Key (OPENAI_API_KEY) is missing in Streamlit secrets. Required for embeddings.")
    # Log the error as well
    logger.error("OpenAI API Key not found in Streamlit secrets!")
    st.stop() # Stop the app if the critical key is missing

# Use logger for API key status messages (showing partial keys is less critical now)
if openai_key:
    logger.info("✅ OpenAI API Key Loaded from Streamlit secrets.")
    # Set the key for OpenAI libraries *if they don't automatically pick it up*
    # LlamaIndex OpenAIEmbedding usually finds it, but good practice for other libs:
    os.environ["OPENAI_API_KEY"] = openai_key
else:
    # This case is handled by st.stop() above, but keep for consistency
    logger.warning("⚠️ OpenAI API Key is missing!")

if huggingface_key:
    logger.info("✅ HuggingFace API Key Loaded from Streamlit secrets.")
    os.environ["HUGGINGFACE_API_KEY"] = huggingface_key # Set for libs that use env var
else:
    logger.warning("⚠️ HuggingFace API Key is missing in Streamlit secrets.")

if openrouter_key:
    logger.info("✅ OpenRouter API Key Loaded from Streamlit secrets.")
    os.environ["OPENROUTER_API_KEY"] = openrouter_key # Set for libs that use env var
else:
    logger.warning("⚠️ OpenRouter API Key is missing in Streamlit secrets.")

# --- End API Key Loading ---

st.set_page_config(
    page_title="Financial Advisor Bot",
    page_icon="💰",
    layout="wide"
)

# --- LlamaIndex Global Settings --- (Optional but recommended)
# Set embedding model globally if desired (requires API key to be available)
Settings.embed_model = OpenAIEmbedding()
logger.info("LlamaIndex Settings.embed_model configured globally.")
# Set LLM globally if used by extractors (like TitleExtractor)
# Settings.llm = get_llm_provider(provider_name, model_name) # Requires provider/model selection
# --- End Global Settings ---

def save_uploaded_file(uploaded_file):
    """Saves uploaded file to the data directory."""
    try:
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")
        
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Successfully saved uploaded file to: {file_path}")
        return True, file_path
    except Exception as e:
        logger.error(f"Error saving uploaded file {uploaded_file.name}: {e}", exc_info=True)
        return False, str(e)

def sidebar_settings(ui_config: dict):
    """Create sidebar with parameter settings."""
    st.sidebar.title("Settings")
    st.sidebar.subheader("LLM Settings")
    # LLM Provider Selection
    provider_name = st.sidebar.selectbox(
        "Select LLM Provider",
        options=["openai", "huggingface", "openrouter"],
        index=0
    )
    
    # Model Selection based on provider
    available_models = ui_config['llm_providers'][provider_name]
    model = st.sidebar.selectbox(
        "Select Model",
        options=available_models,
        index=0
    )

    # Top P Settings(just directly input not from configLoader)
    top_p = st.sidebar.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01
    )
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=ui_config['temperature_range'][0],
        max_value=ui_config['temperature_range'][1],
        value=0.1,
        step=ui_config['temperature_range'][2]
    )
    
    # Get output token range for selected model
    output_token_range = ui_config.get('output_tokens_range', {}).get(provider_name, {}).get(model, [1, 2048, 50])
    # Dynamic max output tokens slider based on model
    max_output_tokens = st.sidebar.slider(
        "Max Output Tokens",
        min_value=output_token_range[0],
        max_value=output_token_range[1],
        value=min(1000, output_token_range[1]),  # Default to 1000 or max allowed
        step=output_token_range[2],
        help="Maximum number of tokens to generate in the response. Adjusts based on model capabilities."
    )
    # Prompt Engineering Settings
    st.sidebar.subheader("Prompt Engineering")
    prompt_type = st.sidebar.selectbox(
        "Prompt Engineering Technique",
        options=["Few-Shot", "Context-Prompting", "Chain-of-Thought", "Persona"],
        index=0
    )
    
    # Persona selection if persona prompting is chosen
    persona = None
    if prompt_type == "Persona":
        persona = st.sidebar.selectbox(
            "Select Persona",
            options=["Financial Advisor", "Risk Analyst", "Investment Strategist"],
            index=0
        )
    
    
    # RAG Parameters
    st.sidebar.subheader("RAG Parameters")
    st.sidebar.caption("Ingestion Settings")
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=ui_config['chunk_size_range'][0],
        max_value=ui_config['chunk_size_range'][1],
        value=1024,
        step=ui_config['chunk_size_range'][2]
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        min_value = 25,
        max_value = 125,
        value = 50,
        step = 5
    )
    st.sidebar.caption("Retriever Settings")
    top_k = st.sidebar.slider(
        "Retrieval Top K",
        min_value=ui_config['top_k_range'][0],
        max_value=ui_config['top_k_range'][1],
        value=3,
        step=ui_config['top_k_range'][2]
    )
    
    # Retriever Selection
    retriever_type = st.sidebar.selectbox(
        "Select Retriever Type",
        options=["Vector", "BM25", "Hybrid Fusion", "Auto-Merging"], # Add more if implemented
        index=0, # Default to Vector
        help="Choose the method for retrieving relevant documents."
    )
    # Verification Settings
    st.sidebar.subheader("Response Verification")
    enable_citation_check = st.sidebar.checkbox("Enable Citation Verification", value=True, help="Need to be implemented in the future")
    enable_hallucination_check = st.sidebar.checkbox("Enable Hallucination Detection", value=True, help="Need to be implemented in the future")
    enable_guardrails = st.sidebar.checkbox("Enable SEC Compliance Check", value=True, help="Need to be implemented in the future")
    
    return {
        "provider_name": provider_name,
        "model": model,
        "prompt_type": prompt_type,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "persona": persona,
        "enable_citation_check": enable_citation_check,
        "enable_hallucination_check": enable_hallucination_check,
        "enable_guardrails": enable_guardrails,
        "top_k": top_k,
        "retriever_type": retriever_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

def display_query_analysis(analysis: Dict):
    """Display query analysis results in the main content area."""
    st.write("### Query Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display intent information
        st.write("**Query Intent:**")
        # Display core intents directly from analysis
        intent_fields = {
            'is_question': 'Question',
            'requires_calculation': 'Calculation Required',
            'is_comparison': 'Comparison',
            'time_sensitive': 'Time Sensitive'
        }
        
        for field, display_name in intent_fields.items():
            value = analysis.get(field, False)
            st.write(f"- {display_name}: {'✅' if value else '❌'}")
        
        # Display retrieval recommendation
        st.write("\n**Retrieval Strategy:**")
        # Use rag_status for more accurate info on what happened
        rag_status = analysis.get('rag_status', 'N/A')
        if "Used RAG" in rag_status:
            st.write("✅ RAG Used")
        elif "RAG Attempted" in rag_status or "RAG Needed" in rag_status:
             st.write("⚠️ RAG Recommended/Attempted (Fallback to Direct)")
        else:
            st.write("❌ Direct Query Used")
        st.info(f"Status: {rag_status}")
    
    with col2:
        # Display companies mentioned
        companies = analysis.get('companies_mentioned', [])
        if companies:
            st.write("**Companies Mentioned:**")
            for company in companies:
                st.write(f"- {company}")
        
        # Display decomposition if available
        decomposition = analysis.get('decomposition')
        if decomposition and isinstance(decomposition, dict):
            st.write("**Query Decomposition:**")
            sub_problems = decomposition.get('sub_problems', [])
            if sub_problems:
                 for i, sp in enumerate(sub_problems):
                      st.write(f"- Sub-problem {i+1}: {sp.get('query', 'N/A')}")
            elif decomposition.get('error'):
                 st.warning(f"Decomposition failed: {decomposition['error']}")
            else:
                 st.write("- No sub-problems identified.")
    
    # Display LLM analysis if available
    llm_analysis = analysis.get('llm_analysis')
    if llm_analysis:
        st.write("\n**Detailed Analysis:**")
        st.info(llm_analysis)
    
    st.markdown("---")  # Add a separator line

def display_verification_results(citations: Dict, hallucination_check: Dict, compliance_check: Dict):
    """Display verification results in an organized manner."""
    st.write("### Response Verification")
    
    if citations:
        with st.expander("Citation Verification", expanded=True):
            score = citations.get('verification_score', 0.0)
            st.write(f"**Verification Score:** {score:.2%}")
            
            st.write("**Verified Statements:**")
            for citation in citations.get('verified', []):
                st.markdown(f"✅ {citation.get('statement', 'N/A')}")
                st.markdown(f"*Source: {citation.get('source', 'N/A')}*", unsafe_allow_html=True)
            
            if citations.get('unverified'):
                st.write("**Unverified Statements:**")
                for statement in citations.get('unverified', []):
                    st.markdown(f"❌ {statement}")
    
    if hallucination_check:
        with st.expander("Hallucination Detection", expanded=True):
            score = hallucination_check.get('confidence_score', 0.0)
            st.write(f"**Confidence Score:** {score:.2%}")
            
            if hallucination_check.get('warning_flags'):
                st.write("**Warning Flags:**")
                for flag in hallucination_check['warning_flags']:
                    st.warning(flag)
    
    if compliance_check:
        with st.expander("SEC Compliance Check", expanded=True):
            if compliance_check.get('compliant', False):
                st.success("✅ Response complies with SEC guidelines")
            else:
                st.error("❌ Response has compliance issues")
            
            if compliance_check.get('violations'):
                st.write("**Violations:**")
                for violation in compliance_check['violations']:
                    st.error(violation)
                
            if compliance_check.get('warnings'):
                st.write("**Warnings:**")
                for warning in compliance_check['warnings']:
                    st.warning(warning)
    
    st.markdown("---")

def main():
    st.title("💰 AI Financial Advisor")
    st.markdown("""
    Welcome to the AI Financial Advisor! Ask questions about financial planning, investments, or get general financial advice.
    Optionally, upload documents to get more context-aware responses.
    """)
    # Initialize configuration
    config_loader = ConfigLoader()
    # get UI config
    # including temperature, top_k, chunk_size, max_tokens, output_tokens, and llm_providers
    ui_config = config_loader.get_ui_config()
    # Get settings from sidebar
    settings = sidebar_settings(ui_config)
    # get LLM
    llm = get_llm_provider_from_toml(
        settings["provider_name"],
        settings["model"],
        settings["temperature"],
        settings["top_p"],
        settings["max_output_tokens"],
        openai_api_key=openai_key,
        huggingface_api_key=huggingface_key,
        openrouter_api_key=openrouter_key
    )
    if llm is None:
        st.error(f"Failed to initialize LLM provider '{settings['provider_name']}'. Check API key and configuration.")
        st.stop()

    # Initialize core components (consider making QueryEngine singleton if needed)
    document_loader = DocumentLoader(config_loader)
    storage_manager = StorageManager(config_loader)
    query_engine = QueryEngine(llm, config_loader, openrouter_key) # Loads index on init now
    prompt_manager = PromptManager(config_loader)
    
    # use settings dictionary in evaluation page
    st.session_state.settings = settings
    st.session_state.llm = llm
    st.session_state.config_loader = config_loader
    st.session_state.storage_manager = storage_manager
  

    # --- Ingestion Pipeline Setup --- 
    # Define transformations based on settings or config
    # Using OpenAIEmbedding, assuming OPENAI_API_KEY is set
    transformations = [
        SentenceSplitter(chunk_size=settings['chunk_size'], chunk_overlap=settings['chunk_overlap']),
        Settings.embed_model # Use the globally set embedding model
        # Add TitleExtractor back if needed and Settings.llm is configured
        # TitleExtractor(llm=Settings.llm), 
    ]
    
    # Create the pipeline *without* a vector_store for transformations only
    pipeline = IngestionPipeline(transformations=transformations)
    logger.info("Ingestion pipeline (transformations only) initialized.")
    # --- End Pipeline Setup --- 

    # File upload section
    st.subheader("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your financial documents",
        accept_multiple_files=True,
        type=['txt', '.pdf', '.docx'] # Make sure this matches DocumentLoader
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            saved_file_paths = []
            with st.spinner("Saving uploaded files..."):
                for uploaded_file in uploaded_files:
                    success, saved_path = save_uploaded_file(uploaded_file)
                    if success:
                        saved_file_paths.append(saved_path)
                    else:
                        st.error(f"Failed to save file: {uploaded_file.name} - {saved_path}")
            
            if saved_file_paths:
                all_raw_docs: List[Document] = []
                with st.spinner("Loading document content..."):
                    for file_path in saved_file_paths:
                        # Load raw documents one by one
                        raw_docs = document_loader.load_single_document(file_path)
                        if raw_docs:
                            all_raw_docs.extend(raw_docs)
                        else:
                             st.warning(f"Could not load content from {os.path.basename(file_path)}")
                
                if all_raw_docs:
                    with st.spinner(f"Running ingestion transformations for {len(all_raw_docs)} document sections..."):
                        try:
                            nodes = pipeline.run(documents=all_raw_docs)
                            logger.info(f"Pipeline transformations completed, generated {len(nodes)} nodes.")
                            
                            if not nodes:
                                 st.error("Pipeline ran but generated no nodes. Check transformations (e.g., chunk size) and input documents.")
                                 return # Stop processing if no nodes
                        
                            # 5. Create Index from Nodes using the prepared Storage Context
                            logger.info("Creating VectorStoreIndex from generated nodes with the prepared storage context...")
                            index = VectorStoreIndex(nodes=nodes, show_progress=True)
                            logger.info(f"Index created and associated with storage context for dir: {storage_manager.persist_dir}")
                            
                            # 6. Persist Index (using StorageManager)
                            logger.info(f"Persisting index to {storage_manager.persist_dir}...")
                            # Use the storage_manager's save function which handles directory creation etc.
                            save_successful = storage_manager.save_index(index) 
                            # index.storage_context.persist(persist_dir=storage_manager.persist_dir) # Old direct way
                            
                            if save_successful:
                                # 7. Update Query Engine with the NEW index
                                logger.info("Updating Query Engine with the newly created index.")
                                query_engine.add_nodes_to_index(nodes)
                                st.success(f"Successfully processed, indexed, and persisted {len(saved_file_paths)} document(s)!")
                            else:
                                 st.error("Index created, but failed to persist it to disk.")
                                
                        except Exception as pipe_err:
                            logger.error(f"Ingestion pipeline/indexing failed: {pipe_err}", exc_info=True)
                            st.error(f"Error during document processing or indexing: {pipe_err}")
                else:
                    st.error("No content could be extracted from the uploaded files.")
            else:
                st.error("No files were successfully saved for processing.")

    # Query section
    st.subheader("Ask Questions")
    query = st.text_input("Enter your financial question:")
    # Store the query in the session state
    st.session_state.query = query

    force_rag = st.checkbox("Force RAG Mode", value=False, 
                           help="Force using document retrieval regardless of query analysis")
    
    if query:
        with st.spinner("Processing query..."):


            # Process query - Pass None for index, QE uses its internal one
            response, success, analysis, context = query_engine.process_query(
                query=query,
                index=storage_manager.load_index(), 
                provider_name=settings["provider_name"],
                model_name=settings["model"],
                retriever_type=settings["retriever_type"],
                force_rag=force_rag,
                prompt_type=settings["prompt_type"],
                top_k=settings["top_k"],
                persona=settings["persona"]
            )
            st.session_state.response = response
            st.session_state.context = context
            
            results_container = st.container()
            with results_container:
                display_query_analysis(analysis)
                st.write("### Response")
                if success:
                    st.write(response)
                else:
                    st.error(response)
                
                # Determine if RAG was used based on analysis status
                rag_status = analysis.get('rag_status', '')
                was_rag_used = "Used RAG" in rag_status 
                
                if was_rag_used and context:
                    with st.expander("🔍 Retrieved Context Snippets"):
                        st.caption(f"Top {len(context)} snippets retrieved based on your query and settings:")
                        for i, node_with_score in enumerate(context):
                            try:
                                # Access node and score
                                node = node_with_score # Assuming context items are NodeWithScore
                                score = node_with_score.score if hasattr(node_with_score, 'score') else None
                                text = node.get_content() if hasattr(node, 'get_content') else getattr(node, 'text', 'Error: Cannot get content')

                                # Display score if available
                                score_display = f"(Score: {score:.3f})" if score is not None else "(Score: N/A)"
                                st.markdown(f"**Snippet {i+1} {score_display}:**")

                                # Display text in a text area
                                st.text_area(f"Context Snippet {i+1}", value=text, height=100, disabled=True, label_visibility="collapsed")
                                st.markdown("---") # Separator

                            except Exception as e:
                                 st.error(f"Error displaying context snippet {i+1}: {e}")
                                 logger.warning(f"Error processing node {i+1} for display", exc_info=True)

                elif was_rag_used and not context:
                     with st.expander("🔍 Retrieved Context Snippets"):
                          st.info("RAG was attempted, but no context snippets were retrieved or returned.")

                # TODO: Implement verification logic
                # # Only show verification if RAG was successfully used
                # if was_rag_used and (settings["enable_citation_check"] or settings["enable_hallucination_check"] or settings["enable_guardrails"]):
                #     # We need the actual context used by the LLM for verification.
                #     # This might require modification in PromptManager or QueryEngine
                #     # to return the context along with the response.
                #     # For now, we'll simulate with an empty context or retrieved nodes if available.
                #
                #     # Placeholder: Get context if QueryEngine provides it in analysis
                #     llm_context = analysis.get("llm_context", "")
                #
                #     citations, hallucination_check, compliance_check = None, None, None
                #     if llm_context:
                #         if settings["enable_citation_check"]:
                #             citations = prompt_manager.verify_citations(response, llm_context)
                #         if settings["enable_hallucination_check"]:
                #             hallucination_check = prompt_manager.detect_hallucination(response, llm_context, citations)
                #         if settings["enable_guardrails"]:
                #             compliance_check = prompt_manager.apply_guardrails(response, []) # Guardrails might not need context
                #     else:
                #          st.warning("Verification requires context used by LLM, which is not currently passed back. Skipping verification steps.")
                #
                #     # Display verification results if checks were run
                #     if citations or hallucination_check or compliance_check:
                #          display_verification_results(citations or {}, hallucination_check or {}, compliance_check or {})
                    
                # Show processing details
                with st.expander("Query Processing Details"):
                    st.write("**Query Processing:**")
                    st.write("Original Query:", analysis.get('original_query', query))
                    st.write("Enhanced Query:", analysis.get('enhanced_query', query))
                    if 'decomposition' in analysis and isinstance(analysis['decomposition'], dict) and 'sub_problems' in analysis['decomposition']:
                         st.write("Sub-problems:", [sp.get('query') for sp in analysis['decomposition']['sub_problems']])
                    
                    st.write("\n**Generation Settings:**")
                    st.write("RAG Status:", rag_status)
                    st.write("Retriever Type:", settings["retriever_type"])
                    st.write("Model:", settings["model"])
                    st.write("Temperature:", settings["temperature"])
                    st.write("Prompt Type:", settings["prompt_type"])
                    if settings["persona"]:
                        st.write("Persona:", settings["persona"])

if __name__ == "__main__":
    main() 