import os
from typing import List, Tuple
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import Any, Optional
from llm_providers import get_llm_provider, BaseLLMProvider

# Load environment variables
load_dotenv()

class CustomLLMWrapper(CustomLLM):
    """Custom LLM wrapper for different providers."""
    
    # Define these as class attributes
    context_window: int = 4096
    num_output: int = 512
    provider_name: str = "openai"  # Default value
    model_name: str = "custom_llm_openai"
    provider: Optional[BaseLLMProvider] = None
    has_error: bool = False
    error_msg: str = ""
    
    def __init__(self, provider_name: str = "openai"):
        """Initialize the custom LLM wrapper."""
        super().__init__()
        # Update the class attribute with the provided value
        self.provider_name = provider_name
        self.model_name = f"custom_llm_{provider_name}"
    
        # Initialize provider
        try:
            self.provider = get_llm_provider(provider_name)
            if self.provider is None:
                print(f"Warning: Provider {provider_name} returned None")
                self.error_msg = f"Invalid provider name: {provider_name}"
                self.has_error = True
            else:
                self.has_error = False
                self.error_msg = ""
        except Exception as e:
            print(f"Error initializing provider {provider_name}: {str(e)}")
            self.error_msg = str(e)
            self.has_error = True
            self.provider = None

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Generate completion for the prompt."""
        try:
            if self.has_error or self.provider is None:
                return CompletionResponse(
                    text=f"Error: Failed to initialize LLM provider {self.provider_name}. {self.error_msg}"
                )
                
            response_text = self.provider.generate_response(prompt, **kwargs)
            return CompletionResponse(text=response_text)
        except Exception as e:
            error_msg = f"Error in CustomLLMWrapper.complete: {str(e)}"
            print(error_msg)
            return CompletionResponse(text=error_msg)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion for the prompt."""
        try:
            if self.has_error or self.provider is None:
                error_msg = f"Error: Failed to initialize LLM provider {self.provider_name}. {self.error_msg}"
                yield CompletionResponse(text=error_msg)
                return

            response = ""
            # Since we don't have streaming implemented in providers yet, 
            # we'll simulate it with the complete response
            full_response = self.provider.generate_response(prompt, **kwargs)
            for token in full_response:
                response += token
                yield CompletionResponse(text=response, delta=token)
        except Exception as e:
            error_msg = f"Error in CustomLLMWrapper.stream_complete: {str(e)}"
            print(error_msg)
            yield CompletionResponse(text=error_msg)

def init_llm(provider_name: str = "openai"):
    """Initialize the LLM with specific parameters for financial advice."""
    system_prompt = (
        "You are an AI financial advisor. Provide clear, factual financial advice based "
        "on the given context. Always be transparent about limitations and risks. "
        "If you're unsure about something, say so explicitly."
    )

    llm = CustomLLMWrapper(provider_name)
    return llm

def configure_settings(provider_name: str = "openai"):
    """Configure global settings for LlamaIndex."""
    try:
        llm = init_llm(provider_name)
        
        # Use OpenAI embeddings for vector search
        try:
            embed_model = OpenAIEmbedding()
        except Exception as e:
            print(f"Error initializing OpenAI embedding model: {str(e)}")
            raise

        # Configure global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=1024,
            chunk_overlap=20
        )
        
        return True
    except Exception as e:
        print(f"Error configuring settings: {str(e)}")
        raise

def process_documents(directory: str = "data", provider_name: str = "openai") -> bool:
    """Process documents and create/update the vector index."""
    try:
        # Check if directory exists and has files
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory {directory}, but no files found.")
            return False
        
        if not os.listdir(directory):
            print("No files found in the data directory")
            return False
            
        # Load documents from the data directory
        try:
            documents = SimpleDirectoryReader(directory).load_data()
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return False
        
        if not documents:
            print("No documents were loaded")
            return False
            
        # Configure settings
        try:
            configure_settings(provider_name)
        except Exception as e:
            print(f"Error configuring settings: {str(e)}")
            return False
        
        # Create and save index
        try:
            index = VectorStoreIndex.from_documents(documents)
            
            # Save to disk
            index.storage_context.persist("storage")
            print("Successfully created and saved index")
            return True
        except Exception as e:
            print(f"Error creating or saving index: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        return False

def load_index(provider_name: str = "openai"):
    """Load the index from disk if it exists, otherwise return None."""
    try:
        if not os.path.exists("storage"):
            return None
            
        configure_settings(provider_name)
        storage_context = StorageContext.from_defaults(persist_dir="storage")
        index = load_index_from_storage(storage_context)
        return index
    except Exception as e:
        print(f"Error loading index: {str(e)}")
        return None

def get_response(query: str, provider_name: str = "openai", use_rag: bool = False) -> Tuple[str, bool]:
    """Get response either directly from LLM or using RAG."""
    try:
        system_prompt = (
            "You are an AI financial advisor. Provide clear, factual financial advice based "
            "on the given context. Always be transparent about limitations and risks. "
            "If you're unsure about something, say so explicitly."
        )

        if use_rag:
            index = load_index(provider_name)
            if index is None:
                return "Error: No documents have been processed yet. Please upload documents first.", False
            
            query_engine = index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact"
            )
            response = query_engine.query(query)
            return str(response), True
        else:
            # Direct LLM query with financial advisor context
            llm = get_llm_provider(provider_name)
            if llm is None:
                return "Error: Failed to initialize LLM provider. Please check your API key and try again.", False

            prompt = f"Please provide financial advice about: {query}"
            response = llm.generate_response(
                prompt,
                system_prompt=system_prompt,
                temperature=0.1
            )
            
            if response.startswith("OpenAI Error") or \
               response.startswith("HuggingFace Error") or \
               response.startswith("OpenRouter Error"):
                return response, False
                
            return response, True
    except Exception as e:
        return f"Error generating response: {str(e)}", False 