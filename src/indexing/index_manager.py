from typing import List, Optional
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from src.config.config_loader import ConfigLoader

class IndexManager:
    """Manager for creating and handling document indices."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader
        self.rag_config = config_loader.get_rag_config()
        self.setup_settings()
        
    def setup_settings(self):
        """Configure global settings for LlamaIndex."""
        try:
            # Configure node parser
            Settings.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=self.rag_config.get('chunk_size', 1024),
                chunk_overlap=self.rag_config.get('chunk_overlap', 20)
            )
            
            # Configure embedding model
            embed_config = self.config.get_embedding_config()
            embed_model = OpenAIEmbedding(
                model=embed_config.get('model', "text-embedding-3-small"),
                dimensions=embed_config.get('dimension', 1536)
            )
            Settings.embed_model = embed_model
            
        except Exception as e:
            print(f"Error setting up indexing configuration: {str(e)}")
            raise
            
    def create_index(self, documents: List) -> Optional[VectorStoreIndex]:
        """Create a new index from documents."""
        try:
            if not documents:
                print("No documents provided for indexing")
                return None
                
            index = VectorStoreIndex.from_documents(documents)
            print("Successfully created index")
            return index
            
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return None 