from pathlib import Path
from typing import Optional
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, load_indices_from_storage
from src.config.config_loader import ConfigLoader
import logging

logger = logging.getLogger(__name__)

class StorageManager:
    """Manager for handling index storage and retrieval."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader.get_storage_config()
        self.persist_dir = self.config.get('persist_dir', 'storage')
        logger.info(f"StorageManager initialized. Persist directory: {self.persist_dir}")
        
    def save_index(self, index: VectorStoreIndex) -> bool:
        """Save index to disk."""
        try:
            if index is None:
                logger.warning("No index provided to save.")
                return False
                
            persist_path = Path(self.persist_dir)
            persist_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured persist directory exists: {persist_path}")
            
            # Persist the entire storage context associated with the index
            index.storage_context.persist(persist_dir=str(persist_path))
            logger.info(f"Successfully saved index to {self.persist_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}", exc_info=True)
            return False
            
    def load_index(self) -> Optional[VectorStoreIndex]:
        """Load index from disk using the full storage context."""
        try:
            persist_path = Path(self.persist_dir)
            # Check if the directory exists and contains core index files
            if not persist_path.exists() or not (persist_path / "docstore.json").exists(): 
                logger.warning(f"No storage directory: {self.persist_dir}")
                # if not exists, create a directory
                persist_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created storage directory: {self.persist_dir}")
            logger.info(f"Attempting to load index from storage: {self.persist_dir}")
            # Create a StorageContext pointing to the persistent directory
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
            
            # Load the index using the standard method for persisted contexts
            index = load_index_from_storage(storage_context=storage_context)
            
            logger.info("Successfully loaded index from storage.")
            return index
            
        except FileNotFoundError:
             logger.warning(f"Storage directory not found: {self.persist_dir}")
             return None
        except Exception as e:
             # Catch other potential errors during storage context init or index loading
             logger.error(f"Error loading index from storage: {str(e)}", exc_info=True)
             # Log specific LlamaIndex errors if possible
             if "No index" in str(e) and "storage context" in str(e):
                  logger.warning("load_index_from_storage confirms no index metadata found.")
             return None 