import os
from pathlib import Path
from typing import List, Optional, Union
from llama_index.core import SimpleDirectoryReader, Document
from src.config.config_loader import ConfigLoader
from src.data_ingestion.enhanced_document_loader import EnhancedDocumentLoader
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loads raw documents from various file formats using EnhancedDocumentLoader."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader.get_data_config()
        self.input_dir = self.config.get('input_dir', 'data')
        self.supported_formats = self.config.get('supported_formats', ['.txt', '.pdf', '.docx'])
        
        # Initialize enhanced document loader (just for loading/parsing)
        self.enhanced_loader = EnhancedDocumentLoader()
    
    def load_single_document(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single document, returning raw Document objects."""
        try:
            file_path = Path(file_path)
            logger.info(f"Attempting to load single document: {file_path}")
            
            if not self.validate_file_format(file_path):
                logger.warning(f"Unsupported file format: {file_path.suffix}")
                return []
            
            # Use enhanced loader to get raw documents (text + tables)
            documents = self.enhanced_loader.load_documents(file_path)
            
            if not documents:
                logger.warning(f"No content was extracted from the document: {file_path.name}")
                return []
            
            logger.info(f"Successfully loaded {len(documents)} raw sections from: {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing single document {file_path}: {str(e)}", exc_info=True)
            return []
    
    def load_documents_from_directory(self, directory: Optional[str] = None) -> List[Document]:
        """Load all supported documents from a directory, returning raw Document objects."""
        try:
            target_dir = directory or self.input_dir
            logger.info(f"Attempting to load documents from directory: {target_dir}")
            
            target_path = Path(target_dir)
            if not target_path.exists():
                # Don't create it here, let the user manage the data dir
                logger.warning(f"Data directory not found: {target_dir}")
                return []
            
            if not any(target_path.iterdir()): # Check if directory is empty
                logger.warning(f"No files found in the data directory: {target_dir}")
                return []
            
            # Enhanced loader handles iterating the directory now
            documents = self.enhanced_loader.load_documents(target_path)
            
            if not documents:
                logger.warning(f"No documents were loaded from directory: {target_dir}")
                return []
            
            logger.info(f"Successfully loaded {len(documents)} raw sections from directory: {target_dir}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from directory {target_dir}: {str(e)}", exc_info=True)
            return []
    
    def validate_file_format(self, file_path: Union[str, Path]) -> bool:
        """Validate if file format is supported."""
        return Path(file_path).suffix.lower() in self.supported_formats 