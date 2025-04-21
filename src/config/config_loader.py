import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """Configuration loader for the application."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return {}
            
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for specific LLM provider."""
        return self.config.get('ui', {}).get('llm_providers', {})
        
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG-specific configuration."""
        return self.config.get('rag', {})
        
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return self.config.get('embeddings', {})
        
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        return self.config.get('storage', {})
        
    def get_data_config(self) -> Dict[str, Any]:
        """Get data ingestion configuration."""
        return self.config.get('data', {})
        
    def get_ui_config(self) -> Dict[str, Any]:
        """Get UI configuration."""
        return self.config.get('ui', {})
        
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation-specific configuration."""
        return self.config.get('evaluation', {})
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration including provider and model settings."""
        model_config = self.config.get('model', {})
        return model_config
        
    def get_query_config(self) -> Dict[str, Any]:
        """Get query processing configuration."""
        return self.config.get('query', {
            'preprocessing': {
                'enable_spell_check': True,
                'enable_grammar_check': True,
                'enable_query_expansion': True
            },
            'decomposition': {
                'max_sub_queries': 3,
                'enable_dependency_tracking': True
            },
            'retrieval': {
                'hybrid_search_weights': {
                    'bm25': 0.4,
                    'vector': 0.4,
                    'keyword': 0.2
                },
                'rerank_top_k': 10
            }
        }) 