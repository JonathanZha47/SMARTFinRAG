import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Import specific LLM implementations you want to support as judges
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.llms.openrouter import OpenRouter as LlamaOpenRouter
# Add other imports like HuggingFaceLLM if needed
from llama_index.core.llms import LLM # Base class for type hinting

logger = logging.getLogger(__name__)
load_dotenv()

# Consider making judge LLM config separate if needed, or reuse main config for now
DEFAULT_JUDGE_TEMP = 0.0 # Judges should typically be deterministic
DEFAULT_JUDGE_MAX_TOKENS = 512
def get_judge_llm_from_env(provider_name: str,
                  model_name: str,
                  temperature: float = DEFAULT_JUDGE_TEMP,
                  max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
                  ) -> Optional[LLM]:
    """
    Gets an initialized LlamaIndex LLM instance specifically for judging/evaluation tasks.
    """
    provider_name = provider_name.lower()
    llm_instance: Optional[LLM] = None
    logger.info(f"Attempting to get judge LLM: Provider='{provider_name}', Model='{model_name}'")

    try:
        if provider_name == 'openai':   
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("Judge LLM Error: OPENAI_API_KEY not found.")
                return None
            llm_instance = LlamaOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider_name == 'openrouter':
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                logger.error("Judge LLM Error: OPENROUTER_API_KEY not found.")
                return None
            llm_instance = LlamaOpenRouter(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider_name == 'huggingface':
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not api_key:
                logger.error("Judge LLM Error: HUGGINGFACE_API_KEY not found.")
                return None 
            llm_instance = LlamaHuggingFace(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )           
        else:
            logger.error(f"Unsupported judge LLM provider: {provider_name}")
            return None

        if llm_instance:
            logger.info(f"Successfully initialized judge LLM: {model_name}")
        return llm_instance

    except Exception as e:
        logger.error(f"Error initializing judge LLM '{model_name}' from '{provider_name}': {e}", exc_info=True)
        return None         



def get_judge_llm_from_toml(provider_name: str,
                  model_name: str,
                  temperature: float = DEFAULT_JUDGE_TEMP,
                  max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
                  openai_api_key: Optional[str] = None,
                  huggingface_api_key: Optional[str] = None,
                  openrouter_api_key: Optional[str] = None
                  ) -> Optional[LLM]:
    """
    Gets an initialized LlamaIndex LLM instance specifically for judging/evaluation tasks.

    Args:
        provider_name (str): The name of the provider ('openai', 'openrouter', etc.).
        model_name (str): The specific model identifier.
        temperature (float): Temperature setting (defaults to 0.0 for judging).
        max_tokens (int): Max tokens setting.

    Returns:
        Optional[LLM]: An initialized LlamaIndex LLM instance or None on failure.
    """
    provider_name = provider_name.lower()
    llm_instance: Optional[LLM] = None
    logger.info(f"Attempting to get judge LLM: Provider='{provider_name}', Model='{model_name}'")

    try:
        if provider_name == 'openai':
            api_key = openai_api_key
            if not api_key:
                logger.error("Judge LLM Error: OPENAI_API_KEY not found.")
                return None
            llm_instance = LlamaOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider_name == 'openrouter':
            api_key = openrouter_api_key
            if not api_key:
                logger.error("Judge LLM Error: OPENROUTER_API_KEY not found.")
                return None
            llm_instance = LlamaOpenRouter(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider_name == 'huggingface':
            api_key = huggingface_api_key
            if not api_key:
                logger.error("Judge LLM Error: HUGGINGFACE_API_KEY not found.")
                return None
            llm_instance = LlamaHuggingFace(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            # Add elif for 'huggingface' or other providers if needed
        # elif provider_name == 'huggingface':
        #     # ... HuggingFace instantiation logic ...
        else:
            logger.error(f"Unsupported judge LLM provider: {provider_name}")
            return None

        if llm_instance:
            logger.info(f"Successfully initialized judge LLM: {model_name}")
        return llm_instance

    except Exception as e:
        logger.error(f"Error initializing judge LLM '{model_name}' from '{provider_name}': {e}", exc_info=True)
        return None 