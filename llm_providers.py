from typing import Optional, Dict, Any
import os
from abc import ABC, abstractmethod
import openai
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import tiktoken
import logging
from llama_index.llms.openai import OpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openrouter import OpenRouter
from transformers import BitsAndBytesConfig
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

load_dotenv()


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        try:
            if not api_key.startswith('sk-'):
                raise ValueError("Invalid API key format. OpenAI API keys should start with 'sk-'")
                
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self.max_tokens = 14000 if "gpt-4" in model else 14000  # Conservative limits
        except Exception as e:
            print(f"Error initializing OpenAI client: {str(e)}")
            raise

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            # Prepare messages with system prompt if provided
            messages = [{"role": "user", "content": prompt}]
            if kwargs.get('system_prompt'):
                messages.insert(0, {"role": "system", "content": kwargs['system_prompt']})
                
            # Create chat completion with minimal parameters to avoid errors
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_output_tokens', 2000)  # Use the max_output_tokens from kwargs
            )
            return response.choices[0].message.content
            
        except AuthenticationError as e:
            return f"OpenAI Error: Authentication failed. Please check your API key. Details: {str(e)}"
        except RateLimitError as e:
            return f"OpenAI Error: Rate limit exceeded. Please try again later. Details: {str(e)}"
        except APIConnectionError as e:
            return f"OpenAI Error: Connection failed. Please check your internet connection. Details: {str(e)}"
        except APIError as e:
            return f"OpenAI Error: API error occurred. Details: {str(e)}"
        except OpenAIError as e:
            return f"OpenAI Error: An error occurred. Details: {str(e)}"
        except Exception as e:
            return f"OpenAI Error: Unexpected error occurred. Details: {str(e)}"

class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        try:
            self.client = InferenceClient(token=api_key)
            self.model = model
            self.max_tokens = 24000  # Conservative limit for Mixtral
        except Exception as e:
            print(f"Error initializing HuggingFace client: {str(e)}")
            raise

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:

            # Format prompt for instruction-tuned models
            formatted_prompt = f"""<s>[INST] {prompt} [/INST]"""
            
            response = self.client.text_generation(
                formatted_prompt,
                model=self.model,
                temperature=kwargs.get('temperature', 0.1),
                max_new_tokens=kwargs.get('max_output_tokens', 2000),  # Use the max_output_tokens from kwargs
                repetition_penalty=1.1
            )
            return response
        except Exception as e:
            return f"HuggingFace Error: {str(e)}"

class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "mistralai/mixtral-8x7b-instruct"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        # Set max tokens based on model
        self.max_tokens = 24000 if "mixtral" in model else 8000  # Conservative limits

    def generate_response(self, prompt: str, **kwargs) -> str:
        try:

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Financial Advisor Bot",
                "Content-Type": "application/json",
                "OpenRouter-Referrer": "http://localhost:8501",
            }
            
            messages = [{"role": "user", "content": prompt}]
            if kwargs.get('system_prompt'):
                messages.insert(0, {"role": "system", "content": kwargs['system_prompt']})
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get('temperature', 0.1),
                "max_tokens": kwargs.get('max_output_tokens', 2000),  # Use the max_output_tokens from kwargs
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                response_json = response.json()
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    if 'message' in response_json['choices'][0]:
                        return response_json['choices'][0]['message']['content']
                    elif 'text' in response_json['choices'][0]:
                        return response_json['choices'][0]['text']
                return "Error: Unexpected response format from OpenRouter"
            else:
                error_message = response.json().get('error', {}).get('message', 'Unknown error')
                if "maximum context length" in error_message.lower():
                    return f"OpenRouter Error: Context too long. The response has been truncated. Details: {error_message}"
                return f"OpenRouter Error: {error_message}"
                
        except requests.exceptions.RequestException as e:
            return f"OpenRouter Error: {str(e)}"
        except Exception as e:
            return f"OpenRouter Error: {str(e)}"

# Modified get_llm_provider signature and logic (DEPRECATED)
"""


def get_llm_provider(provider_name: str, model_name: str) -> Optional[BaseLLMProvider]:
    '''
    Gets an initialized LLM provider instance for the specified provider and model.

    Args:
        provider_name (str): The name of the provider (e.g., 'openai', 'huggingface').
        model_name (str): The specific model identifier for the provider.
        temperature (float): The temperature setting for the LLM.

    Returns:
        Optional[BaseLLMProvider]: An initialized provider instance or None on failure.
    '''
    providers_config = {
        'openai': {
            'class': OpenAIProvider,
            'api_key_name': 'OPENAI_API_KEY'
        },
        'huggingface': {
            'class': HuggingFaceProvider,
            'api_key_name': 'HUGGINGFACE_API_KEY'
            # Note: HuggingFace model might be passed differently or validated against allowed models if needed
        },
        'openrouter': {
            'class': OpenRouterProvider,
            'api_key_name': 'OPENROUTER_API_KEY'
        }
    }

    if provider_name not in providers_config:
        logger.error(f"Unsupported provider name: {provider_name}")
        return None

    config = providers_config[provider_name]
    ProviderClass = config['class']
    api_key_name = config['api_key_name']
    api_key = os.getenv(api_key_name)

    if not api_key:
        logger.error(f"API key {api_key_name} not found in environment variables.")
        # Optionally show error in UI if called from Streamlit context, otherwise just log
        # st.error(f"API Key for {provider_name.capitalize()} ({api_key_name}) is missing!")
        return None # Return None to indicate failure

    try:
        logger.info(f"Initializing {provider_name} provider with model '{model_name}'")
        # Pass the specific model_name and temperature during initialization
        return ProviderClass(api_key=api_key, model=model_name)
    except Exception as e:
        logger.error(f"Error initializing {provider_name} provider with model {model_name}: {e}", exc_info=True)
        # Optionally show error in UI
        # st.error(f"Failed to initialize {provider_name.capitalize()} provider: {e}")
        return None
"""
def get_llm_provider_from_env(provider_name: str,
                              model_name: str,
                              temperature: Optional[float] = 0.1,
                              top_p: Optional[float] = 0.3,
                              max_tokens: Optional[int] = 512
                              ) -> Optional[LLM]:
    """
    Gets an initialized LlamaIndex LLM instance for the specified provider and model.
    """
    provider_name = provider_name.lower()
    llm_instance: Optional[LLM] = None
    logger.info(f"Attempting to get LLM: Provider='{provider_name}', Model='{model_name}'")
    
    try:
        if provider_name == 'openai':
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                logger.error("API key OPENAI_API_KEY not found in environment variables.")
                return None
            logger.info(f"Initializing LlamaIndex OpenAI provider with model '{model_name}'")
            llm_instance = OpenAI(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens # Corresponds to max_tokens in OpenAI API
            )
            # when use it can either llm_instance.stream() or llm_instance.complete()

        elif provider_name == 'huggingface':
            # Using HuggingFaceInferenceAPI which requires an API Token (key)
            api_key_huggingface = os.getenv('HUGGINGFACE_API_KEY')
            if not api_key_huggingface:
                logger.error("API key HUGGINGFACE_API_KEY not found in environment variables.")
                return None
            logger.info(f"Initializing LlamaIndex HuggingFace Inference API provider with model '{model_name}'")    

            # quantize to save memory
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            llm_instance = HuggingFaceLLM(
                token=api_key_huggingface, # Parameter name is 'token' for HF API Key
                quantization_config=quantization_config,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens # Parameter name is 'max_new_tokens'
            )
            # when use it will be llm_instance.complete()

        elif provider_name == 'openrouter':
            openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
            if not openrouter_api_key:
                logger.error("API key OPENROUTER_API_KEY not found in environment variables.")
                return None
            logger.info(f"Initializing LlamaIndex OpenRouter provider with model '{model_name}'")
            llm_instance = OpenRouter(
                api_key=openrouter_api_key,
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens # Corresponds to max_tokens in OpenRouter API
            )

        else:
            logger.error(f"Unsupported provider name: {provider_name}")
            return None

        # Basic check to ensure the model was initialized
        if llm_instance:
            logger.info(f"Successfully initialized LlamaIndex {provider_name} provider.")
            return llm_instance
        else:
            logger.error(f"Failed to initialize LlamaIndex {provider_name} provider for unknown reasons.")
            return None
    except Exception as e:
        logger.error(f"Error initializing LlamaIndex {provider_name} provider with model {model_name}: {e}", exc_info=True)
        return None
        

def get_llm_provider_from_toml(provider_name: str,
                     model_name: str,
                     temperature: Optional[float] = 0.1, # Optional temperature
                     top_p: Optional[float] = 0.3, # Optional top_p
                     max_tokens: Optional[int] = 512 ,    # Optional max_tokens
                     openai_api_key: Optional[str] = None,
                     huggingface_api_key: Optional[str] = None,
                     openrouter_api_key: Optional[str] = None
                     ) -> Optional[LLM]:
    """
    Gets an initialized LlamaIndex LLM instance for the specified provider and model.

    Args:
        provider_name (str): The name of the provider ('openai', 'huggingface', 'openrouter').
        model_name (str): The specific model identifier for the provider.
        temperature (float): The default temperature setting for the LLM.
        top_p (float): The default top_p setting for the LLM.
        max_tokens (int): The default maximum number of tokens to generate.

    Returns:
        Optional[LLM]: An initialized LlamaIndex LLM instance or None on failure.
    """
    provider_name = provider_name.lower() # Normalize name
    llm_instance: Optional[LLM] = None

    try:
        if provider_name == 'openai':
            openai.api_key = openai_api_key
            if not openai_api_key:
                logger.error("API key OPENAI_API_KEY not found in environment variables.")
                return None
            logger.info(f"Initializing LlamaIndex OpenAI provider with model '{model_name}'")
            llm_instance = OpenAI(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens # Corresponds to max_tokens in OpenAI API
            )
            # when use it can either llm_instance.stream() or llm_instance.complete()

        elif provider_name == 'huggingface':
            # Using HuggingFaceInferenceAPI which requires an API Token (key)
            api_key_huggingface = huggingface_api_key
            if not api_key_huggingface:
                logger.error("API key HUGGINGFACE_API_KEY not found for HuggingFace Inference API.")
                return None
            logger.info(f"Initializing LlamaIndex HuggingFace Inference API provider with model '{model_name}'")

            # quantize to save memory
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            llm_instance = HuggingFaceLLM(
                token=api_key_huggingface, # Parameter name is 'token' for HF API Key
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens # Parameter name is 'max_new_tokens'
                # Add other params like context_window if needed, e.g., context_window=3900
            )
            # when use it will be llm_instance.complete()

        elif provider_name == 'openrouter':
            api_key_openrouter = openrouter_api_key
            if not api_key_openrouter:
                logger.error("API key OPENROUTER_API_KEY not found in environment variables.")
                return None
            logger.info(f"Initializing LlamaIndex OpenRouter provider with model '{model_name}'")
            llm_instance = OpenRouter(
                api_key=api_key_openrouter,
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens # Corresponds to max_tokens in OpenRouter API
                # Add other params like context_window if needed, e.g., context_window=4096
            )

        else:
            logger.error(f"Unsupported provider name: {provider_name}")
            return None

        # Basic check to ensure the model was initialized
        if llm_instance:
             # You could add a simple test call here if desired, e.g., llm_instance.complete("test")
             # but it might incur cost/time, so maybe skip it.
             logger.info(f"Successfully initialized LlamaIndex {provider_name} provider.")
             return llm_instance
        else:
             # This case should ideally be caught by specific provider checks, but as a fallback:
             logger.error(f"Failed to initialize LlamaIndex {provider_name} provider for unknown reasons.")
             return None

    except Exception as e:
        # Log the full exception details
        logger.error(f"Error initializing LlamaIndex {provider_name} provider with model {model_name}: {e}", exc_info=True)
        return None
