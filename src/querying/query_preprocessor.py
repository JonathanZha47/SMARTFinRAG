from typing import Dict, Tuple
import spacy
from spellchecker import SpellChecker
from src.config.config_loader import ConfigLoader
from llama_index.core import Settings
import re
from llm_providers import get_llm_provider_from_toml
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)
load_dotenv()

class QueryPreprocessor:
    """Preprocesses queries for enhanced understanding and retrieval."""
    
    def __init__(self, config_loader: ConfigLoader):
        """Initialize the query preprocessor with necessary tools."""
        self.config = config_loader
        self.spell_checker = SpellChecker()
        self.llm = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        if not self.llm:
            logger.warning("QueryPreprocessor initialized without an LLM provider. LLM-based features will be disabled.")
        # Load English language model for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            # If model not found, download it
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Financial domain-specific terms to preserve
        self.domain_terms = {
            "401k", "ira", "roth", "etf", "nasdaq", "forex", "cryptocurrency",
            "bitcoin", "ethereum", "blockchain", "fintech", "robo-advisor",
            "aapl", "apple", "msft", "microsoft", "amzn", "amazon", "googl",
            "google", "meta", "tesla", "tsla"
        }
        
    def enhance_query(self, query: str) -> str:
        """Enhance query using instruction-based LLM."""
        if not self.llm:
            logger.warning("LLM enhancement skipped: LLM not available.")
            return query
            
        enhancement_prompt = f"""As a financial query enhancement system, improve the following query while maintaining its core intent.
        
        Original query: "{query}"

        Instructions:
        1. Fix any grammatical errors
        2. Improve clarity and specificity
        3. Preserve financial terms and company names/symbols
        4. Structure as a clear request or question
        5. Add relevant financial context if needed

        Enhanced query:"""
                
        try:
            # Format the prompt correctly for the chat completions API
            messages = [{"role": "user", "content": enhancement_prompt}]
            response = self.llm.chat.completions.create(
                        model="deepseek/deepseek-chat-v3-0324", # Correct model name as per previous request
                        messages=messages # Pass the formatted messages list
                        )
            enhanced_query = response.choices[0].message.content
            print("query after enhancement", enhanced_query)
            return enhanced_query if enhanced_query else query
        except:
            return query
            
    def analyze_query_intent(self, query: str) -> Dict:
        """Analyze query intent using instruction-based LLM.
        Returns a dictionary with the following keys:
        - is_question: bool
        - requires_calculation: bool
        - is_comparison: bool
        - time_sensitive: bool
        - companies_mentioned: list
        - requires_retrieval: bool
        - llm_analysis: str
        """
        logger.info("Analyzing query intent using LLM.")
        if not self.llm:
            return self._basic_intent_analysis(query)
            
        analysis_prompt = f"""Analyze the following financial query and provide structured information about its intent and requirements.

        Query: "{query}"

        Instructions:
        1. Identify the main type of request (e.g., analysis, comparison, explanation, calculation)
        2. Detect any specific financial entities or terms
        3. Determine if it requires retrieval of specific information
        4. Assess if it needs numerical calculations
        5. Check if it involves time-sensitive information

        Provide analysis in a clear, structured format."""
        
        try:
             # Format the prompt correctly for the chat completions API
            messages = [{"role": "user", "content": analysis_prompt}]
            response = self.llm.chat.completions.create(
                        model="deepseek/deepseek-chat-v3-0324", # Correct model name
                        messages=messages # Pass the formatted messages list
                        )
            llm_analysis = response.choices[0].message.content.strip()
            
            # Combine LLM analysis with basic NLP analysis
            basic_analysis = self._basic_intent_analysis(query)
            basic_analysis['llm_analysis'] = llm_analysis
            return basic_analysis
        except:
            return self._basic_intent_analysis(query)
            
    def _basic_intent_analysis(self, query: str) -> Dict:
        """Perform basic intent analysis using spaCy."""
        logger.info("Performing basic intent analysis using spaCy.")
        doc = self.nlp(query)
        
        return {
            'is_question': any(token.tag_ in ['WDT', 'WP', 'WRB'] for token in doc),
            'requires_calculation': any(token.like_num or token.text == '$' for token in doc),
            'is_comparison': any(token.text.lower() in ['versus', 'vs', 'compare', 'difference'] for token in doc),
            'time_sensitive': any(token.ent_type_ in ['DATE', 'TIME'] for token in doc),
            'companies_mentioned': [ent.text for ent in doc.ents if ent.label_ == 'ORG'],
            'requires_retrieval': self._check_if_needs_retrieval(query)[0],
            'reason': self._check_if_needs_retrieval(query)[1]
        }
        
    def _check_if_needs_retrieval(self, query: str) -> bool:
        """Determine if the query requires document retrieval using LLM."""
        # Construct a prompt for the LLM to analyze the query
        analysis_prompt = {
            "query": query,
            "task": "Analyze if this query requires retrieving information from documents.",
            "considerations": [
                "Does it ask for specific facts or data?",
                "Does it require domain knowledge?",
                "Does it reference specific entities or events?"
            ],
            "format": "Respond with a JSON object containing: {needs_retrieval: boolean, reason: string}"
        }
        
        # Format the prompt correctly for the chat completions API
        analysis_prompt_str = json.dumps(analysis_prompt)
        messages = [{"role": "user", "content": analysis_prompt_str}]

        try:
            # Use OpenAI client syntax for consistent results
            response = self.llm.chat.completions.create(
                        model="deepseek/deepseek-chat-v3-0324", # Correct model name
                        messages=messages
                        )
            result_str = response.choices[0].message.content
            # Clean potential markdown fences from the response
            if result_str.startswith("```json"):
                result_str = result_str[7:] # Remove ```json\n
            if result_str.startswith("```"): # Handle case where only ``` is present
                 result_str = result_str[3:]
            if result_str.endswith("```"):
                result_str = result_str[:-3] # Remove trailing ```
            result_str = result_str.strip() # Strip again after removing fences

            # Attempt to parse the potentially cleaned JSON response
            try:
                result = json.loads(result_str)
                # Ensure expected keys exist, provide defaults if not
                needs_retrieval = result.get('needs_retrieval', False)
                reason = result.get('reason', 'LLM analysis successful, reason not specified.')
                # Basic validation - check if needs_retrieval is actually a boolean
                if not isinstance(needs_retrieval, bool):
                    logger.warning(f"LLM returned non-boolean for needs_retrieval: {needs_retrieval}. Defaulting to False.")
                    needs_retrieval = False
                    reason = f"LLM analysis returned invalid format: {result_str}"
                return needs_retrieval, reason
            except json.JSONDecodeError:
                logger.warning(f"LLM response was not valid JSON: {result_str}. Falling back to basic heuristics.")
                # Fallback if JSON parsing fails
                doc = self.nlp(query)
                fallback_needs_retrieval = any(token.text.lower() in ['how', 'what', 'why', 'explain', 'analyze', 'describe'] for token in doc)
                return fallback_needs_retrieval, f"Fallback due to invalid JSON response from LLM: {result_str}"
            
        except Exception as e:
            logger.error(f"Error in LLM retrieval analysis: {str(e)}") # Use logger.error
            # Fallback to basic heuristics if LLM fails
            doc = self.nlp(query)
            fallback_needs_retrieval = any(token.text.lower() in ['how', 'what', 'why', 'explain', 'analyze', 'describe'] 
                      for token in doc)
            # Return tuple in fallback case as well
            return fallback_needs_retrieval, f"Fallback due to LLM API error: {str(e)}"
        
    def preprocess_query(self, query: str) -> Tuple[str, Dict]:
        """Complete query preprocessing pipeline."""
        # Enhance query using instruction-based LLM
        logger.info("Enhancing query using instruction-based LLM.")
        enhanced_query = self.enhance_query(query)
        logger.info("Enhanced query successfully")
        # Analyze query intent
        analysis = self.analyze_query_intent(enhanced_query)
        
        # Prepare complete analysis
        analysis.update({
            'original_query': query,
            'enhanced_query': enhanced_query
        })
        
        return enhanced_query, analysis