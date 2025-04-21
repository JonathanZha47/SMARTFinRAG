from typing import Dict, List, Optional
from src.config.config_loader import ConfigLoader
import logging

logger = logging.getLogger(__name__)

class PromptManager:
    """Manages different prompt engineering techniques and verification."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader
        
        # Few-shot examples for financial advice
        self.few_shot_examples = [
            {
                "question": "What's the difference between a Roth IRA and Traditional IRA?",
                "answer": "A Traditional IRA offers tax-deductible contributions but taxable withdrawals in retirement, while a Roth IRA has after-tax contributions but tax-free qualified withdrawals. Traditional IRAs require minimum distributions (RMDs) starting at age 72, while Roth IRAs don't have RMDs."
            },
            {
                "question": "How should I diversify my investment portfolio?",
                "answer": "A well-diversified portfolio typically includes: 1) Different asset classes (stocks, bonds, cash), 2) Various sectors (technology, healthcare, etc.), 3) Geographic diversity (domestic and international), 4) Different market capitalizations (large-cap, mid-cap, small-cap). The specific allocation depends on your risk tolerance and time horizon."
            }
        ]
        
        # Persona definitions
        self.personas = {
            "financial_advisor": "You are a licensed financial advisor with 15 years of experience, specializing in personal finance and retirement planning. You always consider the client's best interests and adhere to fiduciary responsibilities.",
            "risk_analyst": "You are a risk management specialist with expertise in analyzing market risks and compliance requirements. You focus on identifying potential risks and ensuring regulatory compliance.",
            "investment_strategist": "You are an investment strategist with deep knowledge of market dynamics and portfolio management. You specialize in creating balanced, long-term investment strategies."
        }
        
    def create_few_shot_prompt(self, query: str) -> str:
        """Create a prompt using few-shot learning approach."""
        prompt = "Here are some examples of financial advice questions and answers:\n\n"
        
        for example in self.few_shot_examples:
            prompt += f"Question: {example['question']}\n"
            prompt += f"Answer: {example['answer']}\n\n"
            
        prompt += f"Now, please answer this question in a similar style:\n"
        prompt += f"Question: {query}\n"
        prompt += "Answer:"
        
        return prompt
        
    def create_context_prompt(self, query: str, context: Optional[List[str]] = None) -> str:
        """Create a prompt using context-based approach."""
        prompt = "Using the following context, provide a detailed and accurate answer:\n\n"
        
        if context:
            prompt += "Context:\n"
            for ctx in context:
                prompt += f"- {ctx}\n"
            
        prompt += f"\nQuestion: {query}\n"
        prompt += "Provide a comprehensive answer based on the given context. If any information is not supported by the context, explicitly state so."
        
        return prompt
        
    def create_cot_prompt(self, query: str) -> str:
        """Create a prompt using chain-of-thought reasoning."""
        prompt = "Let's approach this financial question step by step:\n\n"
        prompt += f"Question: {query}\n\n"
        prompt += "1. First, let's identify the key financial concepts involved.\n"
        prompt += "2. Then, let's analyze how these concepts relate to the question.\n"
        prompt += "3. Next, let's consider any relevant regulations or guidelines.\n"
        prompt += "4. Finally, let's formulate a comprehensive answer.\n\n"
        prompt += "Please follow this reasoning process and provide a detailed explanation:"
        
        return prompt
        
    def create_persona_prompt(self, query: str, persona: str) -> str:
        """Create a prompt using persona-based approach."""
        if persona not in self.personas:
            persona = "financial_advisor"  # default persona
            
        prompt = f"{self.personas[persona]}\n\n"
        prompt += "Given your expertise and role, please provide advice on the following question:\n"
        prompt += f"{query}\n\n"
        prompt += "Ensure your response reflects your professional perspective and expertise."
        
        return prompt

    # --- NEW: Method for RAG templates compatible with LlamaIndex ---
    def get_rag_prompt_template(self, prompt_type: str, persona: Optional[str], query: str) -> str:
        """
        Returns a LlamaIndex-compatible QA prompt template string
        with {context_str} and {query_str} placeholders, incorporating
        different prompting strategies.
        """
        logger.debug(f"Generating RAG prompt template for type: {prompt_type}, persona: {persona}")

        # --- Define Structure Placeholders ---
        # These are standard placeholders expected by LlamaIndex ResponseSynthesizer
        context_placeholder = "{context_str}"
        query_placeholder = "{query_str}" # Use the query passed for few-shot examples if needed

        # --- Base Instruction ---
        base_instruction = f"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Provide a detailed and factual response based *only* on the context provided.\n\nContext:\n---------------------\n{context_placeholder}\n---------------------\n\n"
        final_query_instruction = f"Question: {query_placeholder}\nAnswer:"

        # --- Strategy-Specific Prefixes/Suffixes ---
        persona_prefix = ""
        cot_prefix = ""
        few_shot_prefix = ""

        # 1. Persona Logic
        if prompt_type == "Persona" and persona:
            selected_persona = self.personas.get(persona)
            if selected_persona:
                 persona_prefix = f"{selected_persona}\nBased on your role and the context provided below:\n"
                 logger.debug(f"Applied persona: {persona}")
            else:
                 logger.warning(f"Persona '{persona}' not found, using default.")
                 persona_prefix = f"{self.personas.get('financial_advisor', 'You are a helpful AI assistant.')}\nBased on your role and the context provided below:\n"
        elif prompt_type == "Persona": # Persona selected but value is None/empty
             logger.warning("Persona prompt type selected but no persona provided, using default.")
             persona_prefix = f"{self.personas.get('financial_advisor', 'You are a helpful AI assistant.')}\nBased on your role and the context provided below:\n"


        # 2. Few-Shot Logic (Inject before main instruction)
        if prompt_type == "Few-Shot" and self.few_shot_examples:
             few_shot_prefix = "Here are some examples of how to answer financial questions:\n\n"
             for example in self.few_shot_examples:
                  # Note: These examples don't use the {context_str} for the current query
                  few_shot_prefix += f"Example Question: {example['question']}\nExample Answer: {example['answer']}\n\n"
             few_shot_prefix += "Now, using the context provided below, answer the user's question in a similar style:\n"
             logger.debug("Applied few-shot examples.")

        # 3. Chain-of-Thought Logic (Modify base instruction)
        if prompt_type == "Chain-of-Thought":
             cot_prefix = "Let's think step-by-step based *only* on the provided context to answer the question.\n"
             final_query_instruction = f"Question: {query_placeholder}\nStep-by-step thinking and final Answer:" # Modify final instruction
             logger.debug("Applied Chain-of-Thought structure.")

        # 4. Default Context Logic (No prefix needed)
        if prompt_type == "Context-Prompting":
             logger.debug("Using standard context prompting structure.")
             pass # No extra prefix needed, base_instruction is sufficient


        # --- Assemble the Final Template ---
        full_template = (
            persona_prefix +
            few_shot_prefix + # Examples come first if used
            cot_prefix +      # CoT instruction modifies the flow
            base_instruction +
            final_query_instruction
        )

        # --- Verification and Fallback ---
        if "{context_str}" not in full_template or "{query_str}" not in full_template:
            logger.error("CRITICAL: Constructed prompt template is missing required placeholders ({context_str}, {query_str}). Falling back to basic template.")
            full_template = f"Context:\n---------------------\n{{context_str}}\n---------------------\nQuery: {{query_str}}\nAnswer:"

        logger.debug(f"Final RAG template generated (start): {full_template[:200]}...")
        return full_template

    # --- Keep direct prompt generation logic ---
    def get_direct_prompt(self, query: str, prompt_type: str, persona: Optional[str]) -> str:
        """
        Returns a prompt string suitable for a direct LLM call (no context).
        """
        logger.debug(f"Generating direct prompt for type: {prompt_type}, persona: {persona}")
        # Reuse existing methods, ensuring they DON'T add context instructions
        if prompt_type == "Few-Shot":
            # Assuming create_few_shot_prompt is suitable for direct query
            return self.create_few_shot_prompt(query)
        elif prompt_type == "Chain-of-Thought":
             # Assuming create_cot_prompt is suitable for direct query
            return self.create_cot_prompt(query)
        elif prompt_type == "Persona":
            # Assuming create_persona_prompt is suitable for direct query
            return self.create_persona_prompt(query, persona or "financial_advisor")
        else: # Default direct prompt (Context-Prompting type doesn't make sense here)
             logger.debug("Using default direct prompt structure.")
             # Basic instruction for direct query
             default_persona = self.personas.get('financial_advisor', 'You are a helpful AI assistant.')
             return f"{default_persona}\n\nAnswer the following query accurately and concisely:\nQuery: {query}\nAnswer:"

    def verify_citations(self, response: str, context: List[str]) -> Dict:
        """Verify if the response content is supported by the provided context."""
        citations = {
            "verified": [],
            "unverified": [],
            "verification_score": 0.0
        }
        
        # Split response into statements
        statements = response.split(". ")
        
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
                
            # Check if statement is supported by context
            found = False
            for ctx in context:
                if any(sent.lower() in ctx.lower() for sent in [statement]):
                    citations["verified"].append({
                        "statement": statement,
                        "source": ctx[:100] + "..."  # First 100 chars as reference
                    })
                    found = True
                    break
                    
            if not found:
                citations["unverified"].append(statement)
                
        # Calculate verification score
        total_statements = len(statements)
        verified_statements = len(citations["verified"])
        citations["verification_score"] = verified_statements / total_statements if total_statements > 0 else 0.0
        
        return citations
        
    def detect_hallucination(self, response: str, context: List[str], citations: Dict) -> Dict:
        """Detect potential hallucinations in the response."""
        hallucination_check = {
            "potential_hallucinations": [],
            "confidence_score": 0.0,
            "warning_flags": []
        }
        
        # Check for statements without citations
        hallucination_check["potential_hallucinations"] = citations["unverified"]
        
        # Calculate confidence score based on citation verification
        hallucination_check["confidence_score"] = citations["verification_score"]
        
        # Add warning flags for specific cases
        if citations["verification_score"] < 0.7:
            hallucination_check["warning_flags"].append(
                "Low citation verification score - significant content may be unsupported"
            )
            
        if len(citations["unverified"]) > len(citations["verified"]):
            hallucination_check["warning_flags"].append(
                "More unverified statements than verified ones"
            )
            
        return hallucination_check
        
    def apply_guardrails(self, response: str, sec_guidelines: List[str]) -> Dict:
        """Check if response complies with SEC guidelines."""
        compliance_check = {
            "compliant": True,
            "violations": [],
            "warnings": []
        }
        
        # Check against each SEC guideline
        for guideline in sec_guidelines:
            # Add your specific SEC guideline checks here
            # This is a placeholder for the actual implementation
            pass
            
        return compliance_check
