import os
import logging
from typing import List, Optional
from llama_index.core.evaluation import generate_question_context_pairs, EmbeddingQAFinetuneDataset
from llama_index.core.node_parser import SentenceSplitter # Or other node access methods
from llama_index.core.schema import BaseNode
from llama_index.core.llms import LLM
from src.storing.storage_manager import StorageManager

logger = logging.getLogger(__name__)

DEFAULT_SAVE_PATH = "data/evaluation/qa_pairs.json"
DEFAULT_QUESTION_TYPES = [
    "financial_metrics",
    "year_over_year_comparison",
    "segment_performance",
    "future_outlook",
    "executive_commentary"
]

class QuestionGenerator:
    """Generates question-context pairs for evaluation from an existing index."""

    def __init__(self, storage_manager: StorageManager):
        """
        Initializes the QuestionGenerator.

        Args:
            storage_manager: Instance to load the index/nodes.
        """
        self.storage_manager = storage_manager
        logger.info("QuestionGenerator initialized.")

    def _load_nodes(self) -> Optional[List[BaseNode]]:
        """Loads nodes from the persisted index."""
        logger.info("Attempting to load index to extract nodes...")
        index = self.storage_manager.load_index()
        if not index:
            logger.error("Failed to load index. Cannot generate questions without nodes.")
            return None

        try:
            # Access nodes through the docstore associated with the index
            nodes = list(index.docstore.docs.values())
            if not nodes:
                 logger.warning("Index loaded, but no nodes found in the docstore.")
                 return None
            logger.info(f"Successfully loaded {len(nodes)} nodes from index docstore.")
            return nodes
        except Exception as e:
            logger.error(f"Error accessing nodes from loaded index docstore: {e}", exc_info=True)
            return None

    def generate(self,
                 judge_llm: LLM,
                 num_questions_per_chunk: int,
                 save_path: str = DEFAULT_SAVE_PATH
                 ) -> Optional[EmbeddingQAFinetuneDataset]:
        """
        Generates question-context pairs using the specified judge LLM.

        Args:
            judge_llm: The LLM instance to use for generation.
            num_questions_per_chunk: How many questions to generate per node.
            question_types: Optional list of specific question types/categories to guide generation.
            save_path: Path to save the generated dataset JSON file.

        Returns:
            The generated dataset object, or None on failure.
        """
        nodes = self._load_nodes()
        if not nodes:
            return None # Error logged in _load_nodes

        logger.info(f"Generating {num_questions_per_chunk} questions per chunk using LLM: {judge_llm.metadata.model_name}.")

        try:
            qa_dataset = generate_question_context_pairs(
                nodes=nodes,
                llm=judge_llm,
                num_questions_per_chunk=num_questions_per_chunk
            )

            if not qa_dataset.queries:
                 logger.warning("Question generation ran but produced no QA pairs.")
                 return None

            # Ensure directory exists and save
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            qa_dataset.save_json(save_path)
            logger.info(f"Generated {len(qa_dataset.queries)} QA pairs. Saved to {save_path}")
            return qa_dataset

        except Exception as e:
            logger.error(f"Error during LlamaIndex's generate_question_context_pairs: {e}", exc_info=True)
            return None 