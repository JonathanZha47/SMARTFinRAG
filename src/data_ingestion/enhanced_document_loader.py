from typing import List, Optional, Union
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file import PyMuPDFReader
# SentenceSplitter is removed as it will be part of the IngestionPipeline
# from llama_index.core.node_parser import SentenceSplitter 
import pdfplumber
from pathlib import Path
import os
import logging
import pandas as pd
import json
import pymupdf 

# Configure logging - Use standard logging, assume configuration happens in main app
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger for this module

def format_table_raw_string(table: List[List[Optional[str]]]) -> Optional[str]:
    """Converts a list-of-lists table into a simple multi-line string representation."""
    if not table:
        return None
    
    output_lines = []
    for row in table:
        if row: # Skip potentially completely empty rows
            # Convert cells to string, handle None, strip whitespace
            cleaned_cells = [(str(cell).strip() if cell is not None else "") for cell in row]
            output_lines.append(" | ".join(cleaned_cells)) # Join cells with a pipe delimiter
        else:
            output_lines.append("") # Keep empty rows if present in original table data
            
    if not output_lines:
        return None
        
    return "\n".join(output_lines) # Join rows with newline

class EnhancedDocumentLoader:
    """Loads documents and extracts text and tables, preparing them for an ingestion pipeline."""
    
    # Removed SentenceSplitter initialization
    def __init__(self):
        pass # No specific initialization needed here now

    
    def load_documents(self, input_path: Union[str, Path]) -> List[Document]:
        """Load and process documents from a path, returning raw Document objects (unsplit)."""
        documents = []
        input_path = Path(input_path)
        
        try:
            if input_path.is_file():
                if input_path.suffix.lower() in ['.pdf', '.txt', '.docx']:
                    logger.info(f"Processing single file: {input_path.name}")
                    docs = self._process_file(input_path)
                    if docs:
                        documents.extend(docs)
                else:
                     logger.warning(f"Skipping unsupported file format: {input_path.suffix}")
            else:
                logger.info(f"Processing directory: {input_path}")
                for file_path in input_path.glob("**/*"):
                    if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.docx']:
                        logger.info(f"Processing file: {file_path.name}")
                        docs = self._process_file(file_path)
                        if docs:
                            documents.extend(docs)
                    elif file_path.is_file():
                         logger.warning(f"Skipping unsupported file format: {file_path.suffix}")
            
            if not documents:
                logger.warning("No documents were successfully processed or found.")
                return []
            
            logger.info(f"Successfully extracted {len(documents)} raw document sections.")
            # Removed splitting - pipeline will handle it
            # split_docs = self._split_documents(documents)
            # logging.info(f"Split documents into {len(split_docs)} chunks.")
            return documents # Return the unsplit documents
            
        except Exception as e:
            logger.error(f"Error in load_documents for {input_path}: {str(e)}", exc_info=True)
            return []
    
    def _process_file(self, file_path: Path) -> List[Document]:
        """Process a single file based on its type."""
        try:
            if not file_path.exists():
                logger.error(f"File not found during processing: {file_path}")
                return []
                
            if file_path.suffix.lower() == '.pdf':
                return self._process_pdf(file_path)
            else:
                try:
                    reader = SimpleDirectoryReader(input_files=[str(file_path)], file_metadata=lambda filename: {"source": filename})
                    docs = reader.load_data()
                    logger.info(f"Extracted {len(docs)} sections from {file_path.name}")
                    if docs and docs[0].text.strip():
                         logger.debug(f"First 100 chars from {file_path.name}: {docs[0].text[:100]}...")
                    return docs if docs else []
                except Exception as e:
                    logger.error(f"Error reading non-PDF file {file_path}: {str(e)}", exc_info=True)
                    return []
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            return []

    # --- Refactored PDF Processing --- 
    def _process_pdf(self, pdf_path: Path) -> List[Document]:
        """Process PDF: extract text, extract tables. Returns list of Documents."""
        documents = []
        logger.info(f"Processing PDF: {pdf_path}")
        
        # 1. Extract text content
        text_docs = self._extract_text_from_pdf(pdf_path)
        if text_docs:
            documents.extend(text_docs)
            
        # 2. Extract tables 
        table_docs = self._extract_tables_from_pdf(pdf_path)
        if table_docs:
            documents.extend(table_docs)
            
        # 3. OCR Placeholder (remains placeholder)
        # ocr_docs = self._extract_ocr_from_pdf(pdf_path)
        # if ocr_docs:
        #    documents.extend(ocr_docs)
        
        if not documents:
            logger.warning(f"No content (text or tables) successfully extracted from {pdf_path}")
        
        return documents

    def _extract_text_from_pdf(self, pdf_path: Path) -> List[Document]:
        """Extracts text sections from a PDF using the best available reader."""
        base_docs = []
        try:

            pdf_reader = PyMuPDFReader()
            logger.info("PyMuPDFReader initialized")
            file_extractor = {".pdf": pdf_reader}
            pdf_parser_used = "PyMuPDFReader"
            try:
                logger.info(f"Attempting data extraction with SimpleDirectoryReader (using {pdf_parser_used}) for {pdf_path.name}")
                # Instantiate SimpleDirectoryReader targeting the single PDF file
                pdf_loader = SimpleDirectoryReader(
                    input_files=[str(pdf_path)],
                    required_exts=[".pdf"], 
                    file_extractor=file_extractor, # Pass the configured extractor
                    # Define metadata function - source is useful
                    file_metadata=lambda filename: {"source": Path(filename).name}
                )
                # Load data using the configured reader
                loaded_docs = pdf_loader.load_data()
                logger.info(f"SimpleDirectoryReader finished loading data for {pdf_path.name}. Found {len(loaded_docs)} document section(s).")

            except Exception as e:
                # Catch potential errors during loading (e.g., corrupted file)
                logger.error(f"SimpleDirectoryReader failed for {pdf_path.name}: {e}", exc_info=True)
                loaded_docs = []

            # # --- Quality Check ---
            # if loaded_docs:
            #     sample_text = loaded_docs[0].text[:500] if loaded_docs[0].text else ""
            #     sample_len = len(sample_text) # Avoid recalculating length
            #
            #     # --- Modified Quality Check ---
            #     is_long_enough = sample_len > 10
            #     # Calculate ratio of non-printable characters
            #     non_printable_ratio = sum(1 for char in sample_text if not char.isprintable()) / sample_len if sample_len > 0 else 1.0
            #     is_mostly_printable = non_printable_ratio < 0.05 # Allow up to 5% non-printable chars
            #
            #     # Calculate ratio of non-ASCII characters
            #     non_ascii_ratio = sum(1 for char in sample_text if ord(char) > 127) / sample_len if sample_len > 0 else 1.0
            #     is_mostly_ascii = non_ascii_ratio < 0.30 # Allow up to 30% non-ASCII (e.g., symbols, some foreign chars)
            #
            #     # Combine checks
            #     passes_quality_check = is_long_enough and is_mostly_printable and is_mostly_ascii
            #     # --- End Modified Quality Check ---
            #
            #     if passes_quality_check:
            #          logger.info(f"Successfully extracted {len(loaded_docs)} text sections from {pdf_path.name} (passed quality check).")
            #          logger.debug(f"First 100 chars: {sample_text[:100]}...")
            #          base_docs.extend(loaded_docs)
            #     else:
            #          # Log the specific reason for failure
            #          fail_reasons = []
            #          if not is_long_enough: fail_reasons.append(f"too short ({sample_len} chars)")
            #          if not is_mostly_printable: fail_reasons.append(f"too many non-printable chars ({non_printable_ratio:.2%})")
            #          if not is_mostly_ascii: fail_reasons.append(f"too many non-ASCII chars ({non_ascii_ratio:.2%})")
            #          logger.warning(f"Discarding text from {pdf_path.name} due to quality check failure: {', '.join(fail_reasons)}. Sample: '{sample_text[:1000]}...'")
        except Exception as e:
            logger.error(f"Overall text extraction phase failed for {pdf_path}: {str(e)}", exc_info=True)
        return loaded_docs

    def _extract_tables_from_pdf(self, pdf_path: Path) -> List[Document]:
        """Extracts tables from a PDF using pdfplumber and formats as raw strings."""
        table_docs = []
        table_count = 0
        try:
            logger.info(f"Attempting table extraction with pdfplumber from {pdf_path.name}...")
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables_on_page = page.extract_tables(table_settings={})
                    if tables_on_page:
                        logger.info(f"Found {len(tables_on_page)} potential tables on page {page_num + 1}")
                        for table_idx, table_data in enumerate(tables_on_page):
                            if table_data:
                                table_raw_str = format_table_raw_string(table_data)
                                print(table_raw_str)
                                if table_raw_str:
                                    table_text = f"--- Start Table (Page {page_num + 1}, Index {table_idx + 1}) ---\n{table_raw_str}\n--- End Table ---"
                                    doc = Document(
                                        text=table_text,
                                        metadata={
                                            "source": str(pdf_path),
                                            "page_number": page_num + 1,
                                            "type": "table_raw",
                                            "table_index_on_page": table_idx + 1
                                        }
                                    )
                                    table_docs.append(doc)
                                    table_count += 1
                                    logger.debug(f"Extracted Table {table_count} Raw String Snippet:\n{table_raw_str[:300]}{'...' if len(table_raw_str) > 300 else ''}")
                                else:
                                     logger.warning(f"Raw string conversion failed for table {table_idx+1} on page {page_num+1}.")
                            else:
                                logger.debug(f"Skipping empty table object {table_idx+1} on page {page_num+1}.")
                    else:
                         logger.debug(f"No tables found on page {page_num + 1} by pdfplumber.")
            logger.info(f"Extracted {table_count} valid tables as raw strings using pdfplumber from {pdf_path.name}.")
        except ImportError:
            logger.warning(f"pdfplumber not installed. Skipping table extraction. Run 'pip install pdfplumber'.")
        except Exception as e:
            if "No tables found" in str(e):
                 logger.info(f"No tables found by pdfplumber in {pdf_path.name}.")
            else:
                 logger.error(f"pdfplumber table extraction failed for {pdf_path}: {str(e)}", exc_info=True)
        return table_docs
        
    # Removed _split_documents method
    # def _split_documents(self, documents: List[Document]) -> List[Document]:
    #     ... 