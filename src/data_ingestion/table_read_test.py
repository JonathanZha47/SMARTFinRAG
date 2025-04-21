# test_tabula.py
import tabula
import logging
import sys

import pdfplumber

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- IMPORTANT: Replace with the actual path to one of your test PDFs ---
pdf_path = "/Users/zhayiwei/Desktop/simple-financial-rag-chatbot/data/AMZN-Q4-2024-Earnings-Release.pdf" 
# --- Or provide it as a command-line argument ---
# if len(sys.argv) > 1:
#    pdf_path = sys.argv[1]
# else:
#    print("Usage: python test_tabula.py <path_to_pdf>")
#    sys.exit(1)

logging.info(f"Attempting to read: {pdf_path}")
logging.info(f"Table extraction using tabula")
try:
    # Try the simplest possible call first
    dfs = tabula.read_pdf(pdf_path, pages='all') 
    
    if dfs:
        logging.info(f"Successfully read {len(dfs)} DataFrame(s) from page 1.")
        # Optionally print the first table's head
        print(dfs[0]) 
    elif dfs == []:
            logging.info("Tabula ran successfully but found no tables on page 1.")
    else:
            logging.warning(f"Tabula returned an unexpected non-list, non-empty type: {type(dfs)}")

except ImportError as ie:
        logging.error(f"ImportError during Tabula call: {ie}. Is tabula-py installed?", exc_info=True)
except FileNotFoundError as fnf:
    logging.error(f"PDF File not found: {pdf_path}", exc_info=True)
except Exception as e:
    logging.error(f"An error occurred during the Tabula call: {e}", exc_info=True)
    # Try to print Java stderr if possible (might not work depending on how tabula-py handles it)
    if hasattr(e, 'stderr'):
            logging.error(f"Java stderr (if available): {e.stderr}")
logging.info(f"Table extraction using pdfplumber")
try:
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            logging.info(f"Found {len(tables)} tables on this page")
            for table in tables:
                logging.info(f"Table: {table}")
except Exception as e:
    logging.error(f"Error during pdfplumber extraction: {e}", exc_info=True)
logging.info("Test script finished.")