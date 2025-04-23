# SMARTFinRAG: An Interactive Modular Framework for Financial RAG Systems

SMARTFinRAG is a comprehensive, modular live-demo system specifically designed for financial domains that addresses critical challenges in Retrieval-Augmented Generation (RAG) systems. Our framework enables customizable RAG evaluation, real-time component swapping, and document-centric assessment to promote trustworthy, document-grounded financial question answering research.

## Deployed Live-Demo Available!!!

https://smartfinrag-a9cawsutx5f4pl4j8wnox7.streamlit.app/

## Key Capabilities

### Interactive Evaluation Framework
- **Customizable Parameter Configuration**: Adjust both RAG and LLM parameters to dynamically configure the generation process
- **Modular Component Architecture**: Selectively enable/disable components in the RAG pipeline for ablation studies and bottleneck identification
- **Document-Based Evaluation**: Utilize a document-centric evaluation paradigm with LLM-as-a-Judge to generate and assess QA pairs
- **Comprehensive Metrics Suite**: Measure both retrieval quality (hit rate, MRR, precision, recall, NDCG) and response quality (faithfulness, relevancy)

### Financial Domain Specialization
- **Finance-Specific Processing**: Tailored for financial document understanding with domain-specific components
- **Multi-Dataset Support**: Unified QA schema covering multiple financial datasets
- **SEC Filings Support**: Compatible with the "Generative AI rewritten SEC filings" dataset (Lehner, 2024)
- **Timeliness Evaluation**: Just-in-time document ingestion to assess model performance on recent financial information

### Advanced RAG Components
- **QueryPreprocessor**: Named Entity Recognition and query enhancement capabilities
- **RetrieverFactory**: Configure BM25, vector-based, and hybrid retrieval approaches
- **Document Processing Pipeline**: Support for PDF, TXT, DOCX with chunking optimization
- **Extensible Evaluation System**: Automated metrics calculation and result exportation

## Implementation

### Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here

HUGGINGFACE_API_KEY=your_hf_api_key_here

OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload financial documents using the sidebar
3. Click "Process Documents" to index the uploaded files
4. Configure RAG components through the interface
5. Start asking questions in the chat interface
6. View evaluation metrics and experiment with different configurations

## Document Guidelines

- Upload relevant financial documents (PDF, TXT, DOCX)
- For optimal performance, use well-structured financial documents
- The system supports both batch processing and real-time document indexing
- Documents are stored securely in the local environment

## Architecture

SMARTFinRAG implements a modular architecture with the following key components:
![Architecture](https://github.com/user-attachments/assets/460c3151-51ce-4522-8967-c551ae6af4b4)

1. **Document Processing**: Ingestion, parsing, and chunking of financial documents
2. **Retrieval System**: Configurable retrieval mechanisms with multiple strategies
3. **Generation Module**: LLM-based response synthesis with context integration
4. **Evaluation Framework**: Automatic assessment of retrieval and generation quality
5. **Web Interface**: Interactive UI for real-time experimentation and visualization

## Security Considerations

- Never upload documents containing sensitive personal information
- API keys should be kept secure and never shared
- Document storage is local to the application
