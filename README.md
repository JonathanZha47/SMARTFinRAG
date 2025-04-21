# Financial Advisor RAG Chatbot

A simple RAG-based chatbot that provides financial advice based on uploaded documents. Built with LlamaIndex and Streamlit.

## Features

- Document upload support (PDF, TXT, DOCX)
- Real-time document processing
- Interactive chat interface
- Context-aware financial advice
- Persistent document storage
- Chat history

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload financial documents using the sidebar
3. Click "Process Documents" to index the uploaded files
4. Start asking questions in the chat interface

## Document Guidelines

- Upload relevant financial documents (PDF, TXT, DOCX)
- Documents should contain financial information, reports, or advice
- Larger documents will be automatically chunked for processing

## Notes

- The system uses GPT-4 for response generation
- Documents are processed using LlamaIndex for efficient retrieval
- Chat history is maintained during the session
- Uploaded documents are stored in the `data` directory
- Vector store is persisted in the `storage` directory

## Security

- Never upload documents containing sensitive personal information
- API keys should be kept secure and never shared
- Document storage is local to the application 