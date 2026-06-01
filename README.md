# DocuMind AI – Context-Aware Document Q&A Bot

## Overview
DocuMind AI is a Context-Aware Document Q&A Bot built using a Retrieval-Augmented Generation (RAG) architecture. It allows users to upload PDF documents or paste text directly, and then ask questions about the provided content. The bot strictly answers based on the provided context and gracefully handles situations where the information is unavailable.

## Features
- **PDF & Text Input**: Accepts both PDF file uploads and raw pasted text.
- **RAG Architecture**: Uses text chunking, embeddings, and vector similarity search for context retrieval.
- **Strict Answering**: The LLM is prompted to answer ONLY from the retrieved context.
- **Hallucination Prevention**: If the answer is not found, the bot will return exactly: `"This information is not available in the document"`.
- **Clean UI**: Built with Streamlit for a simple and intuitive user experience.

## Tech Stack
- **Frontend**: Streamlit
- **Language**: Python
- **LLM**: Google Gemini (via `langchain-google-genai`)
- **Embeddings**: HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **PDF Processing**: PyPDF2
- **Orchestration**: LangChain

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/piyushxbhardwaj/azentrix-fullstack-task1.git
cd azentrix-fullstack-task1
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. API Key Setup
Create a `.env` file in the root directory (you can copy `.env.example`):
```bash
cp .env.example .env
```
Add your Google Gemini API key to the `.env` file:
```
GOOGLE_API_KEY="your_gemini_api_key_here"
```

## Run Instructions
Start the Streamlit application:
```bash
streamlit run app.py
```

## Folder Structure
```
azentrix-fullstack-task1
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── .env.example           # Example environment variables
├── utils/                 # Utility modules
│   ├── pdf_loader.py      # PDF text extraction logic
│   ├── chunking.py        # Text splitting and chunking logic
│   ├── embeddings.py      # Embedding generation and vector store setup
│   └── retrieval.py       # QA chain and retrieval logic
├── screenshots/           # Application screenshots
└── sample_docs/           # Sample PDF documents for testing
```

## Screenshots
*(Add screenshots of the application here)*

## Demo
*(Add a link to a video demo or hosted application here)*

## Challenges and Approach
- **Hallucination Prevention**: To ensure the bot doesn't hallucinate, a strict prompt template is provided to the LangChain QA chain, explicitly instructing the LLM to output a specific phrase if the information is missing.
- **Handling Large Documents**: PDF text extraction and processing can take time. We split text into smaller overlapping chunks to preserve context across boundaries and use FAISS for efficient similarity search.
- **User Experience**: The UI is divided into a sidebar for data ingestion and a main area for questioning to keep the interface clean and intuitive.
