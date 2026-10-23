# GoTravel RAG Assistant

A document-based Retrieval-Augmented Generation (RAG) application that answers travel-policy questions using information retrieved from a knowledge base and Google's Gemini large language model.

## Overview

GoTravel RAG combines semantic search with large language model generation to produce answers grounded in relevant document passages.

Instead of relying only on the language model's existing knowledge, the application retrieves relevant text chunks from a travel-policy document and supplies them as context to Gemini.

## Features

- **Document loading:** Reads travel-policy information from a text file.
- **Text chunking:** Splits documents into overlapping chunks to preserve context between sections.
- **Semantic embeddings:** Uses the `all-MiniLM-L6-v2` sentence-transformer model to represent text as numerical vectors.
- **Vector search:** Uses FAISS to retrieve relevant chunks for a user question.
- **LLM integration:** Uses the Gemini API to generate answers from the retrieved context.
- **Source-aware responses:** Includes retrieved source chunk references in the prompt.
- **Interactive terminal interface:** Allows users to ask multiple questions in one session.

## How It Works

1. Load the travel-policy knowledge base.
2. Split the document into overlapping chunks.
3. Generate embeddings for each chunk.
4. Store the embeddings in a FAISS index.
5. Convert the user's question into an embedding.
6. Retrieve the most relevant chunks.
7. Send the question and retrieved context to Gemini.
8. Display the generated answer.

## Technology Stack

- Python
- Google Gemini API
- Sentence Transformers
- FAISS
- NumPy
- python-dotenv

## Project Structure

```text
GoTravel-RAG-Project/
├── app.py
├── baseline_gemini.py
├── embeddings.py
├── embed_chunks.py
├── faiss_test.py
├── rag_gemini.py
├── read_document.py
├── retrieve_chunks.py
├── test_gemini.py
├── test_imports.py
├── External_KB_for_RAG.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/israel77798755/gotravel-rag.git
cd gotravel-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key. Never commit or publish the `.env` file.

### 5. Run the application

```bash
python app.py
```

Enter a travel-policy question when prompted. Type `exit` or `quit` to end the session.

## Example Questions

- What changes occurred when Romania and Bulgaria joined the Schengen Area?
- What does the knowledge base say about ETIAS requirements?
- What travel-policy information is available for a particular destination?

Answers depend on the contents and accuracy of the supplied knowledge base.

## Current Limitations

- The application uses a single text document as its knowledge base.
- Chunking uses a fixed character size and overlap.
- Retrieval quality depends on the embedding model, chunk boundaries, and source document.
- Generated answers can contain unsupported claims, even when relevant chunks are retrieved.
- The project does not yet include automated evaluation, a web interface, or production deployment.

Travel information should be verified against official government and EU sources before being used for real travel decisions.

## Future Improvements

- Improve chunking and retrieval quality.
- Add retrieval and answer evaluation metrics.
- Add automated tests for the RAG pipeline.
- Build a web interface using FastAPI or another suitable framework.
- Add document upload and support for multiple file formats.
- Add source-level citations and stronger evidence checks.
- Deploy the application with appropriate monitoring and error handling.

## Security

Store API credentials in environment variables and keep `.env` out of version control. Do not expose API keys in source code, logs, or public repositories.

## Author

**Israel Tsegaye Abino**

Information Science Student | Aspiring AI Engineer

GitHub: [@israel77798755](https://github.com/israel77798755)
