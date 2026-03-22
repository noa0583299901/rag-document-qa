# RAG Document QA System 📄🤖

A Retrieval-Augmented Generation (RAG) system that answers questions based on a collection of documents and provides source citations for each response.

The system combines vector embeddings, semantic search, and LLM reasoning to retrieve relevant information and generate accurate, grounded answers.

------------------------------------------------------------

## Features

• Semantic search using vector embeddings  
• Retrieval-Augmented Generation (RAG)  
• Answers grounded in real documents  
• Source citation for each response  
• Multi-document support  
• Persistent vector index for fast queries  

------------------------------------------------------------

## Technologies

- Python  
- LlamaIndex  
- OpenAI Models  
- Embeddings (text-embedding-3-small)  
- HTTPX  

------------------------------------------------------------

## How It Works

1. Documents (PDFs) are loaded from a data directory  
2. The system converts documents into vector embeddings  
3. A vector index is created and stored locally  
4. When a query is asked:
   - Relevant chunks are retrieved using similarity search  
   - The LLM generates an answer based only on retrieved data  
   - The system returns the answer along with source references  

------------------------------------------------------------

## Example Queries

- What are the ethical challenges of LLMs?  
- What applications exist in healthcare and education?  
- How can LLM performance be improved after training?  

------------------------------------------------------------

## Output Example

The system returns:

- A generated answer  
- A list of source documents and page references  

------------------------------------------------------------

## Project Structure

rag-document-qa

main.py  
data/                (PDF files)  
storage_ai/          (vector index storage)  
README.md  
.env  
.gitignore  

------------------------------------------------------------

## Installation

Install dependencies:

pip install llama-index openai python-dotenv httpx

------------------------------------------------------------
## Data

Add your PDF files into the /data folder before running the project.

## Run the Project

python main.py

------------------------------------------------------------

## Configuration

The system uses environment variables for API access:

OPENAI_API_KEY=your_api_key_here

------------------------------------------------------------

## Why This Project Stands Out

- Implements advanced RAG architecture  
- Combines retrieval + generation  
- Uses vector databases and embeddings  
- Provides explainability via source citations  
- Demonstrates real-world AI application design  

------------------------------------------------------------

## Future Improvements

- Web interface for user interaction  
- Support for more file types  
- Improved ranking and retrieval strategies  
- Integration with external databases  
- Real-time document updates  
