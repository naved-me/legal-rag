# Motor Legal Chatbot (RAG System)

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about Motor Laws. It extracts text from local PDF documents, stores them in a local vector database, and uses a cloud-based LLM (Groq) to provide factually accurate, hallucination-free legal answers.

## Features
- **Offline Embeddings**: Uses HuggingFace (`all-MiniLM-L6-v2`) to generate document embeddings completely locally.
- **Vector Database**: Uses ChromaDB to store and retrieve text vectors locally.
- **Fast Cloud LLM**: Uses Groq (Llama-3.1-8b-instant) for blazing-fast answer generation.
- **LCEL Architecture**: Built using modern LangChain Expression Language for clean, readable pipelines.
- **Mathematical Evaluation**: Includes a custom LLM-as-a-Judge script to mathematically score the bot on Faithfulness and Answer Relevance.

## Project Files
- `ingest.py`: Parses the PDFs, splits the text into chunks, creates embeddings, and saves them to the local Chroma database.
- `chat.py`: The main RAG retrieval chain. Takes user questions, retrieves relevant chunks from Chroma, and asks the LLM to generate an answer.
- `evaluate.py`: An automated testing script that grades the chatbot's answers to ensure they don't hallucinate.
- `plan.md`: The roadmap and architecture phases for the project.
- `learn.md`: A learning log of architectural decisions and concepts.

## Setup Instructions

1. **Create and Activate a Virtual Environment**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   - Create a file named `.env` in the root folder.
   - Add your Groq API key:
     ```env
     GROQ_API_KEY=gsk_your_api_key_here
     ```

4. **Ingest the Data**
   Run the ingestion script to build the local database:
   ```powershell
   python ingest.py
   ```

5. **Chat with the Bot**
   Run the chat script to ask questions (you can change the hardcoded question at the bottom of the file):
   ```powershell
   python chat.py
   ```

6. **Run the Evaluator**
   Run the test script to ensure the bot is generating faithful answers:
   ```powershell
   python evaluate.py
   ```