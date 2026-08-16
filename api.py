from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uuid

# Import our existing RAG function
from chat import query_rag

app = FastAPI(
    title="Motor Legal Chatbot API",
    description="An API to ask questions about Motor Laws using RAG.",
    version="1.0.0"
)

# In-memory dictionary to store chat history per session
# In production, this would be a database like Redis or PostgreSQL
sessions = {}

# --- Pydantic Models ---

class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class SourceDoc(BaseModel):
    page: int
    content: str

class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources: List[SourceDoc]

# --- Endpoints ---

@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    # 1. Handle Session ID
    session_id = req.session_id
    if not session_id:
        # Generate a new session ID if the client didn't provide one
        session_id = str(uuid.uuid4())
        sessions[session_id] = []
    
    if session_id not in sessions:
        # If they provided an ID but we don't have it (e.g. server restarted)
        sessions[session_id] = []
        
    chat_history = sessions[session_id]
    
    try:
        # 2. Call the RAG pipeline
        # Pass api_mode=True so it doesn't print all the terminal outputs
        response = query_rag(req.question, history=chat_history, api_mode=True)
        
        # 3. Format the sources
        sources = []
        for doc in response["context"]:
            sources.append(SourceDoc(
                page=doc.metadata.get("page", 0),
                content=doc.page_content
            ))
            
        # 4. Return the clean JSON
        return AskResponse(
            answer=response["answer"],
            session_id=session_id,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# --- Static File Serving ---
# Mount the static directory to serve index.html at the root
app.mount("/", StaticFiles(directory="static", html=True), name="static")
