from fastapi import FastAPI

# 1. Initialize the FastAPI application
app = FastAPI(title="Legal RAG API", version="1.0")

# 2. Basic GET endpoint to check if the server is alive
@app.get("/health")
async def health_check():
    return {"status": "online", "message": "Legal RAG API is running"}