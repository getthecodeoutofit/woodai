from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
from rag_engine import RAGEngine
from agent_engine import AgentEngine

app = FastAPI(title="WoodAI Backend")

# CORS middleware - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
rag_engine = RAGEngine()
agent_engine = AgentEngine()

class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    files: Optional[List[Dict]] = None

class ChatRequest(BaseModel):
    message: str
    mode: str
    context_length: int = 4096
    memory_enabled: bool = True
    temperature: float = 0.7
    system_prompt: str = "You are a helpful AI assistant."
    history: List[Message] = []

class ChatResponse(BaseModel):
    response: str
    mode: str
    tokens_used: Optional[int] = None

@app.get("/")
async def root():
    return {
        "message": "WoodAI Backend is running", 
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if request.mode == "rag":
            # RAG mode using Ollama
            response = await rag_engine.generate_response(
                message=request.message,
                context_length=request.context_length,
                memory_enabled=request.memory_enabled,
                temperature=request.temperature,
                system_prompt=request.system_prompt,
                history=[msg.dict() for msg in request.history]
            )
        else:
            # Agent mode
            response = await agent_engine.generate_response(
                message=request.message,
                system_prompt=request.system_prompt,
                history=[msg.dict() for msg in request.history]
            )
        
        return ChatResponse(
            response=response, 
            mode=request.mode,
            tokens_used=len(response.split()) * 4  # Rough estimate
        )
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    rag_available = rag_engine.is_available()
    agent_available = agent_engine.is_available()
    
    return {
        "status": "healthy",
        "rag_available": rag_available,
        "agent_available": agent_available,
        "rag_model": rag_engine.model_name if rag_available else None,
        "knowledge_base_docs": len(rag_engine.knowledge_base)
    }

@app.get("/models")
async def get_models():
    """Get available Ollama models"""
    return {
        "rag_model": rag_engine.model_name,
        "available": rag_engine.is_available()
    }

if __name__ == "__main__":
    print("🚀 Starting WoodAI Backend on http://localhost:8000")
    print(f"📚 Knowledge base docs loaded: {len(rag_engine.knowledge_base)}")
    print(f"🤖 RAG Engine available: {rag_engine.is_available()}")
    
    if not rag_engine.is_available():
        print("\n⚠️  WARNING: Ollama is not running!")
        print("Please run 'ollama serve' in another terminal\n")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)