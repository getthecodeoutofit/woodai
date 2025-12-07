import os
# Disable TensorFlow in transformers (we use PyTorch only)
# This must be set before importing any transformers-related modules
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_TORCH', '1')

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
from rag_engine import RAGEngine
from agent_engine import AgentEngine
from database import get_database
import json

# Load environment variables from .env file
load_dotenv()

# Initialize engines and database
db = get_database()
rag_engine = RAGEngine()
agent_engine = AgentEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    db.connect()
    print("🚀 WoodAI Backend started successfully")
    yield
    # Shutdown
    db.close()
    print("👋 WoodAI Backend shut down")

app = FastAPI(title="WoodAI Backend", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    user_id: str = "default_user"
    chat_id: Optional[str] = None
    selected_doc_ids: Optional[List[str]] = None

class ChatResponse(BaseModel):
    response: str
    mode: str
    tokens_used: Optional[int] = None
    chat_id: Optional[str] = None

class SwitchModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    mode: str = "rag"

class DownloadModelRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str

@app.get("/")
async def root():
    return {
        "message": "WoodAI Backend is running", 
        "status": "healthy",
        "version": "2.0.0",
        "features": ["RAG", "Agent", "MongoDB", "OCR", "Multi-format documents"]
    }

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint - returns Server-Sent Events (SSE)"""
    async def generate():
        try:
            full_response = ""
            
            # Generate streaming response
            if request.mode == "rag":
                # Use async for since generate_response_stream is an async generator
                async for chunk_data in rag_engine.generate_response_stream(
                    message=request.message,
                    context_length=request.context_length,
                    memory_enabled=request.memory_enabled,
                    temperature=request.temperature,
                    system_prompt=request.system_prompt,
                    history=[msg.dict() for msg in request.history],
                    filter_doc_ids=request.selected_doc_ids
                ):
                    # Extract chunk from SSE format
                    if chunk_data.startswith("data: "):
                        try:
                            data = json.loads(chunk_data[6:])  # Remove "data: " prefix
                            chunk = data.get('chunk', '')
                            done = data.get('done', False)
                            
                            if chunk:
                                full_response += chunk
                            
                            # Send chunk to client immediately
                            yield chunk_data
                            
                            # If done, save to MongoDB
                            if done:
                                # Save/update chat in MongoDB
                                chat_id = request.chat_id
                                updated_history = [msg.dict() for msg in request.history]
                                updated_history.append({
                                    "role": "user",
                                    "content": request.message,
                                    "timestamp": ""
                                })
                                updated_history.append({
                                    "role": "assistant", 
                                    "content": full_response,
                                    "timestamp": ""
                                })
                                
                                if chat_id:
                                    await db.update_chat(chat_id, updated_history, request.selected_doc_ids)
                                else:
                                    title = request.message[:50] + "..." if len(request.message) > 50 else request.message
                                    chat_id = await db.save_chat(
                                        user_id=request.user_id,
                                        title=title,
                                        messages=updated_history,
                                        selected_doc_ids=request.selected_doc_ids
                                    )
                                
                                # Send final metadata
                                yield f"data: {json.dumps({'chat_id': chat_id, 'tokens_used': len(full_response.split()) * 4, 'done': True})}\n\n"
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                # Agent mode streaming
                async for chunk_data in agent_engine.generate_response_stream(
                    message=request.message,
                    system_prompt=request.system_prompt,
                    history=[msg.dict() for msg in request.history],
                    temperature=request.temperature,
                    context_length=request.context_length
                ):
                    if chunk_data.startswith("data: "):
                        try:
                            data = json.loads(chunk_data[6:])
                            chunk = data.get('chunk', '')
                            done = data.get('done', False)
                            
                            if chunk:
                                full_response += chunk
                            
                            # Send chunk immediately
                            yield chunk_data
                            
                            if done:
                                # Save/update chat in MongoDB
                                chat_id = request.chat_id
                                updated_history = [msg.dict() for msg in request.history]
                                updated_history.append({
                                    "role": "user",
                                    "content": request.message,
                                    "timestamp": ""
                                })
                                updated_history.append({
                                    "role": "assistant", 
                                    "content": full_response,
                                    "timestamp": ""
                                })
                                
                                if chat_id:
                                    await db.update_chat(chat_id, updated_history, request.selected_doc_ids)
                                else:
                                    title = request.message[:50] + "..." if len(request.message) > 50 else request.message
                                    chat_id = await db.save_chat(
                                        user_id=request.user_id,
                                        title=title,
                                        messages=updated_history,
                                        selected_doc_ids=request.selected_doc_ids
                                    )
                                
                                yield f"data: {json.dumps({'chat_id': chat_id, 'tokens_used': len(full_response.split()) * 4, 'done': True})}\n\n"
                                break
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            print(f"Error in streaming chat endpoint: {str(e)}")
            import traceback
            traceback.print_exc()
            error_data = json.dumps({"chunk": f"Error: {str(e)}", "done": True})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with MongoDB integration (non-streaming, for backward compatibility)"""
    try:
        # Generate response
        if request.mode == "rag":
            response = await rag_engine.generate_response(
                message=request.message,
                context_length=request.context_length,
                memory_enabled=request.memory_enabled,
                temperature=request.temperature,
                system_prompt=request.system_prompt,
                history=[msg.dict() for msg in request.history],
                filter_doc_ids=request.selected_doc_ids
            )
        else:
            # Agent mode - ensure we get a string response (not structured)
            response = await agent_engine.generate_response(
                message=request.message,
                system_prompt=request.system_prompt,
                history=[msg.dict() for msg in request.history],
                return_structured=False  # Explicitly request string response
            )
            # Ensure response is a string (in case of any edge cases)
            if isinstance(response, dict):
                response = response.get('response', str(response))
            response = str(response)
        
        # Save/update chat in MongoDB
        chat_id = request.chat_id
        updated_history = [msg.dict() for msg in request.history]
        updated_history.append({
            "role": "user",
            "content": request.message,
            "timestamp": ""
        })
        updated_history.append({
            "role": "assistant", 
            "content": response,
            "timestamp": ""
        })
        
        if chat_id:
            # Update existing chat
            await db.update_chat(chat_id, updated_history, request.selected_doc_ids)
        else:
            # Create new chat
            title = request.message[:50] + "..." if len(request.message) > 50 else request.message
            chat_id = await db.save_chat(
                user_id=request.user_id,
                title=title,
                messages=updated_history,
                selected_doc_ids=request.selected_doc_ids
            )
        
        return ChatResponse(
            response=response,
            mode=request.mode,
            tokens_used=len(response.split()) * 4,
            chat_id=chat_id
        )
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/{user_id}")
async def get_user_chats(user_id: str, limit: int = 50):
    """Get all chats for a user"""
    try:
        chats = await db.get_user_chats(user_id, limit)
        return {"chats": chats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/{chat_id}")
async def get_chat(chat_id: str):
    """Get specific chat by ID"""
    try:
        chat = await db.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat"""
    try:
        success = await db.delete_chat(chat_id)
        if not success:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"success": True, "message": "Chat deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("default_user")
):
    """Upload and index a document"""
    try:
        # Read file content
        content = await file.read()
        
        # Process and index
        result = await rag_engine.index_document(content, file.filename)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result.get('error', 'Processing failed'))
        
        return {
            "success": True,
            "message": f"Document indexed successfully",
            "doc_id": result['doc_id'],
            "filename": result['filename'],
            "chunks": result['chunks']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents():
    """Get all indexed documents"""
    try:
        docs = await db.get_all_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its vectors"""
    try:
        # Delete from vector store (Qdrant)
        if rag_engine.vector_store:
            success = rag_engine.vector_store.delete_documents([doc_id])
            if not success:
                print(f"⚠️ Warning: Failed to delete vectors for doc_id: {doc_id}")
        
        # Delete document metadata from MongoDB
        success = await db.delete_document(doc_id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "message": f"Document {doc_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    rag_available = rag_engine.is_available()
    agent_available = agent_engine.is_available()
    
    try:
        # Check MongoDB
        await db.chats.count_documents({}, limit=1)
        db_connected = True
    except:
        db_connected = False
    
    return {
        "status": "healthy",
        "rag_available": rag_available,
        "agent_available": agent_available,
        "mongodb_connected": db_connected,
        "rag_model": rag_engine.model_name if rag_available else None,
        "knowledge_base_docs": len(rag_engine.knowledge_base)
    }

@app.get("/models")
async def get_models():
    """Get current model info and available Ollama models"""
    try:
        available_models = rag_engine.get_available_models()
        return {
            "rag_model": rag_engine.model_name,
            "agent_model": agent_engine.model_name,
            "available_models": available_models,
            "ollama_available": rag_engine.is_available()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/switch")
async def switch_model(request: SwitchModelRequest):
    """Switch the Ollama model for RAG or Agent mode"""
    try:
        if request.mode == "rag":
            success = rag_engine.switch_model(request.model_name)
        elif request.mode == "agent":
            success = agent_engine.switch_model(request.model_name)
        else:
            raise HTTPException(status_code=400, detail="Mode must be 'rag' or 'agent'")
        
        if success:
            return {
                "success": True,
                "message": f"Switched {request.mode} model to {request.model_name}",
                "current_model": request.model_name
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to switch model to {request.model_name}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/download")
async def download_model(request: DownloadModelRequest):
    """Download an Ollama model"""
    try:
        result = rag_engine.download_model(request.model_name)
        if result.get('success'):
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Download failed'))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting WoodAI Backend on http://localhost:8000")
    print(f"📚 Knowledge base docs loaded: {len(rag_engine.knowledge_base)}")
    print(f"🤖 RAG Engine available: {rag_engine.is_available()}")
    
    if not rag_engine.is_available():
        print("\n⚠️  WARNING: Ollama is not running!")
        print("Please run 'ollama serve' in another terminal\n")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)