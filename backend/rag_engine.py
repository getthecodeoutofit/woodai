import requests
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import time

class RAGEngine:
    def __init__(self, model_name="gemma3:4b", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        print("🔄 Loading embedding model...")
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = []
        self.embeddings = []
        self._load_knowledge_base()
        print(f"✅ RAG Engine initialized with {len(self.knowledge_base)} documents")
    
    def _load_knowledge_base(self):
        """Load documents from knowledge base directory"""
        kb_path = Path("knowledge_base")
        
        # Create knowledge_base directory if it doesn't exist
        if not kb_path.exists():
            kb_path.mkdir(parents=True)
            print("📁 Created knowledge_base directory")
            
            # Create a sample document
            sample_doc = kb_path / "sample.txt"
            sample_doc.write_text(
                "Welcome to WoodAI!\n\n"
                "This is a sample knowledge base document. "
                "You can add your own documents here to enhance the AI's knowledge.\n\n"
                "Features:\n"
                "- RAG (Retrieval Augmented Generation)\n"
                "- Agent mode with tool calling\n"
                "- Conversation memory\n"
                "- File attachments\n"
                "- Dark/Light themes"
            )
            print("📄 Created sample knowledge base document")
        
        # Load all text files
        for file_path in kb_path.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty documents
                        self.knowledge_base.append({
                            'filename': file_path.name,
                            'content': content
                        })
                        print(f"📄 Loaded: {file_path.name}")
            except Exception as e:
                print(f"⚠️ Error loading {file_path.name}: {e}")
        
        # Generate embeddings for knowledge base
        if self.knowledge_base:
            print("🔄 Generating embeddings...")
            texts = [doc['content'] for doc in self.knowledge_base]
            self.embeddings = self.embeddings_model.encode(texts)
            print(f"✅ Generated embeddings for {len(self.embeddings)} documents")
    
    def is_available(self):
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def retrieve_context(self, query: str, top_k: int = 3):
        """Retrieve relevant context from knowledge base"""
        if not self.knowledge_base:
            return ""
        
        query_embedding = self.embeddings_model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        context_parts = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Threshold for relevance
                doc = self.knowledge_base[idx]
                context_parts.append(f"[From {doc['filename']}]\n{doc['content']}")
        
        return "\n\n".join(context_parts)
    
    async def generate_response(
        self, 
        message: str, 
        context_length: int,
        memory_enabled: bool,
        temperature: float,
        system_prompt: str,
        history: List[Dict]
    ):
        """Generate response using RAG with Ollama"""
        try:
            # Check if Ollama is running
            if not self.is_available():
                return (
                    "⚠️ Ollama is not running. Please start it with:\n\n"
                    "1. Open a terminal\n"
                    "2. Run: ollama serve\n"
                    "3. In another terminal, ensure model is pulled: ollama pull gemma2:2b"
                )
            
            # Retrieve relevant context
            context = self.retrieve_context(message)
            
            # Build conversation history if memory is enabled
            conversation = []
            if memory_enabled and len(history) > 1:
                # Include last few messages based on context length
                max_history = min(len(history) - 1, context_length // 512)
                for msg in history[-max_history:]:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'user':
                        conversation.append(f"User: {content}")
                    elif role == 'assistant':
                        conversation.append(f"Assistant: {content}")
            
            # Build prompt with context
            prompt = f"{system_prompt}\n\n"
            
            if context:
                prompt += f"=== Relevant Knowledge ===\n{context}\n\n"
            
            if conversation:
                prompt += f"=== Conversation History ===\n" + "\n".join(conversation) + "\n\n"
            
            prompt += f"=== Current Query ===\nUser: {message}\n\nAssistant:"
            
            print(f"📤 Sending request to Ollama ({self.model_name})...")
            start_time = time.time()
            
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": context_length,
                        "temperature": temperature,
                    }
                },
                timeout=120
            )
            
            elapsed = time.time() - start_time
            print(f"✅ Response received in {elapsed:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                return f"Error: Ollama returned status code {response.status_code}"
        
        except requests.exceptions.ConnectionError:
            return (
                "❌ Cannot connect to Ollama.\n\n"
                "Please ensure:\n"
                "1. Ollama is installed (https://ollama.ai)\n"
                "2. Run 'ollama serve' in a terminal\n"
                "3. Run 'ollama pull gemma2:2b' to download the model"
            )
        except requests.exceptions.Timeout:
            return (
                "⏱️ Request timed out. This can happen if:\n"
                "- The model is still loading (first time)\n"
                "- The query is very complex\n"
                "- Your system is under heavy load\n\n"
                "Try again in a moment."
            )
        except Exception as e:
            return f"❌ Error: {str(e)}"