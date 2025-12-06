import requests
from typing import List, Dict
import json

class AgentEngine:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "gemma2:2b"
        print("✅ Agent Engine initialized")
    
    def is_available(self):
        """Check if Ollama is available for agent mode"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    async def generate_response(
        self,
        message: str,
        system_prompt: str,
        history: List[Dict]
    ):
        """Generate response in agent mode with potential tool calling"""
        try:
            if not self.is_available():
                return (
                    "⚠️ Ollama is not running for Agent mode.\n\n"
                    "Please start it with: ollama serve"
                )
            
            # Build conversation context
            conversation = []
            for msg in history[-10:]:  # Last 10 messages
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    conversation.append(f"User: {content}")
                elif role == 'assistant':
                    conversation.append(f"Assistant: {content}")
            
            # Agent system prompt with tool descriptions
            agent_system = (
                f"{system_prompt}\n\n"
                "You are an AI agent with access to various tools:\n"
                "- Web search (for current information)\n"
                "- Calculator (for math operations)\n"
                "- Code executor (for running code)\n"
                "- File operations (read/write files)\n\n"
                "When responding:\n"
                "1. Analyze the user's request\n"
                "2. Determine if tools are needed\n"
                "3. Provide helpful, accurate responses\n"
                "4. If you need real-time data or calculations, mention what tools would help"
            )
            
            # Build prompt
            prompt = f"{agent_system}\n\n"
            if conversation:
                prompt += "=== Conversation History ===\n" + "\n".join(conversation) + "\n\n"
            prompt += f"=== Current Query ===\nUser: {message}\n\nAssistant:"
            
            print(f"📤 Agent mode: Processing request...")
            
            # Call Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 4096,
                        "temperature": 0.8,
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                agent_response = result.get('response', 'No response generated')
                
                # Add agent capabilities note
                if any(keyword in message.lower() for keyword in ['search', 'calculate', 'run', 'execute']):
                    agent_response += (
                        "\n\n💡 *Note: This is a simulated agent response. "
                        "Full tool integration would require additional MCP server setup.*"
                    )
                
                return agent_response
            else:
                return f"Error: Agent mode returned status code {response.status_code}"
        
        except requests.exceptions.ConnectionError:
            return (
                "❌ Cannot connect to Ollama for Agent mode.\n\n"
                "Please ensure Ollama is running: ollama serve"
            )
        except Exception as e:
            return f"❌ Agent Error: {str(e)}"