import requests
import asyncio
from typing import List, Dict, Optional
import json
from functools import lru_cache
from task_orchestrator import TaskOrchestrator

# Try to use async HTTP for better performance
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

class AgentEngine:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "gemma3:4b"
        self.task_orchestrator = TaskOrchestrator(ollama_url=ollama_url, model_name=self.model_name)
        
        # Create async HTTP client for faster requests (if available)
        self.async_client = None
        if HTTPX_AVAILABLE:
            self.async_client = httpx.AsyncClient(timeout=120.0)
        
        # Cache for task detection (faster keyword matching)
        self._task_keywords_set = frozenset([
            "write", "send", "email", "create", "file", "essay",
            "open", "youtube", "browser", "visit", "navigate",
            "then", "after that", "and then",
            "notion", "github", "slack", "mcp",
            "python", "code", "program"
        ])
        
        print("✅ Agent Engine initialized with task execution capabilities")
        if HTTPX_AVAILABLE:
            print("   ⚡ Using async HTTP for faster requests")
    
    async def close(self):
        """Close async HTTP client"""
        if self.async_client:
            await self.async_client.aclose()
    
    def is_available(self):
        """Check if Ollama is available for agent mode"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _is_task_request(self, message: str) -> bool:
        """Detect if the message contains task execution requests (optimized for speed)"""
        message_lower = message.lower()
        # Fast set intersection check (much faster than list iteration)
        words = set(message_lower.split())
        return bool(words & self._task_keywords_set)
    
    async def generate_response(
        self,
        message: str,
        system_prompt: str,
        history: List[Dict],
        return_structured: bool = False
    ):
        """Generate response in agent mode with task execution capabilities"""
        try:
            if not self.is_available():
                error_msg = (
                    "⚠️ Ollama is not running for Agent mode.\n\n"
                    "Please start it with: ollama serve"
                )
                return {"response": error_msg, "browser_actions": [], "task_results": None} if return_structured else error_msg
            
            # Check if this is a task execution request
            if self._is_task_request(message):
                print(f"🤖 Agent mode: Detected task execution request")
                return await self._execute_tasks(message, system_prompt, return_structured)
            
            # Otherwise, use regular chat mode
            response = await self._generate_chat_response(message, system_prompt, history)
            return {"response": response, "browser_actions": [], "task_results": None} if return_structured else response
        
        except requests.exceptions.ConnectionError:
            error_msg = (
                "❌ Cannot connect to Ollama for Agent mode.\n\n"
                "Please ensure Ollama is running: ollama serve"
            )
            return {"response": error_msg, "browser_actions": [], "task_results": None} if return_structured else error_msg
        except Exception as e:
            error_msg = f"❌ Agent Error: {str(e)}"
            return {"response": error_msg, "browser_actions": [], "task_results": None} if return_structured else error_msg
    
    async def _execute_tasks(
        self,
        message: str,
        system_prompt: str,
        return_structured: bool = False
    ):
        """Execute tasks from user input"""
        try:
            print(f"📋 Parsing and executing tasks...")
            result = await self.task_orchestrator.execute_task_sequence(
                user_input=message,
                system_prompt=system_prompt
            )
            
            # Extract browser actions
            browser_actions = []
            for task_result in result.get("execution_results", []):
                task = task_result["task"]
                exec_result = task_result["result"]
                if task.get("type") == "browser" and exec_result.get("success") and exec_result.get("url"):
                    browser_actions.append({
                        "url": exec_result["url"],
                        "action": exec_result.get("action", "open")
                    })
            
            # Format response
            response_parts = [
                "🤖 **Task Execution Results**\n",
                f"📊 **Summary:** {result['tasks_parsed']} task(s) processed\n\n"
            ]
            
            # Add execution summary
            if result.get("summary"):
                response_parts.append("**Execution Summary:**\n")
                response_parts.append(result["summary"])
                response_parts.append("\n")
            
            # Add detailed results
            response_parts.append("\n**Detailed Results:**\n")
            for task_result in result.get("execution_results", []):
                task = task_result["task"]
                exec_result = task_result["result"]
                
                status = "✅" if exec_result.get("success") else "❌"
                task_type = task.get("type", "unknown")
                
                response_parts.append(f"\n{status} **Task {task_result['task_number']}** ({task_type}):")
                response_parts.append(f"   - Instruction: {task.get('raw', 'N/A')}")
                
                if exec_result.get("success"):
                    response_parts.append(f"   - Result: {exec_result.get('message', 'Completed')}")
                    
                    # Add specific details based on task type
                    if task_type == "email" and exec_result.get("recipient"):
                        response_parts.append(f"   - Email sent to: {exec_result['recipient']}")
                    elif task_type == "file" and exec_result.get("file_path"):
                        response_parts.append(f"   - File created: {exec_result['file_path']}")
                    elif task_type == "browser" and exec_result.get("url"):
                        response_parts.append(f"   - URL to open: {exec_result['url']}")
                        response_parts.append(f"   - Note: Browser will open this URL in your frontend")
                else:
                    response_parts.append(f"   - Error: {exec_result.get('error', 'Unknown error')}")
            
            # Add browser action instructions if needed
            if browser_actions:
                response_parts.append("\n\n🌐 **Browser Actions:**")
                response_parts.append("The frontend will automatically open the specified URLs.")
            
            response_text = "\n".join(response_parts)
            
            if return_structured:
                return {
                    "response": response_text,
                    "browser_actions": browser_actions,
                    "task_results": result
                }
            else:
                return response_text
            
        except Exception as e:
            error_msg = f"❌ Task execution error: {str(e)}"
            return {"response": error_msg, "browser_actions": [], "task_results": None} if return_structured else error_msg
    
    async def _generate_chat_response(
        self,
        message: str,
        system_prompt: str,
        history: List[Dict]
    ) -> str:
        """Generate regular chat response (optimized for speed)"""
        # Build conversation context (reduced to last 5 for speed)
        conversation = []
        for msg in history[-5:]:  # Last 5 messages (reduced from 10)
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                conversation.append(f"User: {content}")
            elif role == 'assistant':
                conversation.append(f"Assistant: {content}")
        
        # Simplified system prompt (faster processing)
        agent_system = f"{system_prompt}\n\nYou are a helpful AI assistant with task execution capabilities."
        
        # Build prompt (more concise)
        prompt = f"{agent_system}\n\n"
        if conversation:
            prompt += "\n".join(conversation[-3:]) + "\n\n"  # Only last 3 messages
        prompt += f"User: {message}\nAssistant:"
        
        print(f"📤 Agent mode: Processing chat request...")
        
        # Use async HTTP if available, otherwise fallback to requests
        if self.async_client:
            try:
                response = await self.async_client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_ctx": 2048,  # Reduced from 4096 for speed
                            "temperature": 0.8,
                            "num_predict": 500,  # Limit response length
                        }
                    }
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', 'No response generated')
                else:
                    return f"Error: Agent mode returned status code {response.status_code}"
            except Exception as e:
                print(f"⚠️ Async request failed, falling back to sync: {e}")
                # Fallback to sync
        
        # Fallback to synchronous requests
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 2048,  # Reduced from 4096 for speed
                    "temperature": 0.8,
                    "num_predict": 500,  # Limit response length
                }
            },
            timeout=60  # Reduced from 120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No response generated')
        else:
            return f"Error: Agent mode returned status code {response.status_code}"