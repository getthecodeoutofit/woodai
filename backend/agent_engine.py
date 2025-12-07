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
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different Ollama model and preload it"""
        try:
            # Verify model exists
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                if model_name in model_names:
                    old_model = self.model_name
                    self.model_name = model_name
                    # Update task orchestrator model
                    self.task_orchestrator.model_name = model_name
                    print(f"🔄 Switching Agent model from {old_model} to {model_name}...")
                    
                    # Give Ollama a moment to unload the previous model
                    import time
                    time.sleep(0.5)
                    
                    # Preload the new model to avoid EOF errors on first request
                    if self._preload_model():
                        print(f"✅ Switched Agent model to: {model_name} (preloaded)")
                        return True
                    else:
                        print(f"⚠️ Switched Agent model to: {model_name} (preload failed, but will try on first request)")
                        # Still return True - the model switch succeeded, preload is just optimization
                        return True
                else:
                    print(f"⚠️ Model {model_name} not found. Available models: {model_names}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Error switching model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _preload_model(self) -> bool:
        """Preload the model to avoid EOF errors during first request"""
        try:
            print(f"🔄 Preloading model {self.model_name}...")
            # Make a small test request to load the model
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "test",
                    "stream": False,
                    "options": {
                        "num_predict": 1,  # Just generate 1 token to load the model
                        "num_ctx": 128
                    }
                },
                timeout=60
            )
            if response.status_code == 200:
                print(f"✅ Model {self.model_name} preloaded successfully")
                return True
            else:
                error_text = response.text
                # Check for EOF/model loading errors
                if "load" in error_text.lower() and ("EOF" in error_text or "connection" in error_text.lower()):
                    print(f"⚠️ Model preload failed with loading error: {error_text}")
                    # Retry once after a short delay
                    import time
                    time.sleep(1)
                    try:
                        response = requests.post(
                            f"{self.ollama_url}/api/generate",
                            json={
                                "model": self.model_name,
                                "prompt": "test",
                                "stream": False,
                                "options": {
                                    "num_predict": 1,
                                    "num_ctx": 128
                                }
                            },
                            timeout=60
                        )
                        if response.status_code == 200:
                            print(f"✅ Model {self.model_name} preloaded successfully on retry")
                            return True
                    except:
                        pass
                print(f"⚠️ Model preload returned status {response.status_code}: {error_text}")
                return False
        except Exception as e:
            print(f"⚠️ Model preload failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m.get('name', '') for m in models]
            return []
        except:
            return []
    
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
    
    async def generate_response_stream(
        self,
        message: str,
        system_prompt: str,
        history: List[Dict],
        temperature: float = 0.8,
        context_length: int = 2048
    ):
        """Generate response with streaming support - yields chunks as they arrive"""
        try:
            if not self.is_available():
                yield "data: " + json.dumps({"chunk": "⚠️ Ollama is not running for Agent mode.\n\nPlease start it with: ollama serve", "done": True}) + "\n\n"
                return
            
            # Check if this is a task execution request
            if self._is_task_request(message):
                # For task execution, we'll stream the results as they come
                # But task execution is complex, so we'll do it non-streaming for now
                # and just stream the final result
                print(f"🤖 Agent mode: Detected task execution request")
                result = await self._execute_tasks(message, system_prompt, return_structured=False)
                yield "data: " + json.dumps({"chunk": result, "done": True}) + "\n\n"
                return
            
            # Build conversation context
            conversation = []
            for msg in history[-5:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    conversation.append(f"User: {content}")
                elif role == 'assistant':
                    conversation.append(f"Assistant: {content}")
            
            # Simplified system prompt
            agent_system = f"{system_prompt}\n\nYou are a helpful AI assistant with task execution capabilities."
            
            # Build prompt
            prompt = f"{agent_system}\n\n"
            if conversation:
                prompt += "\n".join(conversation[-3:]) + "\n\n"
            prompt += f"User: {message}\nAssistant:"
            
            print(f"📤 Agent mode: Processing chat request with streaming...")
            
            # Use async HTTP if available for streaming
            if self.async_client:
                try:
                    async with self.async_client.stream(
                        'POST',
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "stream": True,
                            "options": {
                                "num_ctx": context_length,
                                "temperature": temperature,
                                "num_predict": min(context_length, 2048),
                            }
                        },
                        timeout=120.0
                    ) as response:
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if line:
                                    try:
                                        data = json.loads(line)
                                        chunk = data.get('response', '')
                                        if chunk:
                                            # Yield immediately for each chunk
                                            yield "data: " + json.dumps({"chunk": chunk, "done": False}) + "\n\n"
                                        
                                        if data.get('done', False):
                                            yield "data: " + json.dumps({"chunk": "", "done": True}) + "\n\n"
                                            break
                                    except json.JSONDecodeError:
                                        continue
                        else:
                            yield "data: " + json.dumps({"chunk": f"Error: Agent mode returned status code {response.status_code}", "done": True}) + "\n\n"
                    return
                except Exception as e:
                    print(f"⚠️ Async streaming failed, falling back to sync: {e}")
            
            # Fallback to synchronous streaming
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_ctx": 2048,
                        "temperature": 0.8,
                        "num_predict": 500,
                    }
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get('response', '')
                            if chunk:
                                # Yield immediately for each chunk
                                yield "data: " + json.dumps({"chunk": chunk, "done": False}) + "\n\n"
                            
                            if data.get('done', False):
                                yield "data: " + json.dumps({"chunk": "", "done": True}) + "\n\n"
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                yield "data: " + json.dumps({"chunk": f"Error: Agent mode returned status code {response.status_code}", "done": True}) + "\n\n"
        
        except requests.exceptions.ConnectionError:
            yield "data: " + json.dumps({"chunk": "❌ Cannot connect to Ollama for Agent mode.\n\nPlease ensure Ollama is running: ollama serve", "done": True}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"chunk": f"❌ Agent Error: {str(e)}", "done": True}) + "\n\n"