"""
Task Orchestrator
Coordinates task parsing, AI generation, and execution
"""

import requests
import asyncio
from typing import List, Dict, Any, Optional
import json
from task_parser import TaskParser
from task_executor import TaskExecutor
from mcp_client import MCPClient

# Try to use async HTTP for better performance
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None


class TaskOrchestrator:
    """Orchestrates task execution workflow"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "gemma3:4b"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.task_parser = TaskParser()
        self.task_executor = TaskExecutor()
        self.mcp_client = MCPClient()
        
        # Create async HTTP client for faster requests (if available)
        self.async_client = None
        if HTTPX_AVAILABLE:
            self.async_client = httpx.AsyncClient(timeout=60.0)
    
    async def execute_task_sequence(
        self,
        user_input: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a sequence of tasks from user input
        
        Args:
            user_input: Natural language task description
            system_prompt: Optional system prompt for AI generation
        
        Returns:
            Execution results with status and details
        """
        # Parse tasks
        tasks = self.task_parser.parse_tasks(user_input)
        
        results = {
            "success": True,
            "tasks_parsed": len(tasks),
            "execution_results": [],
            "summary": ""
        }
        
        execution_summary = []
        
        # Execute tasks - parallel where possible, sequential when dependencies exist
        # Check if tasks can run in parallel (no dependencies)
        can_parallelize = self._can_parallelize(tasks)
        
        if can_parallelize and len(tasks) > 1:
            # Execute in parallel for speed
            print(f"⚡ Executing {len(tasks)} tasks in parallel...")
            task_coroutines = [self._execute_single_task(task, system_prompt) for task in tasks]
            task_results_list = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            for i, (task, task_result) in enumerate(zip(tasks, task_results_list), 1):
                if isinstance(task_result, Exception):
                    task_result = {"success": False, "error": str(task_result)}
                
                results["execution_results"].append({
                    "task_number": i,
                    "task": task,
                    "result": task_result
                })
                
                # Build summary
                if task_result.get("success"):
                    execution_summary.append(f"✅ Task {i} ({task.get('type', 'unknown')}): {task_result.get('message', 'Completed')}")
                else:
                    execution_summary.append(f"❌ Task {i} ({task.get('type', 'unknown')}): {task_result.get('error', 'Failed')}")
                    results["success"] = False
        else:
            # Execute sequentially (when dependencies exist)
            for i, task in enumerate(tasks, 1):
                task_result = await self._execute_single_task(task, system_prompt)
                results["execution_results"].append({
                    "task_number": i,
                    "task": task,
                    "result": task_result
                })
                
                # Build summary
                if task_result.get("success"):
                    execution_summary.append(f"✅ Task {i} ({task.get('type', 'unknown')}): {task_result.get('message', 'Completed')}")
                else:
                    execution_summary.append(f"❌ Task {i} ({task.get('type', 'unknown')}): {task_result.get('error', 'Failed')}")
                    results["success"] = False
        
        results["summary"] = "\n".join(execution_summary)
        return results
    
    def _can_parallelize(self, tasks: List[Dict]) -> bool:
        """Check if tasks can be executed in parallel (no dependencies)"""
        # Tasks that can't be parallelized:
        # - Email tasks (might depend on previous file creation)
        # - File tasks that reference previous files
        # For now, allow parallelization for independent tasks like browser, simple files
        
        independent_types = {"browser", "mcp"}
        dependent_types = {"email", "file"}  # These might have dependencies
        
        # If all tasks are independent, can parallelize
        task_types = {task.get("type") for task in tasks}
        if task_types.issubset(independent_types):
            return True
        
        # If only one task, can always execute
        if len(tasks) == 1:
            return True
        
        # For mixed tasks, be conservative and execute sequentially
        return False
    
    async def _execute_single_task(
        self,
        task: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a single task"""
        task_type = task.get("type")
        
        try:
            if task_type == "email":
                return await self._execute_email_task(task, system_prompt)
            
            elif task_type == "file":
                return await self._execute_file_task(task, system_prompt)
            
            elif task_type == "browser":
                return await self._execute_browser_task(task)
            
            elif task_type == "mcp":
                return await self._execute_mcp_task(task)
            
            elif task_type == "general":
                # For general instructions, try to generate a response
                return await self._execute_general_task(task, system_prompt)
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task_type}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Task execution error: {str(e)}"
            }
    
    async def _execute_email_task(
        self,
        task: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute email task with AI-generated content"""
        recipient = task.get("recipient")
        subject = task.get("subject", "No Subject")
        body_context = task.get("body_context", "")
        
        # Generate email body using AI if context provided
        if body_context:
            email_prompt = f"""Write a professional email about: {body_context}

Requirements:
- Be concise and clear
- Professional tone
- Include appropriate greeting and closing
- Address the topic: {body_context}

Email:"""
            
            body = await self._generate_text(email_prompt, system_prompt)
        else:
            # Generate generic email
            email_prompt = f"""Write a professional email with subject: {subject}

Email:"""
            body = await self._generate_text(email_prompt, system_prompt)
        
        # Execute email task
        return await self.task_executor.execute_email_task(
            recipient=recipient,
            subject=subject,
            body=body
        )
    
    async def _execute_file_task(
        self,
        task: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute file task with AI-generated content"""
        file_path = task.get("file_path", "output.txt")
        content_type = task.get("content_type", "text")
        topic = task.get("topic", "")
        
        # Generate content using AI
        if content_type == "cpp_code" or (content_type == "code" and file_path.endswith(('.cpp', '.c', '.h', '.hpp'))):
            # Generate C++ code with proper syntax
            code_prompt = f"""Write a complete, syntactically correct C++ program about: {topic if topic else 'a useful C++ program'}

Requirements:
- Write complete, runnable C++ code
- Use proper C++ syntax and formatting
- Include necessary headers (#include)
- Use standard C++ style
- Add comments to explain key parts
- Make sure the code is functional and can be compiled
- Include main() function if it's a program
- Handle edge cases appropriately

C++ Code:"""
            
            content = await self._generate_text(code_prompt, system_prompt)
            
            # Clean up the response - remove markdown code blocks if present
            if content.startswith('```cpp'):
                content = content.replace('```cpp', '').replace('```', '').strip()
            elif content.startswith('```c++'):
                content = content.replace('```c++', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
        elif content_type == "python_code" or (content_type == "code" and file_path.endswith('.py')):
            # Generate Python code with proper syntax
            code_prompt = f"""Write a complete, syntactically correct Python program about: {topic if topic else 'a useful Python program'}

Requirements:
- Write complete, runnable Python code
- Use proper Python syntax and indentation (4 spaces)
- Include necessary imports at the top
- Add comments to explain key parts
- Follow PEP 8 style guidelines
- Make sure the code is functional and can be executed
- Include docstrings for functions/classes if applicable
- Handle edge cases appropriately

Python Code:"""
            
            content = await self._generate_text(code_prompt, system_prompt)
            
            # Clean up the response - remove markdown code blocks if present
            if content.startswith('```python'):
                content = content.replace('```python', '').replace('```', '').strip()
            elif content.startswith('```'):
                content = content.replace('```', '').strip()
            
        elif content_type == "code":
            # Generate code in other languages
            code_prompt = f"""Write complete, syntactically correct code about: {topic if topic else 'a useful program'}

Requirements:
- Write complete, runnable code
- Use proper syntax and formatting
- Include necessary imports/headers
- Add comments to explain key parts
- Make sure the code is functional

Code:"""
            
            content = await self._generate_text(code_prompt, system_prompt)
            
            # Clean up markdown code blocks if present
            if '```' in content:
                lines = content.split('\n')
                # Remove first and last lines if they are code block markers
                if lines[0].strip().startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                content = '\n'.join(lines).strip()
                
        elif content_type == "essay":
            essay_prompt = f"""Write a well-structured essay on the topic: {topic if topic else 'a general topic'}

Requirements:
- Introduction paragraph
- 2-3 body paragraphs with supporting details
- Conclusion paragraph
- Clear thesis statement
- Professional writing style

Essay:"""
            
            content = await self._generate_text(essay_prompt, system_prompt)
        else:
            # Generate general text
            text_prompt = f"""Write content about: {topic if topic else 'general information'}

Content:"""
            content = await self._generate_text(text_prompt, system_prompt)
        
        # Execute file task
        return await self.task_executor.execute_file_task(
            file_path=file_path,
            content=content,
            mode="write"
        )
    
    async def _execute_browser_task(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute browser task"""
        url = task.get("url")
        return await self.task_executor.execute_browser_task(url=url, action="open")
    
    async def _execute_mcp_task(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute MCP task"""
        mcp_server = task.get("mcp_server")
        action_text = task.get("action", "")
        
        # Parse action and params from action_text
        # This is a simplified parser - can be enhanced
        params = {"action_text": action_text}
        
        return await self.mcp_client.execute_mcp_action(
            server_name=mcp_server,
            action="custom",
            params=params
        )
    
    async def _execute_general_task(
        self,
        task: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute general task (AI response)"""
        instruction = task.get("instruction", "")
        
        response = await self._generate_text(instruction, system_prompt)
        
        return {
            "success": True,
            "message": "General task processed",
            "response": response
        }
    
    async def _generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 500
    ) -> str:
        """Generate text using Ollama (optimized for speed)"""
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Use async HTTP if available for better performance
            if self.async_client:
                try:
                    response = await self.async_client.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": full_prompt,
                            "stream": False,
                            "options": {
                                "num_ctx": 2048,  # Reduced from 4096 for speed
                                "temperature": 0.7,
                                "num_predict": max_tokens,  # Limit response length
                            }
                        }
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return result.get('response', '')
                    else:
                        return f"Error generating text: {response.status_code}"
                except Exception as e:
                    print(f"⚠️ Async request failed, falling back to sync: {e}")
                    # Fallback to sync
            
            # Fallback to synchronous requests
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 2048,  # Reduced from 4096 for speed
                        "temperature": 0.7,
                        "num_predict": max_tokens,  # Limit response length
                    }
                },
                timeout=60  # Reduced from 120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                return f"Error generating text: {response.status_code}"
                
        except Exception as e:
            return f"Error generating text: {str(e)}"

