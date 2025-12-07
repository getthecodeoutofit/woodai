"""
Task Parser Module
Parses user input to extract tasks and their parameters
"""

import re
from typing import List, Dict, Any, Optional
import json


class TaskParser:
    """Parses natural language input to extract structured tasks"""
    
    def __init__(self):
        self.task_patterns = {
            "email": [
                r"(?:write|send|compose|create)\s+(?:an?\s+)?email\s+(?:to\s+)?([^\s]+@[^\s]+)",
                r"email\s+(?:to\s+)?([^\s]+@[^\s]+)\s+(?:about|regarding|on)\s+(.+)",
                r"send\s+(?:an?\s+)?email\s+(?:to\s+)?([^\s]+@[^\s]+)",
            ],
            "file": [
                r"(?:create|write|make)\s+(?:a\s+)?python\s+file\s+(?:named\s+)?(?:called\s+)?['\"]?([^\s'\"]+\.py)['\"]?",
                r"(?:create|write|make)\s+(?:a\s+)?python\s+file\s+(?:named\s+)?(?:called\s+)?['\"]?([^\s'\"]+)['\"]?",
                r"(?:write|create)\s+python\s+code\s+(?:for|about|on)\s+(.+?)(?:\s+in\s+file\s+['\"]?([^\s'\"]+\.py)['\"]?)?",
                r"['\"]?([^\s'\"]+\.py)['\"]?\s+(?:with|and)\s+(?:code|python)",
                r"(?:create|write|make)\s+(?:a\s+)?(?:text\s+)?file\s+(?:named\s+)?(?:called\s+)?['\"]?([^\s'\"]+)['\"]?",
                r"(?:write|create)\s+(?:an?\s+)?essay\s+(?:in\s+)?(?:a\s+)?(?:file\s+)?(?:named\s+)?['\"]?([^\s'\"]+)?['\"]?",
                r"save\s+(?:to\s+)?(?:file\s+)?['\"]?([^\s'\"]+)['\"]?",
            ],
            "browser": [
                r"(?:open|visit|go\s+to|navigate\s+to)\s+(?:youtube|https?://[^\s]+|www\.[^\s]+)",
                r"(?:open|visit)\s+['\"]?([^\s'\"']+)['\"]?",
                r"youtube\s+(?:video\s+)?(?:link\s+)?['\"]?([^\s'\"']+)['\"]?",
            ],
            "mcp": [
                r"(?:use|connect\s+to|integrate\s+with)\s+(notion|github|slack|discord)\s+(?:to\s+)?(.+)",
                r"mcp\s+(?:server\s+)?(?:for\s+)?(\w+)\s+(.+)",
            ]
        }
    
    def parse_tasks(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Parse user input to extract tasks
        
        Args:
            user_input: Natural language input from user
        
        Returns:
            List of parsed tasks with their parameters
        """
        tasks = []
        input_lower = user_input.lower()
        
        # Extract file path if specified in format "(save to: path)"
        file_path_override = None
        path_match = re.search(r'\(save\s+to:\s+([^)]+)\)', user_input, re.IGNORECASE)
        if path_match:
            file_path_override = path_match.group(1).strip()
            # Remove the path instruction from input for cleaner parsing
            user_input = re.sub(r'\s*\(save\s+to:[^)]+\)', '', user_input, flags=re.IGNORECASE)
            input_lower = user_input.lower()
        
        # Split by common task separators
        task_segments = re.split(r'\s+then\s+|\s+and\s+then\s+|\s+after\s+that\s+|\n+', user_input, flags=re.IGNORECASE)
        
        for segment in task_segments:
            segment = segment.strip()
            if not segment:
                continue
            
            # Try to match each task type
            task = self._parse_single_task(segment, file_path_override if len(tasks) == 0 else None)
            if task:
                tasks.append(task)
            else:
                # If no specific task pattern matches, treat as general instruction
                tasks.append({
                    "type": "general",
                    "instruction": segment,
                    "raw": segment
                })
        
        return tasks if tasks else [{"type": "general", "instruction": user_input, "raw": user_input}]
    
    def _parse_single_task(self, text: str, file_path_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Parse a single task from text"""
        text_lower = text.lower()
        
        # Email task
        for pattern in self.task_patterns["email"]:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                recipient = match.group(1) if match.lastindex and match.lastindex >= 1 else None
                subject_context = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                
                # Extract subject from context
                subject = self._extract_subject(text, subject_context)
                body_context = self._extract_body_context(text, subject_context)
                
                return {
                    "type": "email",
                    "recipient": recipient,
                    "subject": subject or "No Subject",
                    "body_context": body_context,
                    "raw": text
                }
        
        # File task
        # First, try to extract filename directly from common patterns
        filename_match = re.search(r"(?:file\s+)?(?:named\s+)?(?:called\s+)?['\"]?([^\s'\"]+\.(?:py|txt|md|js|html|css|json))['\"]?", text, re.IGNORECASE)
        explicit_filename = filename_match.group(1) if filename_match else None
        
        for pattern in self.task_patterns["file"]:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                file_path = None
                
                # Use explicitly found filename if available
                if explicit_filename:
                    file_path = explicit_filename
                elif match.lastindex and match.lastindex >= 1:
                    file_path = match.group(1).strip()
                    # Clean up - take only first word if it looks like a filename
                    if ' ' in file_path:
                        # If it contains a dot, it's likely a filename
                        if '.' in file_path:
                            file_path = file_path.split()[0]
                        else:
                            # Otherwise, might be part of a longer phrase
                            words = file_path.split()
                            # Check if first word looks like a filename
                            if len(words[0]) <= 20 and not any(word in words[0].lower() for word in ['about', 'for', 'with', 'and', 'the']):
                                file_path = words[0]
                            else:
                                file_path = None
                
                # Use override path if provided (from frontend dialog)
                if file_path_override:
                    file_path = file_path_override
                # Default filename if not specified
                elif not file_path or file_path == "essay":
                    # Determine default extension based on content type
                    if "python" in text_lower:
                        file_path = f"code_{self._generate_timestamp()}.py"
                    elif "code" in text_lower:
                        file_path = f"code_{self._generate_timestamp()}.py"
                    else:
                        file_path = f"essay_{self._generate_timestamp()}.txt"
                else:
                    # Clean up file path
                    file_path = file_path.strip('"\'')
                    # Preserve file extension if specified, otherwise add appropriate extension
                    # Check for common code file extensions
                    code_extensions = ('.py', '.js', '.ts', '.cpp', '.c', '.h', '.hpp', '.java', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.html', '.css', '.json', '.yaml', '.yml', '.xml', '.sql', '.sh', '.bash')
                    text_extensions = ('.txt', '.md', '.text', '.rst', '.log')
                    
                    if not any(file_path.endswith(ext) for ext in code_extensions + text_extensions):
                        # Determine extension based on content type or language mentioned
                        if "python" in text_lower or file_path.endswith('.py'):
                            file_path = f"{file_path}.py" if not file_path.endswith('.py') else file_path
                        elif "c++" in text_lower or "cpp" in text_lower or file_path.endswith('.cpp'):
                            file_path = f"{file_path}.cpp" if not file_path.endswith('.cpp') else file_path
                        elif "javascript" in text_lower or "js" in text_lower or file_path.endswith('.js'):
                            file_path = f"{file_path}.js" if not file_path.endswith('.js') else file_path
                        elif "code" in text_lower:
                            file_path = f"{file_path}.py"  # Default to Python for generic "code"
                        else:
                            file_path = f"{file_path}.txt"
                    # Don't add .py if file already has another extension
                    elif file_path.endswith(('.cpp', '.c', '.h', '.hpp', '.js', '.ts', '.java', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.html', '.css', '.json', '.xml', '.sql', '.sh', '.bash')):
                        # File already has a code extension, don't modify it
                        pass
                
                # Extract content type based on file extension or language mentioned
                if "python" in text_lower or file_path.endswith('.py'):
                    content_type = "python_code"
                elif "c++" in text_lower or "cpp" in text_lower or file_path.endswith(('.cpp', '.c', '.h', '.hpp')):
                    content_type = "cpp_code"
                elif "javascript" in text_lower or "js" in text_lower or file_path.endswith('.js'):
                    content_type = "js_code"
                elif "code" in text_lower or file_path.endswith(('.js', '.ts', '.java', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.html', '.css', '.sql', '.sh')):
                    content_type = "code"
                elif "essay" in text_lower:
                    content_type = "essay"
                else:
                    content_type = "text"
                
                # Extract topic
                topic = self._extract_topic(text)
                
                return {
                    "type": "file",
                    "file_path": file_path,
                    "content_type": content_type,
                    "topic": topic,
                    "raw": text
                }
        
        # Browser task
        for pattern in self.task_patterns["browser"]:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                url = match.group(1) if match.lastindex and match.lastindex >= 1 else None
                
                # Extract URL from text
                if not url:
                    url_match = re.search(r'(https?://[^\s]+|www\.[^\s]+|youtube\.com/[^\s]+|youtu\.be/[^\s]+)', text, re.IGNORECASE)
                    if url_match:
                        url = url_match.group(1)
                    elif "youtube" in text_lower:
                        url = "https://www.youtube.com"
                
                if url:
                    return {
                        "type": "browser",
                        "url": url,
                        "raw": text
                    }
        
        # MCP task
        for pattern in self.task_patterns["mcp"]:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                mcp_server = match.group(1) if match.lastindex and match.lastindex >= 1 else None
                action_text = match.group(2) if match.lastindex and match.lastindex >= 2 else None
                
                return {
                    "type": "mcp",
                    "mcp_server": mcp_server,
                    "action": action_text,
                    "raw": text
                }
        
        return None
    
    def _extract_subject(self, text: str, context: Optional[str] = None) -> Optional[str]:
        """Extract email subject from text"""
        # Look for "about", "regarding", "on" keywords
        patterns = [
            r"(?:about|regarding|on|subject:)\s+(.+?)(?:\s+then|\s+and|$)",
            r"subject['\"]?\s*[:=]\s*['\"]?([^'\"]+)['\"]?",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        if context:
            return context.strip()
        
        return None
    
    def _extract_body_context(self, text: str, subject_context: Optional[str] = None) -> Optional[str]:
        """Extract email body context from text"""
        # Remove the subject part and get the rest
        if subject_context:
            # Find where subject_context appears and get text after it
            idx = text.lower().find(subject_context.lower())
            if idx != -1:
                return text[idx + len(subject_context):].strip()
        
        # Look for body indicators
        body_patterns = [
            r"(?:body|content|message|text):\s*(.+)",
            r"saying\s+(.+)",
        ]
        
        for pattern in body_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract essay/topic/code topic from text"""
        patterns = [
            r"(?:about|on|topic:|for)\s+(.+?)(?:\s+then|\s+and|$)",
            r"essay\s+(?:about|on)\s+(.+)",
            r"write\s+(?:an?\s+)?essay\s+(?:about|on)\s+(.+)",
            r"(?:python\s+)?code\s+(?:for|about|on)\s+(.+?)(?:\s+in\s+file|\s+then|\s+and|$)",
            r"write\s+(?:python\s+)?code\s+(?:for|about|on)\s+(.+)",
            r"create\s+(?:python\s+)?(?:code\s+)?(?:file\s+)?(?:for|about|on)\s+(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _generate_timestamp(self) -> str:
        """Generate timestamp for default filenames"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

