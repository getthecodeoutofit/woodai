"""
MCP (Model Context Protocol) Client
Framework for integrating with MCP servers like Notion, GitHub, etc.
"""

import json
import requests
from typing import Dict, Any, Optional, List
import os


class MCPClient:
    """Client for interacting with MCP servers"""
    
    def __init__(self):
        self.servers = {}
        self._initialize_servers()
    
    def _initialize_servers(self):
        """Initialize MCP server configurations"""
        # Notion integration
        self.servers["notion"] = {
            "type": "notion",
            "api_key": os.getenv("NOTION_API_KEY", ""),
            "api_url": "https://api.notion.com/v1",
            "enabled": bool(os.getenv("NOTION_API_KEY"))
        }
        
        # GitHub integration
        self.servers["github"] = {
            "type": "github",
            "token": os.getenv("GITHUB_TOKEN", ""),
            "api_url": "https://api.github.com",
            "enabled": bool(os.getenv("GITHUB_TOKEN"))
        }
        
        # Slack integration
        self.servers["slack"] = {
            "type": "slack",
            "token": os.getenv("SLACK_TOKEN", ""),
            "webhook_url": os.getenv("SLACK_WEBHOOK_URL", ""),
            "enabled": bool(os.getenv("SLACK_TOKEN") or os.getenv("SLACK_WEBHOOK_URL"))
        }
    
    async def execute_notion_action(
        self,
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Notion action"""
        if not self.servers["notion"]["enabled"]:
            return {
                "success": False,
                "error": "Notion API key not configured. Set NOTION_API_KEY environment variable."
            }
        
        headers = {
            "Authorization": f"Bearer {self.servers['notion']['api_key']}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        
        try:
            if action == "create_page":
                # Create a new page
                url = f"{self.servers['notion']['api_url']}/pages"
                data = {
                    "parent": {"database_id": params.get("database_id")},
                    "properties": params.get("properties", {})
                }
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": "Notion page created",
                        "data": response.json()
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Notion API error: {response.text}"
                    }
            
            elif action == "read_page":
                # Read a page
                page_id = params.get("page_id")
                url = f"{self.servers['notion']['api_url']}/pages/{page_id}"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "data": response.json()
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Notion API error: {response.text}"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown Notion action: {action}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute_github_action(
        self,
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute GitHub action"""
        if not self.servers["github"]["enabled"]:
            return {
                "success": False,
                "error": "GitHub token not configured. Set GITHUB_TOKEN environment variable."
            }
        
        headers = {
            "Authorization": f"token {self.servers['github']['token']}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            if action == "create_issue":
                # Create an issue
                repo = params.get("repo")
                url = f"{self.servers['github']['api_url']}/repos/{repo}/issues"
                data = {
                    "title": params.get("title"),
                    "body": params.get("body", "")
                }
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 201:
                    return {
                        "success": True,
                        "message": "GitHub issue created",
                        "data": response.json()
                    }
                else:
                    return {
                        "success": False,
                        "error": f"GitHub API error: {response.text}"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown GitHub action: {action}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute_slack_action(
        self,
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Slack action"""
        webhook_url = self.servers["slack"]["webhook_url"]
        
        if not webhook_url:
            return {
                "success": False,
                "error": "Slack webhook URL not configured. Set SLACK_WEBHOOK_URL environment variable."
            }
        
        try:
            if action == "send_message":
                # Send a message via webhook
                data = {
                    "text": params.get("message", "")
                }
                response = requests.post(webhook_url, json=data)
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": "Slack message sent"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Slack API error: {response.text}"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown Slack action: {action}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def execute_mcp_action(
        self,
        server_name: str,
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute action on specified MCP server"""
        if server_name not in self.servers:
            return {
                "success": False,
                "error": f"Unknown MCP server: {server_name}"
            }
        
        server = self.servers[server_name]
        
        if not server["enabled"]:
            return {
                "success": False,
                "error": f"{server_name} is not configured. Please set the required environment variables."
            }
        
        # Route to appropriate handler
        if server_name == "notion":
            return await self.execute_notion_action(action, params)
        elif server_name == "github":
            return await self.execute_github_action(action, params)
        elif server_name == "slack":
            return await self.execute_slack_action(action, params)
        else:
            return {
                "success": False,
                "error": f"No handler for server: {server_name}"
            }
    
    def get_available_servers(self) -> List[str]:
        """Get list of available and configured MCP servers"""
        return [name for name, config in self.servers.items() if config["enabled"]]



