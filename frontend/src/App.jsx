import React, { useState, useRef, useEffect } from 'react';
import { Menu, Settings, History, MessageSquare, Brain, Workflow, Send, Plus, ChevronLeft, Moon, Sun, Trash2, User, Paperclip, Camera, X, Download, Copy, Check, Volume2, VolumeX, Maximize2, Minimize2, RefreshCw, Zap, FileText, Image as ImageIcon, FolderOpen, Save } from 'lucide-react';

export default function AIChat() {
  // Initialize sidebar state based on screen size
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    // Default to closed on mobile, open on desktop
    if (typeof window !== 'undefined') {
      return window.innerWidth >= 768;
    }
    return true;
  });
  const [darkMode, setDarkMode] = useState(false);
  const [activeMode, setActiveMode] = useState('rag');
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m your AI assistant. How can I help you today?', timestamp: new Date().toLocaleTimeString() }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [chatHistory, setChatHistory] = useState([
    { id: 1, title: 'Getting Started', timestamp: '2 hours ago', preview: 'How do I use RAG mode?' },
    { id: 2, title: 'Project Discussion', timestamp: 'Yesterday', preview: 'Help me with my React project' },
    { id: 3, title: 'Code Review', timestamp: '2 days ago', preview: 'Review this Python code' }
  ]);
  const [activeView, setActiveView] = useState('chat');
  const [contextLength, setContextLength] = useState(4096);
  const [memoryEnabled] = useState(true);
  const [temperature, setTemperature] = useState(0.7);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploadingDocument, setIsUploadingDocument] = useState(false);
  const [uploadingFileName, setUploadingFileName] = useState('');
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [showAttachMenu, setShowAttachMenu] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showSystemPrompt, setShowSystemPrompt] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful AI assistant.');
  const [searchQuery, setSearchQuery] = useState('');
  const [tokenCount, setTokenCount] = useState({ input: 0, output: 0, total: 0 });
  const [exportFormat, setExportFormat] = useState('txt');
  const [userId] = useState('default_user'); // Or get from auth
  const [currentChatId, setCurrentChatId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [_showDocuments, _setShowDocuments] = useState(false);
  const [showFileLocationDialog, setShowFileLocationDialog] = useState(false);
  const [pendingMessage, setPendingMessage] = useState(null);
  const [fileLocation, setFileLocation] = useState({ filename: '', directory: 'agent_outputs' });
  const [currentRagModel, setCurrentRagModel] = useState('gemma3:4b');
  const [currentAgentModel, setCurrentAgentModel] = useState('gemma3:4b');
  const [availableModels, setAvailableModels] = useState([]);
  const [showModelSelector, setShowModelSelector] = useState(false);
  const [downloadingModel, setDownloadingModel] = useState(null);
  const [downloadProgress, setDownloadProgress] = useState('');
  const [apiBaseUrl] = useState(() => {
    // API base URL - detect automatically or use environment variable
    if (import.meta.env.VITE_API_URL) {
      return import.meta.env.VITE_API_URL;
    }
    // Auto-detect: if on mobile/network, use the host's IP, otherwise localhost
    if (typeof window !== 'undefined') {
      const hostname = window.location.hostname;
      if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return 'http://localhost:8000';
      }
      // For mobile devices on same network, use the current host with port 8000
      return `http://${hostname}:8000`;
    }
    return 'http://localhost:8000';
  });
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);
  const chatContainerRef = useRef(null);
  const streamingMessageRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handle window resize to auto-close sidebar on mobile
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      } else {
        setSidebarOpen(true);
      }
    };

    window.addEventListener('resize', handleResize);
    // Initial check
    handleResize();

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Load chat history from backend
  useEffect(() => {
    loadChatHistory();
    loadDocuments();
    loadModels();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load models from backend
  const loadModels = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/models`);
      const data = await response.json();
      
      if (data.rag_model) {
        setCurrentRagModel(data.rag_model);
      }
      if (data.agent_model) {
        setCurrentAgentModel(data.agent_model);
      }
      if (data.available_models) {
        setAvailableModels(data.available_models);
      }
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  // Switch model
  const switchModel = async (modelName, mode) => {
    try {
      const response = await fetch(`${apiBaseUrl}/models/switch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: modelName,
          mode: mode
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        if (mode === 'rag') {
          setCurrentRagModel(modelName);
        } else {
          setCurrentAgentModel(modelName);
        }
        
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `✅ Switched ${mode} model to ${modelName}`,
          timestamp: new Date().toLocaleTimeString()
        }]);
        
        // Reload models to update available list
        loadModels();
      } else {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `❌ Failed to switch model: ${data.detail || 'Unknown error'}`,
          timestamp: new Date().toLocaleTimeString()
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Error switching model: ${error.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    }
  };

  // Download model
  const downloadModel = async (modelName) => {
    setDownloadingModel(modelName);
    setDownloadProgress('Starting download...');
    
    try {
      const response = await fetch(`${apiBaseUrl}/models/download`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: modelName
        })
      });
      
      const data = await response.json();
      
      if (data.success) {
        setDownloadProgress('Download completed!');
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `✅ Model ${modelName} downloaded successfully!`,
          timestamp: new Date().toLocaleTimeString()
        }]);
        
        // Reload models to update available list
        setTimeout(() => {
          loadModels();
          setDownloadingModel(null);
          setDownloadProgress('');
        }, 1000);
      } else {
        setDownloadProgress(`Error: ${data.error || 'Download failed'}`);
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `❌ Failed to download model: ${data.error || 'Unknown error'}`,
          timestamp: new Date().toLocaleTimeString()
        }]);
        setTimeout(() => {
          setDownloadingModel(null);
          setDownloadProgress('');
        }, 3000);
      }
    } catch (error) {
      setDownloadProgress(`Error: ${error.message}`);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Error downloading model: ${error.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
      setTimeout(() => {
        setDownloadingModel(null);
        setDownloadProgress('');
      }, 3000);
    }
  };

  // Estimate token count (rough approximation)
  useEffect(() => {
    const calculateTokens = () => {
      const inputTokens = Math.ceil(inputValue.length / 4);
      const outputTokens = messages.reduce((acc, msg) => 
        acc + Math.ceil(msg.content.length / 4), 0
      );
      setTokenCount({
        input: inputTokens,
        output: outputTokens,
        total: inputTokens + outputTokens
      });
    };
    calculateTokens();
  }, [messages, inputValue]);

  // Detect if message contains file creation task
  const isFileCreationTask = (message) => {
    if (activeMode !== 'agent') return false;
    const lower = message.toLowerCase();
    const fileKeywords = ['create', 'write', 'make'].some(kw => lower.includes(kw));
    const fileTypeKeywords = ['file', 'python', 'code', 'essay', 'text'].some(kw => lower.includes(kw));
    return fileKeywords && fileTypeKeywords;
  };

  // Extract suggested filename from message
  const extractSuggestedFilename = (message) => {
    // Try to find filename in quotes or after "called" or "named"
    const patterns = [
      /(?:called|named)\s+['"]?([^\s'"]+\.(?:py|txt|md|js|html|css|json))['"]?/i,
      /['"]([^\s'"]+\.(?:py|txt|md|js|html|css|json))['"]/i,
      /(?:file|file\s+called)\s+['"]?([^\s'"]+\.(?:py|txt|md|js|html|css|json))['"]?/i,
    ];
    
    for (const pattern of patterns) {
      const match = message.match(pattern);
      if (match && match[1]) {
        return match[1];
      }
    }
    
    // Default based on content type
    if (message.toLowerCase().includes('python') || message.toLowerCase().includes('.py')) {
      return 'code.py';
    }
    return 'output.txt';
  };

  const handleSend = async (customMessage = null, customFilePath = null) => {
    const messageToSend = customMessage || inputValue;
    
    if (messageToSend.trim() || attachedFiles.length > 0) {
      // Check if this is a file creation task and show dialog
      if (isFileCreationTask(messageToSend) && !customFilePath && activeMode === 'agent') {
        const suggestedFilename = extractSuggestedFilename(messageToSend);
        setFileLocation({ filename: suggestedFilename, directory: 'agent_outputs' });
        setPendingMessage(messageToSend);
        setShowFileLocationDialog(true);
        return;
      }

      const userMessage = {
        role: 'user',
        content: messageToSend,
        files: attachedFiles,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setMessages(prev => [...prev, userMessage]);
      if (!customMessage) {
        setInputValue('');
        setAttachedFiles([]);
      }
      setIsLoading(true);

      // Store assistantMessageIndex in a ref for use in catch block
      const assistantIndexRef = { current: undefined };

      try {
        // Build file path if provided
        let finalMessage = messageToSend;
        if (customFilePath) {
          const filePath = customFilePath.directory 
            ? `${customFilePath.directory}/${customFilePath.filename}`
            : customFilePath.filename;
          // Append file path instruction to message
          finalMessage = `${messageToSend} (save to: ${filePath})`;
        }

        // Add placeholder assistant message for streaming
        let assistantMessageIndex;
        setMessages(prev => {
          assistantMessageIndex = prev.length; // Index after user message
          return [...prev, {
            role: 'assistant',
            content: '',
            timestamp: new Date().toLocaleTimeString()
          }];
        });
        
        // Store assistantMessageIndex in ref for use in catch block
        assistantIndexRef.current = assistantMessageIndex;

        // Use streaming endpoint
        // Get updated messages after adding user and assistant messages
        const updatedMessages = [...messages, userMessage];
        const response = await fetch(`${apiBaseUrl}/chat/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: finalMessage,
            mode: activeMode,
            context_length: contextLength,
            memory_enabled: memoryEnabled,
            temperature: temperature,
            system_prompt: systemPrompt,
            history: updatedMessages,
            user_id: userId,
            chat_id: currentChatId,
            selected_doc_ids: selectedDocuments.length > 0 ? selectedDocuments : null
          })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullResponse = '';
        streamingMessageRef.current = { index: assistantMessageIndex, content: '' };

        // Function to update message immediately
        const updateMessage = (content) => {
          setMessages(prev => {
            const updated = [...prev];
            if (assistantMessageIndex !== undefined && updated[assistantMessageIndex]) {
              updated[assistantMessageIndex] = {
                ...updated[assistantMessageIndex],
                content: content
              };
            }
            return updated;
          });
          // Force scroll to bottom
          setTimeout(() => scrollToBottom(), 0);
        };

        while (true) {
          const { done, value } = await reader.read();
          
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.trim() && line.startsWith('data: ')) {
              try {
                const jsonStr = line.slice(6).trim(); // Remove 'data: ' prefix
                if (!jsonStr) continue;
                
                const data = JSON.parse(jsonStr);
                const chunk = data.chunk || '';
                
                if (chunk) {
                  fullResponse += chunk;
                  streamingMessageRef.current.content = fullResponse;
                  // Update immediately for real-time display
                  updateMessage(fullResponse);
                }

                // Handle completion
                if (data.done) {
                  // Update current chat ID if new chat was created
                  if (data.chat_id && !currentChatId) {
                    setCurrentChatId(data.chat_id);
                  }
                  
                  // Refresh chat history
                  loadChatHistory();

                  // Text-to-speech if enabled
                  if (voiceEnabled && 'speechSynthesis' in window && fullResponse) {
                    const utterance = new SpeechSynthesisUtterance(fullResponse);
                    window.speechSynthesis.speak(utterance);
                  }
                  
                  streamingMessageRef.current = null;
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e, 'Line:', line);
              }
            }
          }
        }
      } catch (error) {
        console.error('Streaming error:', error);
        setMessages(prev => {
          const updated = [...prev];
          // Update the assistant message with error
          const assistantIndex = assistantIndexRef?.current;
          if (assistantIndex !== undefined && updated[assistantIndex]) {
            updated[assistantIndex] = {
              role: 'assistant',
              content: `Sorry, I couldn't connect to the server. Please make sure the backend is running on ${apiBaseUrl}`,
              timestamp: new Date().toLocaleTimeString()
            };
          } else {
            // Fallback: add error message
            updated.push({
              role: 'assistant',
              content: `Sorry, I couldn't connect to the server. Please make sure the backend is running on ${apiBaseUrl}`,
              timestamp: new Date().toLocaleTimeString()
            });
          }
          return updated;
        });
        streamingMessageRef.current = null;
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleFileLocationConfirm = () => {
    if (fileLocation.filename.trim()) {
      setShowFileLocationDialog(false);
      handleSend(pendingMessage, fileLocation);
      setPendingMessage(null);
      setFileLocation({ filename: '', directory: 'agent_outputs' });
    }
  };

  const handleFileLocationCancel = () => {
    setShowFileLocationDialog(false);
    setPendingMessage(null);
    setFileLocation({ filename: '', directory: 'agent_outputs' });
  };

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files);
    
    files.forEach(file => {
      // Check if it's a document that should be indexed
      const docExtensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.txt', '.csv'];
      const fileExt = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
      
      if (docExtensions.includes(fileExt)) {
        // Upload to backend for indexing
        handleDocumentUpload(file);
      } else {
        // Just attach to message
        setAttachedFiles(prev => [...prev, { 
          name: file.name, 
          type: 'document', 
          size: (file.size / 1024).toFixed(2) + ' KB',
          file: file 
        }]);
      }
    });
    
    setShowAttachMenu(false);
  };

  const handleCameraCapture = (e) => {
    const files = Array.from(e.target.files);
    setAttachedFiles(prev => [...prev, ...files.map(f => ({ 
      name: f.name, 
      type: 'image',
      size: (f.size / 1024).toFixed(2) + ' KB',
      file: f 
    }))]);
    setShowAttachMenu(false);
  };

  const removeFile = (index) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleNewChat = () => {
    setMessages([{ 
      role: 'assistant', 
      content: 'Hello! I\'m your AI assistant. How can I help you today?', 
      timestamp: new Date().toLocaleTimeString() 
    }]);
    setCurrentChatId(null);
    setSelectedDocuments([]); // Clear selected documents for new chat
    setActiveView('chat');
  };

  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const downloadMessage = (message, index) => {
    const blob = new Blob([message.content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `message-${index}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportChat = () => {
    let content = '';
    const timestamp = new Date().toLocaleString();
    
    if (exportFormat === 'txt') {
      content = `Chat Export - ${timestamp}\n${'='.repeat(50)}\n\n`;
      messages.forEach(msg => {
        content += `[${msg.timestamp}] ${msg.role.toUpperCase()}:\n${msg.content}\n\n`;
      });
    } else if (exportFormat === 'json') {
      content = JSON.stringify({ timestamp, messages }, null, 2);
    } else if (exportFormat === 'md') {
      content = `# Chat Export\n**Date:** ${timestamp}\n\n`;
      messages.forEach(msg => {
        content += `### ${msg.role === 'user' ? '👤 User' : '🤖 Assistant'} (${msg.timestamp})\n${msg.content}\n\n`;
      });
    }

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${Date.now()}.${exportFormat}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleFullscreen = () => {
    if (!isFullscreen) {
      chatContainerRef.current?.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
    setIsFullscreen(!isFullscreen);
  };

  const regenerateResponse = async () => {
    if (messages.length < 2) return;
    
    const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
    if (!lastUserMessage) return;

    // Remove last assistant message
    setMessages(prev => prev.slice(0, -1));
    setIsLoading(true);

    try {
      const response = await fetch(`${apiBaseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: lastUserMessage.content,
          mode: activeMode,
          context_length: contextLength,
          memory_enabled: memoryEnabled,
          temperature: temperature + 0.1,
          system_prompt: systemPrompt,
          history: messages.slice(0, -1)
        })
      });

      const data = await response.json();
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Failed to regenerate response.',
        timestamp: new Date().toLocaleTimeString()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Load chat history from backend
  const loadChatHistory = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/chats/${userId}`);
      const data = await response.json();
      
      if (data.chats) {
        setChatHistory(data.chats.map(chat => ({
          id: chat._id,
          title: chat.title,
          timestamp: new Date(chat.updated_at).toLocaleDateString(),
          preview: chat.preview,
          messages: chat.messages
        })));
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  // Load specific chat when clicked
  const loadChat = async (chatId) => {
    try {
      const response = await fetch(`${apiBaseUrl}/chat/${chatId}`);
      const chat = await response.json();
      
      if (chat && chat.messages) {
        setMessages(chat.messages);
        setActiveView('chat');
        setCurrentChatId(chatId);
        
        // Restore selected documents if they exist
        if (chat.selected_doc_ids && chat.selected_doc_ids.length > 0) {
          setSelectedDocuments(chat.selected_doc_ids);
          // Only show restore message if chat has messages (not a new chat)
          if (chat.messages && chat.messages.length > 1) {
            setMessages(prev => [...prev, {
              role: 'assistant',
              content: `📚 Restored ${chat.selected_doc_ids.length} document${chat.selected_doc_ids.length > 1 ? 's' : ''} from this chat's context.`,
              timestamp: new Date().toLocaleTimeString()
            }]);
          }
        } else {
          setSelectedDocuments([]);
        }
      }
    } catch (error) {
      console.error('Failed to load chat:', error);
    }
  };

  // Delete chat
  const deleteChat = async (chatId, e) => {
    e.stopPropagation(); // Prevent click from triggering loadChat
    
    if (!confirm('Delete this chat?')) return;
    
    try {
      const response = await fetch(`${apiBaseUrl}/chat/${chatId}`, {
        method: 'DELETE'
      });
      
      if (response.ok) {
        setChatHistory(prev => prev.filter(chat => chat.id !== chatId));
        
        // If deleted chat is currently active, start new chat
        if (currentChatId === chatId) {
          handleNewChat();
        }
      }
    } catch (error) {
      console.error('Failed to delete chat:', error);
    }
  };

  // Load documents from backend
  const loadDocuments = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/documents`);
      const data = await response.json();
      
      if (data.documents) {
        setDocuments(data.documents);
      }
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  // Upload document
  const handleDocumentUpload = async (file) => {
    setIsUploadingDocument(true);
    setUploadingFileName(file.name);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('user_id', userId);
      
      const response = await fetch(`${apiBaseUrl}/upload`, {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `✅ Document "${data.filename}" uploaded and indexed successfully!\n\n` +
                   `- Chunks created: ${data.chunks}\n` +
                   `- The AI can now use this document in RAG mode.`,
          timestamp: new Date().toLocaleTimeString()
        }]);
        
        // Reload documents list
        loadDocuments();
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Failed to upload document: ${error.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    } finally {
      setIsUploadingDocument(false);
      setUploadingFileName('');
    }
  };

  // Delete document
  const deleteDocument = async (docId, filename) => {
    if (!confirm(`Delete document "${filename}"? This will remove all its context from the knowledge base.`)) {
      return;
    }
    
    try {
      const response = await fetch(`${apiBaseUrl}/document/${docId}`, {
        method: 'DELETE'
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Remove from documents list
        setDocuments(prev => prev.filter(doc => doc.doc_id !== docId));
        
        // Remove from selected documents if selected
        setSelectedDocuments(prev => prev.filter(id => id !== docId));
        
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `✅ Document "${filename}" deleted successfully!`,
          timestamp: new Date().toLocaleTimeString()
        }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Failed to delete document: ${error.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    }
  };

  // Toggle document selection
  const toggleDocumentSelection = (docId) => {
    setSelectedDocuments(prev => {
      if (prev.includes(docId)) {
        return prev.filter(id => id !== docId);
      } else {
        return [...prev, docId];
      }
    });
  };

  const filteredHistory = chatHistory.filter(chat => 
    chat.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    chat.preview.toLowerCase().includes(searchQuery.toLowerCase())
  );

  

  return (
    <div 
      ref={chatContainerRef}
      className={`flex ${darkMode ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900' : 'bg-gradient-to-br from-amber-50 via-orange-50 to-amber-100'} overflow-hidden transition-all duration-500`}
      style={{
        height: '100dvh', // Dynamic viewport height for better mobile support
        minHeight: '-webkit-fill-available',
        maxHeight: '100dvh',
        paddingTop: 'env(safe-area-inset-top, 0px)',
        paddingBottom: 'env(safe-area-inset-bottom, 0px)'
      }}
    >
      
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-30 md:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}
      
      {/* Sidebar */}
      <div 
        className={`${
          sidebarOpen 
            ? 'w-80 md:w-80 translate-x-0' 
            : 'w-0 -translate-x-full md:translate-x-0'
        } fixed md:relative z-40 md:z-auto transition-all duration-300 ${darkMode ? 'glass-effect-dark md:bg-gray-800/40' : 'glass-effect md:bg-gradient-to-b md:from-amber-900/20 md:to-orange-900/30'} border-r ${darkMode ? 'border-gray-700/50' : 'border-amber-800/20'} overflow-hidden rounded-r-3xl md:rounded-none flex-shrink-0`}
        style={{
          boxShadow: sidebarOpen ? '4px 0 20px rgba(0,0,0,0.1)' : 'none',
          height: '100%',
          maxHeight: '100dvh'
        }}
        aria-hidden={!sidebarOpen}
      >
        <div className="h-full flex flex-col p-4 overflow-y-auto">
          {/* Logo & New Chat */}
          <div className="mb-6 space-y-3">
            <div className="flex items-center justify-between">
              <h1 className={`text-2xl font-bold ${darkMode ? 'text-amber-400' : 'text-amber-900'} tracking-tight`}>
                WoodAI
              </h1>
              <button
                onClick={handleNewChat}
                className={`p-2 rounded-lg ${darkMode ? 'bg-amber-600/20 hover:bg-amber-600/30 text-amber-400' : 'bg-white/60 hover:bg-white/80 text-amber-900'} transition-all duration-200 backdrop-blur-sm`}
              >
                <Plus size={20} />
              </button>
            </div>
          </div>

          {/* Mode Selection */}
          <div className={`mb-6 p-3 rounded-xl ${darkMode ? 'bg-gray-700/40' : 'bg-white/50'} backdrop-blur-sm border ${darkMode ? 'border-gray-600/30' : 'border-amber-800/10'}`}>
            <p className={`text-xs font-semibold ${darkMode ? 'text-gray-400' : 'text-amber-800'} mb-2 uppercase tracking-wider`}>AI Mode</p>
            <div className="space-y-2">
              <button
                onClick={() => setActiveMode('rag')}
                className={`w-full p-3 rounded-lg flex items-center gap-3 transition-all duration-200 ${
                  activeMode === 'rag' 
                    ? darkMode ? 'bg-amber-600/30 text-amber-300 shadow-lg' : 'bg-amber-600/80 text-white shadow-lg'
                    : darkMode ? 'bg-gray-600/20 text-gray-300 hover:bg-gray-600/30' : 'bg-white/40 text-amber-900 hover:bg-white/60'
                }`}
              >
                <Brain size={20} />
                <div className="text-left flex-1">
                  <div className="font-semibold text-sm">RAG Mode</div>
                  <div className={`text-xs ${activeMode === 'rag' ? 'opacity-90' : 'opacity-60'} truncate`} title={currentRagModel}>
                    {currentRagModel}
                  </div>
                </div>
                <button
                  onClick={() => setShowModelSelector(true)}
                  className={`p-1.5 rounded ${activeMode === 'rag' ? (darkMode ? 'hover:bg-amber-700/40' : 'hover:bg-amber-700/20') : (darkMode ? 'hover:bg-gray-600/40' : 'hover:bg-white/60')} transition-all`}
                  title="Change model"
                >
                  <Settings size={14} />
                </button>
              </button>
              
              <button
                onClick={() => setActiveMode('agent')}
                className={`w-full p-3 rounded-lg flex items-center gap-3 transition-all duration-200 ${
                  activeMode === 'agent' 
                    ? darkMode ? 'bg-amber-600/30 text-amber-300 shadow-lg' : 'bg-amber-600/80 text-white shadow-lg'
                    : darkMode ? 'bg-gray-600/20 text-gray-300 hover:bg-gray-600/30' : 'bg-white/40 text-amber-900 hover:bg-white/60'
                }`}
              >
                <Workflow size={20} />
                <div className="text-left flex-1">
                  <div className="font-semibold text-sm">Agent Mode</div>
                  <div className={`text-xs ${activeMode === 'agent' ? 'opacity-90' : 'opacity-60'} truncate`} title={currentAgentModel}>
                    {currentAgentModel}
                  </div>
                </div>
                <button
                  onClick={() => setShowModelSelector(true)}
                  className={`p-1.5 rounded ${activeMode === 'agent' ? (darkMode ? 'hover:bg-amber-700/40' : 'hover:bg-amber-700/20') : (darkMode ? 'hover:bg-gray-600/40' : 'hover:bg-white/60')} transition-all`}
                  title="Change model"
                >
                  <Settings size={14} />
                </button>
              </button>
            </div>
          </div>

          {/* Navigation */}
          <nav className="space-y-1 mb-6">
            {[
              { id: 'chat', icon: MessageSquare, label: 'Chat' },
              { id: 'history', icon: History, label: 'History' },
              { id: 'documents', icon: FileText, label: 'Documents' },
              { id: 'settings', icon: Settings, label: 'Settings' }
            ].map(item => (
              <button
                key={item.id}
                onClick={() => setActiveView(item.id)}
                className={`w-full p-3 rounded-lg flex items-center gap-3 transition-all duration-200 ${
                  activeView === item.id
                    ? darkMode ? 'bg-gray-700/60 text-amber-400' : 'bg-white/60 text-amber-900'
                    : darkMode ? 'text-gray-400 hover:bg-gray-700/40' : 'text-amber-800/70 hover:bg-white/40'
                }`}
              >
                <item.icon size={20} />
                <span className="font-medium">{item.label}</span>
              </button>
            ))}
          </nav>

          {/* Chat History */}
          {activeView === 'history' && (
            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
              {/* Sticky Search Bar */}
              <div className="flex-shrink-0 mb-3 sticky top-0 z-10 bg-inherit pb-2">
                <input
                  type="text"
                  placeholder="Search chats..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className={`w-full px-3 py-2 rounded-lg ${darkMode ? 'bg-gray-700/40 text-gray-200 placeholder-gray-400' : 'bg-white/40 text-amber-900 placeholder-amber-700/50'} backdrop-blur-sm border ${darkMode ? 'border-gray-600/30' : 'border-amber-800/10'} outline-none focus:ring-2 ${darkMode ? 'focus:ring-amber-500/50' : 'focus:ring-amber-500/30'} transition-all`}
                />
              </div>
              
              {/* Scrollable Chat List */}
              <div className="flex-1 overflow-y-auto min-h-0">
                <p className={`text-xs font-semibold ${darkMode ? 'text-gray-400' : 'text-amber-800'} mb-3 uppercase tracking-wider sticky top-0 ${darkMode ? 'bg-gray-800/95' : 'bg-amber-50/95'} backdrop-blur-sm py-1 z-10`}>Recent Chats</p>
                <div className="space-y-2">
                  {filteredHistory.map(chat => (
                  <div
                    key={chat.id}
                    onClick={() => loadChat(chat.id)}
                    className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700/40 hover:bg-gray-700/60' : 'bg-white/40 hover:bg-white/60'} cursor-pointer transition-all duration-200 backdrop-blur-sm group relative`}
                  >
                    <div className={`font-medium text-sm ${darkMode ? 'text-gray-200' : 'text-amber-900'} mb-1`}>
                      {chat.title}
                    </div>
                    <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'} mb-1 opacity-70`}>
                      {chat.preview}
                    </div>
                    <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-amber-600'}`}>
                      {chat.timestamp}
                    </div>
                    
                    {/* Delete button */}
                    <button
                      onClick={(e) => deleteChat(chat.id, e)}
                      className={`absolute top-2 right-2 p-1.5 rounded opacity-0 group-hover:opacity-100 transition-opacity ${darkMode ? 'bg-red-600/20 hover:bg-red-600/30 text-red-400' : 'bg-red-600/20 hover:bg-red-600/30 text-red-600'}`}
                      title="Delete chat"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Documents Management */}
          {activeView === 'documents' && (
            <div className="flex-1 overflow-y-auto">
              <p className={`text-xs font-semibold ${darkMode ? 'text-gray-400' : 'text-amber-800'} mb-3 uppercase tracking-wider`}>Uploaded Documents</p>
              
              {documents.length === 0 ? (
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700/40' : 'bg-white/40'} backdrop-blur-sm text-center`}>
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>
                    No documents uploaded yet. Upload a PDF or document to get started.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {documents.map(doc => (
                    <div
                      key={doc.doc_id}
                      className={`p-3 rounded-lg ${darkMode ? 'bg-gray-700/40' : 'bg-white/40'} backdrop-blur-sm group relative`}
                    >
                      <div className="flex items-start gap-3">
                        <input
                          type="checkbox"
                          checked={selectedDocuments.includes(doc.doc_id)}
                          onChange={() => toggleDocumentSelection(doc.doc_id)}
                          className={`mt-1 w-4 h-4 rounded ${darkMode ? 'bg-gray-600 border-gray-500' : 'bg-white border-amber-300'} cursor-pointer`}
                          title="Select to add context to chat"
                        />
                        <div className="flex-1 min-w-0">
                          <div className={`font-medium text-sm ${darkMode ? 'text-gray-200' : 'text-amber-900'} mb-1 truncate`}>
                            {doc.filename}
                          </div>
                          <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>
                            {doc.chunk_count || 0} chunks • {new Date(doc.created_at).toLocaleDateString()}
                          </div>
                        </div>
                        <button
                          onClick={() => deleteDocument(doc.doc_id, doc.filename)}
                          className={`p-1.5 rounded opacity-0 group-hover:opacity-100 transition-opacity ${darkMode ? 'bg-red-600/20 hover:bg-red-600/30 text-red-400' : 'bg-red-600/20 hover:bg-red-600/30 text-red-600'}`}
                          title="Delete document"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              {selectedDocuments.length > 0 && (
                <div className={`mt-4 p-3 rounded-lg ${darkMode ? 'bg-amber-600/20 border border-amber-600/30' : 'bg-amber-100/60 border border-amber-300/50'}`}>
                  <p className={`text-xs font-semibold ${darkMode ? 'text-amber-400' : 'text-amber-800'} mb-2`}>
                    {selectedDocuments.length} document{selectedDocuments.length > 1 ? 's' : ''} selected
                  </p>
                  <button
                    onClick={() => {
                      setMessages(prev => [...prev, {
                        role: 'assistant',
                        content: `✅ Added ${selectedDocuments.length} document${selectedDocuments.length > 1 ? 's' : ''} to context. The AI will now use these documents when answering questions.`,
                        timestamp: new Date().toLocaleTimeString()
                      }]);
                      setActiveView('chat');
                    }}
                    className={`w-full px-3 py-2 rounded-lg text-sm font-medium ${darkMode ? 'bg-amber-600/30 hover:bg-amber-600/40 text-amber-300' : 'bg-amber-600/80 hover:bg-amber-600/90 text-white'} transition-all`}
                  >
                    Use in Chat
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Settings */}
          {activeView === 'settings' && (
            <div className="flex-1 overflow-y-auto space-y-4">
              <p className={`text-xs font-semibold ${darkMode ? 'text-gray-400' : 'text-amber-800'} mb-3 uppercase tracking-wider`}>Settings</p>
              
              {/* Dark Mode */}
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700/40' : 'bg-white/40'} backdrop-blur-sm`}>
                <div className="flex items-center justify-between">
                  <span className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-amber-900'}`}>Dark Mode</span>
                  <button
                    onClick={() => setDarkMode(!darkMode)}
                    className={`p-2 rounded-lg ${darkMode ? 'bg-amber-600/20 text-amber-400' : 'bg-amber-600/80 text-white'} transition-all duration-200`}
                  >
                    {darkMode ? <Moon size={16} /> : <Sun size={16} />}
                  </button>
                </div>
              </div>

              {/* Context Length */}
              <div className={`p-3 md:p-4 rounded-lg ${darkMode ? 'bg-gray-700/40' : 'bg-white/40'} backdrop-blur-sm`}>
                <div className="flex items-center justify-between mb-3">
                  <span className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-amber-900'}`}>Context Length</span>
                  <span className={`text-xs md:text-sm ${darkMode ? 'text-amber-400' : 'text-amber-600'} font-semibold`}>{contextLength.toLocaleString()}</span>
                </div>
                <input 
                  type="range" 
                  min="1024" 
                  max="8192" 
                  step="1024"
                  value={contextLength}
                  onChange={(e) => setContextLength(parseInt(e.target.value))}
                  className="w-full h-3 md:h-2 rounded-lg appearance-none cursor-pointer touch-manipulation slider-mobile"
                  style={{
                    background: darkMode 
                      ? `linear-gradient(to right, #f59e0b ${(contextLength - 1024) / 7168 * 100}%, #374151 ${(contextLength - 1024) / 7168 * 100}%)`
                      : `linear-gradient(to right, #92400e ${(contextLength - 1024) / 7168 * 100}%, #fed7aa ${(contextLength - 1024) / 7168 * 100}%)`
                  }}
                  aria-label="Context length slider"
                />
                <div className="flex justify-between mt-2">
                  <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>1K</span>
                  <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>8K</span>
                </div>
                <p className={`text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-amber-600/70'}`}>
                  Controls how much context the model can use
                </p>
              </div>

              {/* Temperature */}
              <div className={`p-3 md:p-4 rounded-lg ${darkMode ? 'bg-gray-700/40' : 'bg-white/40'} backdrop-blur-sm`}>
                <div className="flex items-center justify-between mb-3">
                  <span className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-amber-900'}`}>Temperature</span>
                  <span className={`text-xs md:text-sm ${darkMode ? 'text-amber-400' : 'text-amber-600'} font-semibold`}>{temperature.toFixed(1)}</span>
                </div>
                <input 
                  type="range" 
                  min="0" 
                  max="2" 
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-3 md:h-2 rounded-lg appearance-none cursor-pointer touch-manipulation slider-mobile"
                  style={{
                    background: darkMode 
                      ? `linear-gradient(to right, #f59e0b ${temperature / 2 * 100}%, #374151 ${temperature / 2 * 100}%)`
                      : `linear-gradient(to right, #92400e ${temperature / 2 * 100}%, #fed7aa ${temperature / 2 * 100}%)`
                  }}
                  aria-label="Temperature slider"
                />
                <div className="flex justify-between mt-2">
                  <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>Concise</span>
                  <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>Very Detailed</span>
                </div>
                <p className={`text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-amber-600/70'}`}>
                  Lower = concise, Higher = very detailed responses
                </p>
              </div>

              {/* Memory Control */}
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700/40' : 'bg-white/40'} backdrop-blur-sm`}>
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <div className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-amber-900'}`}>Conversation Memory</div>
                    <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'} mt-1`}>
                      Read responses aloud
                    </div>
                  </div>
                  <button
                    onClick={() => setVoiceEnabled(!voiceEnabled)}
                    className={`p-2 rounded-lg ${voiceEnabled ? (darkMode ? 'bg-amber-600/20 text-amber-400' : 'bg-amber-600/80 text-white') : (darkMode ? 'bg-gray-600/40 text-gray-400' : 'bg-gray-400/60 text-gray-700')} transition-all duration-200`}
                  >
                    {voiceEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
                  </button>
                </div>
              </div>

              {/* System Prompt */}
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700/40' : 'bg-white/40'} backdrop-blur-sm`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-amber-900'}`}>System Prompt</span>
                  <button
                    onClick={() => setShowSystemPrompt(!showSystemPrompt)}
                    className={`text-xs ${darkMode ? 'text-amber-400' : 'text-amber-600'}`}
                  >
                    {showSystemPrompt ? 'Hide' : 'Edit'}
                  </button>
                </div>
                {showSystemPrompt && (
                  <textarea
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    className={`w-full mt-2 p-2 rounded-lg ${darkMode ? 'bg-gray-600/40 text-gray-200' : 'bg-white/60 text-amber-900'} text-xs outline-none`}
                    rows={3}
                  />
                )}
              </div>

              {/* Export Chat */}
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700/40' : 'bg-white/40'} backdrop-blur-sm`}>
                <div className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-amber-900'} mb-3`}>Export Chat</div>
                <div className="flex gap-2 mb-2">
                  {['txt', 'json', 'md'].map(format => (
                    <button
                      key={format}
                      onClick={() => setExportFormat(format)}
                      className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                        exportFormat === format
                          ? darkMode ? 'bg-amber-600/30 text-amber-300' : 'bg-amber-600/80 text-white'
                          : darkMode ? 'bg-gray-600/20 text-gray-400' : 'bg-white/40 text-amber-900'
                      }`}
                    >
                      .{format}
                    </button>
                  ))}
                </div>
                <button
                  onClick={exportChat}
                  className={`w-full p-2 rounded-lg flex items-center justify-center gap-2 ${darkMode ? 'bg-amber-600/20 hover:bg-amber-600/30 text-amber-400' : 'bg-amber-600/80 hover:bg-amber-600/90 text-white'} transition-all`}
                >
                  <Download size={16} />
                  <span className="text-sm">Export</span>
                </button>
              </div>
            </div>
          )}

          {/* Theme Toggle at Bottom */}
          <div className={`mt-auto pt-4 border-t ${darkMode ? 'border-gray-700/50' : 'border-amber-800/20'}`}>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`w-full p-3 rounded-lg flex items-center gap-3 ${darkMode ? 'bg-gray-700/60 text-amber-400' : 'bg-white/60 text-amber-900'} transition-all duration-200`}
            >
              {darkMode ? <Moon size={20} /> : <Sun size={20} />}
              <span className="font-medium">{darkMode ? 'Dark' : 'Light'} Theme</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
        {/* Header - Sticky on Mobile, Normal on Desktop */}
        <div 
          className={`${darkMode ? 'bg-gray-800/40' : 'bg-white/40'} backdrop-blur-xl border-b ${darkMode ? 'border-gray-700/50' : 'border-amber-800/20'} flex items-center gap-2 md:gap-4 z-50 rounded-b-3xl md:rounded-none sticky md:static md:p-4`}
          style={{
            top: 'env(safe-area-inset-top, 0px)',
            paddingTop: `calc(env(safe-area-inset-top, 0px) + 0.75rem)`,
            paddingBottom: `calc(env(safe-area-inset-bottom, 0px) + 0.75rem)`,
            paddingLeft: '0.75rem',
            paddingRight: '0.75rem'
          }}
        >
          {/* Menu Button */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              setSidebarOpen(!sidebarOpen);
            }}
            className={`liquid-button md:p-2 p-3.5 rounded-2xl md:rounded-lg ${darkMode ? 'bg-amber-600/50 hover:bg-amber-600/60 md:bg-gray-700/60 md:hover:bg-gray-700/80 text-amber-200 md:text-amber-400' : 'bg-amber-600 hover:bg-amber-700 md:bg-amber-900/10 md:hover:bg-amber-900/20 text-white md:text-amber-900'} touch-manipulation shadow-xl md:shadow-none flex-shrink-0 relative z-[100] flex items-center justify-center backdrop-blur-md md:backdrop-blur-sm min-w-[48px] md:min-w-0 min-h-[48px] md:min-h-0 transition-all duration-200`}
            aria-label="Toggle sidebar"
            title={sidebarOpen ? 'Close menu' : 'Open menu'}
          >
            {sidebarOpen ? <ChevronLeft size={22} className="md:w-5 md:h-5" /> : <Menu size={22} className="md:w-5 md:h-5" />}
            {!sidebarOpen && (
              <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse md:hidden shadow-lg" style={{ boxShadow: '0 0 8px rgba(239, 68, 68, 0.6)' }} />
            )}
          </button>
          
          {/* Product Name and Mode Info */}
          <div className="flex-1 min-w-0 flex flex-col">
            {/* Product Name - Visible on Mobile */}
            <div className="flex items-center gap-2 md:hidden">
              <h1 className={`text-lg font-bold ${darkMode ? 'text-amber-400' : 'text-amber-900'} truncate`}>
                WoodAI
              </h1>
              <span className={`text-xs px-2 py-0.5 rounded ${darkMode ? 'bg-amber-600/30 text-amber-300' : 'bg-amber-600/20 text-amber-800'}`}>
                {activeMode === 'rag' ? 'RAG' : 'Agent'}
              </span>
            </div>
            
            {/* Desktop Title */}
            <div className="hidden md:block">
              <h2 className={`text-lg font-bold ${darkMode ? 'text-amber-400' : 'text-amber-900'}`}>
                {activeMode === 'rag' ? 'RAG Assistant' : 'Agent Assistant'}
              </h2>
              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>
                {activeMode === 'rag' 
                  ? `Model: ${currentRagModel} • Context: ${contextLength} • Tokens: ${tokenCount.total}` 
                  : `Model: ${currentAgentModel} • Tool Enhanced`}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1 md:gap-2 flex-shrink-0">
            <button 
              onClick={toggleFullscreen}
              className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700/60 hover:bg-gray-700/80 text-gray-400' : 'bg-white/60 hover:bg-white/80 text-amber-900'} transition-all duration-200 touch-manipulation hidden md:flex`}
              title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
            >
              {isFullscreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
            </button>
            
            <button 
              onClick={handleNewChat}
              className={`p-2.5 md:p-2 rounded-2xl md:rounded-lg ${darkMode ? 'bg-gray-700/50 hover:bg-gray-700/70 md:bg-gray-700/60 md:hover:bg-gray-700/80 text-gray-300 md:text-gray-400' : 'bg-white/70 hover:bg-white/90 md:bg-white/60 md:hover:bg-white/80 text-amber-900'} touch-manipulation backdrop-blur-md md:backdrop-blur-sm min-w-[44px] md:min-w-0 min-h-[44px] md:min-h-0 transition-all duration-200`}
              title="New chat"
            >
              <Trash2 size={20} />
            </button>
          </div>
        </div>

        {/* Selected Documents Indicator */}
        {selectedDocuments.length > 0 && (
          <div className={`${darkMode ? 'bg-amber-600/20 border-b border-amber-600/30' : 'bg-amber-100/60 border-b border-amber-300/50'} px-4 py-2 flex items-center justify-between`}>
            <div className="flex items-center gap-2">
              <FileText size={16} className={darkMode ? 'text-amber-400' : 'text-amber-800'} />
              <span className={`text-sm font-medium ${darkMode ? 'text-amber-400' : 'text-amber-800'}`}>
                {selectedDocuments.length} document{selectedDocuments.length > 1 ? 's' : ''} in context
              </span>
            </div>
            <button
              onClick={() => setSelectedDocuments([])}
              className={`text-xs px-2 py-1 rounded ${darkMode ? 'hover:bg-amber-600/30 text-amber-400' : 'hover:bg-amber-200 text-amber-800'}`}
            >
              Clear
            </button>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-3 md:p-6 space-y-3 md:space-y-4 min-h-0 max-h-full">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex gap-2 md:gap-3 animate-[fadeIn_0.3s_ease-in] ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className={`w-8 h-8 md:w-10 md:h-10 rounded-xl ${darkMode ? 'bg-amber-600/30' : 'bg-amber-600/80'} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                  <Brain className={darkMode ? 'text-amber-400' : 'text-white'} size={16} />
                </div>
              )}
              
              <div className="flex flex-col gap-1 md:gap-2 max-w-[85%] md:max-w-2xl">
                <div
                  className={`p-3 md:p-4 rounded-2xl md:rounded-2xl text-sm md:text-base transition-all duration-200 ${
                    msg.role === 'user'
                      ? darkMode ? 'bg-amber-600/30 text-amber-100' : 'bg-amber-600/80 text-white'
                      : darkMode ? 'bg-gray-700/60 text-gray-100' : 'bg-white/60 text-amber-900'
                  } backdrop-blur-sm shadow-lg hover:shadow-xl group relative break-words`}
                >
                  {msg.files && msg.files.length > 0 && (
                    <div className="mb-2 flex flex-wrap gap-2">
                      {msg.files.map((file, i) => (
                        <div key={i} className={`text-xs px-2 py-1 rounded flex items-center gap-1 ${darkMode ? 'bg-gray-600/40' : 'bg-white/40'}`}>
                          {file.type === 'image' ? <ImageIcon size={12} /> : <FileText size={12} />}
                          <span>{file.name}</span>
                          <span className="opacity-60">({file.size})</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="whitespace-pre-wrap">
                    {msg.content}
                    {isLoading && idx === messages.length - 1 && msg.role === 'assistant' && (
                      <span className="inline-block w-2 h-5 ml-1 bg-amber-600 animate-pulse" />
                    )}
                  </div>
                  
                  {/* Message Actions */}
                  {msg.role === 'assistant' && (
                    <div className={`flex items-center gap-2 mt-3 pt-3 border-t ${darkMode ? 'border-gray-600/30' : 'border-amber-800/20'} opacity-0 group-hover:opacity-100 transition-opacity`}>
                      <button
                        onClick={() => copyToClipboard(msg.content, idx)}
                        className={`p-1.5 rounded ${darkMode ? 'hover:bg-gray-600/40' : 'hover:bg-white/40'} transition-all`}
                        title="Copy"
                      >
                        {copiedIndex === idx ? <Check size={14} className="text-green-500" /> : <Copy size={14} />}
                      </button>
                      <button
                        onClick={() => downloadMessage(msg, idx)}
                        className={`p-1.5 rounded ${darkMode ? 'hover:bg-gray-600/40' : 'hover:bg-white/40'} transition-all`}
                        title="Download"
                      >
                        <Download size={14} />
                      </button>
                      {idx === messages.length - 1 && (
                        <button
                          onClick={regenerateResponse}
                          className={`p-1.5 rounded ${darkMode ? 'hover:bg-gray-600/40' : 'hover:bg-white/40'} transition-all`}
                          title="Regenerate"
                        >
                          <RefreshCw size={14} />
                        </button>
                      )}
                    </div>
                  )}
                </div>
                
                <div className={`text-xs ${darkMode ? 'text-gray-500' : 'text-amber-600'} ${msg.role === 'user' ? 'text-right' : 'text-left'} px-2`}>
                  {msg.timestamp}
                </div>
              </div>

              {msg.role === 'user' && (
                <div className={`w-8 h-8 md:w-10 md:h-10 rounded-xl ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                  <User className={darkMode ? 'text-amber-400' : 'text-amber-900'} size={16} />
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="flex gap-3 animate-[fadeIn_0.3s_ease-in]">
              <div className={`w-10 h-10 rounded-xl ${darkMode ? 'bg-amber-600/30' : 'bg-amber-600/80'} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                <Brain className={darkMode ? 'text-amber-400' : 'text-white'} size={20} />
              </div>
              <div className={`p-4 rounded-2xl ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} backdrop-blur-sm shadow-lg`}>
                <div className="flex gap-1">
                  <div className={`w-2 h-2 rounded-full ${darkMode ? 'bg-amber-400' : 'bg-amber-900'} animate-bounce`} style={{animationDelay: '0ms'}} />
                  <div className={`w-2 h-2 rounded-full ${darkMode ? 'bg-amber-400' : 'bg-amber-900'} animate-bounce`} style={{animationDelay: '150ms'}} />
                  <div className={`w-2 h-2 rounded-full ${darkMode ? 'bg-amber-400' : 'bg-amber-900'} animate-bounce`} style={{animationDelay: '300ms'}} />
                </div>
              </div>
            </div>
          )}
          
          {isUploadingDocument && (
            <div className="flex gap-3 animate-[fadeIn_0.3s_ease-in]">
              <div className={`w-10 h-10 rounded-xl ${darkMode ? 'bg-blue-600/30' : 'bg-blue-600/80'} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                <FileText className={darkMode ? 'text-blue-400' : 'text-white'} size={20} />
              </div>
              <div className={`p-4 rounded-2xl ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} backdrop-blur-sm shadow-lg`}>
                <div className="flex items-center gap-3">
                  <RefreshCw className={`${darkMode ? 'text-blue-400' : 'text-blue-600'} animate-spin`} size={20} />
                  <div>
                    <div className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                      Processing document...
                    </div>
                    <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {uploadingFileName}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {isUploadingDocument && (
            <div className="flex gap-3 animate-[fadeIn_0.3s_ease-in]">
              <div className={`w-10 h-10 rounded-xl ${darkMode ? 'bg-blue-600/30' : 'bg-blue-600/80'} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                <FileText className={darkMode ? 'text-blue-400' : 'text-white'} size={20} />
              </div>
              <div className={`p-4 rounded-2xl ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} backdrop-blur-sm shadow-lg`}>
                <div className="flex items-center gap-3">
                  <RefreshCw className={`${darkMode ? 'text-blue-400' : 'text-blue-600'} animate-spin`} size={20} />
                  <div>
                    <div className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                      Processing document...
                    </div>
                    <div className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {uploadingFileName}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className={`${darkMode ? 'bg-gray-800/40' : 'bg-white/40'} backdrop-blur-xl border-t ${darkMode ? 'border-gray-700/50' : 'border-amber-800/20'} p-3 md:p-4 flex-shrink-0`}>
          {/* Attached Files */}
          {attachedFiles.length > 0 && (
            <div className="mb-2 md:mb-3 flex flex-wrap gap-2">
              {attachedFiles.map((file, idx) => (
                <div key={idx} className={`flex items-center gap-2 px-2 md:px-3 py-1.5 md:py-2 rounded-lg ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} backdrop-blur-sm text-xs md:text-sm`}>
                  {file.type === 'image' ? <Camera size={14} /> : <Paperclip size={14} />}
                  <div className="flex flex-col min-w-0">
                    <span className={`${darkMode ? 'text-gray-200' : 'text-amber-900'} truncate max-w-[120px] md:max-w-none`}>{file.name}</span>
                    <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>{file.size}</span>
                  </div>
                  <button onClick={() => removeFile(idx)} className={`${darkMode ? 'text-gray-400 hover:text-gray-200' : 'text-amber-700 hover:text-amber-900'} touch-manipulation`}>
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Token Counter */}
          <div className={`mb-2 flex items-center justify-between text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'} flex-wrap gap-1`}>
            <div className="flex items-center gap-2 md:gap-4 flex-wrap">
              <span className="whitespace-nowrap">Input: {tokenCount.input}</span>
              <span className="whitespace-nowrap">Output: {tokenCount.output}</span>
              <span className="font-semibold whitespace-nowrap">Total: {tokenCount.total}</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap size={12} />
              <span>{activeMode === 'rag' ? 'RAG' : 'Agent'}</span>
            </div>
          </div>

          <div className={`flex gap-2 md:gap-3 p-2 md:p-2 rounded-xl md:rounded-2xl ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} backdrop-blur-sm shadow-lg relative w-full`}>
            {/* Attachment Menu */}
            {showAttachMenu && (
              <div className={`absolute bottom-full left-0 mb-2 p-2 rounded-xl ${darkMode ? 'bg-gray-700/90' : 'bg-white/90'} backdrop-blur-xl shadow-xl z-10`}>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg ${darkMode ? 'hover:bg-gray-600/60 text-gray-200' : 'hover:bg-amber-50 text-amber-900'} transition-all w-full text-left`}
                >
                  <Paperclip size={18} />
                  <span>Upload Document</span>
                </button>
                <button
                  onClick={() => cameraInputRef.current?.click()}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg ${darkMode ? 'hover:bg-gray-600/60 text-gray-200' : 'hover:bg-amber-50 text-amber-900'} transition-all w-full text-left`}
                >
                  <Camera size={18} />
                  <span>Take Photo</span>
                </button>
              </div>
            )}

            <button
              onClick={() => setShowAttachMenu(!showAttachMenu)}
              className={`liquid-button md:p-3 p-3 rounded-2xl md:rounded-xl ${darkMode ? 'hover:bg-gray-600/40 text-amber-400' : 'hover:bg-amber-50 text-amber-900'} touch-manipulation min-w-[44px] md:min-w-0 min-h-[44px] md:min-h-0`}
              aria-label="Attach file"
            >
              <Paperclip size={18} />
            </button>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (!isLoading && (inputValue.trim() || attachedFiles.length > 0)) {
                  handleSend();
                }
              }}
              className="flex-1 flex gap-2 md:gap-3"
            >
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
                    e.preventDefault();
                    if (inputValue.trim() || attachedFiles.length > 0) {
                      handleSend();
                    }
                  }
                }}
                placeholder="Type your message..."
                className={`flex-1 px-3 md:px-4 py-2 md:py-3 bg-transparent outline-none text-sm md:text-base ${darkMode ? 'text-gray-100 placeholder-gray-400' : 'text-amber-900 placeholder-amber-700/50'}`}
                disabled={isLoading}
                autoComplete="off"
              />
              
              <button
                type="submit"
                disabled={isLoading || (!inputValue.trim() && attachedFiles.length === 0)}
                className={`liquid-button md:px-6 px-5 md:py-3 py-2.5 rounded-2xl md:rounded-xl ${
                  isLoading || (!inputValue.trim() && attachedFiles.length === 0)
                    ? darkMode ? 'bg-gray-600/40 text-gray-500' : 'bg-amber-400/40 text-amber-700'
                    : darkMode ? 'bg-amber-600/50 active:bg-amber-600/70 md:bg-amber-600/30 md:hover:bg-amber-600/40 text-amber-200 md:text-amber-400' : 'bg-amber-600 active:bg-amber-700 md:bg-amber-600/80 md:hover:bg-amber-600/90 text-white'
                } font-medium flex items-center justify-center gap-2 shadow-xl md:shadow-lg disabled:cursor-not-allowed touch-manipulation backdrop-blur-md md:backdrop-blur-sm transition-all duration-200 min-w-[48px] md:min-w-0 min-h-[48px] md:min-h-0`}
                aria-label="Send message"
              >
                <Send size={18} />
              </button>
            </form>
          </div>
        </div>
      </div>

      {/* Hidden file inputs */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.doc,.docx,.pptx,.xlsx,.txt,.csv"
        onChange={handleFileSelect}
        className="hidden"
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleCameraCapture}
        className="hidden"
      />

      {/* Model Selector Dialog */}
      {showModelSelector && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-2xl shadow-2xl max-w-2xl w-full p-6 space-y-4 max-h-[80vh] overflow-y-auto`}>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className={`text-lg font-bold ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                  Select Ollama Model
                </h3>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Choose a model for {activeMode === 'rag' ? 'RAG' : 'Agent'} mode
                </p>
              </div>
              <button
                onClick={() => setShowModelSelector(false)}
                className={`p-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'}`}
              >
                <X size={20} className={darkMode ? 'text-gray-400' : 'text-gray-600'} />
              </button>
            </div>

            {/* Current Model */}
            <div className={`p-4 rounded-lg ${darkMode ? 'bg-amber-600/20 border border-amber-600/30' : 'bg-amber-50 border border-amber-200'}`}>
              <p className={`text-sm font-medium ${darkMode ? 'text-amber-400' : 'text-amber-800'} mb-1`}>
                Current {activeMode === 'rag' ? 'RAG' : 'Agent'} Model:
              </p>
              <p className={`text-lg font-bold ${darkMode ? 'text-amber-300' : 'text-amber-900'}`}>
                {activeMode === 'rag' ? currentRagModel : currentAgentModel}
              </p>
            </div>

            {/* Available Models */}
            <div>
              <p className={`text-sm font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-3`}>
                Available Models ({availableModels.length})
              </p>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {availableModels.length === 0 ? (
                  <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} text-center py-4`}>
                    No models found. Make sure Ollama is running.
                  </p>
                ) : (
                  availableModels.map((model) => {
                    const isCurrent = activeMode === 'rag' 
                      ? model === currentRagModel 
                      : model === currentAgentModel;
                    return (
                      <div
                        key={model}
                        className={`p-3 rounded-lg flex items-center justify-between ${
                          isCurrent
                            ? darkMode ? 'bg-amber-600/30 border border-amber-600/50' : 'bg-amber-100 border border-amber-300'
                            : darkMode ? 'bg-gray-700/40 hover:bg-gray-700/60 border border-gray-600/30' : 'bg-white/60 hover:bg-white/80 border border-gray-200'
                        } transition-all cursor-pointer`}
                        onClick={() => {
                          if (!isCurrent) {
                            switchModel(model, activeMode);
                            setShowModelSelector(false);
                          }
                        }}
                      >
                        <div className="flex-1">
                          <div className={`font-medium ${darkMode ? 'text-gray-200' : 'text-gray-900'}`}>
                            {model}
                          </div>
                          {isCurrent && (
                            <div className={`text-xs mt-1 ${darkMode ? 'text-amber-400' : 'text-amber-700'}`}>
                              Currently active
                            </div>
                          )}
                        </div>
                        {isCurrent && (
                          <Check size={20} className={darkMode ? 'text-amber-400' : 'text-amber-600'} />
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>

            {/* Popular Small Models to Download */}
            <div>
              <p className={`text-sm font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-3`}>
                Popular Small Models (Download)
              </p>
              <div className="grid grid-cols-1 gap-2">
                {[
                  { name: 'gemma2:2b', desc: 'Google Gemma 2B - Fast & Efficient' },
                  { name: 'llama3.2:1b', desc: 'Meta Llama 3.2 1B - Ultra Small' },
                  { name: 'llama3.2:3b', desc: 'Meta Llama 3.2 3B - Balanced' },
                  { name: 'phi3:mini', desc: 'Microsoft Phi-3 Mini - Fast' },
                  { name: 'qwen2.5:0.5b', desc: 'Qwen 2.5 0.5B - Tiny' },
                  { name: 'tinyllama', desc: 'TinyLlama - Smallest' }
                ].map((model) => {
                  const isDownloading = downloadingModel === model.name;
                  const isInstalled = availableModels.includes(model.name);
                  return (
                    <div
                      key={model.name}
                      className={`p-3 rounded-lg flex items-center justify-between ${
                        darkMode ? 'bg-gray-700/40 border border-gray-600/30' : 'bg-white/60 border border-gray-200'
                      }`}
                    >
                      <div className="flex-1">
                        <div className={`font-medium ${darkMode ? 'text-gray-200' : 'text-gray-900'}`}>
                          {model.name}
                        </div>
                        <div className={`text-xs mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          {model.desc}
                        </div>
                        {isDownloading && (
                          <div className={`text-xs mt-1 ${darkMode ? 'text-amber-400' : 'text-amber-600'}`}>
                            {downloadProgress}
                          </div>
                        )}
                      </div>
                      <button
                        onClick={() => !isInstalled && !isDownloading && downloadModel(model.name)}
                        disabled={isInstalled || isDownloading}
                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                          isInstalled
                            ? darkMode ? 'bg-gray-600/40 text-gray-400 cursor-not-allowed' : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                            : isDownloading
                            ? darkMode ? 'bg-amber-600/40 text-amber-300 cursor-not-allowed' : 'bg-amber-200 text-amber-700 cursor-not-allowed'
                            : darkMode ? 'bg-amber-600/30 hover:bg-amber-600/40 text-amber-400' : 'bg-amber-600/80 hover:bg-amber-600/90 text-white'
                        }`}
                      >
                        {isInstalled ? (
                          <>
                            <Check size={16} className="inline mr-1" />
                            Installed
                          </>
                        ) : isDownloading ? (
                          <>
                            <RefreshCw size={16} className="inline mr-1 animate-spin" />
                            Downloading...
                          </>
                        ) : (
                          <>
                            <Download size={16} className="inline mr-1" />
                            Download
                          </>
                        )}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Custom Model Input */}
            <div>
              <p className={`text-sm font-semibold ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-3`}>
                Download Custom Model
              </p>
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="e.g., llama3.2:1b"
                  id="customModelInput"
                  className={`flex-1 px-4 py-2 rounded-lg border ${
                    darkMode 
                      ? 'bg-gray-700 border-gray-600 text-gray-100 placeholder-gray-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                  } focus:outline-none focus:ring-2 focus:ring-amber-500`}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      const input = document.getElementById('customModelInput');
                      if (input && input.value.trim()) {
                        downloadModel(input.value.trim());
                        input.value = '';
                      }
                    }
                  }}
                />
                <button
                  onClick={() => {
                    const input = document.getElementById('customModelInput');
                    if (input && input.value.trim()) {
                      downloadModel(input.value.trim());
                      input.value = '';
                    }
                  }}
                  className={`px-4 py-2 rounded-lg font-medium ${
                    darkMode ? 'bg-amber-600/30 hover:bg-amber-600/40 text-amber-400' : 'bg-amber-600/80 hover:bg-amber-600/90 text-white'
                  } transition-all`}
                >
                  <Download size={16} className="inline mr-1" />
                  Download
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* File Location Dialog */}
      {showFileLocationDialog && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-2xl shadow-2xl max-w-md w-full p-6 space-y-4`}>
            <div className="flex items-center gap-3 mb-4">
              <div className={`p-2 rounded-lg ${darkMode ? 'bg-amber-600/20' : 'bg-amber-100'}`}>
                <FolderOpen className={darkMode ? 'text-amber-400' : 'text-amber-600'} size={24} />
              </div>
              <div>
                <h3 className={`text-lg font-bold ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
                  Specify File Location
                </h3>
                <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  Choose where to save the file
                </p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  Filename
                </label>
                <input
                  type="text"
                  value={fileLocation.filename}
                  onChange={(e) => setFileLocation(prev => ({ ...prev, filename: e.target.value }))}
                  placeholder="e.g., calculator.py, notes.txt"
                  className={`w-full px-4 py-2 rounded-lg border ${
                    darkMode 
                      ? 'bg-gray-700 border-gray-600 text-gray-100 placeholder-gray-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                  } focus:outline-none focus:ring-2 focus:ring-amber-500`}
                  autoFocus
                />
                <p className={`text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                  Include file extension (.py, .txt, .md, etc.)
                </p>
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  Directory/Path
                </label>
                <input
                  type="text"
                  value={fileLocation.directory}
                  onChange={(e) => setFileLocation(prev => ({ ...prev, directory: e.target.value }))}
                  placeholder="e.g., agent_outputs, projects/myapp, /home/user/docs"
                  className={`w-full px-4 py-2 rounded-lg border ${
                    darkMode 
                      ? 'bg-gray-700 border-gray-600 text-gray-100 placeholder-gray-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                  } focus:outline-none focus:ring-2 focus:ring-amber-500`}
                />
                <p className={`text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                  Relative to backend directory or absolute path. Default: agent_outputs
                </p>
              </div>

              <div className={`p-3 rounded-lg ${darkMode ? 'bg-amber-600/10 border border-amber-600/30' : 'bg-amber-50 border border-amber-200'}`}>
                <p className={`text-xs ${darkMode ? 'text-amber-300' : 'text-amber-800'}`}>
                  <strong>Full path:</strong> {fileLocation.directory ? `${fileLocation.directory}/` : ''}{fileLocation.filename || 'filename'}
                </p>
              </div>
            </div>

            <div className="flex gap-3 pt-4">
              <button
                onClick={handleFileLocationCancel}
                className={`flex-1 px-4 py-2 rounded-lg font-medium transition-all ${
                  darkMode 
                    ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' 
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                }`}
              >
                Cancel
              </button>
              <button
                onClick={handleFileLocationConfirm}
                disabled={!fileLocation.filename.trim()}
                className={`flex-1 px-4 py-2 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                  !fileLocation.filename.trim()
                    ? darkMode 
                      ? 'bg-gray-700 text-gray-500 cursor-not-allowed' 
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    : darkMode 
                      ? 'bg-amber-600 hover:bg-amber-700 text-white' 
                      : 'bg-amber-600 hover:bg-amber-700 text-white'
                }`}
              >
                <Save size={18} />
                Create File
              </button>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        /* Mobile-optimized sliders */
        input[type="range"] {
          -webkit-appearance: none;
          appearance: none;
          touch-action: pan-x;
        }
        
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: #d97706;
          cursor: pointer;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
          transition: all 0.2s ease;
          border: 2px solid white;
        }
        
        @media (min-width: 768px) {
          input[type="range"]::-webkit-slider-thumb {
            width: 18px;
            height: 18px;
          }
        }
        
        input[type="range"]::-webkit-slider-thumb:hover,
        input[type="range"]::-webkit-slider-thumb:active {
          transform: scale(1.15);
          box-shadow: 0 4px 12px rgba(217, 119, 6, 0.5);
        }
        
        input[type="range"]::-moz-range-thumb {
          width: 24px;
          height: 24px;
          border-radius: 50%;
          background: #d97706;
          cursor: pointer;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
          transition: all 0.2s ease;
          border: 2px solid white;
        }
        
        @media (min-width: 768px) {
          input[type="range"]::-moz-range-thumb {
            width: 18px;
            height: 18px;
          }
        }
        
        input[type="range"]::-moz-range-thumb:hover,
        input[type="range"]::-moz-range-thumb:active {
          transform: scale(1.15);
          box-shadow: 0 4px 12px rgba(217, 119, 6, 0.5);
        }
        
        /* Touch-friendly buttons */
        .touch-manipulation {
          touch-action: manipulation;
          -webkit-tap-highlight-color: transparent;
        }
        
        /* Mobile sidebar improvements */
        @media (max-width: 767px) {
          .sidebar-mobile {
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            z-index: 50;
          }
        }
      `}</style>
    </div>
  );
  
}