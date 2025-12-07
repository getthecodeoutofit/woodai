import React, { useState, useRef, useEffect } from 'react';
import { Menu, Settings, History, MessageSquare, Brain, Workflow, Send, Plus, ChevronLeft, Moon, Sun, Trash2, User, Paperclip, Camera, X, Download, Copy, Check, Volume2, VolumeX, Maximize2, Minimize2, RefreshCw, Zap, FileText, Image as ImageIcon } from 'lucide-react';

export default function AIChat() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
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
  const [memoryEnabled, setMemoryEnabled] = useState(true);
  const [temperature, setTemperature] = useState(0.7);
  const [isLoading, setIsLoading] = useState(false);
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
  
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);
  const chatContainerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load chat history from backend
  useEffect(() => {
    loadChatHistory();
  }, []);

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

  const handleSend = async () => {
    if (inputValue.trim() || attachedFiles.length > 0) {
      const userMessage = {
        role: 'user',
        content: inputValue,
        files: attachedFiles,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setMessages(prev => [...prev, userMessage]);
      setInputValue('');
      setAttachedFiles([]);
      setIsLoading(true);

      try {
        const response = await fetch('http://localhost:8000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message: userMessage.content,
            mode: activeMode,
            context_length: contextLength,
            memory_enabled: memoryEnabled,
            temperature: temperature,
            system_prompt: systemPrompt,
            history: messages,
            user_id: userId,
            chat_id: currentChatId // Include current chat ID
          })
        });

        const data = await response.json();
        
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.response,
          timestamp: new Date().toLocaleTimeString()
        }]);
        
        // Update current chat ID if new chat was created
        if (data.chat_id && !currentChatId) {
          setCurrentChatId(data.chat_id);
        }
        
        // Refresh chat history
        loadChatHistory();

        // Text-to-speech if enabled
        if (voiceEnabled && 'speechSynthesis' in window) {
          const utterance = new SpeechSynthesisUtterance(data.response);
          window.speechSynthesis.speak(utterance);
        }
      } catch (error) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'Sorry, I couldn\'t connect to the server. Please make sure the backend is running on http://localhost:8000',
          timestamp: new Date().toLocaleTimeString()
        }]);
      } finally {
        setIsLoading(false);
      }
    }
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
      const response = await fetch('http://localhost:8000/chat', {
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
    } catch (error) {
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
      const response = await fetch(`http://localhost:8000/chats/${userId}`);
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
      const response = await fetch(`http://localhost:8000/chat/${chatId}`);
      const chat = await response.json();
      
      if (chat && chat.messages) {
        setMessages(chat.messages);
        setActiveView('chat');
        setCurrentChatId(chatId);
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
      const response = await fetch(`http://localhost:8000/chat/${chatId}`, {
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

  // Upload document
  const handleDocumentUpload = async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('user_id', userId);
      
      const response = await fetch('http://localhost:8000/upload', {
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
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Failed to upload document: ${error.message}`,
        timestamp: new Date().toLocaleTimeString()
      }]);
    }
  };

  const filteredHistory = chatHistory.filter(chat => 
    chat.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    chat.preview.toLowerCase().includes(searchQuery.toLowerCase())
  );

  

  return (
    <div 
      ref={chatContainerRef}
      className={`flex h-screen ${darkMode ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900' : 'bg-gradient-to-br from-amber-50 via-orange-50 to-amber-100'} overflow-hidden transition-all duration-500`}
    >
      
      {/* Sidebar */}
      <div 
        className={`${sidebarOpen ? 'w-80' : 'w-0'} transition-all duration-300 ${darkMode ? 'bg-gray-800/40' : 'bg-gradient-to-b from-amber-900/20 to-orange-900/30'} backdrop-blur-xl border-r ${darkMode ? 'border-gray-700/50' : 'border-amber-800/20'} overflow-hidden`}
        style={{
          boxShadow: sidebarOpen ? '4px 0 20px rgba(0,0,0,0.1)' : 'none'
        }}
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
                <div className="text-left">
                  <div className="font-semibold text-sm">RAG Mode</div>
                  <div className={`text-xs ${activeMode === 'rag' ? 'opacity-90' : 'opacity-60'}`}>Ollama Gemma 2B</div>
                </div>
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
                <div className="text-left">
                  <div className="font-semibold text-sm">Agent Mode</div>
                  <div className={`text-xs ${activeMode === 'agent' ? 'opacity-90' : 'opacity-60'}`}>Tool Enhanced</div>
                </div>
              </button>
            </div>
          </div>

          {/* Navigation */}
          <nav className="space-y-1 mb-6">
            {[
              { id: 'chat', icon: MessageSquare, label: 'Chat' },
              { id: 'history', icon: History, label: 'History' },
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
            <div className="flex-1 overflow-y-auto">
              <div className="mb-3">
                <input
                  type="text"
                  placeholder="Search chats..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className={`w-full px-3 py-2 rounded-lg ${darkMode ? 'bg-gray-700/40 text-gray-200 placeholder-gray-400' : 'bg-white/40 text-amber-900 placeholder-amber-700/50'} backdrop-blur-sm border ${darkMode ? 'border-gray-600/30' : 'border-amber-800/10'} outline-none`}
                />
              </div>
              <p className={`text-xs font-semibold ${darkMode ? 'text-gray-400' : 'text-amber-800'} mb-3 uppercase tracking-wider`}>Recent Chats</p>
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
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700/40' : 'bg-white/40'} backdrop-blur-sm`}>
                <div className="flex items-center justify-between mb-3">
                  <span className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-amber-900'}`}>Context Length</span>
                  <span className={`text-xs ${darkMode ? 'text-amber-400' : 'text-amber-600'} font-semibold`}>{contextLength}</span>
                </div>
                <input 
                  type="range" 
                  min="1024" 
                  max="8192" 
                  step="1024"
                  value={contextLength}
                  onChange={(e) => setContextLength(parseInt(e.target.value))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: darkMode 
                      ? `linear-gradient(to right, #f59e0b ${(contextLength - 1024) / 7168 * 100}%, #374151 ${(contextLength - 1024) / 7168 * 100}%)`
                      : `linear-gradient(to right, #92400e ${(contextLength - 1024) / 7168 * 100}%, #fed7aa ${(contextLength - 1024) / 7168 * 100}%)`
                  }}
                />
                <div className="flex justify-between mt-2">
                  <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>1K</span>
                  <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>8K</span>
                </div>
              </div>

              {/* Temperature */}
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700/40' : 'bg-white/40'} backdrop-blur-sm`}>
                <div className="flex items-center justify-between mb-3">
                  <span className={`text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-amber-900'}`}>Temperature</span>
                  <span className={`text-xs ${darkMode ? 'text-amber-400' : 'text-amber-600'} font-semibold`}>{temperature.toFixed(1)}</span>
                </div>
                <input 
                  type="range" 
                  min="0" 
                  max="2" 
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-2 rounded-lg appearance-none cursor-pointer"
                  style={{
                    background: darkMode 
                      ? `linear-gradient(to right, #f59e0b ${temperature / 2 * 100}%, #374151 ${temperature / 2 * 100}%)`
                      : `linear-gradient(to right, #92400e ${temperature / 2 * 100}%, #fed7aa ${temperature / 2 * 100}%)`
                  }}
                />
                <div className="flex justify-between mt-2">
                  <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>Precise</span>
                  <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>Creative</span>
                </div>
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
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className={`${darkMode ? 'bg-gray-800/40' : 'bg-white/40'} backdrop-blur-xl border-b ${darkMode ? 'border-gray-700/50' : 'border-amber-800/20'} p-4 flex items-center gap-4`}>
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700/60 hover:bg-gray-700/80 text-amber-400' : 'bg-amber-900/10 hover:bg-amber-900/20 text-amber-900'} transition-all duration-200`}
          >
            {sidebarOpen ? <ChevronLeft size={20} /> : <Menu size={20} />}
          </button>
          
          <div className="flex-1">
            <h2 className={`text-lg font-bold ${darkMode ? 'text-amber-400' : 'text-amber-900'}`}>
              {activeMode === 'rag' ? 'RAG Assistant' : 'Agent Assistant'}
            </h2>
            <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>
              {activeMode === 'rag' ? `Context: ${contextLength} • Tokens: ${tokenCount.total}` : 'Tool Enhanced • Multi-capability'}
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button 
              onClick={toggleFullscreen}
              className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700/60 hover:bg-gray-700/80 text-gray-400' : 'bg-white/60 hover:bg-white/80 text-amber-900'} transition-all duration-200`}
              title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
            >
              {isFullscreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
            </button>
            
            <button 
              onClick={handleNewChat}
              className={`p-2 rounded-lg ${darkMode ? 'bg-gray-700/60 hover:bg-gray-700/80 text-gray-400' : 'bg-white/60 hover:bg-white/80 text-amber-900'} transition-all duration-200`}
              title="New chat"
            >
              <Trash2 size={20} />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex gap-3 animate-[fadeIn_0.3s_ease-in] ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className={`w-10 h-10 rounded-xl ${darkMode ? 'bg-amber-600/30' : 'bg-amber-600/80'} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                  <Brain className={darkMode ? 'text-amber-400' : 'text-white'} size={20} />
                </div>
              )}
              
              <div className="flex flex-col gap-2 max-w-2xl">
                <div
                  className={`p-4 rounded-2xl ${
                    msg.role === 'user'
                      ? darkMode ? 'bg-amber-600/30 text-amber-100' : 'bg-amber-600/80 text-white'
                      : darkMode ? 'bg-gray-700/60 text-gray-100' : 'bg-white/60 text-amber-900'
                  } backdrop-blur-sm shadow-lg transition-all duration-200 hover:shadow-xl group relative`}
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
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                  
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
                <div className={`w-10 h-10 rounded-xl ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} flex items-center justify-center flex-shrink-0 shadow-lg`}>
                  <User className={darkMode ? 'text-amber-400' : 'text-amber-900'} size={20} />
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
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className={`${darkMode ? 'bg-gray-800/40' : 'bg-white/40'} backdrop-blur-xl border-t ${darkMode ? 'border-gray-700/50' : 'border-amber-800/20'} p-4`}>
          {/* Attached Files */}
          {attachedFiles.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              {attachedFiles.map((file, idx) => (
                <div key={idx} className={`flex items-center gap-2 px-3 py-2 rounded-lg ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} backdrop-blur-sm`}>
                  {file.type === 'image' ? <Camera size={16} /> : <Paperclip size={16} />}
                  <div className="flex flex-col">
                    <span className={`text-sm ${darkMode ? 'text-gray-200' : 'text-amber-900'}`}>{file.name}</span>
                    <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>{file.size}</span>
                  </div>
                  <button onClick={() => removeFile(idx)} className={`${darkMode ? 'text-gray-400 hover:text-gray-200' : 'text-amber-700 hover:text-amber-900'}`}>
                    <X size={16} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Token Counter */}
          <div className={`mb-2 flex items-center justify-between text-xs ${darkMode ? 'text-gray-400' : 'text-amber-700'}`}>
            <div className="flex items-center gap-4">
              <span>Input: {tokenCount.input} tokens</span>
              <span>Output: {tokenCount.output} tokens</span>
              <span className="font-semibold">Total: {tokenCount.total}</span>
            </div>
            <div className="flex items-center gap-2">
              <Zap size={14} />
              <span>{activeMode === 'rag' ? 'RAG' : 'Agent'}</span>
            </div>
          </div>

          <div className={`flex gap-3 p-2 rounded-2xl ${darkMode ? 'bg-gray-700/60' : 'bg-white/60'} backdrop-blur-sm shadow-lg relative`}>
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
              className={`p-3 rounded-xl ${darkMode ? 'hover:bg-gray-600/40 text-amber-400' : 'hover:bg-amber-50 text-amber-900'} transition-all duration-200`}
            >
              <Paperclip size={20} />
            </button>

            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !isLoading && !e.shiftKey && handleSend()}
              placeholder="Type your message... (Shift+Enter for new line)"
              className={`flex-1 px-4 py-3 bg-transparent outline-none ${darkMode ? 'text-gray-100 placeholder-gray-400' : 'text-amber-900 placeholder-amber-700/50'}`}
              disabled={isLoading}
            />
            
            <button
              onClick={handleSend}
              disabled={isLoading || (!inputValue.trim() && attachedFiles.length === 0)}
              className={`px-6 py-3 rounded-xl ${
                isLoading || (!inputValue.trim() && attachedFiles.length === 0)
                  ? darkMode ? 'bg-gray-600/40 text-gray-500' : 'bg-amber-400/40 text-amber-700'
                  : darkMode ? 'bg-amber-600/30 hover:bg-amber-600/40 text-amber-400' : 'bg-amber-600/80 hover:bg-amber-600/90 text-white'
              } transition-all duration-200 font-medium flex items-center gap-2 shadow-lg hover:shadow-xl disabled:cursor-not-allowed`}
            >
              <Send size={20} />
            </button>
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
        
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: #d97706;
          cursor: pointer;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
          transition: all 0.2s ease;
        }
        
        input[type="range"]::-webkit-slider-thumb:hover {
          transform: scale(1.1);
          box-shadow: 0 4px 12px rgba(217, 119, 6, 0.4);
        }
      `}</style>
    </div>
  );
  
}