from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import os
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

class MongoDB:
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        self.connection_string = connection_string
        self.client = None
        self.db = None
        self.async_client = None
        self.async_db = None
        
    def connect(self):
        """Connect to MongoDB"""
        try:
            # Sync client for initialization
            self.client = MongoClient(self.connection_string)
            self.db = self.client['woodai']
            
            # Async client for FastAPI
            self.async_client = AsyncIOMotorClient(self.connection_string)
            self.async_db = self.async_client['woodai']
            
            # Create collections
            self.chats = self.async_db['chats']
            self.vectors = self.async_db['vectors']
            self.documents = self.async_db['documents']
            
            # Create indexes
            self.client['woodai']['chats'].create_index([("user_id", 1), ("created_at", -1)])
            self.client['woodai']['vectors'].create_index([("doc_id", 1)])
            self.client['woodai']['documents'].create_index([("filename", 1)])
            
            print("✅ MongoDB connected successfully")
            return True
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            return False
    
    async def save_chat(self, user_id: str, title: str, messages: List[Dict]) -> str:
        """Save chat history to MongoDB"""
        chat_doc = {
            "user_id": user_id,
            "title": title,
            "messages": messages,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "message_count": len(messages)
        }
        result = await self.chats.insert_one(chat_doc)
        return str(result.inserted_id)
    
    async def get_user_chats(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get all chats for a user"""
        cursor = self.chats.find(
            {"user_id": user_id}
        ).sort("updated_at", -1).limit(limit)
        
        chats = []
        async for chat in cursor:
            chat['_id'] = str(chat['_id'])
            chat['created_at'] = chat['created_at'].isoformat()
            chat['updated_at'] = chat['updated_at'].isoformat()
            
            # Create preview from last message
            if chat['messages']:
                last_msg = chat['messages'][-1]
                chat['preview'] = last_msg.get('content', '')[:100] + '...'
            else:
                chat['preview'] = 'Empty conversation'
                
            chats.append(chat)
        
        return chats
    
    async def get_chat_by_id(self, chat_id: str) -> Optional[Dict]:
        """Get specific chat by ID"""
        from bson import ObjectId
        try:
            chat = await self.chats.find_one({"_id": ObjectId(chat_id)})
            if chat:
                chat['_id'] = str(chat['_id'])
                chat['created_at'] = chat['created_at'].isoformat()
                chat['updated_at'] = chat['updated_at'].isoformat()
            return chat
        except:
            return None
    
    async def update_chat(self, chat_id: str, messages: List[Dict]) -> bool:
        """Update chat with new messages"""
        from bson import ObjectId
        try:
            result = await self.chats.update_one(
                {"_id": ObjectId(chat_id)},
                {
                    "$set": {
                        "messages": messages,
                        "updated_at": datetime.utcnow(),
                        "message_count": len(messages)
                    }
                }
            )
            return result.modified_count > 0
        except:
            return False
    
    async def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat"""
        from bson import ObjectId
        try:
            result = await self.chats.delete_one({"_id": ObjectId(chat_id)})
            return result.deleted_count > 0
        except:
            return False
    
    async def save_document_vectors(self, doc_id: str, filename: str, chunks: List[Dict]) -> bool:
        """Save document chunks with their embeddings"""
        try:
            # Save document metadata
            doc_meta = {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_count": len(chunks),
                "created_at": datetime.utcnow()
            }
            await self.documents.insert_one(doc_meta)
            
            # Save vectors
            vector_docs = []
            for i, chunk in enumerate(chunks):
                vector_doc = {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunk['text'],
                    "embedding": chunk['embedding'].tolist(),  # Convert numpy to list
                    "metadata": chunk.get('metadata', {})
                }
                vector_docs.append(vector_doc)
            
            if vector_docs:
                await self.vectors.insert_many(vector_docs)
            
            return True
        except Exception as e:
            print(f"Error saving vectors: {e}")
            return False
    
    async def search_vectors(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar vectors using cosine similarity"""
        try:
            # Get all vectors (in production, use vector search index)
            cursor = self.vectors.find({})
            
            results = []
            async for doc in cursor:
                # Calculate cosine similarity
                doc_embedding = np.array(doc['embedding'])
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                
                if similarity > 0.3:  # Threshold
                    results.append({
                        'text': doc['text'],
                        'similarity': float(similarity),
                        'doc_id': doc['doc_id'],
                        'metadata': doc.get('metadata', {})
                    })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"Error searching vectors: {e}")
            return []
    
    async def get_all_documents(self) -> List[Dict]:
        """Get all indexed documents"""
        cursor = self.documents.find({})
        docs = []
        async for doc in cursor:
            doc['_id'] = str(doc['_id'])
            doc['created_at'] = doc['created_at'].isoformat()
            docs.append(doc)
        return docs
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
        if self.async_client:
            self.async_client.close()

# Global database instance
db = MongoDB()

def get_database():
    """Get database instance"""
    return db