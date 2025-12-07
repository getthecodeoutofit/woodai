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
            self.client['woodai']['vectors'].create_index([("chunk_index", 1)])
            self.client['woodai']['documents'].create_index([("filename", 1)])
            
            # Create text index for keyword search
            try:
                self.client['woodai']['vectors'].create_index([("text", "text")])
            except:
                pass  # Text index might already exist
            
            print("✅ MongoDB connected successfully")
            return True
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            return False
    
    async def save_chat(self, user_id: str, title: str, messages: List[Dict], selected_doc_ids: Optional[List[str]] = None) -> str:
        """Save chat history to MongoDB"""
        chat_doc = {
            "user_id": user_id,
            "title": title,
            "messages": messages,
            "selected_doc_ids": selected_doc_ids or [],
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
    
    async def update_chat(self, chat_id: str, messages: List[Dict], selected_doc_ids: Optional[List[str]] = None) -> bool:
        """Update chat with new messages"""
        from bson import ObjectId
        try:
            update_data = {
                "messages": messages,
                "updated_at": datetime.utcnow(),
                "message_count": len(messages)
            }
            if selected_doc_ids is not None:
                update_data["selected_doc_ids"] = selected_doc_ids
            
            result = await self.chats.update_one(
                {"_id": ObjectId(chat_id)},
                {"$set": update_data}
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
        """Save document chunks with their embeddings - with validation"""
        try:
            if not chunks or len(chunks) == 0:
                print(f"❌ No chunks to save for {filename}")
                return False
            
            # Validate chunks before saving
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                # Validate required fields
                if 'text' not in chunk or not chunk['text'] or not chunk['text'].strip():
                    print(f"⚠️  Skipping chunk {i}: missing or empty text")
                    continue
                
                if 'embedding' not in chunk or chunk['embedding'] is None:
                    print(f"⚠️  Skipping chunk {i}: missing embedding")
                    continue
                
                # Validate embedding is numpy array and convert
                try:
                    embedding = chunk['embedding']
                    if hasattr(embedding, 'tolist'):
                        embedding_list = embedding.tolist()
                    elif isinstance(embedding, list):
                        embedding_list = embedding
                    else:
                        embedding_list = np.array(embedding).tolist()
                    
                    # Validate embedding dimensions
                    if not isinstance(embedding_list, list) or len(embedding_list) == 0:
                        print(f"⚠️  Skipping chunk {i}: invalid embedding format")
                        continue
                    
                    # Validate text length
                    text = chunk['text'].strip()
                    if len(text) < 10:
                        print(f"⚠️  Skipping chunk {i}: text too short ({len(text)} chars)")
                        continue
                    
                    valid_chunks.append({
                        "doc_id": doc_id,
                        "chunk_index": len(valid_chunks),
                        "text": text,
                        "embedding": embedding_list,
                        "metadata": chunk.get('metadata', {})
                    })
                except Exception as e:
                    print(f"⚠️  Error processing chunk {i}: {e}")
                    continue
            
            if not valid_chunks:
                print(f"❌ No valid chunks to save for {filename}")
                return False
            
            # Save document metadata
            doc_meta = {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_count": len(valid_chunks),
                "created_at": datetime.utcnow()
            }
            await self.documents.insert_one(doc_meta)
            print(f"✅ Saved document metadata: {filename} with {len(valid_chunks)} chunks")
            
            # Save vectors in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(valid_chunks), batch_size):
                batch = valid_chunks[i:i + batch_size]
                await self.vectors.insert_many(batch)
                print(f"  Saved batch {i//batch_size + 1}/{(len(valid_chunks)-1)//batch_size + 1}")
            
            print(f"✅ Successfully saved {len(valid_chunks)} vectors to database")
            return True
        except Exception as e:
            print(f"❌ Error saving vectors: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def search_vectors(self, query_embedding: np.ndarray, top_k: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """Search for similar vectors using efficient cosine similarity"""
        try:
            # Validate query embedding
            if query_embedding is None or len(query_embedding) == 0:
                print("❌ Invalid query embedding")
                return []
            
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                print("❌ Query embedding has zero norm")
                return []
            
            # Normalize query embedding for efficient dot product
            query_normalized = query_embedding / query_norm
            
            # Get all vectors and compute similarity efficiently
            cursor = self.vectors.find({})
            
            results = []
            async for doc in cursor:
                try:
                    # Validate document embedding
                    if 'embedding' not in doc or not doc['embedding']:
                        continue
                    
                    doc_embedding = np.array(doc['embedding'])
                    if len(doc_embedding) != len(query_normalized):
                        continue
                    
                    # Calculate cosine similarity (both normalized, so dot product = cosine)
                    doc_norm = np.linalg.norm(doc_embedding)
                    if doc_norm == 0:
                        continue
                    
                    doc_normalized = doc_embedding / doc_norm
                    similarity = float(np.dot(query_normalized, doc_normalized))
                    
                    if similarity >= min_similarity:
                        results.append({
                            'text': doc.get('text', ''),
                            'similarity': similarity,
                            'doc_id': doc.get('doc_id', ''),
                            'chunk_index': doc.get('chunk_index', 0),
                            'metadata': doc.get('metadata', {}),
                            '_id': str(doc.get('_id', ''))
                        })
                except Exception as e:
                    continue
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"❌ Error searching vectors: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    async def search_keywords(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search using MongoDB text index for keyword matching"""
        try:
            # Use MongoDB text search
            cursor = self.vectors.find(
                {"$text": {"$search": query}},
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k * 2)
            
            results = []
            async for doc in cursor:
                results.append({
                    'text': doc.get('text', ''),
                    'similarity': float(doc.get('score', 0.0)),
                    'doc_id': doc.get('doc_id', ''),
                    'chunk_index': doc.get('chunk_index', 0),
                    'metadata': doc.get('metadata', {}),
                    '_id': str(doc.get('_id', ''))
                })
            
            return results[:top_k]
        except Exception as e:
            # If text index doesn't exist, return empty
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
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its metadata"""
        try:
            result = await self.documents.delete_one({"doc_id": doc_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return False
    
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