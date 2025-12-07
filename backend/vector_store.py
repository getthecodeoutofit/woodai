"""
Advanced Vector Store using Qdrant
Qdrant is a modern, high-performance vector database optimized for similarity search
"""

from typing import List, Dict, Optional, Any
import numpy as np
from pathlib import Path
import uuid
from datetime import datetime
import os
import time

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
        CollectionStatus, UpdateStatus
    )
    # Try to import MatchAny (may not be available in older versions)
    try:
        from qdrant_client.models import MatchAny
        MATCH_ANY_AVAILABLE = True
    except ImportError:
        MatchAny = None
        MATCH_ANY_AVAILABLE = False
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    MATCH_ANY_AVAILABLE = False
    MatchAny = None
    print("⚠️ Qdrant not available. Install with: pip install qdrant-client")


def cleanup_qdrant_lock(qdrant_path: str, force: bool = False, max_age_seconds: int = 180) -> bool:
    """
    Clean up Qdrant lock file
    
    Args:
        qdrant_path: Path to Qdrant database directory
        force: If True, remove lock file regardless of age (use with caution)
        max_age_seconds: Maximum age of lock file before considering it stale (default: 5 minutes)
    
    Returns:
        True if lock was cleaned up, False otherwise
    """
    try:
        qdrant_path_obj = Path(qdrant_path)
        # Check for lock file in the directory
        lock_file = qdrant_path_obj / ".lock"
        
        # Also check in collection subdirectories (some Qdrant versions store locks there)
        collection_lock_files = list(qdrant_path_obj.glob("**/.lock"))
        
        all_lock_files = [lock_file] if lock_file.exists() else []
        all_lock_files.extend(collection_lock_files)
        
        cleaned = False
        for lock_file_path in all_lock_files:
            if lock_file_path.exists():
                try:
                    lock_age = time.time() - lock_file_path.stat().st_mtime
                    if force or lock_age > max_age_seconds:
                        if force:
                            print(f"🧹 Force removing Qdrant lock: {lock_file_path}")
                        else:
                            print(f"🧹 Removing stale Qdrant lock: {lock_file_path} (age: {lock_age:.0f}s)")
                        lock_file_path.unlink()
                        cleaned = True
                    else:
                        print(f"⚠️ Lock file exists but is recent (age: {lock_age:.0f}s). Use force=True to remove.")
                except (OSError, PermissionError) as e:
                    print(f"⚠️ Could not remove lock file {lock_file_path}: {e}")
                except Exception as e:
                    print(f"⚠️ Error checking lock file {lock_file_path}: {e}")
        
        return cleaned
    except Exception as e:
        print(f"⚠️ Error in cleanup_qdrant_lock: {e}")
        return False


def cleanup_stale_qdrant_lock(qdrant_path: str, max_age_seconds: int = 180) -> bool:
    """
    Clean up stale Qdrant lock file if it's older than max_age_seconds
    (Legacy function - use cleanup_qdrant_lock for more options)
    
    Args:
        qdrant_path: Path to Qdrant database directory
        max_age_seconds: Maximum age of lock file before considering it stale (default: 5 minutes)
    
    Returns:
        True if lock was cleaned up, False otherwise
    """
    return cleanup_qdrant_lock(qdrant_path, force=False, max_age_seconds=max_age_seconds)
    """
    Clean up stale Qdrant lock file if it's older than max_age_seconds
    
    Args:
        qdrant_path: Path to Qdrant database directory
        max_age_seconds: Maximum age of lock file before considering it stale (default: 5 minutes)
    
    Returns:
        True if lock was cleaned up, False otherwise
    """
    try:
        qdrant_path_obj = Path(qdrant_path)
        # Check for lock file in the directory
        lock_file = qdrant_path_obj / ".lock"
        
        # Also check in collection subdirectories (some Qdrant versions store locks there)
        collection_lock_files = list(qdrant_path_obj.glob("**/.lock"))
        
        all_lock_files = [lock_file] if lock_file.exists() else []
        all_lock_files.extend(collection_lock_files)
        
        cleaned = False
        for lock_file_path in all_lock_files:
            if lock_file_path.exists():
                try:
                    lock_age = time.time() - lock_file_path.stat().st_mtime
                    if lock_age > max_age_seconds:
                        print(f"🧹 Removing stale Qdrant lock: {lock_file_path} (age: {lock_age:.0f}s)")
                        lock_file_path.unlink()
                        cleaned = True
                except (OSError, PermissionError) as e:
                    print(f"⚠️ Could not remove lock file {lock_file_path}: {e}")
                    # Try to check if process is still running
                    try:
                        import psutil
                        # Check if any process is using the lock file
                        for proc in psutil.process_iter(['pid', 'name']):
                            try:
                                if 'qdrant' in proc.info['name'].lower() or 'python' in proc.info['name'].lower():
                                    # Process might be holding the lock
                                    pass
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                    except ImportError:
                        pass  # psutil not available
                except Exception as e:
                    print(f"⚠️ Error checking lock file {lock_file_path}: {e}")
        
        return cleaned
    except Exception as e:
        print(f"⚠️ Error in cleanup_stale_qdrant_lock: {e}")
        return False


class QdrantVectorStore:
    """Advanced vector store using Qdrant for high-performance similarity search"""
    
    def __init__(
        self,
        collection_name: str = "woodai_documents",
        path: Optional[str] = None,
        embedding_dim: int = 768,
        distance: Distance = Distance.COSINE,
        use_memory_fallback: bool = True
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
        
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.distance = distance
        self.path = Path(path) if path else None
        self.use_memory = False
        
        # Initialize Qdrant client
        if path:
            # Local persistent storage
            self.path = Path(path)
            self.path.mkdir(parents=True, exist_ok=True)
            
            # Try to initialize with local storage
            # First check if lock exists and clean up stale locks
            lock_file = self.path / ".lock"
            # Use environment variable or default to 180 seconds (3 minutes) for stale lock threshold
            stale_lock_threshold = int(os.getenv('QDRANT_STALE_LOCK_SECONDS', '180'))
            
            if lock_file.exists():
                lock_age = time.time() - lock_file.stat().st_mtime
                if lock_age > stale_lock_threshold:  # Stale lock - clean it up
                    print(f"🧹 Found stale lock (age: {lock_age:.0f}s), cleaning up...")
                    if cleanup_qdrant_lock(str(self.path), force=False, max_age_seconds=stale_lock_threshold):
                        # Lock was cleaned, continue with initialization
                        print("   Lock cleaned, retrying initialization...")
                    else:
                        # Could not clean lock, use fallback if enabled
                        if use_memory_fallback:
                            print("   Could not clean lock, using in-memory mode")
                            self.client = QdrantClient(":memory:")
                            self.use_memory = True
                            self._ensure_collection()
                            return
                elif use_memory_fallback:  # Recent lock - use fallback immediately
                    print(f"⚠️ Qdrant database is locked (lock age: {lock_age:.0f}s)")
                    print("   Using in-memory mode to avoid conflicts")
                    print(f"   (Lock will be cleaned automatically if older than {stale_lock_threshold}s)")
                    self.client = QdrantClient(":memory:")
                    self.use_memory = True
                    self._ensure_collection()
                    return
            
            try:
                self.client = QdrantClient(path=str(self.path))
                print(f"✅ Qdrant initialized (local): {self.path}")
            except (RuntimeError, Exception) as e:
                error_str = str(e).lower()
                error_type = type(e).__name__
                # Check for lock-related errors
                if ("already accessed" in error_str or 
                    "alreadylocked" in error_type.lower() or
                    "resource temporarily unavailable" in error_str or
                    "lock" in error_str):
                    print(f"⚠️ Qdrant database is locked by another instance")
                    if use_memory_fallback:
                        print("   Falling back to in-memory mode (data will not persist)")
                        self.client = QdrantClient(":memory:")
                        self.use_memory = True
                    else:
                        # Try to clean up stale lock
                        if cleanup_stale_qdrant_lock(str(self.path)):
                            try:
                                # Retry after cleanup
                                self.client = QdrantClient(path=str(self.path))
                                print(f"✅ Qdrant initialized (local, after lock cleanup): {self.path}")
                            except Exception as retry_error:
                                if use_memory_fallback:
                                    print("   Retry after cleanup failed, using in-memory mode")
                                    self.client = QdrantClient(":memory:")
                                    self.use_memory = True
                                else:
                                    raise RuntimeError(
                                        f"Qdrant database still locked after cleanup: {retry_error}"
                                    )
                        else:
                            # Lock exists but is not stale
                            lock_file = self.path / ".lock"
                            if lock_file.exists():
                                lock_age = time.time() - lock_file.stat().st_mtime
                                raise RuntimeError(
                                    f"Qdrant database is locked by another instance. "
                                    f"Close other instances or wait. Lock age: {lock_age:.0f}s"
                                )
                            else:
                                # No lock file but still getting error - try in-memory
                                if use_memory_fallback:
                                    print("   Using in-memory mode as fallback")
                                    self.client = QdrantClient(":memory:")
                                    self.use_memory = True
                                else:
                                    raise
                else:
                    # Not a lock error, re-raise
                    raise
        else:
            # In-memory (for testing)
            self.client = QdrantClient(":memory:")
            self.use_memory = True
            print("✅ Qdrant initialized (in-memory)")
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=self.distance
                    )
                )
                print(f"✅ Created Qdrant collection: {self.collection_name}")
                print(f"   Collection is empty - ready for document indexing")
            else:
                # Verify collection configuration
                collection_info = self.client.get_collection(self.collection_name)
                existing_dim = collection_info.config.params.vectors.size
                points_count = collection_info.points_count
                
                if existing_dim != self.embedding_dim:
                    print(f"⚠️ Collection dimension mismatch detected!")
                    print(f"   Existing collection: {existing_dim}D, Required: {self.embedding_dim}D")
                    print(f"   Collection has {points_count} indexed documents")
                    
                    if points_count > 0:
                        print(f"⚠️ WARNING: Dimension mismatch with {points_count} existing documents!")
                        print(f"   This usually happens when the embedding model changed.")
                        print(f"   To preserve data: Use the same embedding model or manually delete qdrant_db/")
                        print(f"   To recreate: Delete backend/qdrant_db/ directory and restart")
                        raise RuntimeError(
                            f"Collection dimension mismatch ({existing_dim}D vs {self.embedding_dim}D) with {points_count} existing documents. "
                            f"To fix: Use the same embedding model, or delete backend/qdrant_db/ to recreate collection."
                        )
                    
                    print(f"   Deleting and recreating empty collection...")
                    self.client.delete_collection(self.collection_name)
                    self._ensure_collection()
                else:
                    print(f"✅ Using existing Qdrant collection: {self.collection_name}")
                    print(f"   📊 Collection statistics:")
                    print(f"      - Vector dimension: {existing_dim}D")
                    print(f"      - Indexed documents (points): {points_count}")
                    
                    if points_count > 0:
                        print(f"   ✅ {points_count} document chunks are available for RAG queries")
                    else:
                        print(f"   ⚠️ Collection is empty - no documents indexed yet")
        except Exception as e:
            print(f"❌ Error ensuring collection: {e}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            texts: List of text chunks
            embeddings: numpy array of embeddings (n_chunks, embedding_dim)
            metadatas: Optional list of metadata dicts
            ids: Optional list of document IDs
        
        Returns:
            List of document IDs
        """
        if len(texts) != len(embeddings):
            raise ValueError(f"Texts ({len(texts)}) and embeddings ({len(embeddings)}) must have same length")
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Prepare points for Qdrant
        points = []
        for i, (text, embedding, metadata, doc_id) in enumerate(zip(texts, embeddings, metadatas, ids)):
            # Convert embedding to list
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
            
            # Prepare payload (metadata + text for keyword search)
            payload = {
                "text": text,
                "doc_id": metadata.get("doc_id", doc_id),
                "chunk_index": metadata.get("chunk_index", i),
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", ""),
                "section": metadata.get("section", ""),
                "type": metadata.get("type", "unknown"),
                "chunk_size_tokens": metadata.get("chunk_size_tokens", 0),
                "total_chunks": metadata.get("total_chunks", 0),
                "split_method": metadata.get("split_method", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
                **{k: v for k, v in metadata.items() if k not in ["doc_id", "chunk_index", "source", "page", "section", "type", "chunk_size_tokens", "total_chunks", "split_method"]}
            }
            
            # Generate unique ID (use hash of doc_id + chunk_index, ensure positive)
            point_id = abs(hash(doc_id + str(i))) % (2**63 - 1)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding_list,
                    payload=payload
                )
            )
        
        # Batch upsert
        try:
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                print(f"✅ Added {len(points)} documents to Qdrant")
            else:
                print(f"⚠️ Qdrant upsert status: {operation_info.status}")
            
            return ids
        except AttributeError as e:
            error_msg = (
                f"❌ Qdrant API Error (upsert): {e}\n"
                f"   The 'upsert' method is not available on Qdrant client.\n"
                f"   This might indicate an incompatible qdrant-client version.\n"
                f"   Solution: pip install --upgrade qdrant-client"
            )
            print(error_msg)
            raise RuntimeError(f"Qdrant upsert failed: {e}") from e
        except Exception as e:
            error_msg = (
                f"❌ Error adding documents to Qdrant: {type(e).__name__}: {e}\n"
                f"   Collection: {self.collection_name}\n"
                f"   Points to add: {len(points)}\n"
                f"   Check if collection exists and Qdrant is accessible."
            )
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter conditions (e.g., {"source": "document.pdf"})
        
        Returns:
            List of search results with text, similarity, and metadata
        """
        # Convert embedding to list
        query_vector = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Build filter if conditions provided
        qdrant_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                # Support list of values (for multiple doc_ids)
                if isinstance(value, list):
                    if len(value) == 1:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value[0])
                            )
                        )
                    elif len(value) > 1:
                        # Use MatchAny if available, otherwise filter after search
                        if MATCH_ANY_AVAILABLE and MatchAny:
                            conditions.append(
                                FieldCondition(
                                    key=key,
                                    match=MatchAny(any=value)
                                )
                            )
                        else:
                            # Fallback: will filter after search
                            # For now, just use first value and filter later
                            conditions.append(
                                FieldCondition(
                                    key=key,
                                    match=MatchValue(value=value[0])
                                )
                            )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        try:
            # Search in Qdrant - use query_points() method (qdrant-client 1.16.1+)
            # For older versions, try fallback methods
            search_results = None
            error_messages = []
            
            # Method 1: query_points() method (qdrant-client 1.16.1+)
            try:
                query_params = {
                    "collection_name": self.collection_name,
                    "query": query_vector,  # Can be list of floats or numpy array
                    "limit": top_k,
                    "with_payload": True,
                    "with_vectors": False
                }
                if score_threshold is not None:
                    query_params["score_threshold"] = score_threshold
                if qdrant_filter is not None:
                    query_params["query_filter"] = qdrant_filter
                
                response = self.client.query_points(**query_params)
                # Extract points from QueryResponse
                if hasattr(response, 'points'):
                    search_results = response.points
                elif hasattr(response, 'result'):
                    search_results = response.result
                else:
                    # Try to iterate directly
                    search_results = list(response) if response else []
            except AttributeError as e1:
                error_messages.append(f"Method 'query_points' failed: {e1}")
                # Method 2: Try search() method (qdrant-client 1.7.0-1.15.x)
                try:
                    search_params = {
                        "collection_name": self.collection_name,
                        "query_vector": query_vector,
                        "limit": top_k,
                        "with_payload": True,
                        "with_vectors": False
                    }
                    if score_threshold is not None:
                        search_params["score_threshold"] = score_threshold
                    if qdrant_filter is not None:
                        search_params["query_filter"] = qdrant_filter
                    
                    search_results = self.client.search(**search_params)
                except AttributeError as e2:
                    error_messages.append(f"Method 'search' failed: {e2}")
                    # Method 3: Try search_points (very old API)
                    try:
                        search_params = {
                            "collection_name": self.collection_name,
                            "query_vector": query_vector,
                            "limit": top_k,
                            "with_payload": True,
                            "with_vectors": False
                        }
                        if score_threshold is not None:
                            search_params["score_threshold"] = score_threshold
                        if qdrant_filter is not None:
                            search_params["query_filter"] = qdrant_filter
                        
                        search_results = self.client.search_points(**search_params)
                    except AttributeError as e3:
                        error_messages.append(f"Method 'search_points' failed: {e3}")
                        raise RuntimeError(
                            f"All Qdrant search methods failed. Errors: {'; '.join(error_messages)}\n"
                            f"Available query methods: {[m for m in dir(self.client) if 'query' in m.lower() or 'search' in m.lower()]}\n"
                            f"Qdrant client type: {type(self.client).__name__}\n"
                            f"Please check qdrant-client version compatibility."
                        )
            except Exception as e:
                error_messages.append(f"Unexpected error in search: {e}")
                raise RuntimeError(
                    f"Qdrant search error: {e}\n"
                    f"Previous errors: {'; '.join(error_messages) if error_messages else 'None'}"
                )
            
            if search_results is None:
                raise RuntimeError("Search returned None - no results and no error")
            
            # Format results - handle both QueryResponse.points and direct list of ScoredPoint
            results = []
            for result in search_results:
                # Handle different response formats
                if hasattr(result, 'payload'):
                    payload = result.payload
                    score = float(result.score) if hasattr(result, 'score') else 0.0
                    point_id = str(result.id) if hasattr(result, 'id') else ''
                elif hasattr(result, 'point'):
                    # QueryResponse format
                    payload = result.point.payload if hasattr(result.point, 'payload') else {}
                    score = float(result.score) if hasattr(result, 'score') else 0.0
                    point_id = str(result.point.id) if hasattr(result.point, 'id') else ''
                else:
                    # Fallback
                    payload = getattr(result, 'payload', {})
                    score = float(getattr(result, 'score', 0.0))
                    point_id = str(getattr(result, 'id', ''))
                
                results.append({
                    'text': payload.get('text', '') if isinstance(payload, dict) else '',
                    'similarity': score,
                    'doc_id': payload.get('doc_id', '') if isinstance(payload, dict) else '',
                    'chunk_index': payload.get('chunk_index', 0) if isinstance(payload, dict) else 0,
                    'metadata': {
                        'source': payload.get('source', 'unknown') if isinstance(payload, dict) else 'unknown',
                        'page': payload.get('page', '') if isinstance(payload, dict) else '',
                        'section': payload.get('section', '') if isinstance(payload, dict) else '',
                        'type': payload.get('type', 'unknown') if isinstance(payload, dict) else 'unknown',
                        **{k: v for k, v in (payload.items() if isinstance(payload, dict) else []) 
                           if k not in ['text', 'doc_id', 'chunk_index', 'source', 'page', 'section', 'type']}
                    },
                    'id': point_id
                })
            
            return results
        except AttributeError as e:
            error_msg = (
                f"❌ Qdrant API Error: {e}\n"
                f"   The Qdrant client method is not available.\n"
                f"   This might be due to:\n"
                f"   1. Incompatible qdrant-client version\n"
                f"   2. Qdrant client not properly initialized\n"
                f"   Solution: Try 'pip install --upgrade qdrant-client'\n"
                f"   Available methods: {[m for m in dir(self.client) if not m.startswith('_')][:10]}"
            )
            print(error_msg)
            import traceback
            traceback.print_exc()
            return []
        except RuntimeError as e:
            error_msg = (
                f"❌ Qdrant Runtime Error: {e}\n"
                f"   This indicates a problem with the Qdrant client or database.\n"
                f"   Solutions:\n"
                f"   1. Check if Qdrant database is accessible\n"
                f"   2. Verify collection exists: {self.collection_name}\n"
                f"   3. Try restarting the backend\n"
                f"   4. Check qdrant-client version: pip show qdrant-client"
            )
            print(error_msg)
            import traceback
            traceback.print_exc()
            return []
        except Exception as e:
            error_msg = (
                f"❌ Unexpected error searching Qdrant: {type(e).__name__}: {e}\n"
                f"   Collection: {self.collection_name}\n"
                f"   Top K: {top_k}\n"
                f"   This is an unexpected error. Please report this issue."
            )
            print(error_msg)
            import traceback
            traceback.print_exc()
            return []
    
    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword matching
        
        Args:
            query_embedding: Query embedding vector
            query_text: Original query text for keyword matching
            top_k: Number of results to return
            semantic_weight: Weight for semantic similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            filter_conditions: Optional filter conditions (e.g., {"doc_id": ["id1", "id2"]})
        
        Returns:
            Combined and re-ranked results
        """
        # Semantic search with filter
        semantic_results = self.search(
            query_embedding, 
            top_k=top_k * 2,
            filter_conditions=filter_conditions
        )
        
        # Keyword search (simple text matching in payload)
        keyword_results = []
        try:
            # Build filter for scroll if filter_conditions provided
            qdrant_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    if isinstance(value, list):
                        if len(value) == 1:
                            conditions.append(
                                FieldCondition(
                                    key=key,
                                    match=MatchValue(value=value[0])
                                )
                            )
                        elif len(value) > 1:
                            if MATCH_ANY_AVAILABLE and MatchAny:
                                conditions.append(
                                    FieldCondition(
                                        key=key,
                                        match=MatchAny(any=value)
                                    )
                                )
                            else:
                                # Fallback: use first value, will filter after
                                conditions.append(
                                    FieldCondition(
                                        key=key,
                                        match=MatchValue(value=value[0])
                                    )
                                )
                    else:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Get documents with filter applied
            scroll_params = {
                "collection_name": self.collection_name,
                "limit": 1000,  # Adjust based on collection size
                "with_payload": True,
                "with_vectors": False
            }
            if qdrant_filter:
                scroll_params["query_filter"] = qdrant_filter
            
            scroll_results = self.client.scroll(**scroll_params)
            
            query_words = set(query_text.lower().split())
            for point in scroll_results[0]:
                payload = point.payload
                
                # Apply post-filter if MatchAny not available and we have multiple doc_ids
                if filter_conditions:
                    doc_id = payload.get('doc_id', '')
                    filter_doc_ids = filter_conditions.get('doc_id', [])
                    if isinstance(filter_doc_ids, list) and len(filter_doc_ids) > 1:
                        if not (MATCH_ANY_AVAILABLE and MatchAny):
                            # Post-filter if MatchAny not available
                            if doc_id not in filter_doc_ids:
                                continue
                    elif isinstance(filter_doc_ids, list) and len(filter_doc_ids) == 1:
                        if doc_id != filter_doc_ids[0]:
                            continue
                
                text = payload.get('text', '').lower()
                text_words = set(text.split())
                
                # Calculate keyword score (Jaccard similarity)
                if query_words and text_words:
                    intersection = len(query_words & text_words)
                    union = len(query_words | text_words)
                    keyword_score = intersection / union if union > 0 else 0.0
                    
                    if keyword_score > 0:
                        keyword_results.append({
                            'text': payload.get('text', ''),
                            'similarity': keyword_score,
                            'doc_id': payload.get('doc_id', ''),
                            'chunk_index': payload.get('chunk_index', 0),
                            'metadata': {
                                'source': payload.get('source', 'unknown'),
                                'page': payload.get('page', ''),
                                'section': payload.get('section', ''),
                                'type': payload.get('type', 'unknown')
                            },
                            'id': str(point.id),
                            'search_type': 'keyword'
                        })
        except AttributeError as e:
            print(f"⚠️ Keyword search API error: {e}")
            print(f"   Qdrant scroll method not available. Skipping keyword search.")
        except Exception as e:
            print(f"⚠️ Keyword search error: {type(e).__name__}: {e}")
            print(f"   Continuing with semantic search only.")
        
        # Combine and re-rank
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result.get('doc_id', '') + '_' + str(result.get('chunk_index', 0))
            combined_results[doc_id] = {
                **result,
                'semantic_score': result['similarity'],
                'keyword_score': 0.0,
                'combined_score': result['similarity'] * semantic_weight
            }
        
        # Add/update with keyword results
        for result in keyword_results:
            doc_id = result.get('doc_id', '') + '_' + str(result.get('chunk_index', 0))
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = result['similarity']
                combined_results[doc_id]['combined_score'] = (
                    combined_results[doc_id]['semantic_score'] * semantic_weight +
                    result['similarity'] * keyword_weight
                )
            else:
                combined_results[doc_id] = {
                    **result,
                    'semantic_score': 0.0,
                    'keyword_score': result['similarity'],
                    'combined_score': result['similarity'] * keyword_weight
                }
        
        # Sort by combined score and return top_k
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        # Update similarity to combined score for consistency
        for result in sorted_results[:top_k]:
            result['similarity'] = result['combined_score']
        
        return sorted_results[:top_k]
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents by doc_id"""
        try:
            # Delete points matching filter for each doc_id
            for doc_id in doc_ids:
                # Filter by doc_id in payload
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                )
                
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=filter_condition
                )
            
            print(f"✅ Deleted {len(doc_ids)} documents from Qdrant")
            return True
        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            info = {
                'name': self.collection_name,
                'points_count': collection_info.points_count,
                'indexed_vectors_count': getattr(collection_info, 'indexed_vectors_count', collection_info.points_count),
                'status': collection_info.status.value if hasattr(collection_info.status, 'value') else str(collection_info.status)
            }
            # Try to get vectors_count if available (older Qdrant versions)
            if hasattr(collection_info, 'vectors_count'):
                info['vectors_count'] = collection_info.vectors_count
            else:
                # In newer versions, vectors_count == points_count
                info['vectors_count'] = collection_info.points_count
            return info
        except AttributeError as e:
            # Handle missing attributes gracefully
            print(f"⚠️ CollectionInfo missing attribute: {e}")
            try:
                collection_info = self.client.get_collection(self.collection_name)
                return {
                    'name': self.collection_name,
                    'points_count': getattr(collection_info, 'points_count', 0),
                    'vectors_count': getattr(collection_info, 'points_count', 0),
                    'indexed_vectors_count': getattr(collection_info, 'indexed_vectors_count', 0),
                    'status': 'unknown'
                }
            except Exception as e2:
                print(f"❌ Error getting collection info: {e2}")
                return {}
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            print(f"✅ Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
            return False

