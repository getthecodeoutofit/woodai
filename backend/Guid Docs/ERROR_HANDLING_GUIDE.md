# Error Handling Guide

## Overview

This document describes the comprehensive error handling system implemented throughout the WoodAI backend to ensure robust operation and clear error messages.

## Qdrant API Compatibility Fix

### Issue Fixed
The error `'QdrantClient' object has no attribute 'search_points'` has been resolved with a multi-method fallback system.

### Solution
The code now tries multiple Qdrant API methods in order:
1. `client.search()` - Standard method (qdrant-client 1.7.0+)
2. `client.search_points()` - Alternative method (older versions)
3. `collection.search()` - Collection-based method (if available)

### Error Messages
All errors now include:
- Clear error type identification
- Suggested solutions
- Available methods list
- Version compatibility information

## Error Handling Categories

### 1. Qdrant API Errors

**AttributeError**: Method not available
```
❌ Qdrant API Error: 'QdrantClient' object has no attribute 'search'
   Solutions:
   1. Upgrade qdrant-client: pip install --upgrade qdrant-client
   2. Check Qdrant client initialization
   3. Verify qdrant-client version compatibility
```

**RuntimeError**: Database access issues
```
❌ Qdrant Runtime Error: [error details]
   Solutions:
   1. Check if Qdrant database is accessible
   2. Verify collection exists
   3. Try restarting the backend
   4. Check database lock files
```

### 2. Search Errors

**Hybrid Search Errors**:
- Falls back to semantic search only
- Logs warning but continues operation
- Returns partial results if possible

**Semantic Search Errors**:
- Returns empty results list
- Logs detailed error information
- Continues with other search methods if available

### 3. Collection Management Errors

**Collection Creation Errors**:
- Checks for existing collections
- Handles dimension mismatches
- Provides clear error messages with solutions

**Collection Access Errors**:
- Verifies collection exists before operations
- Handles lock file issues
- Falls back to in-memory mode if needed

### 4. Document Addition Errors

**Validation Errors**:
- Validates text/embedding length matching
- Checks metadata format
- Provides clear validation messages

**Upsert Errors**:
- Handles API compatibility issues
- Provides version upgrade suggestions
- Logs detailed error information

## Error Handling Patterns

### Pattern 1: Multi-Method Fallback

```python
try:
    # Try primary method
    result = client.primary_method()
except AttributeError:
    try:
        # Fallback to alternative method
        result = client.alternative_method()
    except AttributeError:
        # Final fallback
        result = client.fallback_method()
```

### Pattern 2: Graceful Degradation

```python
try:
    # Preferred method
    results = hybrid_search()
except Exception as e:
    print(f"⚠️ Hybrid search failed: {e}")
    # Fallback to simpler method
    results = semantic_search()
```

### Pattern 3: Detailed Error Messages

```python
except Exception as e:
    error_msg = (
        f"❌ Error Type: {type(e).__name__}\n"
        f"   Error: {e}\n"
        f"   Context: [relevant context]\n"
        f"   Solutions:\n"
        f"   1. [Solution 1]\n"
        f"   2. [Solution 2]"
    )
    print(error_msg)
```

## Common Errors and Solutions

### Error: Qdrant API Method Not Found

**Symptoms**:
- `AttributeError: 'QdrantClient' object has no attribute 'search'`
- `AttributeError: 'QdrantClient' object has no attribute 'search_points'`

**Solutions**:
1. Upgrade qdrant-client: `pip install --upgrade qdrant-client`
2. Check version: `pip show qdrant-client`
3. Verify compatibility with Qdrant server version

### Error: Qdrant Database Locked

**Symptoms**:
- `Resource temporarily unavailable`
- `Already locked` errors

**Solutions**:
1. Wait for lock to clear (automatic after 5 minutes)
2. Manually remove lock file: `rm backend/qdrant_db/.lock`
3. Restart backend server
4. System will auto-fallback to in-memory mode

### Error: Collection Dimension Mismatch

**Symptoms**:
- Collection exists but has wrong embedding dimension
- Search returns no results

**Solutions**:
1. System auto-recreates collection with correct dimensions
2. Re-index documents after dimension change
3. Check embedding model configuration

### Error: Empty Search Results

**Symptoms**:
- Search returns no results
- No error message but no data

**Solutions**:
1. Check if documents are indexed
2. Verify collection has data: `client.get_collection_info()`
3. Lower similarity threshold
4. Check query embedding generation

## Error Logging

All errors are logged with:
- **Error Type**: AttributeError, RuntimeError, etc.
- **Error Message**: Detailed error description
- **Context**: Relevant operation context
- **Solutions**: Actionable fix suggestions
- **Stack Trace**: Full traceback for debugging

## Testing Error Handling

### Test Qdrant API Compatibility

```python
from vector_store import QdrantVectorStore

try:
    store = QdrantVectorStore()
    results = store.search(query_embedding, top_k=5)
    print(f"✅ Search successful: {len(results)} results")
except Exception as e:
    print(f"❌ Error: {e}")
```

### Test Error Recovery

```python
# Test graceful degradation
try:
    results = hybrid_search()
except:
    results = semantic_search()  # Fallback
```

## Best Practices

1. **Always provide solutions**: Every error message includes actionable solutions
2. **Graceful degradation**: System continues operation when possible
3. **Detailed logging**: Log enough information for debugging
4. **Version compatibility**: Handle multiple API versions
5. **User-friendly messages**: Clear, non-technical error descriptions
6. **Fallback mechanisms**: Always have a backup plan

## Future Error Prevention

### Version Checking
- Check qdrant-client version on startup
- Warn about incompatible versions
- Suggest upgrades when needed

### Health Checks
- Verify Qdrant connection on startup
- Check collection status
- Validate embedding dimensions

### Automatic Recovery
- Auto-cleanup stale locks
- Auto-recreate corrupted collections
- Auto-fallback to safe modes

## Summary

The error handling system ensures:
- ✅ **Robustness**: System continues operation despite errors
- ✅ **Clarity**: Clear error messages with solutions
- ✅ **Compatibility**: Works with multiple API versions
- ✅ **Recovery**: Automatic fallback mechanisms
- ✅ **Debugging**: Detailed logging for troubleshooting

All errors are now handled gracefully with clear solutions! 🛡️✨



