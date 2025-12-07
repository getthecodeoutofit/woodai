# Qdrant Troubleshooting Guide

## Lock File Issues

### Problem: "Storage folder is already accessed by another instance"

This happens when:
- Multiple instances of the application try to access the same Qdrant database
- A previous instance didn't close properly
- The application is restarted while another instance is running

### Solutions

#### 1. Automatic Fallback (Default)
The system automatically falls back to **in-memory mode** if the database is locked:
- ✅ Application continues to work
- ⚠️ Data will not persist between restarts
- ✅ No manual intervention needed

#### 2. Clean Up Stale Locks
If a lock file is older than 5 minutes, it's considered stale and will be automatically cleaned up.

To manually clean up:
```python
from vector_store import cleanup_stale_qdrant_lock
cleanup_stale_qdrant_lock('qdrant_db', max_age_seconds=300)
```

Or manually:
```bash
rm backend/qdrant_db/.lock
```

#### 3. Close Other Instances
- Check for running Python processes: `ps aux | grep python`
- Close other instances of the application
- Restart the application

#### 4. Use Qdrant Server (For Production)
For concurrent access, use Qdrant server instead of local storage:

```bash
# Install Qdrant server
docker run -p 6333:6333 qdrant/qdrant

# Or use Qdrant Cloud
```

Then connect to server:
```python
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
```

## Current Behavior

- **Lock detected**: Automatically falls back to in-memory mode
- **Stale lock (>5 min)**: Automatically cleaned up
- **Active lock (<1 min)**: Uses in-memory mode to avoid conflicts

## Checking Lock Status

```python
from pathlib import Path
import time

lock_file = Path('qdrant_db/.lock')
if lock_file.exists():
    age = time.time() - lock_file.stat().st_mtime
    print(f"Lock age: {age:.0f} seconds")
else:
    print("No lock file")
```

## Best Practices

1. **Single Instance**: Run only one instance of the application per Qdrant database
2. **Graceful Shutdown**: Ensure the application closes Qdrant connections properly
3. **Production**: Use Qdrant server for multiple instances
4. **Development**: In-memory fallback is fine for testing



