#!/usr/bin/env python3
"""
Fix Qdrant dimension mismatch by deleting and recreating the collection
"""

import sys
from pathlib import Path

def fix_qdrant_dimension():
    """Delete Qdrant database to fix dimension mismatch"""
    qdrant_path = Path(__file__).parent / "qdrant_db"
    
    if qdrant_path.exists():
        print(f"🗑️  Deleting Qdrant database at: {qdrant_path}")
        import shutil
        try:
            shutil.rmtree(qdrant_path)
            print(f"✅ Qdrant database deleted successfully")
            print(f"   The collection will be recreated with the correct dimension on next run")
            return True
        except Exception as e:
            print(f"❌ Error deleting Qdrant database: {e}")
            return False
    else:
        print(f"ℹ️  Qdrant database not found at: {qdrant_path}")
        print(f"   No action needed")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("Qdrant Dimension Mismatch Fix")
    print("=" * 60)
    print()
    
    if fix_qdrant_dimension():
        print()
        print("✅ Fix complete! Restart your application to recreate the database.")
        sys.exit(0)
    else:
        print()
        print("❌ Fix failed. Please delete the Qdrant database manually:")
        print("   rm -rf backend/qdrant_db")
        sys.exit(1)



