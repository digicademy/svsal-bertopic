#!/usr/bin/env python3
"""
Diagnostic script to check cache and manifest status.
Run this to understand what the notebook found on startup.
"""

import os
import pickle
import json

# File paths (adjust if needed)
cache_path = "./out-data/embeddings_cache.pkl"
manifest_path = "./out-data/embeddings_manifest.json"

print("=" * 80)
print("EMBEDDING CACHE DIAGNOSTIC")
print("=" * 80)

# Check cache
if os.path.exists(cache_path):
    print(f"\n✓ Cache file exists: {cache_path}")
    print(f"  Size: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB")
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"\n  Providers in cache: {len(cache_data)}")
        for provider_id, embeddings in cache_data.items():
            print(f"    - {provider_id}: {len(embeddings)} embeddings")
    except Exception as e:
        print(f"  ✗ Error loading cache: {e}")
else:
    print(f"\n✗ Cache file NOT found: {cache_path}")
    print("  This is normal for first run")

# Check manifest
if os.path.exists(manifest_path):
    print(f"\n✓ Manifest file exists: {manifest_path}")
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        print(f"\n  Completed providers: {len(manifest.get('providers', {}))}")
        for provider_id, info in manifest.get('providers', {}).items():
            print(f"    - {provider_id}:")
            print(f"        Embeddings: {info['num_embeddings']}")
            print(f"        Completed: {info['completed_at']}")
            print(f"        File: {info['file']}")
    except Exception as e:
        print(f"  ✗ Error loading manifest: {e}")
else:
    print(f"\n✗ Manifest file NOT found: {manifest_path}")
    print("  This is normal for first run")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if os.path.exists(cache_path) or os.path.exists(manifest_path):
    print("""
The notebook found existing cache/manifest files from a previous run.
All documents were already processed, so it completed instantly.

TO START FRESH:
1. Delete (or rename) the cache and manifest files:
   rm ./out-data/embeddings_cache.pkl
   rm ./out-data/embeddings_manifest.json

2. Re-run the notebook

TO CONTINUE WITH EXISTING DATA:
- The notebook is working correctly
- All embeddings are in the cache
- Check the parquet files in ./out-data/ for results
""")
else:
    print("""
No cache or manifest files found.
The notebook should process all documents on next run.

If it still shows 0 embeddings:
1. Check that documents are being loaded (should print document count)
2. Check for errors in the provider initialization
3. Check API keys are set correctly
""")

print("=" * 80)
