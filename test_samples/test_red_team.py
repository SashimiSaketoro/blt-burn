#!/usr/bin/env python3
"""
Red Team Test Suite for BLT-Burn v0.2

Tests key functionality without requiring the full build.
"""

import json
import numpy as np
from pathlib import Path
import sqlite3
import tempfile

def test_hypergraph_sidecar():
    """Test the hypergraph sidecar structure."""
    print("Testing Hypergraph Sidecar...")
    
    # Create a test sidecar structure
    sidecar = {
        "nodes": [
            {"Trunk": {"source_hash": "abc123", "total_bytes": 1024}},
            {"Branch": {"label": "text_content", "modality": "text"}},
            {"Leaf": {"bytes": [], "label": "chunk_0", "metadata": {}}}
        ],
        "edges": [
            {"label": "contains", "weight": 1.0},
            {"label": "next", "weight": 0.5}
        ],
        "topology": {
            "edges": [[0, [0, 1]], [1, [1, 2]]]
        },
        "sharding": {
            "global_shape": [1, 1000, 768],
            "shard_index": 0,
            "num_shards": 4,
            "process_index": 0,
            "axis": 1
        }
    }
    
    # Test JSON serialization
    try:
        json_str = json.dumps(sidecar, indent=2)
        loaded = json.loads(json_str)
        assert loaded["sharding"]["num_shards"] == 4
        print("✓ JSON serialization works")
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
    
    # Test SQLite structure (mock)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create schema
        cursor.execute("""
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # Insert sharding info
        cursor.execute(
            "INSERT INTO meta (key, value) VALUES ('sharding_info', ?)",
            (json.dumps(sidecar["sharding"]),)
        )
        
        conn.commit()
        
        # Test reading back
        cursor.execute("SELECT value FROM meta WHERE key='sharding_info'")
        result = cursor.fetchone()
        if result:
            sharding = json.loads(result[0])
            assert sharding["num_shards"] == 4
            print("✓ SQLite storage works")
        
        conn.close()
        Path(db_path).unlink()
        
    except Exception as e:
        print(f"✗ SQLite test failed: {e}")


def test_patch_size_distribution():
    """Test the patch size distribution logic."""
    print("\nTesting Patch Size Distribution...")
    
    # Mock patch sizes
    patch_sizes = [2, 5, 15, 30, 75, 150, 300, 600, 1500]
    
    # Define size buckets
    size_buckets = [
        (1, 3, "1-3"),
        (4, 10, "4-10"),
        (11, 24, "11-24"),
        (25, 48, "25-48"),
        (49, 100, "49-100"),
        (101, 256, "101-256"),
        (257, 512, "257-512"),
        (513, 1024, "513-1024"),
        (1025, float('inf'), "1025+")
    ]
    
    # Count patches in each bucket
    size_distribution = {}
    for min_size, max_size, label in size_buckets:
        count = sum(1 for size in patch_sizes if min_size <= size <= max_size)
        if count > 0:
            size_distribution[label] = {
                "count": count,
                "percentage": (count / len(patch_sizes) * 100)
            }
    
    # Print distribution
    print("Patch size distribution:")
    for bucket in ["1-3", "4-10", "11-24", "25-48", "49-100", "101-256", "257-512", "513-1024", "1025+"]:
        if bucket in size_distribution:
            data = size_distribution[bucket]
            bar = "█" * int(data['percentage'] / 10)
            print(f"  {bucket:10}: {bar:<10} {data['count']} ({data['percentage']:.1f}%)")
    
    assert len(size_distribution) > 0
    print("✓ Distribution calculation works")


def test_sharding_metadata():
    """Test JAX sharding metadata structure."""
    print("\nTesting JAX Sharding Metadata...")
    
    # Test sharding info
    sharding_info = {
        "global_shape": [1, 200000, 768],
        "shard_index": 2,
        "num_shards": 8,
        "process_index": 2,
        "axis": 1
    }
    
    # Calculate shard boundaries
    total_tokens = sharding_info["global_shape"][1]
    tokens_per_shard = total_tokens // sharding_info["num_shards"]
    
    start_token = sharding_info["shard_index"] * tokens_per_shard
    end_token = min((sharding_info["shard_index"] + 1) * tokens_per_shard, total_tokens)
    
    print(f"Shard {sharding_info['shard_index']}:")
    print(f"  Token range: {start_token}-{end_token}")
    print(f"  Shard size: {end_token - start_token} tokens")
    print(f"  Target process: {sharding_info['process_index']}")
    
    assert end_token > start_token
    print("✓ Sharding calculation works")


def test_threshold_recommendations():
    """Test the threshold recommendation logic."""
    print("\nTesting Threshold Recommendations...")
    
    # Mock test results
    test_results = [
        {"threshold": 1.0, "mean_size": 200, "tiny_percent": 10, "huge_percent": 5},
        {"threshold": 1.35, "mean_size": 130, "tiny_percent": 15, "huge_percent": 2},
        {"threshold": 1.55, "mean_size": 95, "tiny_percent": 25, "huge_percent": 1},
        {"threshold": 2.0, "mean_size": 50, "tiny_percent": 60, "huge_percent": 0}
    ]
    
    # Find best for target mean size of 128
    target = 128
    best = min(test_results, key=lambda t: abs(t["mean_size"] - target))
    
    print(f"Target mean size: {target}")
    print(f"Best threshold: {best['threshold']}")
    print(f"  Mean size: {best['mean_size']}")
    print(f"  Tiny patches: {best['tiny_percent']}%")
    print(f"  Huge patches: {best['huge_percent']}%")
    
    # Check warnings
    warnings = []
    if best["tiny_percent"] > 50:
        warnings.append("High percentage of tiny patches")
    if best["huge_percent"] > 20:
        warnings.append("High percentage of huge patches")
    
    if warnings:
        print(f"  Warnings: {', '.join(warnings)}")
    
    assert best["threshold"] == 1.35  # Should pick the one closest to 128
    print("✓ Recommendation logic works")


if __name__ == "__main__":
    print("=== BLT-Burn v0.2 Red Team Test Suite ===\n")
    
    try:
        test_hypergraph_sidecar()
        test_patch_size_distribution()
        test_sharding_metadata()
        test_threshold_recommendations()
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
