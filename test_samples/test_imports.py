#!/usr/bin/env python3
"""
Test that all required Python imports work correctly.
"""

print("Testing Python imports...")

try:
    print("  numpy...", end="")
    import numpy as np
    print(" ✓")
    
    print("  jax...", end="")
    import jax
    import jax.numpy as jnp
    print(" ✓")
    
    print("  safetensors...", end="")
    from safetensors import safe_open
    from safetensors.numpy import load_file
    print(" ✓")
    
    print("  json...", end="")
    import json
    print(" ✓")
    
    print("  pathlib...", end="")
    from pathlib import Path
    print(" ✓")
    
    print("  sqlite3...", end="")
    import sqlite3
    print(" ✓")
    
    print("  tempfile...", end="")
    import tempfile
    print(" ✓")
    
    print("  argparse...", end="")
    import argparse
    print(" ✓")
    
    print("\n✅ All imports successful!")
    
    # Test JAX device detection
    print(f"\nJAX devices detected: {len(jax.devices())}")
    for i, device in enumerate(jax.devices()):
        print(f"  Device {i}: {device}")
    
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    print("\nYou may need to install: pip install numpy jax safetensors")
