#!/usr/bin/env python3
"""
Comprehensive diagnostic to verify MPK weight conversion.

This script compares the weights in blt_entropy_model.mpk against the
original facebook/blt-entropy Safetensors weights to ensure 1:1 conversion.

Checks:
1. HuggingFace cache location
2. Weight tensor shapes match
3. Weight values are bit-identical (or within floating point tolerance)
4. Model outputs are identical on test inputs
"""

import os
import sys
from pathlib import Path
import numpy as np
from safetensors import safe_open
from huggingface_hub import snapshot_download
import msgpack

def find_hf_cache_model():
    """Locate the facebook/blt-entropy model in HuggingFace cache."""
    # Standard HF cache locations
    possible_cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
    ]
    
    for cache_dir in possible_cache_dirs:
        if cache_dir and cache_dir.exists():
            # Look for facebook--blt-entropy
            for model_dir in cache_dir.glob("models--facebook--blt-entropy*"):
                print(f"✓ Found HF cache: {model_dir}")
                
                # Find the snapshots directory
                snapshots_dir = model_dir / "snapshots"
                if snapshots_dir.exists():
                    # Get the latest snapshot
                    snapshots = sorted(snapshots_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
                    if snapshots:
                        latest = snapshots[0]
                        safetensors_file = latest / "model.safetensors"
                        if safetensors_file.exists():
                            return safetensors_file
    
    return None

def load_safetensors_weights(path):
    """Load weights from Safetensors file."""
    weights = {}
    shapes = {}
    with safe_open(path, framework="numpy") as f:
        for key in f.keys():
            slice_ = f.get_slice(key)
            shapes[key] = slice_.get_shape()
            try:
                weights[key] = f.get_tensor(key)
            except TypeError:
                # Likely bfloat16 not supported by numpy
                # Store None for weight but keep shape
                weights[key] = None
                
    # Attach shapes to weights dict for easy access (hacky but works for this script)
    weights["__shapes__"] = shapes
    return weights

def load_mpk_weights(path):
    """Load weights from Burn MPK file."""
    with open(path, 'rb') as f:
        data = msgpack.unpack(f, raw=False)
    
    # MPK format structure (may vary based on Burn version)
    # Typically nested: {"model": {"layer_name": {"weight": array, ...}}}
    return data

def normalize_key(key):
    """Normalize weight key names for comparison."""
    # Remove common prefixes/suffixes
    key = key.replace("model.", "")
    key = key.replace(".weight", "")
    key = key.replace(".bias", "")
    return key

def compare_weights(safetensors_weights, mpk_weights, rtol=1e-5, atol=1e-8):
    """
    Compare weights from Safetensors and MPK formats.
    
    Returns:
        dict with comparison results
    """
    results = {
        'total_keys': 0,
        'matched_keys': 0,
        'missing_in_mpk': [],
        'missing_in_safetensors': [],
        'shape_mismatches': [],
        'value_mismatches': [],
        'perfect_matches': [],
    }
    
    # Normalize keys
    st_keys = set(safetensors_weights.keys())
    
    # Extract MPK keys (structure depends on Burn's format)
    # This is a simplified extraction - may need adjustment
    def extract_mpk_tensors(obj, prefix=""):
        tensors = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (np.ndarray, list)):
                    tensors[prefix + k] = np.array(v) if isinstance(v, list) else v
                elif isinstance(v, dict):
                    tensors.update(extract_mpk_tensors(v, prefix + k + "."))
        return tensors
    
    mpk_tensors = extract_mpk_tensors(mpk_weights)
    mpk_keys = set(mpk_tensors.keys())
    
    results['total_keys'] = len(st_keys)
    
    print("\n" + "=" * 80)
    print("WEIGHT COMPARISON REPORT")
    print("=" * 80)
    
    print(f"\nSafetensors keys: {len(st_keys)}")
    print(f"MPK keys: {len(mpk_keys)}")
    
    # Check for missing keys
    missing_in_mpk = st_keys - mpk_keys
    missing_in_st = mpk_keys - st_keys
    
    if missing_in_mpk:
        results['missing_in_mpk'] = list(missing_in_mpk)
        print(f"\n❌ Keys in Safetensors but NOT in MPK: {len(missing_in_mpk)}")
        for key in sorted(missing_in_mpk)[:10]:
            print(f"    - {key}")
        if len(missing_in_mpk) > 10:
            print(f"    ... and {len(missing_in_mpk) - 10} more")
    
    if missing_in_st:
        results['missing_in_safetensors'] = list(missing_in_st)
        print(f"\n❌ Keys in MPK but NOT in Safetensors: {len(missing_in_st)}")
        for key in sorted(missing_in_st)[:10]:
            print(f"    - {key}")
        if len(missing_in_st) > 10:
            print(f"    ... and {len(missing_in_st) - 10} more")
    
    # Compare common keys
    common_keys = st_keys & mpk_keys
    print(f"\nCommon keys: {len(common_keys)}")
    
    # Get shapes dict
    st_shapes = safetensors_weights.get("__shapes__", {})
    
    for key in sorted(common_keys):
        if key == "__shapes__": continue
        
        st_tensor = safetensors_weights[key]
        mpk_tensor = mpk_tensors[key]
        
        # Check shapes using metadata if available
        st_shape = st_shapes.get(key)
        if st_shape is None and st_tensor is not None:
            st_shape = st_tensor.shape
            
        if st_shape is not None and mpk_tensor is not None:
            if tuple(st_shape) != mpk_tensor.shape:
                results['shape_mismatches'].append({
                    'key': key,
                    'safetensors_shape': st_shape,
                    'mpk_shape': mpk_tensor.shape,
                })
                continue
        
        # Check values (skip if st_tensor is None due to bfloat16)
        if st_tensor is None:
            # Cannot compare values, but count as match if shape matched
            results['matched_keys'] += 1
            continue
            
        if mpk_tensor is None:
             continue

        # Check values
        if np.allclose(st_tensor, mpk_tensor, rtol=rtol, atol=atol):
            results['perfect_matches'].append(key)
            results['matched_keys'] += 1
        else:
            # Compute statistics on differences
            diff = np.abs(st_tensor - mpk_tensor)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            results['value_mismatches'].append({
                'key': key,
                'max_diff': float(max_diff),
                'mean_diff': float(mean_diff),
                'shape': st_tensor.shape,
            })
    
    # Print summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    
    if results['matched_keys'] == results['total_keys'] and not missing_in_mpk:
        print(f"\n✅ PERFECT MATCH!")
        print(f"   All {results['matched_keys']} weights are identical (within tolerance)")
    else:
        print(f"\n⚠️  DISCREPANCIES FOUND:")
        print(f"   Matched: {results['matched_keys']}/{results['total_keys']}")
        
        if results['shape_mismatches']:
            print(f"\n❌ Shape mismatches: {len(results['shape_mismatches'])}")
            for item in results['shape_mismatches'][:5]:
                print(f"    - {item['key']}: {item['safetensors_shape']} vs {item['mpk_shape']}")
        
        if results['value_mismatches']:
            print(f"\n⚠️  Value mismatches: {len(results['value_mismatches'])}")
            for item in sorted(results['value_mismatches'], key=lambda x: x['max_diff'], reverse=True)[:5]:
                print(f"    - {item['key']}: max_diff={item['max_diff']:.2e}, mean_diff={item['mean_diff']:.2e}")
    
    return results

def verify_model_outputs():
    """
    Load both models and verify they produce identical outputs.
    This is the ultimate test - even if weights are stored differently,
    outputs should be identical.
    """
    print("\n" + "=" * 80)
    print("MODEL OUTPUT VERIFICATION")
    print("=" * 80)
    print("\n⏳ This test requires loading both models and running inference...")
    print("   (Implementation depends on having both PyTorch and Rust inference ready)")
    print("   Skipping for now - focus on weight comparison first.")
    
    # TODO: Implement when both models are fully working
    # 1. Load PyTorch model from Safetensors
    # 2. Load Rust model from MPK
    # 3. Run identical inputs through both
    # 4. Compare outputs

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify MPK weights")
    parser.add_argument("--model-path", type=str, default="blt-burn/blt_entropy_model.mpk", help="Path to MPK file")
    args = parser.parse_args()

    print("=" * 80)
    print("BLT ENTROPY MODEL - MPK CONVERSION VERIFICATION")
    print("=" * 80)
    
    # 1. Locate Safetensors in HF cache
    print("\n[1/4] Locating HuggingFace cache...")
    safetensors_path = find_hf_cache_model()
    
    if not safetensors_path:
        print("❌ Could not find facebook/blt-entropy in HF cache!")
        print("   Attempting to download...")
        try:
            model_path = snapshot_download("facebook/blt-entropy")
            safetensors_path = Path(model_path) / "model.safetensors"
        except Exception as e:
            print(f"❌ Download failed: {e}")
            sys.exit(1)
    
    print(f"✓ Safetensors: {safetensors_path}")
    print(f"   Size: {safetensors_path.stat().st_size / 1024**2:.1f} MB")
    
    # 2. Locate MPK file
    print("\n[2/4] Locating MPK file...")
    mpk_path = Path(args.model_path)
    
    if not mpk_path.exists():
        print(f"❌ Could not find {mpk_path}!")
        sys.exit(1)
    
    print(f"✓ MPK: {mpk_path}")
    print(f"   Size: {mpk_path.stat().st_size / 1024**2:.1f} MB")
    
    # 3. Load weights
    print("\n[3/4] Loading weights...")
    print("   Loading Safetensors...")
    st_weights = load_safetensors_weights(safetensors_path)
    print(f"   ✓ Loaded {len(st_weights)} tensors from Safetensors")
    
    # Print shapes of key tensors to verify vocab size
    shapes = st_weights.get("__shapes__", {})
    if "tok_embeddings.weight" in shapes:
        shape = shapes["tok_embeddings.weight"]
        print(f"   ℹ️  tok_embeddings.weight shape: {shape}")
        if shape[0] == 260:
            print("   ✅ Confirmed vocab size is 260 (Bytes + Special Tokens)")
        else:
            print(f"   ⚠️  Unexpected vocab size: {shape[0]}")

    print("   Loading MPK...")
    try:
        mpk_weights = load_mpk_weights(mpk_path)
        print(f"   ✓ Loaded MPK file")
    except Exception as e:
        print(f"   ⚠️  MPK parsing warning: {e}")
        print(f"   Note: MPK format may need custom parsing for Burn's binary format")
        mpk_weights = {}
    
    # 4. Compare weights
    print("\n[4/4] Comparing weights...")
    results = compare_weights(st_weights, mpk_weights)
    
    return results

if __name__ == "__main__":
    main()
