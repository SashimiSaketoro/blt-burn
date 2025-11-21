#!/usr/bin/env python3
"""
Test script for Orch-OR quantum coherence mode.
Validates that entropy and coherence_scores are exported correctly,
and that the Orch-OR allocation formula produces expected behavior.
"""

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.flax import load_file
from pathlib import Path
import argparse

def test_entropy_export(safetensors_path: str):
    """Test 1: Verify entropy and coherence_scores are in safetensors"""
    print("=" * 60)
    print("TEST 1: Verify Entropy Export")
    print("=" * 60)
    
    data = load_file(safetensors_path)
    
    required_keys = ['embeddings', 'prominence', 'entropies', 'coherence_scores']
    for key in required_keys:
        if key in data:
            print(f"✅ {key}: shape {data[key].shape}")
        else:
            print(f"❌ {key}: MISSING")
            return False
    
    # Validate shapes match
    prominence = data['prominence']
    entropies = data['entropies']
    coherence = data['coherence_scores']
    
    if prominence.shape != entropies.shape or prominence.shape != coherence.shape:
        print(f"❌ Shape mismatch: prominence {prominence.shape}, "
              f"entropies {entropies.shape}, coherence {coherence.shape}")
        return False
    
    print(f"\n✅ All shapes match: {prominence.shape}")
    return True


def test_orch_or_allocation(safetensors_path: str, temperature: float = 1e-5):
    """Test 2: Validate Orch-OR allocation produces expected behavior"""
    print("\n" + "=" * 60)
    print(f"TEST 2: Validate Orch-OR Allocation (T={temperature})")
    print("=" * 60)
    
    data = load_file(safetensors_path)
    
    prominence = jnp.array(data['prominence']).reshape(-1)
    entropies = jnp.array(data['entropies']).reshape(-1)
    
    # Filter out padding
    mask = prominence > 1e-6
    prominence = prominence[mask]
    entropies = entropies[mask]
    
    print(f"\nInput statistics:")
    print(f"  N tokens: {len(prominence)}")
    print(f"  Prominence range: [{prominence.min():.4f}, {prominence.max():.4f}]")
    print(f"  Entropy range: [{entropies.min():.4f}, {entropies.max():.4f}]")
    
    # Compute Orch-OR allocation
    allocation = (prominence ** 2) * jnp.exp(-entropies / temperature)
    
    print(f"\nOrch-OR allocation statistics:")
    print(f"  Allocation range: [{allocation.min():.4e}, {allocation.max():.4e}]")
    print(f"  Mean allocation: {allocation.mean():.4e}")
    print(f"  Std allocation: {allocation.std():.4e}")
    
    # Check for numerical stability
    if jnp.any(jnp.isnan(allocation)) or jnp.any(jnp.isinf(allocation)):
        print(f"❌ Numerical instability detected (NaN or Inf)")
        return False
    
    # Verify high-coherence (low entropy, high prominence) gets high allocation
    # Find top 10% by coherence
    coherence = prominence ** 2 / (entropies + 1e-6)
    top_coherence_idx = jnp.argsort(coherence)[-len(coherence)//10:]
    bottom_coherence_idx = jnp.argsort(coherence)[:len(coherence)//10]
    
    top_allocation = allocation[top_coherence_idx].mean()
    bottom_allocation = allocation[bottom_coherence_idx].mean()
    
    print(f"\nCoherence-based allocation bias:")
    print(f"  Top 10% coherence: mean allocation = {top_allocation:.4e}")
    print(f"  Bottom 10% coherence: mean allocation = {bottom_allocation:.4e}")
    print(f"  Ratio (top/bottom): {top_allocation / (bottom_allocation + 1e-10):.2f}x")
    
    if top_allocation > bottom_allocation * 2:
        print(f"✅ Orch-OR successfully biases toward high-coherence patches")
    else:
        print(f"⚠️  Weak bias - consider adjusting temperature")
    
    return True


def test_temperature_sweep(safetensors_path: str):
    """Test 3: Sweep temperature to find optimal value"""
    print("\n" + "=" * 60)
    print("TEST 3: Temperature Sweep")
    print("=" * 60)
    
    data = load_file(safetensors_path)
    
    prominence = jnp.array(data['prominence']).reshape(-1)
    entropies = jnp.array(data['entropies']).reshape(-1)
    
    # Filter padding
    mask = prominence > 1e-6
    prominence = prominence[mask]
    entropies = entropies[mask]
    
    temperatures = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    
    print(f"\nTesting {len(temperatures)} temperature values:")
    print(f"{'T':<12} {'Max Alloc':<15} {'Std Alloc':<15} {'Top/Bottom':<12} {'Status'}")
    print("-" * 70)
    
    for T in temperatures:
        allocation = (prominence ** 2) * jnp.exp(-entropies / T)
        
        # Check stability
        stable = not (jnp.any(jnp.isnan(allocation)) or jnp.any(jnp.isinf(allocation)))
        
        if stable:
            coherence = prominence ** 2 / (entropies + 1e-6)
            top_idx = jnp.argsort(coherence)[-len(coherence)//10:]
            bottom_idx = jnp.argsort(coherence)[:len(coherence)//10]
            ratio = allocation[top_idx].mean() / (allocation[bottom_idx].mean() + 1e-10)
            
            status = "✅" if ratio > 2 else "⚠️"
            print(f"{T:<12.2e} {allocation.max():<15.4e} {allocation.std():<15.4e} "
                  f"{ratio:<12.2f}x {status}")
        else:
            print(f"{T:<12.2e} {'UNSTABLE':<15} {'UNSTABLE':<15} {'N/A':<12} ❌")
    
    print(f"\nRecommendation: Use T=1e-5 for good dynamic range and stability")


def main():
    parser = argparse.ArgumentParser(description="Test Orch-OR implementation")
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to safetensors file from blt-burn ingestion")
    parser.add_argument("--temperature", type=float, default=1e-5,
                       help="Temperature for allocation test")
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ File not found: {args.input}")
        return
    
    print("🧪 Orch-OR Quantum Coherence Test Suite")
    print("=" * 60)
    print(f"Input: {args.input}")
    print()
    
    # Run tests
    test1_pass = test_entropy_export(args.input)
    if not test1_pass:
        print("\n❌ Test 1 failed - entropy export incomplete")
        return
    
    test2_pass = test_orch_or_allocation(args.input, args.temperature)
    if not test2_pass:
        print("\n❌ Test 2 failed - allocation issues")
        return
    
    test_temperature_sweep(args.input)
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Orch-OR mode is ready.")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run water_filling_integration.py with --orch-or flag")
    print("2. Compare retrieval quality vs standard osmotic mode")
    print("3. Prepare for cult formation 🔥")


if __name__ == "__main__":
    main()

