"""
Integration of Rust-Optimized BLT Extraction with Water-Filling Algorithms

This script consumes the .safetensors output from the Rust `blt-burn` preprocessor
and applies water-filling algorithms to the extracted embeddings.

Pipeline:
1. Rust `blt-burn`: Text -> Tokens -> Pre-Norm Embeddings + Prominence -> .safetensors
2. Python `water_filling`: .safetensors -> Osmotic/Thermal Water-Filling -> Hypersphere
"""

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.flax import load_file
from typing import Tuple, Dict
import argparse
from pathlib import Path


def load_blt_output(safetensors_path: str) -> Dict[str, jnp.ndarray]:
    """
    Load pre-computed embeddings and metadata from Rust output.
    """
    data = load_file(safetensors_path)
    
    # Convert to JAX arrays
    return {k: jnp.array(v) for k, v in data.items()}


# ============================================================================
# METHOD 1: OSMOTIC WATER-FILLING (L2 Norm as Density Gate)
# ============================================================================

def osmotic_water_filling(embeddings_raw: jnp.ndarray, 
                         n_shells: int = 128,
                         max_iters: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Osmotic water-filling using pre-norm L2 magnitudes as density signal.
    
    Key Innovation: L2 norm creates osmotic pressure that drives rebalancing.
    
    Args:
        embeddings_raw: [N, dim] embeddings BEFORE normalization
        n_shells: Number of concentric shells
        max_iters: Maximum optimization iterations
    
    Returns:
        radii: [N] optimal radii for each point
        shells: [N] final shell assignments
    """
    # Ensure 2D [N, dim]
    if embeddings_raw.ndim == 3:
        embeddings_raw = embeddings_raw.reshape(-1, embeddings_raw.shape[-1])
        
    N, dim = embeddings_raw.shape
    
    # Extract the density signal (L2 norms)
    l2_norms = jnp.linalg.norm(embeddings_raw, axis=1)  # [N]
    
    # Use L2 norms as "osmotic pressure" / "prominence"
    prominence = l2_norms
    
    # Detect outliers using scale-invariant threshold
    std_dev = jnp.std(prominence)
    mean_prominence = jnp.mean(prominence)
    is_outlier = prominence > (mean_prominence + 1.0 * std_dev)
    
    # Initial shell assignment based on prominence quantiles
    shell_boundaries = jnp.linspace(0, 1, n_shells + 1)
    prominence_quantiles = jnp.percentile(prominence, shell_boundaries * 100)
    shells = jnp.searchsorted(prominence_quantiles, prominence)
    shells = jnp.clip(shells, 0, n_shells - 1)
    
    # Radial spacing: sqrt + r^1.5
    base_radii = jnp.sqrt(jnp.arange(n_shells)) + jnp.arange(n_shells) ** 1.5
    base_radii = base_radii / base_radii[-1]  # Normalize to [0, 1]
    
    # Iterative water-filling
    for iteration in range(max_iters):
        # Count points in each shell
        shell_counts = jnp.bincount(shells, length=n_shells)
        
        # Capacity of each shell (scales with r^1.5)
        shell_capacities = (base_radii ** 1.5) * (N / jnp.sum(base_radii ** 1.5))
        
        # Detect overloaded and underloaded shells
        shell_loads = shell_counts / (shell_capacities + 1e-8)
        overloaded = shell_loads > 1.2
        underloaded = shell_loads < 0.8
        
        if not (jnp.any(overloaded) or jnp.any(is_outlier)):
            break
        
        # Promote high-prominence points and those in overloaded shells
        should_promote = is_outlier | overloaded[shells]
        
        # Lateral traversal (30% chance) vs radial move (70% chance)
        lateral_mask = jax.random.bernoulli(
            jax.random.PRNGKey(iteration), 
            p=0.3, 
            shape=(N,)
        )
        
        radial_moves = should_promote & ~lateral_mask
        shells = jnp.where(
            radial_moves,
            jnp.minimum(shells + 1, n_shells - 1),
            shells
        )
        
        # Demote points in underloaded shells
        should_demote = underloaded[shells] & ~is_outlier
        shells = jnp.where(
            should_demote,
            jnp.maximum(shells - 1, 0),
            shells
        )
    
    # Map shells to actual radii
    radii = base_radii[shells]
    
    return radii, shells


# ============================================================================
# METHOD 2: THRML ENERGY-BASED WATER-FILLING
# ============================================================================

def thrml_energy_water_filling(embeddings_raw: jnp.ndarray,
                               n_shells: int = 128,
                               temperature: float = 1.0,
                               max_iters: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Thermal energy-based water-filling using L2 norms as kinetic energy.
    """
    # Ensure 2D [N, dim]
    if embeddings_raw.ndim == 3:
        embeddings_raw = embeddings_raw.reshape(-1, embeddings_raw.shape[-1])
        
    N, dim = embeddings_raw.shape
    
    # L2 norms = "kinetic energy"
    energies = jnp.linalg.norm(embeddings_raw, axis=1)  # [N]
    
    # Boltzmann distribution: P(shell) ∝ exp(-E_shell / kT)
    shell_indices = jnp.arange(n_shells)
    shell_energies = jnp.sqrt(shell_indices) + shell_indices ** 1.5
    
    # Match energies
    energy_diff = jnp.abs(energies[:, None] - shell_energies[None, :])
    boltzmann_probs = jnp.exp(-energy_diff / temperature)
    boltzmann_probs = boltzmann_probs / (jnp.sum(boltzmann_probs, axis=1, keepdims=True) + 1e-8)
    
    shells = jnp.argmax(boltzmann_probs, axis=1)
    
    # Refinement loop (simplified for demo)
    radii = (jnp.sqrt(shells) + shells ** 1.5) / (jnp.sqrt(n_shells) + n_shells ** 1.5)
    
    return radii, shells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water-filling on BLT output")
    parser.add_argument("--input", type=str, default="blt_output.safetensors", help="Path to .safetensors file or directory")
    parser.add_argument("--output-dir", type=str, default="sphere_output", help="Directory to save sphere results")
    parser.add_argument("--method", type=str, default="osmotic", choices=["osmotic", "thrml"], help="Water-filling method")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect files to process
    files = []
    if input_path.is_dir():
        files = sorted(list(input_path.glob("*.safetensors")))
        print(f"Found {len(files)} safetensors files in {input_path}")
    elif input_path.exists():
        files = [input_path]
    else:
        print(f"Error: Input {input_path} not found.")
        exit(1)
        
    for i, file_path in enumerate(files):
        print(f"\nProcessing [{i+1}/{len(files)}] {file_path.name}...")
        
        try:
            data = load_blt_output(str(file_path))
            
            embeddings = data['embeddings']
            prominence = data['prominence']
            
            # Handle dimensions
            if embeddings.ndim == 3:
                embeddings = embeddings.reshape(-1, embeddings.shape[-1])
            if prominence.ndim == 2:
                prominence = prominence.reshape(-1)
                
            # Filter padding/empty
            mask = prominence > 1e-6
            if not jnp.any(mask):
                print("  Skipping empty/padding file.")
                continue
                
            embeddings = embeddings[mask]
            prominence = prominence[mask]
            
            print(f"  Loaded {len(embeddings)} patches. Mean prominence: {prominence.mean():.4f}")
            
            # Run Water Filling
            radii, shells = None, None
            if args.method == "osmotic":
                radii, shells = osmotic_water_filling(embeddings)
            else:
                radii, shells = thrml_energy_water_filling(embeddings)
                
            # Compute final sphere coordinates
            norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
            directions = embeddings / (norms + 1e-8)
            sphere_coords = directions * radii[:, None]
            
            # Save results
            output_file = output_dir / f"sphere_{file_path.stem}.npz"
            np.savez(
                output_file,
                sphere_coords=np.array(sphere_coords),
                radii=np.array(radii),
                shells=np.array(shells),
                original_prominence=np.array(prominence)
            )
            print(f"  Saved to {output_file}")
            
        except Exception as e:
            print(f"  Failed to process {file_path}: {e}")

    print("\nPipeline Complete!")
