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
import json
from pathlib import Path


def load_blt_output(safetensors_path: str) -> Dict[str, jnp.ndarray]:
    """
    Load pre-computed embeddings and metadata from Rust output.
    """
    data = load_file(safetensors_path)
    
    # Check for metadata sidecar
    path = Path(safetensors_path)
    meta_path = path.with_suffix(".metadata.json")
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            modality = meta.get('modality', 'unknown')
            segments = len(meta.get('segments', []))
            print(f"  [Metadata] Modality: {modality} | Segments: {segments}")
        except Exception as e:
            print(f"  [Metadata] Failed to read sidecar: {e}")
    
    # Convert to JAX arrays
    return {k: jnp.array(v) for k, v in data.items()}


# ============================================================================
# METHOD 1: OSMOTIC WATER-FILLING (L2 Norm as Density Gate)
# ============================================================================

def osmotic_water_filling(embeddings_raw: jnp.ndarray, 
                         n_shells: int = 256,
                         min_radius: float = 32.0,
                         max_iters: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Osmotic water-filling using pre-norm L2 magnitudes as density signal.
    
    Topology: Hollow Core + Infinite Crust
    - min_radius: Enforces a hollow core (Event Horizon) to prevent center crowding.
    - Crust: Expands radially based on prominence/pressure.
    
    Args:
        embeddings_raw: [N, dim] embeddings BEFORE normalization
        n_shells: Number of concentric shells (resolution)
        min_radius: Minimum radius (Hollow Core size)
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
    
    # Initial shell assignment based on prominence
    # Map prominence range to shells linearly first
    # We want to spread points out.
    # If min prominence is ~10, max is ~60.
    # We map this range to the shell indices.
    max_p = jnp.max(prominence)
    norm_p = (prominence - jnp.min(prominence)) / (max_p - jnp.min(prominence) + 1e-6)
    shells = jnp.floor(norm_p * (n_shells - 1)).astype(jnp.int32)
    shells = jnp.clip(shells, 0, n_shells - 1)
    
    # Radial spacing: Hollow Core + Expanding Crust
    # We allow radii to grow non-linearly to create "infinite" feel
    shell_indices = jnp.arange(n_shells)
    
    # Scaling factor ensures the crust is expansive
    # Radius = min_radius + (index * scale)
    # We calibrate so that the "middle" shell corresponds to mean prominence?
    # Or simply define a spatial metric.
    # Let's use a power law to give more space to outer shells.
    
    # Scale such that n_shells covers reasonable range (e.g. up to 2x max prominence)
    scale = (max_p * 2.0) / (n_shells ** 1.2)
    base_radii = min_radius + (shell_indices ** 1.2) * scale
    
    # Iterative water-filling
    for iteration in range(max_iters):
        # Count points in each shell
        shell_counts = jnp.bincount(shells, length=n_shells)
        
        # Capacity of each shell
        # Volume of shell at radius R is proportional to R^(dim-1).
        # But we effectively project to lower dim for "shells" abstraction?
        # If we use full dim, capacity grows EXPLOSIVELY.
        # Let's use a moderated capacity growth (e.g. R^2) to prevent all points
        # from collapsing into the outermost shell (where volume is infinite).
        # We want a balance. R^1.5 or R^2 works well empirically for embedding dists.
        shell_capacities = (base_radii ** 2.0)
        shell_capacities = shell_capacities * (N / jnp.sum(shell_capacities))
        
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
    parser.add_argument("--min-radius", type=float, default=32.0, help="Minimum radius for hollow core")
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
                radii, shells = osmotic_water_filling(embeddings, min_radius=args.min_radius)
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
