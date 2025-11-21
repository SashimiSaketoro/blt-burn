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
import sqlite3


def load_blt_output(safetensors_path: str) -> Dict[str, jnp.ndarray]:
    """
    Load pre-computed embeddings and metadata from Rust output.
    Supports both single files and JAX-compatible sharded files.
    """
    path = Path(safetensors_path)
    
    # Check if this is a sharded file
    if "_shard_" in path.stem:
        # Extract shard info from filename
        parts = path.stem.split("_shard_")
        if len(parts) == 2:
            shard_info = parts[1].split("_of_")
            if len(shard_info) == 2:
                shard_idx = int(shard_info[0])
                num_shards = int(shard_info[1])
                print(f"Loading shard {shard_idx + 1}/{num_shards}")
    
    data = load_file(safetensors_path)
    
    # Check for metadata/hypergraph sidecar
    # Try hypergraph SQLite first (new primary format)
    meta_db_path = path.with_suffix(".hypergraph.db")
    meta_json_path = path.with_suffix(".hypergraph.json")
    format_type = None
    meta_path = None
    
    if meta_db_path.exists():
        format_type = "hypergraph-sqlite"
        meta_path = meta_db_path
    elif meta_json_path.exists():
        format_type = "hypergraph-json"
        meta_path = meta_json_path
    else:
        # Fallback to old metadata format
        meta_path = path.with_suffix(".metadata.json")
        if meta_path.exists():
            format_type = "legacy"
        
    if meta_path and meta_path.exists():
        try:
            if format_type == "hypergraph-sqlite":
                print(f"  Found hypergraph sidecar (SQLite): {meta_path.name}")
                # For now, just note that it exists - full deserialization would need bincode
                conn = sqlite3.connect(meta_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM nodes")
                node_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM hyperedges")
                edge_count = cursor.fetchone()[0]
                conn.close()
                print(f"  Nodes: {node_count}, Edges: {edge_count}")
                
                # Check for sharding info in SQLite metadata
                try:
                    cursor.execute("SELECT value FROM meta WHERE key='sharding_info'")
                    result = cursor.fetchone()
                    if result:
                        sharding_info = json.loads(result[0])
                        print(f"  Sharding Info:")
                        print(f"    Global shape: {sharding_info['global_shape']}")
                        print(f"    Shard {sharding_info['shard_index'] + 1}/{sharding_info['num_shards']}")
                        print(f"    Axis: {sharding_info['axis']}")
                        if 'process_index' in sharding_info:
                            print(f"    Target process: {sharding_info['process_index']}")
                        data["sharding_info"] = sharding_info
                except:
                    pass
                
                conn.close()
                
                # Set defaults since we can't fully deserialize yet
                segments_count = 0
                modality = "unknown"
            
            elif format_type == "hypergraph-json":
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                # Parse hypergraph nodes to find Branch info
                nodes = meta.get('nodes', [])
                modality = "unknown"
                segments_count = 0
                
                # Find Branch for modality
                for node in nodes:
                    if "Branch" in node:
                        modality = node["Branch"].get('modality', 'unknown')
                    if "Leaf" in node:
                        segments_count += 1
                        
                print(f"  [Sidecar] Format: Hypergraph | Modality: {modality} | Leaves: {segments_count}")
                
            elif format_type == "legacy":
                # Legacy flat format
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                modality = meta.get('modality', 'unknown')
                segments = len(meta.get('segments', []))
                print(f"  [Sidecar] Format: Legacy | Modality: {modality} | Segments: {segments}")
                
        except Exception as e:
            print(f"  [Sidecar] Failed to read sidecar: {e}")
    
    # After loading, if hypergraph db exists and coherence present, inject
    if format_type == "hypergraph-sqlite" and 'coherence_scores' in data:
        conn = sqlite3.connect(meta_path)
        cursor = conn.cursor()
        
        # Assume patch_indices maps to node_ids (simplified)
        patch_indices = data.get('patch_indices', jnp.arange(len(data['coherence_scores'])))
        
        # Aggregate mean coherence per patch (group by patch_indices)
        unique_patches = jnp.unique(patch_indices)
        agg_coherence = [jnp.mean(data['coherence_scores'][patch_indices == p]) for p in unique_patches]
        
        # Update nodes (assume node_id starts from 1, matches unique_patches)
        for node_id, score in enumerate(agg_coherence, start=1):
            cursor.execute(
                "UPDATE nodes SET metadata = json_set(metadata, '$.coherence_score', ?) WHERE id = ?",
                (float(score), node_id)
            )
        
        conn.commit()
        conn.close()
        print("  Injected aggregated coherence scores into hypergraph nodes.")

    # Convert to JAX arrays
    return {k: jnp.array(v) for k, v in data.items()}


def load_sharded_blt_output_jax(prefix: str, process_index: int = None, num_processes: int = None):
    """
    Load BLT output sharded for JAX distributed processing.
    
    Args:
        prefix: Base path without shard suffix (e.g., "output/data" for "output/data_shard_0_of_4.safetensors")
        process_index: Current process index (defaults to jax.process_index())
        num_processes: Total processes (defaults to jax.process_count())
    
    Returns:
        dict: Dictionary containing sharded JAX arrays and metadata
    """
    if process_index is None:
        process_index = jax.process_index()
    if num_processes is None:
        num_processes = jax.process_count()
    
    # Load process-local shard
    shard_path = f"{prefix}_shard_{process_index}_of_{num_processes}.safetensors"
    local_data = load_blt_output(shard_path)
    
    # Extract sharding info
    sharding_info = local_data.get("sharding_info", None)
    if not sharding_info:
        raise ValueError(f"No sharding info found in {shard_path}")
    
    # Create JAX mesh and sharding
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ['batch'])
    
    # Determine partition spec based on sharding axis
    if sharding_info['axis'] == 0:
        partition_spec = jax.sharding.PartitionSpec('batch', None, None)
    elif sharding_info['axis'] == 1:
        partition_spec = jax.sharding.PartitionSpec(None, 'batch', None)
    else:
        partition_spec = jax.sharding.PartitionSpec(None, None, 'batch')
    
    sharding = jax.sharding.NamedSharding(mesh, partition_spec)
    
    # Create global arrays
    embeddings = jax.make_array_from_process_local_data(
        sharding, local_data['embeddings']
    )
    
    prominence = jax.make_array_from_process_local_data(
        sharding, local_data['embedding_norms']
    )
    
    # Patch indices and mask are typically not sharded
    patch_indices = local_data['patch_indices']
    patch_mask = local_data['patch_mask']
    
    return {
        'embeddings': embeddings,
        'prominence': prominence,
        'patch_indices': patch_indices,
        'patch_mask': patch_mask,
        'sharding_info': sharding_info
    }


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


# ============================================================================
# METHOD 3: ORCH-OR QUANTUM COHERENCE (Penrose-Hameroff Inspired)
# ============================================================================

def orch_or_water_filling(
    embeddings_raw: jnp.ndarray,
    prominence: jnp.ndarray,
    entropies: jnp.ndarray,
    temperature: float = 1e-5,
    min_radius: float = 32.0,
    max_radius: float = 512.0,
    n_shells: int = 256,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Orch-OR inspired allocation: power ∝ pre_norm^2 * exp(-entropy / T)
    
    Penrose-Hameroff mapping:
    - pre_norm^2: Gravitational self-energy (superposition size)
    - exp(-entropy/T): Quantum decoherence rate
    - High coherence (low entropy) + high prominence = larger "conscious volume"
    
    This biases the hypersphere toward patches with high coherence AND high amplitude,
    exactly like Hameroff's claim that significant moments feel "brighter" because
    more tubulins are involved.
    
    Args:
        embeddings_raw: [N, dim] embeddings BEFORE normalization
        prominence: [N] pre-norm L2 norms (superposition mass)
        entropies: [N] Shannon entropy from logits (decoherence)
        temperature: Planck temperature T (default: 1e-5)
        min_radius: Minimum radius (hollow core)
        max_radius: Maximum radius
        n_shells: Number of shells for discretization
    
    Returns:
        radii: [N] optimal radii (conscious volume allocation)
        shells: [N] shell assignments
    """
    # Ensure 1D arrays
    prominence = prominence.reshape(-1)
    entropies = entropies.reshape(-1)
    
    # Orch-OR allocation formula
    # allocation ∝ pre_norm^2 * exp(-entropy / T)
    allocation = (prominence ** 2) * jnp.exp(-entropies / temperature)
    
    # Softmax normalization for proper probabilistic interpretation
    # (fraction of total conscious volume)
    allocation_softmax = jax.nn.softmax(allocation / jnp.max(allocation))
    
    # Map to radii (use softmax for smooth distribution)
    radii = min_radius + allocation_softmax * (max_radius - min_radius) * len(allocation)
    radii = jnp.clip(radii, min_radius, max_radius)
    
    # Shell assignment based on radii
    shell_boundaries = jnp.linspace(min_radius, max_radius, n_shells + 1)
    shells = jnp.searchsorted(shell_boundaries, radii) - 1
    shells = jnp.clip(shells, 0, n_shells - 1)
    
    return radii, shells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water-filling on BLT output")
    parser.add_argument("--input", type=str, default="blt_output.safetensors", help="Path to .safetensors file or directory")
    parser.add_argument("--output-dir", type=str, default="sphere_output", help="Directory to save sphere results")
    parser.add_argument("--method", type=str, default="osmotic", choices=["osmotic", "thrml"], help="Water-filling method")
    parser.add_argument("--min-radius", type=float, default=32.0, help="Minimum radius for hollow core")
    parser.add_argument("--orch-or", action="store_true", help="Enable Orch-OR quantum coherence allocation mode")
    parser.add_argument("--orch-or-temperature", type=float, default=None, help="Planck temperature T for Orch-OR mode (default: auto-tune)")
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
            
            # Load entropies and coherence if available
            entropies = data.get('entropies', None)
            coherence_scores = data.get('coherence_scores', None)
            
            if entropies is not None:
                if entropies.ndim == 2:
                    entropies = entropies.reshape(-1)
                entropies = entropies[mask]
            
            if coherence_scores is not None:
                if coherence_scores.ndim == 2:
                    coherence_scores = coherence_scores.reshape(-1)
                coherence_scores = coherence_scores[mask]
            
            print(f"  Loaded {len(embeddings)} patches. Mean prominence: {prominence.mean():.4f}")
            if entropies is not None:
                print(f"  Mean entropy: {entropies.mean():.4f}")
            if coherence_scores is not None:
                print(f"  Mean coherence: {coherence_scores.mean():.4f}")
            
            # Run Water Filling
            radii, shells = None, None
            if args.orch_or:
                if entropies is None:
                    raise ValueError("--orch-or requires entropies in safetensors. "
                                   "Re-run blt-burn ingestion with updated code.")
                print(f"  Using Orch-OR mode (T={args.orch_or_temperature})")
                radii, shells = orch_or_water_filling(
                    embeddings,
                    prominence,
                    entropies,
                    temperature=args.orch_or_temperature,
                    min_radius=args.min_radius
                )
            elif args.method == "osmotic":
                radii, shells = osmotic_water_filling(embeddings, min_radius=args.min_radius)
            else:
                radii, shells = thrml_energy_water_filling(embeddings)
                
            # Compute final sphere coordinates
            norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
            directions = embeddings / (norms + 1e-8)
            sphere_coords = directions * radii[:, None]
            
            # Save results
            output_file = output_dir / f"sphere_{file_path.stem}.npz"
            save_dict = {
                'sphere_coords': np.array(sphere_coords),
                'radii': np.array(radii),
                'shells': np.array(shells),
                'original_prominence': np.array(prominence),
            }
            
            # Include entropies and coherence if available
            if entropies is not None:
                save_dict['original_entropies'] = np.array(entropies)
            if coherence_scores is not None:
                save_dict['original_coherence'] = np.array(coherence_scores)
            
            np.savez(output_file, **save_dict)
            print(f"  Saved to {output_file}")
            
        except Exception as e:
            print(f"  Failed to process {file_path}: {e}")

    print("\nPipeline Complete!")
