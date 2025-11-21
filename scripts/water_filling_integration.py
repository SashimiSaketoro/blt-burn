"""
BLT Output Loader Utility (blt_loader.py)

This script provides functions to load .safetensors files and hypergraph sidecars produced by blt-burn's ingest pipeline.

Use this as a drop-in for consuming applications to load pre-norm embeddings, prominence, coherence, etc.
"""

import jax
import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file
from typing import Dict
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
        try:
            conn = sqlite3.connect(meta_path)
            cursor = conn.cursor()
            
            # Use numpy for indexing (avoid JAX/Metal issues)
            coherence_np = np.array(data['coherence_scores'])
            # Flatten if needed (handle (1, N) or (N,) shapes)
            if coherence_np.ndim > 1:
                coherence_np = coherence_np.flatten()
            
            patch_indices = data.get('patch_indices', None)
            if patch_indices is not None:
                patch_indices = np.array(patch_indices)
                if patch_indices.ndim > 1:
                    patch_indices = patch_indices.flatten()
                
                # patch_indices contains START positions of patches, not per-token assignments
                # We need to aggregate token-level coherence into patch-level coherence
                if len(patch_indices) > 0 and len(patch_indices) < len(coherence_np):
                    # Sort patch starts to ensure order
                    patch_starts = np.sort(patch_indices)
                    
                    # Compute patch boundaries: each patch goes from start[i] to start[i+1] (or end)
                    num_patches = len(patch_starts)
                    patch_coherences = []
                    
                    for i in range(num_patches):
                        start_idx = int(patch_starts[i])
                        # End index is next patch start, or end of sequence
                        if i + 1 < num_patches:
                            end_idx = int(patch_starts[i + 1])
                        else:
                            end_idx = len(coherence_np)
                        
                        # Aggregate coherence for tokens in this patch
                        if start_idx < len(coherence_np) and end_idx <= len(coherence_np):
                            patch_coherence = np.mean(coherence_np[start_idx:end_idx])
                            patch_coherences.append(float(patch_coherence))
                    
                    # Update hypergraph nodes with patch-level coherence
                    # Note: node_id mapping assumes leaves are in order (may need adjustment based on actual hypergraph structure)
                    for patch_idx, score in enumerate(patch_coherences, start=1):
                        # Try to find leaf nodes and update them
                        # This is a simplified approach - full implementation would need to map patches to leaf IDs
                        cursor.execute(
                            "UPDATE nodes SET metadata = json_set(metadata, '$.coherence_score', ?) WHERE id = ?",
                            (score, patch_idx)
                        )
                    
                    conn.commit()
                    print(f"  Injected {len(patch_coherences)} patch-level coherence scores into hypergraph nodes.")
                else:
                    print(f"  Note: patch_indices shape ({patch_indices.shape}) unexpected for coherence ({coherence_np.shape}), skipping aggregation")
            else:
                print("  Note: No patch_indices found, skipping coherence aggregation")
            
            conn.close()
        except Exception as e:
            print(f"  Warning: Could not inject coherence into hypergraph: {e}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLT Output Loader Utility")
    parser.add_argument("--input", type=str, default="blt_output.safetensors", help="Path to .safetensors file or directory")
    args = parser.parse_args()
    
    # Example usage
    data = load_blt_output(args.input)
    print("Loaded keys:", list(data.keys()))
    print("Embeddings shape:", data['embeddings'].shape)
