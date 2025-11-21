import numpy as np
import sys
import json
from pathlib import Path
from safetensors import safe_open
import sqlite3

def inspect_safetensors(path):
    path = Path(path)
    print(f"Inspecting {path}...")
    
    data = {}
    try:
        with safe_open(path, framework="numpy") as f:
            keys = f.keys()
            print("Keys:", list(keys))
        
            for key in keys:
                data[key] = f.get_tensor(key)
        
        for key in keys:
            arr = data[key]
            print(f"\n{key}: shape={arr.shape}, dtype={arr.dtype}")
            if arr.size > 0:
                print(f"  Min: {arr.min()}")
                print(f"  Max: {arr.max()}")
                print(f"  Mean: {arr.mean()}")
                
    except Exception as e:
        print(f"Error loading safetensors: {e}")
        return

    # Look for hypergraph sidecar - try SQLite first, then JSON
    meta_db_path = path.with_suffix(".hypergraph.db")
    meta_json_path = path.with_suffix(".hypergraph.json")
    
    # Handle multi-part files
    name = path.stem
    if "_part_" in name:
        base_name = name.split("_part_")[0]
        meta_db_path = path.parent / f"{base_name}.hypergraph.db"
        meta_json_path = path.parent / f"{base_name}.hypergraph.json"
    
    if meta_db_path.exists():
        try:
            print("\n✅ Found Hypergraph Sidecar (SQLite):")
            print(f"  File: {meta_db_path.name}")
            
            conn = sqlite3.connect(meta_db_path)
            cursor = conn.cursor()
            
            # Read metadata
            cursor.execute("SELECT value FROM meta WHERE key='schema_version'")
            result = cursor.fetchone()
            if result:
                print(f"  Schema Version: {result[0]}")
            
            # Count nodes and edges
            cursor.execute("SELECT COUNT(*) FROM nodes")
            node_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM hyperedges")
            edge_count = cursor.fetchone()[0]
            
            print(f"  Nodes: {node_count}")
            print(f"  Hyperedges: {edge_count}")
            
            # Check for sharding info
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
                        print(f"      Target process: {sharding_info['process_index']}")
            except:
                pass
            
            # Note: Actual deserialization would require bincode-compatible Python library
            print("  (Note: Full inspection requires bincode deserialization)")
            
            conn.close()
            
        except Exception as e:
            print(f"Error reading SQLite hypergraph: {e}")
    
    elif meta_json_path.exists():
        try:
            print("\n✅ Found Hypergraph Sidecar (JSON):")
            print(f"  File: {meta_json_path.name}")
            
            with open(meta_json_path, 'r') as f:
                meta = json.load(f)
            
            nodes = meta.get('nodes', [])
            edges = meta.get('edges', [])
            topology = meta.get('topology', {}).get('edges', [])
            
            print(f"  Nodes: {len(nodes)}")
            print(f"  Hyperedges: {len(edges)}")
            print(f"  Topology Entries: {len(topology)}")
            
            # Find Trunk
            trunk_node = None
            for i, node in enumerate(nodes):
                if "Trunk" in node:
                    trunk_node = node["Trunk"]
                    print(f"  Trunk: Source Hash={trunk_node.get('source_hash', 'N/A')}")
                    break
                    
            # Find Branches
            branches = []
            for i, node in enumerate(nodes):
                if "Branch" in node:
                    branches.append(node["Branch"])
            
            print(f"  Branches: {len(branches)}")
            for b in branches:
                print(f"    - {b.get('label')} ({b.get('modality')})")
                
            # Map patches to Leaf segments
            # We need to filter nodes to find only Leaves
            leaves = []
            for i, node in enumerate(nodes):
                if "Leaf" in node:
                    leaves.append(node["Leaf"])
            
            print(f"  Leaves (Segments): {len(leaves)}")
            
            if "patch_indices" in data:
                patch_indices = data["patch_indices"]
                print(f"\nMapping first 10 patches (of {len(patch_indices)}) to semantic segments:")
                
                count = 0
                for i, idx in enumerate(patch_indices):
                    if count >= 10: break
                    
                    # Find segment containing this byte index
                    match = None
                    for seg in leaves:
                        m = seg.get('metadata', {})
                        start = m.get('start_offset')
                        end = m.get('end_offset')
                        
                        if start is not None and end is not None:
                            if start <= idx < end:
                                match = seg
                                break
                    
                    if match:
                        label = match.get('label', 'unknown')
                        # Get extra info if available
                        extra = match.get('metadata', {}).get('extra')
                        print(f"  Patch {i} (Byte {idx}) -> {label} {extra if extra else ''}")
                        count += 1
                    else:
                        pass
                        
                if count == 0:
                    print("  (No patches mapped to segments - indices might be out of bounds or segments are generated/frames)")
            
            # Print Topology Sample
            print("\nHypergraph Topology (Sample):")
            for i, (edge_weight_idx, vertex_indices) in enumerate(topology[:5]):
                edge_data = edges[edge_weight_idx] if edge_weight_idx < len(edges) else {}
                label = edge_data.get('label', 'unknown')
                
                # Map vertex indices to types
                v_types = []
                for v_idx in vertex_indices:
                    if v_idx < len(nodes):
                        n = nodes[v_idx]
                        if "Trunk" in n: v_types.append("Trunk")
                        elif "Branch" in n: v_types.append("Branch")
                        elif "Leaf" in n: v_types.append("Leaf")
                        else: v_types.append("Unknown")
                    else:
                        v_types.append("Invalid")
                        
                print(f"  Edge {i}: [{label}] connects {v_types}")


        except Exception as e:
            print(f"Error reading hypergraph sidecar: {e}")
    else:
        print("\n⚠️ No hypergraph sidecar found.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_safetensors(sys.argv[1])
    else:
        # Find first safetensors in ingest_output
        results_dir = Path("ingest_output")
        if results_dir.exists():
            files = list(results_dir.glob("*.safetensors"))
        if files:
                inspect_safetensors(files[0])
            else:
                print("No .safetensors files found in ingest_output.")
        else:
            print("ingest_output directory not found.")
