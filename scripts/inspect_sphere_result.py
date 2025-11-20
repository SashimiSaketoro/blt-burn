import numpy as np
import sys
import json
from pathlib import Path
from safetensors import safe_open

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

    # Look for metadata sidecar
    # Try direct name match or check header (if accessible, but name match is robust here)
    meta_path = path.with_suffix(".metadata.json")
    if not meta_path.exists():
        # Check if filename matches pattern item_ID_part_X.safetensors -> item_ID.metadata.json
        # This is a heuristic.
        name = path.stem
        if "_part_" in name:
            base_name = name.split("_part_")[0]
            meta_path = path.parent / f"{base_name}.metadata.json"
            
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            print("\n✅ Found Metadata Sidecar:")
            print(f"  File: {meta_path.name}")
            print(f"  Source Hash: {meta.get('source_hash', 'N/A')}")
            print(f"  Modality: {meta.get('modality', 'N/A')}")
            print(f"  Total Bytes: {meta.get('total_bytes', 'N/A')}")
            
            segments = meta.get('segments', [])
            print(f"  Segments: {len(segments)}")
            
            # Map patches to segments if patch_indices exist
            if "patch_indices" in data:
                patch_indices = data["patch_indices"]
                print(f"\nMapping first 10 patches (of {len(patch_indices)}) to semantic segments:")
                
                count = 0
                for i, idx in enumerate(patch_indices):
                    if count >= 10: break
                    
                    # Find segment containing this byte index
                    match = None
                    for seg in segments:
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
                        # If no match, it might be between segments or generated
                        pass
                        
                if count == 0:
                    print("  (No patches mapped to segments - indices might be out of bounds or segments are generated/frames)")

        except Exception as e:
            print(f"Error reading metadata sidecar: {e}")
    else:
        print("\n⚠️ No metadata sidecar found.")

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
