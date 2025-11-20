import numpy as np
import sys
from pathlib import Path
from safetensors import safe_open

def inspect_safetensors(path):
    print(f"Inspecting {path}...")
    try:
        with safe_open(path, framework="numpy") as f:
            keys = f.keys()
            print("Keys:", list(keys))
            
            data = {}
            for key in keys:
                data[key] = f.get_tensor(key)
        
        for key in keys:
            arr = data[key]
            print(f"\n{key}: shape={arr.shape}, dtype={arr.dtype}")
            if arr.size > 0:
                print(f"  Min: {arr.min()}")
                print(f"  Max: {arr.max()}")
                print(f"  Mean: {arr.mean()}")
                if key == 'shells':
                    print(f"  Unique shells: {len(np.unique(arr))}")
                    print(f"  Distribution: {np.bincount(arr, minlength=128)[:10]} ...")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_safetensors(sys.argv[1])
    else:
        # Find first safetensors in test_output
        results_dir = Path("test_output")
        files = list(results_dir.glob("*.safetensors"))
        if files:
            inspect_safetensors(files[0])
        else:
            print("No .safetensors files found.")
