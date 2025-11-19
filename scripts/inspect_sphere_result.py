import numpy as np
import sys
from pathlib import Path

def inspect_npz(path):
    print(f"Inspecting {path}...")
    try:
        data = np.load(path)
        print("Keys:", list(data.keys()))
        
        for key in data.keys():
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
        inspect_npz(sys.argv[1])
    else:
        # Find first npz in sphere_results
        results_dir = Path("blt-burn/sphere_results")
        files = list(results_dir.glob("*.npz"))
        if files:
            inspect_npz(files[0])
        else:
            print("No .npz files found.")
