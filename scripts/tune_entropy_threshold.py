#!/usr/bin/env python3
"""
Entropy Threshold Tuning Script

This script helps find optimal entropy thresholds for different modalities
by running the ingest binary with various threshold values and analyzing
the resulting patch sizes and distributions.
"""

import subprocess
import json
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple
import argparse
from safetensors import safe_open

def run_ingest(input_path: Path, threshold: float, output_dir: Path) -> Dict:
    """Run the ingest binary with a specific threshold."""
    cmd = [
        "cargo", "run", "--release", "--bin", "ingest", "--",
        "--file", str(input_path),
        "--output", str(output_dir),
        "--threshold", str(threshold),
        "--no-audio-video",  # Skip FFmpeg for testing
    ]
    
    # TODO: Add --max-patch-length support when implemented in ingest binary
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"success": False, "stdout": e.stdout, "stderr": e.stderr}

def analyze_patches(safetensors_path: Path) -> Dict:
    """Analyze patch characteristics from the output."""
    with safe_open(str(safetensors_path), framework="numpy") as f:
        patch_indices = f.get_tensor("patch_indices")
        patch_mask = f.get_tensor("patch_mask")
        
    # Calculate patch sizes
    patch_sizes = []
    if len(patch_indices) > 1:
        for i in range(len(patch_indices) - 1):
            size = patch_indices[i + 1] - patch_indices[i]
            patch_sizes.append(int(size))
        # Last patch
        patch_sizes.append(int(len(patch_mask[0]) - patch_indices[-1]))
    
    # Define size buckets
    size_buckets = [
        (1, 3, "1-3"),
        (4, 10, "4-10"),
        (11, 24, "11-24"),
        (25, 48, "25-48"),
        (49, 100, "49-100"),
        (101, 256, "101-256"),
        (257, 512, "257-512"),
        (513, 1024, "513-1024"),
        (1025, float('inf'), "1025+")
    ]
    
    # Count patches in each bucket
    size_distribution = {}
    for min_size, max_size, label in size_buckets:
        count = sum(1 for size in patch_sizes if min_size <= size <= max_size)
        if count > 0:
            size_distribution[label] = {
                "count": count,
                "percentage": (count / len(patch_sizes) * 100) if patch_sizes else 0
            }
    
    # Calculate percentiles
    percentiles = {}
    if patch_sizes:
        for p in [10, 25, 50, 75, 90, 95, 99]:
            percentiles[f"p{p}"] = int(np.percentile(patch_sizes, p))
    
    return {
        "num_patches": len(patch_indices),
        "patch_sizes": patch_sizes,
        "mean_size": np.mean(patch_sizes) if patch_sizes else 0,
        "std_size": np.std(patch_sizes) if patch_sizes else 0,
        "min_size": np.min(patch_sizes) if patch_sizes else 0,
        "max_size": np.max(patch_sizes) if patch_sizes else 0,
        "total_tokens": len(patch_mask[0]),
        "size_distribution": size_distribution,
        "percentiles": percentiles
    }

def test_thresholds(input_path: Path, thresholds: List[float], modality: str) -> Dict:
    """Test multiple thresholds and analyze patch size distributions."""
    results = {
        "input": str(input_path),
        "modality": modality,
        "tests": []
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for threshold in thresholds:
            print(f"Testing threshold={threshold}...")
            
            output_dir = Path(temp_dir) / f"threshold_{threshold}"
            output_dir.mkdir(exist_ok=True)
            
            # Run ingest
            ingest_result = run_ingest(input_path, threshold, output_dir)
            
            if not ingest_result["success"]:
                print(f"  Failed: {ingest_result['stderr']}")
                continue
            
            # Find output file
            safetensors_files = list(output_dir.glob("*.safetensors"))
            if not safetensors_files:
                print("  No output file found")
                continue
            
            # Analyze patches
            patch_info = analyze_patches(safetensors_files[0])
            
            results["tests"].append({
                "threshold": threshold,
                "patches": patch_info
            })
            
            print(f"  Patches: {patch_info['num_patches']}")
            print(f"  Mean size: {patch_info['mean_size']:.1f} tokens")
            print(f"  Size range: {patch_info['min_size']}-{patch_info['max_size']} tokens")
            
            # Print size distribution
            if patch_info['size_distribution']:
                print("  Size distribution:")
                # Sort buckets by their starting number
                bucket_order = ["1-3", "4-10", "11-24", "25-48", "49-100", "101-256", "257-512", "513-1024", "1025+"]
                for bucket in bucket_order:
                    if bucket in patch_info['size_distribution']:
                        data = patch_info['size_distribution'][bucket]
                        bar = "█" * int(data['percentage'] / 2)
                        print(f"    {bucket:10}: {bar:<25} {data['count']:3} ({data['percentage']:4.1f}%)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Tune entropy thresholds for different modalities")
    parser.add_argument("--text", type=str, help="Text file to test")
    parser.add_argument("--image", type=str, help="Image file to test")
    parser.add_argument("--audio", type=str, help="Audio file to test")
    parser.add_argument("--code", type=str, help="Code file to test")
    parser.add_argument("--thresholds", type=str, default="1.0,1.1,1.2,1.3,1.35,1.4,1.5,1.55,1.6,1.7,1.8,2.0",
                       help="Comma-separated list of thresholds to test")
    # TODO: Add max-patch-length support when implemented in ingest binary
    # parser.add_argument("--max-patch-lengths", type=str, 
    #                    help="Comma-separated list of max patch lengths to test")
    parser.add_argument("--output", type=str, default="threshold_tuning_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [float(t) for t in args.thresholds.split(",")]
    
    all_results = []
    
    # Test each modality
    if args.text:
        print(f"\nTesting text file: {args.text}")
        results = test_thresholds(Path(args.text), thresholds, "text")
        all_results.append(results)
    
    if args.image:
        print(f"\nTesting image file: {args.image}")
        results = test_thresholds(Path(args.image), thresholds, "image")
        all_results.append(results)
    
    if args.audio:
        print(f"\nTesting audio file: {args.audio}")
        results = test_thresholds(Path(args.audio), thresholds, "audio")
        all_results.append(results)
    
    if args.code:
        print(f"\nTesting code file: {args.code}")
        results = test_thresholds(Path(args.code), thresholds, "code")
        all_results.append(results)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for result in all_results:
        print(f"\n{result['modality'].upper()}: {result['input']}")
        print("\nThreshold | Patches | Mean Size | P50  | P90  | P99  | Max | Most Common Bucket")
        print("-" * 85)
        
        for test in result["tests"]:
            p = test["patches"]
            
            # Find most common size bucket
            most_common_bucket = "N/A"
            if p['size_distribution']:
                most_common = max(p['size_distribution'].items(), 
                                key=lambda x: x[1]['count'])
                most_common_bucket = f"{most_common[0]} ({most_common[1]['percentage']:.0f}%)"
            
            # Get percentiles
            p50 = p['percentiles'].get('p50', 0)
            p90 = p['percentiles'].get('p90', 0)
            p99 = p['percentiles'].get('p99', 0)
            
            print(f"{test['threshold']:8.2f} | {p['num_patches']:7} | {p['mean_size']:9.1f} | "
                  f"{p50:4} | {p90:4} | {p99:4} | {p['max_size']:4} | {most_common_bucket}")
        
        # Print detailed size distribution for the best configuration
        if result["tests"]:
            print("\nDetailed size distribution for last configuration:")
            last_test = result["tests"][-1]
            dist = last_test["patches"]["size_distribution"]
            bucket_order = ["1-3", "4-10", "11-24", "25-48", "49-100", "101-256", "257-512", "513-1024", "1025+"]
            for bucket in bucket_order:
                if bucket in dist:
                    data = dist[bucket]
                    bar = "█" * int(data['percentage'] / 2)  # Simple bar chart
                    print(f"  {bucket:10}: {bar:<30} {data['count']:4} patches ({data['percentage']:5.1f}%)")
        
        # Print recommendation
        recommended, reason = find_recommended_threshold(result)
        if recommended:
            print(f"\n  Recommended threshold: {recommended:.2f} ({reason})")

def find_recommended_threshold(results: Dict, target_mean_size: int = None) -> Tuple[float, str]:
    """Find the recommended threshold based on patch size characteristics."""
    if not results["tests"]:
        return None, "No valid tests"
    
    # Default target mean sizes per modality
    default_targets = {
        "text": 128,    # ~1 paragraph
        "code": 256,    # ~1 function
        "image": 196,   # ~1 patch
        "audio": 240    # ~15ms at 16kHz
    }
    
    if target_mean_size is None:
        target_mean_size = default_targets.get(results["modality"], 128)
    
    # Find threshold closest to target mean size
    best_test = min(results["tests"], 
                    key=lambda t: abs(t["patches"]["mean_size"] - target_mean_size))
    
    reason = f"Mean size {best_test['patches']['mean_size']:.1f} closest to target {target_mean_size}"
    
    # Also check for good distribution (not too many tiny or huge patches)
    dist = best_test["patches"]["size_distribution"]
    tiny_percent = sum(dist.get(b, {}).get("percentage", 0) for b in ["1-3", "4-10"])
    huge_percent = dist.get("1025+", {}).get("percentage", 0)
    
    if tiny_percent > 50:
        reason += " (Warning: >50% tiny patches)"
    if huge_percent > 20:
        reason += " (Warning: >20% huge patches)"
    
    return best_test["threshold"], reason

if __name__ == "__main__":
    main()
