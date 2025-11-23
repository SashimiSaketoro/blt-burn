"""
Quick demonstration of pre-norm vs post-norm signal richness.

Run this to see why extracting embeddings BEFORE L2 normalization
gives you infinitely more signal for water-filling.
"""

import numpy as np
import matplotlib.pyplot as plt

def demonstrate_signal_loss():
    """Visual demonstration of information loss during normalization."""
    
    # Simulate pre-norm embeddings from a transformer
    # These have natural variance in L2 magnitude
    np.random.seed(42)
    n_points = 1000
    dim = 512
    
    # Create embeddings with varying "confidence" (L2 norm)
    embeddings_raw = np.random.randn(n_points, dim)
    
    # Some embeddings are high-confidence (model is certain)
    high_conf_mask = np.random.rand(n_points) < 0.15
    embeddings_raw[high_conf_mask] *= 2.5  # Boost L2 norm
    
    # Some are low-confidence (model is uncertain)
    low_conf_mask = np.random.rand(n_points) < 0.15
    embeddings_raw[low_conf_mask] *= 0.4  # Reduce L2 norm
    
    # Compute pre-norm L2 norms (the "signal")
    l2_norms_pre = np.linalg.norm(embeddings_raw, axis=1)
    
    # Apply L2 normalization (what typically happens)
    embeddings_normalized = embeddings_raw / (l2_norms_pre[:, None] + 1e-8)
    l2_norms_post = np.linalg.norm(embeddings_normalized, axis=1)
    
    # ============================================================================
    # ANALYSIS
    # ============================================================================
    
    print("=" * 70)
    print("PRE-NORM vs POST-NORM SIGNAL ANALYSIS")
    print("=" * 70)
    
    print(f"\n📊 Dataset: {n_points} embeddings of dimension {dim}")
    
    print("\n" + "-" * 70)
    print("L2 NORM STATISTICS")
    print("-" * 70)
    
    print(f"\n  PRE-NORMALIZATION:")
    print(f"    Mean:     {l2_norms_pre.mean():.4f}")
    print(f"    Std Dev:  {l2_norms_pre.std():.4f}  ← RICH SIGNAL")
    print(f"    Min:      {l2_norms_pre.min():.4f}")
    print(f"    Max:      {l2_norms_pre.max():.4f}")
    print(f"    Range:    {l2_norms_pre.max() - l2_norms_pre.min():.4f}")
    
    print(f"\n  POST-NORMALIZATION:")
    print(f"    Mean:     {l2_norms_post.mean():.4f}")
    print(f"    Std Dev:  {l2_norms_post.std():.10f}  ← NO SIGNAL (numerical error only)")
    print(f"    Min:      {l2_norms_post.min():.4f}")
    print(f"    Max:      {l2_norms_post.max():.4f}")
    print(f"    Range:    {l2_norms_post.max() - l2_norms_post.min():.10f}")
    
    # Signal-to-noise ratio
    variance_ratio = l2_norms_pre.std() / (l2_norms_post.std() + 1e-10)
    print(f"\n  ⚡ SIGNAL RATIO: {variance_ratio:.1e}x more variance in pre-norm")
    
    print("\n" + "-" * 70)
    print("PROMINENCE/OUTLIER DETECTION")
    print("-" * 70)
    
    # Detect outliers (scale-invariant threshold from TheSphere)
    threshold_pre = l2_norms_pre.mean() + 1.0 * l2_norms_pre.std()
    outliers_pre = np.sum(l2_norms_pre > threshold_pre)
    
    threshold_post = l2_norms_post.mean() + 1.0 * l2_norms_post.std()
    outliers_post = np.sum(l2_norms_post > threshold_post)
    
    print(f"\n  PRE-NORM:")
    print(f"    Threshold (μ + 1σ): {threshold_pre:.4f}")
    print(f"    Outliers detected:  {outliers_pre} ({100*outliers_pre/n_points:.1f}%)")
    print(f"    ✅ Can identify prominent points!")
    
    print(f"\n  POST-NORM:")
    print(f"    Threshold (μ + 1σ): {threshold_post:.4f}")
    print(f"    Outliers detected:  {outliers_post} ({100*outliers_post/n_points:.1f}%)")
    print(f"    ❌ All points look identical!")
    
    print("\n" + "-" * 70)
    print("WATER-FILLING IMPLICATIONS")
    print("-" * 70)
    
    print(f"\n  Using PRE-NORM embeddings:")
    print(f"    • Can detect high-prominence points (outliers)")
    print(f"    • Can use L2 norm as osmotic pressure")
    print(f"    • Can use L2 norm as kinetic energy (THRML)")
    print(f"    • Natural density signal for shell assignment")
    print(f"    • {variance_ratio:.0e}x more information!")
    
    print(f"\n  Using POST-NORM embeddings:")
    print(f"    • All points have L2=1.0 (within numerical error)")
    print(f"    • No prominence signal")
    print(f"    • No density gradient")
    print(f"    • Random shell assignment")
    print(f"    • Water-filling cannot converge!")
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram of L2 norms - PRE
        ax = axes[0, 0]
        ax.hist(l2_norms_pre, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(threshold_pre, color='red', linestyle='--', linewidth=2, 
                   label=f'Prominence threshold (μ+1σ)')
        ax.set_xlabel('L2 Norm', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('PRE-NORM: Rich Distribution (Signal!)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Histogram of L2 norms - POST
        ax = axes[0, 1]
        ax.hist(l2_norms_post, bins=50, alpha=0.7, color='gray', edgecolor='black')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('L2 Norm', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('POST-NORM: All ≈1.0 (No Signal!)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim([0.99, 1.01])  # Zoom in to show it's all ~1.0
        
        # Scatter: Index vs L2 Norm - PRE
        ax = axes[1, 0]
        colors = ['red' if norm > threshold_pre else 'blue' for norm in l2_norms_pre]
        ax.scatter(range(n_points), l2_norms_pre, c=colors, alpha=0.6, s=20)
        ax.axhline(threshold_pre, color='red', linestyle='--', linewidth=2, 
                   label='Prominence threshold')
        ax.set_xlabel('Point Index', fontsize=12)
        ax.set_ylabel('L2 Norm', fontsize=12)
        ax.set_title(f'PRE-NORM: {outliers_pre} Outliers Detected', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Scatter: Index vs L2 Norm - POST
        ax = axes[1, 1]
        ax.scatter(range(n_points), l2_norms_post, alpha=0.6, s=20, color='gray')
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Point Index', fontsize=12)
        ax.set_ylabel('L2 Norm', fontsize=12)
        ax.set_title('POST-NORM: No Outliers (No Signal)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim([0.99, 1.01])
        
        plt.tight_layout()
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, '..', 'docs', 'prenorm_signal_demo.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print("\n" + "=" * 70)
        print(f"📊 Visualization saved to: {output_path}")
        print("=" * 70)
        
    except ImportError:
        print("\n(matplotlib not available - skipping visualization)")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\n  ✅ Extract embeddings BEFORE final normalization")
    print("  ✅ Use L2 norms as prominence/density/energy signal")
    print("  ✅ Important for water-filling optimization")
    print("  ✅ Enables outlier detection and intelligent shell placement")
    print("\n" + "=" * 70)
    
    return {
        'pre_norm_norms': l2_norms_pre,
        'post_norm_norms': l2_norms_post,
        'outliers_pre': outliers_pre,
        'outliers_post': outliers_post,
        'signal_ratio': variance_ratio,
    }


if __name__ == "__main__":
    results = demonstrate_signal_loss()
