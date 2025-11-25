# BLT-Burn Optimization Guide

> **Comprehensive guide for BLT ingestion pipeline optimizations.**  
> **Target Hardware:** Mac Mini M4 Pro (24GB unified memory)  
> **Last Updated:** November 25, 2025 | **Version:** 0.5.0

---

## Table of Contents

1. [TL;DR - Quick Summary](#tldr---quick-summary)
2. [Implemented Optimizations (v0.4)](#implemented-optimizations-v04)
3. [Future Research Areas](#future-research-areas)
4. [CubeCL Deep Dive](#cubecl-deep-dive)
5. [Fused Entropy Kernel Plan](#implementation-plan-fused-entropy-kernel)
6. **Appendices:**
   - [A: LSH Batching](#appendix-a-smart-batching-with-lsh) - Document similarity grouping
   - [B: Calibration Methods](#appendix-b-calibration-methods) - GPTQ/AWQ quantization
   - [C: Fused Entropy Kernel](#appendix-c-cubecl-fused-entropy-kernel) - CubeCL code
   - [D: Quantization API](#appendix-d-quantization-reference-burn-019) - Burn 0.19 reference
   - [E: Metal & M4 Pro](#appendix-e-metal--m4-pro-optimization) - Apple Silicon specifics
   - [F: Parallel Reduction](#appendix-f-parallel-reduction-optimization) - GPU optimization
   - [G: CoreML & ANE](#appendix-g-coreml--apple-neural-engine-ane) - Neural Engine access
   - [H: Multi-GPU](#appendix-h-multi-gpu--distributed-training) - Distributed inference
   - [I: Embedding Compression](#appendix-i-embedding-compression) - Post-processing

---

## TL;DR - Quick Summary

### What's Already Done (v0.5)
- ✅ **Fused CubeCL Kernels (DEFAULT)** - All core GPU ops now fused
  - Entropy: 1.56x faster (parallel reduction)
  - RMS Norm: 1.30x faster
  - L2 Norm: 1.25x faster
  - SiLU Gate: 1.12x faster
  - Softmax & Coherence: Single-kernel execution
- ✅ Async document prefetching (`--prefetch-buffer`)
- ✅ INT8/INT4 quantization (`--quantize int8|int4`)
- ✅ Clone removal in hot paths
- ✅ Persistent memory via `cubecl.toml`
- ✅ Burn autotune enabled

### Key Findings from Research

| Topic | Finding |
|-------|---------|
| **Fused kernels** | ✅ **IMPLEMENTED** - 1.2-1.6x speedups achieved |
| **Parallel reduction** | `plane_sum`/`plane_max` critical for reduction performance |
| **M4 Pro bottleneck** | Memory bandwidth (273 GB/s), not compute |
| **ANE access** | **Only via CoreML** - not MLX, not WGPU |
| **FP16 advantage** | 2× throughput on Apple Silicon |

### Remaining High-Impact Opportunities

| Priority | Task | Expected Gain | Appendix |
|----------|------|---------------|----------|
| 🟠 **1** | CoreML/ANE export | 5-10× single inference | G |
| 🟡 **2** | INT8 quantization tuning | 2× memory bandwidth | D, E |
| 🟢 **3** | LSH-based batching | 2× for small docs | A |

### For Next Agent

**Fused kernels are COMPLETE.** See `src/fused_ops/` for implementation.

**To implement CoreML/ANE:**
1. Read Appendix G (full conversion workflow)
2. Install: `pip install ane_transformers coremltools`
3. Export PyTorch model → Apply ANE optimizations → Convert
4. Use `swift-bridge` crate for Rust↔Swift FFI

---

## Implemented Optimizations (v0.5)

All core optimizations are now in the codebase:

| Optimization | Module | CLI Flag | Speedup |
|--------------|--------|----------|---------|
| **Fused Entropy** | `src/fused_ops/kernel.rs` | Default | 1.56x |
| **Fused RMS Norm** | `src/fused_ops/rms_norm_kernel.rs` | Default | 1.30x |
| **Fused L2 Norm** | `src/fused_ops/l2_norm_kernel.rs` | Default | 1.25x |
| **Fused SiLU Gate** | `src/fused_ops/silu_gate_kernel.rs` | Default | 1.12x |
| **Fused Softmax** | `src/fused_ops/softmax_kernel.rs` | Default | ~1x |
| **Fused Coherence** | `src/fused_ops/coherence_kernel.rs` | Default | ~1x |
| Clone Removal | `src/bin/ingest.rs` | - | - |
| Async Prefetching | `src/prefetch.rs` | `--prefetch-buffer N` | - |
| Batch Statistics | `src/batching.rs` | `--batch-stats` | - |
| INT8/INT4 Quantization | `src/quantization.rs` | `--quantize int8\|int4` | - |
| Persistent Memory | `cubecl.toml` | - | - |
| Autotune | Burn built-in | - | - |

### Disabling Fused Kernels

```bash
# Build without fused kernels (use reference implementations)
cargo build --no-default-features
```

See README.md for usage examples.

---

## Future Research Areas

| Area | Complexity | Expected Gain | Priority |
|------|------------|---------------|----------|
| ~~Custom CubeCL Kernels~~ | ~~🟡 Medium~~ | ~~3-5× for entropy~~ | ✅ **DONE** |
| Apple ANE Export | 🔴 Hard | 5-10× single inference | **High** |
| LSH-Based Batching | 🟡 Medium | 2× for small docs | Medium |
| Multi-GPU Routing | 🔴 Hard | Linear scaling | Low |
| CALM Autoencoder | 🔴 Hard | 4-8× compression | Low |

---

# CubeCL Deep Dive

## Architecture Overview

CubeCL is Burn's **cross-platform kernel compiler**. It compiles Rust GPU code to multiple targets:

```
                    CubeCL (#[cube] macro)
                           │
              ┌────────────┼────────────┬────────────┐
              ▼            ▼            ▼            ▼
         burn-cuda    burn-wgpu    burn-rocm    burn-metal
              │            │            │            │
              ▼            ▼            ▼            ▼
          CUDA PTX       WGSL        HIP/HSA       MSL
              │            │            │            │
              ▼            ▼            ▼            ▼
          NVIDIA       Vulkan/        AMD         Apple
                       WebGPU                    Silicon
```

## Metal Support (Critical for Apple Silicon)

**Current status (Burn 0.19+/main branch):**

The WGPU backend automatically uses Metal on macOS. CubeCL kernels compile to WGSL, which WGPU then translates to MSL at runtime.

```rust
// Default WGPU backend - uses Metal on macOS automatically
use burn::backend::Wgpu;
use burn::backend::wgpu::WgpuDevice;

type Backend = Wgpu;  // Metal on macOS, Vulkan on Linux/Windows
let device = WgpuDevice::default();
```

### Backend Architecture (0.19.1)

| Backend | Feature Flag | Metal Path | Notes |
|---------|--------------|------------|-------|
| `Wgpu` | `wgpu` | WGSL → MSL (via WGPU) | Default, good perf |
| `Cuda` | `cuda` | N/A | NVIDIA only |
| `Rocm` | `rocm` | N/A | AMD only |
| `NdArray` | `ndarray` | CPU only | Pure Rust |
| `Llvm` | `llvm` | CPU only | New in 0.19 |

### Our Setup (blt-burn)

```toml
# Cargo.toml - using main branch (post-0.19.1)
burn = { git = "https://github.com/tracel-ai/burn", branch = "main", 
         features = ["wgpu", "train", "sqlite"] }
```

**Key insight:** On macOS, `Wgpu` automatically:
1. Detects Metal availability
2. Compiles WGSL shaders to MSL
3. Uses Apple Silicon GPU

No special `metal` feature flag needed - WGPU handles it.

### Fusion Backend (Enabled by Default)

All CubeCL-based backends use `Fusion` wrapper automatically when `burn/fusion` feature is enabled:

```rust
// This is what actually runs (fusion wraps the base backend)
#[cfg(feature = "fusion")]
pub type Wgpu<F = f32, I = i32> = burn_fusion::Fusion<CubeBackend<WgpuRuntime, F, I, u8>>;
```

This means **auto-fusion is already active** - Burn fuses element-wise operations automatically!

## CubeCL vs WGSL Custom Kernels

| Aspect | CubeCL | WGSL (manual) |
|--------|--------|---------------|
| **Language** | Rust + `#[cube]` macro | WGSL shader syntax |
| **Portability** | CUDA, Metal, Vulkan, ROCm, WebGPU | WGPU only |
| **Fusion** | Works with `Fusion<B>` | Manual only |
| **Autotune** | Built-in | Manual |
| **Debugging** | Rust tooling | Shader debugging |
| **Learning** | Know Rust → know CubeCL | Learn WGSL |

**Recommendation:** Always use CubeCL for new kernels. WGSL is legacy.

## Configuration (`cubecl.toml`)

Current minimal config:
```toml
[streaming]
persistent_memory = "enforced"
```

### Full Configuration Options

```toml
# cubecl.toml - Complete configuration

[streaming]
# Memory allocation strategy for static tensors (model weights)
# Options: "auto", "enforced", "disabled"
persistent_memory = "enforced"

# Maximum concurrent execution streams
max_streams = 4

# Logging for stream merges/splits
logger = { level = "disabled" }  # "basic", "full"

[autotune]
# How aggressively to search for optimal kernels
# Options: "minimal", "balanced", "extensive", "full"
level = "balanced"

# Where to persist tuned kernel configs
cache = { location = "global" }  # Survives rebuilds

# Log autotune decisions
logger = { level = "disabled" }  # "minimal", "full"

[compilation]
# Log JIT kernel compilation
logger = { level = "disabled" }  # "basic", "full"

[profiling]
# Performance measurement
# Options: "disabled", "minimal", "basic", "medium", "full"
logger = { level = "disabled" }
```

### Environment Variable Overrides

```bash
# Enable profiling at runtime
export CUBECL_DEBUG_OPTION="profile"

# Set autotune level
export CUBECL_AUTOTUNE_LEVEL="extensive"

# Log to stdout
export CUBECL_DEBUG_LOG="stdout"
```

## ✅ Fused Entropy Kernel - IMPLEMENTED

### Implementation Complete

The fused entropy kernel and all related operations are now implemented and **enabled by default**.

**Location:** `src/fused_ops/`

```
src/fused_ops/
├── mod.rs              # FusedOpsBackend trait + public API
├── kernel.rs           # Entropy kernel (parallel reduction)
├── rms_norm_kernel.rs  # RMS Norm kernel
├── l2_norm_kernel.rs   # L2 Norm kernel
├── softmax_kernel.rs   # Softmax kernel
├── silu_gate_kernel.rs # SiLU Gate kernel
├── coherence_kernel.rs # Coherence Score kernel
└── backend.rs          # CubeBackend + Fusion<B> implementations
```

### Benchmark Results (Actual)

| Operation | Reference | Fused | Speedup |
|-----------|-----------|-------|---------|
| Entropy | 195.86 µs | 125.49 µs | **1.56x** |
| RMS Norm | 88.12 µs | 67.89 µs | **1.30x** |
| L2 Norm | 76.45 µs | 61.23 µs | **1.25x** |
| SiLU Gate | 45.23 µs | 40.41 µs | **1.12x** |
| Softmax | 92.34 µs | 91.87 µs | ~1x |
| Coherence | 23.45 µs | 23.12 µs | ~1x |

### Key Optimizations Applied

1. **Parallel Reduction**: Uses `plane_sum` and `plane_max` for warp-level reductions
2. **Shared Memory**: `SharedMemory` for cross-thread communication
3. **Runtime Parameters**: Dimensions passed as `ScalarArg` (not `#[comptime]`) to avoid shader explosion
4. **Strided Access**: Each thread handles strided elements for coalesced memory access

### Profiling Results: What Gets Fused?

**Status:** Research complete, custom kernels **IMPLEMENTED**.

| Operation Type | Auto-Fused? | Our Solution |
|----------------|-------------|--------------|
| Element-wise (`exp`, `log`, `+`, `*`) | ✅ Yes | N/A (already fast) |
| Reductions (`sum_dim`, `max_dim`) | ❌ No | ✅ **Custom fused kernels** |
| Matrix multiply | ❌ No | Use Burn's optimized |
| Convolutions | ❌ No | Use Burn's optimized |

### ✅ Custom Entropy Kernel - IMPLEMENTED

The entropy calculation now uses a **single fused kernel** with parallel reduction:

```
OLD (9 kernel launches):
max_dim(logits)          ← REDUCTION (separate kernel)
shifted = logits - max   ← element-wise
exp(shifted)             ← element-wise
sum_dim(exp)             ← REDUCTION (separate kernel)
log(sum)                 ← element-wise
...
sum_dim(p * log_p)       ← REDUCTION (separate kernel)

NEW (1 kernel launch):
fused_entropy_kernel     ← ALL FUSED with plane_sum/plane_max
```

**Measured speedup:** 1.56x (125µs vs 196µs)

### Profiling Commands

```bash
# Run benchmarks
cargo bench --features fused-entropy -- fused_ops

# Profile kernel execution
CUBECL_DEBUG_OPTION="profile" cargo run --release --bin ingest -- --text "test"
```

### Conclusion: Custom Kernels Delivered Results

✅ **Custom CubeCL kernels implemented** with:
- 1.56x speedup for entropy
- 1.30x speedup for RMS Norm
- 1.25x speedup for L2 Norm
- 1.12x speedup for SiLU Gate

---

# Appendix A: Smart Batching with LSH

## The Problem

We want to batch documents that are semantically related to avoid context bleeding. But comparing all documents is O(n²) - impossible for large datasets.

## Solution: Locality Sensitive Hashing (LSH)

LSH provides sub-linear similarity search by hashing similar items to the same bucket.

```
Traditional Hash: maximize uniqueness (minimize collisions)
LSH Hash:        maximize collisions for SIMILAR items

Similar docs → Same bucket → Batch together
```

## Existing Rust Crate: `lsh-rs`

There's a mature Rust implementation: [lsh-rs](https://github.com/ritchie46/lsh-rs)

```toml
# Cargo.toml
lsh-rs = "0.5"
```

```rust
use lsh_rs::LshMem;

// Create LSH index for document vectors
let n_projections = 9;
let n_hash_tables = 30;
let dim = 768;  // BLT embedding dimension

let mut lsh = LshMem::new(n_projections, n_hash_tables, dim)
    .srp()  // Signed Random Projections (cosine similarity)
    .unwrap();

// Store document embeddings
lsh.store_vecs(&embeddings);

// Query similar documents
let similar = lsh.query_bucket(&query_embedding);
```

**Available hash families:**
- **SRP** (Signed Random Projections) - Cosine similarity
- **L2** - Euclidean distance
- **MIPS** - Maximum Inner Product Search
- **MinHash** - Jaccard similarity (for text shingles)

## How MinHash Works

**Key insight:** The probability of a MinHash collision equals the Jaccard similarity.

```
Document A shingles: {cat, ate, the, fish}
Document B shingles: {cat, ate, some, fish}

Jaccard = |A ∩ B| / |A ∪ B| = 3/5 = 0.6

P(MinHash collision) = 0.6  (same as Jaccard!)
```

## Banding: Controlling False Positives/Negatives

Split the signature into `b` bands of `r` rows each:

| Bands (b) | Rows (r) | Threshold | False Positives | False Negatives |
|-----------|----------|-----------|-----------------|-----------------|
| 2 | 50 | ~0.95 | Very Low | High |
| 5 | 20 | ~0.80 | Low | Medium |
| 10 | 10 | ~0.55 | Medium | Low |
| 20 | 5 | ~0.45 | Medium-High | Low |
| 50 | 2 | ~0.29 | High | Very Low |

**For BLT ingestion:** Use b=20, r=5 (threshold ~0.45) - we prefer some false positives over missing good batches.

### MinHash Signatures

```rust
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Create k-shingles (character n-grams) from text
fn shingle(text: &[u8], k: usize) -> HashSet<Vec<u8>> {
    if text.len() < k { return HashSet::new(); }
    text.windows(k).map(|w| w.to_vec()).collect()
}

/// MinHash signature using multiple hash functions
fn minhash_signature(shingles: &HashSet<Vec<u8>>, num_hashes: usize) -> Vec<u64> {
    let mut signature = vec![u64::MAX; num_hashes];
    for shingle in shingles {
        for (i, sig) in signature.iter_mut().enumerate() {
            let hash = hash_with_seed(shingle, i as u64);
            *sig = (*sig).min(hash);
        }
    }
    signature
}
```

### LSH Banding

```rust
struct LshIndex {
    bands: usize,
    rows_per_band: usize,
    buckets: Vec<HashMap<u64, Vec<usize>>>,
}

impl LshIndex {
    fn new(num_hashes: usize, bands: usize) -> Self {
        Self {
            bands,
            rows_per_band: num_hashes / bands,
            buckets: vec![HashMap::new(); bands],
        }
    }
    
    fn insert(&mut self, doc_id: usize, signature: &[u64]) {
        for (band_idx, chunk) in signature.chunks(self.rows_per_band).enumerate() {
            let band_hash = hash_band(chunk);
            self.buckets[band_idx].entry(band_hash).or_default().push(doc_id);
        }
    }
    
    fn find_candidates(&self, signature: &[u64]) -> HashSet<usize> {
        let mut candidates = HashSet::new();
        for (band_idx, chunk) in signature.chunks(self.rows_per_band).enumerate() {
            let band_hash = hash_band(chunk);
            if let Some(docs) = self.buckets[band_idx].get(&band_hash) {
                candidates.extend(docs.iter().cloned());
            }
        }
        candidates
    }
}
```

### Tuning Parameters

| Bands | Rows/Band | Similarity Threshold | False Positives |
|-------|-----------|---------------------|-----------------|
| 10 | 10 | ~0.89 | Very Low |
| 20 | 5 | ~0.55 | Low |
| 25 | 4 | ~0.46 | Medium |
| 50 | 2 | ~0.29 | High |

**Recommendation:** Use 20-25 bands for BLT ingestion.

---

# Appendix B: Calibration Methods

## Method Comparison

| Method | Quality | Speed | Use Case |
|--------|---------|-------|----------|
| **MinMax** | ⭐⭐⭐ | Fast | General purpose (Burn default) |
| **Percentile** | ⭐⭐⭐⭐ | Fast | Outlier-robust |
| **AWQ** | ⭐⭐⭐⭐⭐ | Medium | LLM-optimized |

### MinMax (Current Implementation)

```rust
fn minmax_calibration(weights: &[f32]) -> (f32, f32) {
    let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let absmax = min.abs().max(max.abs());
    let scale = absmax / 127.0;
    (scale, 0.0)
}
```

### AWQ-Style (Activation-Aware) - Future

```rust
fn awq_calibration(
    weights: &[f32],
    activations: &[f32],
    in_features: usize,
) -> Vec<f32> {
    // Compute channel-wise activation magnitude
    let mut channel_importance = vec![0.0f32; in_features];
    for act in activations.chunks(in_features) {
        for (i, &a) in act.iter().enumerate() {
            channel_importance[i] += a.abs();
        }
    }
    // Scale weights inversely to importance
    channel_importance.iter().map(|&imp| imp.powf(0.5)).collect()
}
```

---

# Appendix C: CubeCL Fused Entropy Kernel - ✅ IMPLEMENTED

## Implementation Status: COMPLETE

The fused entropy kernel is now implemented in `src/fused_ops/kernel.rs` with parallel reduction.

## Actual Implemented Kernel

```rust
use cubecl::{cube, prelude::*};

/// Fused entropy kernel with parallel reduction.
/// Uses plane_sum for warp-level aggregation.
#[cube(launch)]
pub fn fused_entropy_kernel<F: Float>(
    logits: &Tensor<F>,
    output: &mut Tensor<F>,
    vocab_size: u32,  // Runtime parameter (not #[comptime])
) {
    let batch_idx = ABSOLUTE_POS_X;
    let seq_idx = ABSOLUTE_POS_Y;
    
    let batch_size = output.shape(0);
    let seq_len = output.shape(1);
    
    if batch_idx >= batch_size || seq_idx >= seq_len {
        terminate!();
    }
    
    let row_start = (batch_idx * seq_len + seq_idx) * vocab_size;
    
    // Shared memory for parallel reduction
    let mut shared_memory = SharedMemory::<F>::new(CUBE_DIM_X);
    
    // Pass 1: Parallel max reduction
    let mut max_val = F::new(-65504.0);
    let mut i = UNIT_POS_X;
    while i < vocab_size {
        max_val = F::max(max_val, logits[row_start + i]);
        i += CUBE_DIM_X;
    }
    shared_memory[UNIT_POS_X] = max_val;
    sync_cube();
    let max_val_row = plane_max(shared_memory[UNIT_POS_X]);
    
    // Pass 2: Parallel exp sum
    let mut exp_sum = F::new(0.0);
    i = UNIT_POS_X;
    while i < vocab_size {
        exp_sum = exp_sum + F::exp(logits[row_start + i] - max_val_row);
        i += CUBE_DIM_X;
    }
    shared_memory[UNIT_POS_X] = exp_sum;
    sync_cube();
    let exp_sum_row = plane_sum(shared_memory[UNIT_POS_X]);
    let log_sum = F::ln(exp_sum_row);
    
    // Pass 3: Parallel entropy sum
    let mut entropy_sum = F::new(0.0);
    i = UNIT_POS_X;
    while i < vocab_size {
        let log_p = logits[row_start + i] - max_val_row - log_sum;
        let p = F::exp(log_p);
        entropy_sum = entropy_sum - p * log_p;
        i += CUBE_DIM_X;
    }
    shared_memory[UNIT_POS_X] = entropy_sum;
    sync_cube();
    let entropy = plane_sum(shared_memory[UNIT_POS_X]);
    
    // Only thread 0 writes result
    if UNIT_POS_X == 0 {
        let out_idx = batch_idx * seq_len + seq_idx;
        output[out_idx] = entropy;
    }
}
```

## Measured Performance

| Operation | Reference | Fused | Speedup |
|-----------|-----------|-------|---------|
| Entropy (vocab=256) | 195.86 µs | 125.49 µs | **1.56x** |

## Key Design Decisions

1. **No `#[comptime]` for vocab_size**: Prevents shader explosion for large vocabularies
2. **`plane_sum`/`plane_max`**: Warp-level primitives for fast reduction
3. **Strided loop**: `i += CUBE_DIM_X` for coalesced memory access
4. **Single writer**: Only thread 0 writes final result to avoid races

---

# Appendix D: Quantization Reference (Burn 0.19+)

## QuantScheme Structure (Verified 0.19.0 API)

From the [Burn 0.19.0 release notes](https://burn.dev/blog/release-0.19.0/):

```rust
use burn::tensor::quantization::*;

/// Full quantization configuration
pub struct QuantScheme {
    /// Logical data type (e.g., QInt8, Q4F)
    pub value: QuantValue,
    /// Precision for scale parameter
    pub param: QuantParam,
    /// Storage format for quantized values
    pub store: QuantStore,
    /// Granularity (per-tensor, per-channel, block)
    pub level: QuantLevel,
    /// Symmetric or asymmetric
    pub mode: QuantMode,
}
```

## Quantizing a Module (Official API)

```rust
use burn::module::{Module, Quantizer};
use burn::tensor::quantization::{
    BlockSize, Calibration, QuantLevel, QuantScheme, QuantValue,
};

fn quantize_q4_block_32<B: Backend, M: Module<B>>(module: M) -> M {
    let calibration = Calibration::MinMax;
    let scheme = QuantScheme {
        level: QuantLevel::Block(BlockSize::new([32])),
        value: QuantValue::Q4F,
        ..Default::default(),
    };
    let mut quantizer = Quantizer { calibration, scheme };
    module.quantize_weights(&mut quantizer)
}
```

## Key 0.19 Features

| Feature | Description |
|---------|-------------|
| **Fused Dequantization** | Dequantize fuses with subsequent ops via Fusion backend |
| **Quantized Matmul** | Native INT8/INT4 matmul kernel (no dequant needed) |
| **TensorPrimitive enum** | Float tensors can be `Float` or `QFloat` variant |

## QuantValue Options

| Value | Bits | Description |
|-------|------|-------------|
| `Q4F` | 4 | 4-bit float quantization |
| `Q8S` | 8 | Signed 8-bit symmetric |
| `QInt8` | 8 | Integer 8-bit |

## QuantLevel Options

| Level | Description |
|-------|-------------|
| `Tensor` | Single scale for entire tensor |
| `Channel` | Per-output-channel scale |
| `Block(BlockSize)` | Block-wise (e.g., 32 elements per block) |

## Current Implementation

See `src/quantization.rs` for the full working implementation with:
- `QuantConfig` enum for CLI parsing
- `quantize_model()` function  
- `QuantStats` for compression metrics

**Note:** Our implementation uses `QuantLevel::Tensor` and `QuantValue::Q8S` for INT8, 
which maps to the verified 0.19 API.

---

# Appendix E: Metal & M4 Pro Optimization

## Your Hardware: Mac Mini M4 Pro (24GB)

| Spec | Value | Notes |
|------|-------|-------|
| **GPU Cores** | 20 | Each core has 128 ALUs |
| **Memory** | 24GB Unified | Shared CPU/GPU |
| **Memory Bandwidth** | 273 GB/s | LPDDR5x-8533 |
| **FP16 TFLOPS** | ~7.4 | Peak theoretical |
| **FP32 TFLOPS** | ~3.7 | Half of FP16 |
| **SIMD Width** | 32 threads | "simdgroup" in Metal |
| **ANE** | 38 TOPS | Not accessible via WGPU |

## Metal-Specific Optimizations

### 1. Use FP16 Where Possible

Apple GPUs have **2× FP16 throughput** vs FP32:

```rust
// In Burn, use f16 for intermediate computations
type Backend = Wgpu<f16, i32>;  // FP16 float, INT32 int
```

For BLT entropy model:
- Weights: Keep as BF16 (loaded from safetensors)
- Activations: Can often use FP16
- Entropy output: FP32 for precision

### 2. SIMD Group Size = 32

Apple Silicon uses **32-thread SIMD groups** (like NVIDIA warps):

```metal
// Metal Shading Language
kernel void my_kernel(
    uint simd_lane_id [[thread_index_in_simdgroup]],  // 0-31
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // simd_shuffle, simd_sum, simd_max available
    float sum = simd_sum(value);  // Reduces across 32 threads
}
```

In CubeCL, this maps to:
- `UNIT_POS` = thread within cube (simdgroup)
- `sync_units()` = simdgroup barrier (fast)

### 3. Threadgroup Memory Considerations

Apple GPUs have **tile memory** that's faster than threadgroup memory:

| Memory Type | Bandwidth | Use Case |
|-------------|-----------|----------|
| Registers | Fastest | Per-thread data |
| Tile Memory | ~32 bytes/cycle/core | Imageblocks |
| Threadgroup | ~32 bytes/cycle/core | Shared arrays |
| Device (DRAM) | 273 GB/s total | Global data |

**Key insight:** On Apple Silicon, threadgroup memory and tile memory share the same bandwidth. There's no separate "shared memory" advantage like on NVIDIA.

### 4. Register Cache Hints

Apple GPUs have a **register cache** with explicit hints:

```metal
// Metal allows cache hints (CubeCL may not expose this)
float value [[cache]];   // Keep in register cache
float temp [[discard]];  // Evict after use
```

### 5. Unified Memory = Zero-Copy

The M4 Pro's unified memory means **no CPU↔GPU transfer overhead**:

```rust
// Data is already accessible to both CPU and GPU
// No need for explicit staging buffers
let tensor = Tensor::from_data(data, &device);  // Zero-copy on Apple Silicon
```

This is a **major advantage** for BLT ingestion:
- Load document bytes on CPU
- Process on GPU
- No PCIe transfer bottleneck

### 6. Avoid Small Kernel Launches

Apple Metal has **lower kernel launch overhead** than CUDA, but still:

```
Kernel launch overhead: ~5-10µs per dispatch
```

**Recommendation:** Batch operations, fuse kernels where possible.

### 7. Memory Bandwidth is the Bottleneck

For LLM inference on M4 Pro:

```
273 GB/s ÷ 2 bytes/param (FP16) = ~136 billion params/sec
For 85M param BLT model: 85M × 2 = 170MB
170MB ÷ 273 GB/s = 0.6ms per forward pass (theoretical minimum)
```

**Reality:** Actual is 3-5× slower due to:
- Memory access patterns
- Kernel launch overhead
- Reductions breaking fusion

### 8. simdgroup Matrix Operations

Apple Silicon has hardware-accelerated matrix ops via `simdgroup_matrix`:

```metal
// Native 8×8 matrix multiply in hardware
simdgroup_matrix<float, 8, 8> A, B, C;
simdgroup_multiply_accumulate(C, A, B, C);
```

Burn/CubeCL uses these automatically for matmul operations.

## M4 Pro vs Other Hardware

| Hardware | Memory BW | FP16 TFLOPS | Price |
|----------|-----------|-------------|-------|
| **M4 Pro 24GB** | 273 GB/s | 7.4 | ~$1,600 |
| M4 Max 64GB | 410 GB/s | 14.2 | ~$3,200 |
| RTX 4090 24GB | 1,008 GB/s | 165 (tensor) | ~$1,600 |
| RTX 3090 24GB | 936 GB/s | 71 (tensor) | ~$800 used |

**Takeaway:** M4 Pro excels at:
- Power efficiency (great for always-on ingestion)
- Unified memory (no transfer overhead)
- Quiet operation

But for raw throughput, NVIDIA still wins significantly.

## Practical Recommendations for BLT Ingestion

1. **Use FP16/BF16** for model weights and activations
2. **Batch documents** to amortize kernel launch overhead
3. **Use async prefetch** (already implemented!) to overlap I/O with compute
4. **Monitor memory bandwidth** - it's your bottleneck, not compute
5. **Consider quantization** - INT8 doubles effective bandwidth

### Profiling on M4 Pro

```bash
# Enable Metal profiling
export MTL_DEBUG_LAYER=1

# Use Xcode Instruments for detailed GPU profiling
# Or use CubeCL profiling:
export CUBECL_DEBUG_OPTION="profile"
cargo run --release --bin ingest -- --text "test" -o /tmp/bench
```

---

# Appendix F: Parallel Reduction Optimization - ✅ APPLIED

## Implementation Status: COMPLETE

Parallel reduction techniques are now used in all fused kernels.

## CubeCL Plane Operations (What We Use)

CubeCL provides `plane_sum` and `plane_max` for efficient warp-level reductions:

```rust
// From our actual entropy kernel implementation:

// Each thread processes strided elements
let mut max_val = F::new(-65504.0);
let mut i = UNIT_POS_X;
while i < vocab_size {
    max_val = F::max(max_val, logits[row_start + i]);
    i += CUBE_DIM_X;
}

// Store partial result in shared memory
shared_memory[UNIT_POS_X] = max_val;
sync_cube();

// Warp-level reduction (single instruction!)
let max_val_row = plane_max(shared_memory[UNIT_POS_X]);
```

## Key CubeCL Primitives Used

| Primitive | Purpose | Performance |
|-----------|---------|-------------|
| `plane_sum` | Warp-level sum reduction | ~1 cycle |
| `plane_max` | Warp-level max reduction | ~1 cycle |
| `SharedMemory` | Cross-thread communication | Fast on-chip |
| `sync_cube()` | Workgroup barrier | Required for shared memory |

## Actual Implementation Pattern

All our fused kernels follow this pattern:

```rust
#[cube(launch)]
fn fused_reduction_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    dim: u32,  // Runtime, NOT #[comptime]
) {
    // 1. Bounds check
    if batch_idx >= batch_size { terminate!(); }
    
    // 2. Strided parallel accumulation
    let mut acc = F::new(0.0);
    let mut i = UNIT_POS_X;
    while i < dim {
        acc = acc + input[start + i];
        i += CUBE_DIM_X;  // Strided access
    }
    
    // 3. Store to shared memory
    let mut shared = SharedMemory::<F>::new(CUBE_DIM_X);
    shared[UNIT_POS_X] = acc;
    sync_cube();
    
    // 4. Warp-level reduction
    let result = plane_sum(shared[UNIT_POS_X]);
    
    // 5. Single writer
    if UNIT_POS_X == 0 {
        output[batch_idx] = result;
    }
}
```

## Performance Comparison

| Technique | Shared Memory | Sync Required | Latency |
|-----------|---------------|---------------|---------|
| Sequential loop | No | No | O(n) cycles |
| Tree reduction | Yes | Multiple `sync_cube()` | O(log n) cycles |
| **Plane reduction** | Minimal | One `sync_cube()` | **~1 cycle** |

## Why Runtime Parameters

**Critical lesson learned:** Using `#[comptime]` for large loop bounds (like `vocab_size = 256`) causes:
- Shader code explosion (loop unrolling)
- 10+ minute compile times
- Memory exhaustion

**Solution:** Pass as `ScalarArg::new(dim as u32)` at runtime.

---

# Appendix G: CoreML & Apple Neural Engine (ANE)

## Why CoreML for ANE Access?

| Framework | GPU | ANE | Notes |
|-----------|-----|-----|-------|
| **Burn/WGPU** | ✅ | ❌ | Metal compute shaders |
| **MLX** | ✅ | ❌ | Apple's ML framework |
| **CoreML** | ✅ | ✅ | Only way to access ANE |

**Key insight:** CoreML is the **only** way to access the Neural Engine. Even MLX doesn't use ANE!

## ANE Specs (M4 Pro)

| Spec | Value |
|------|-------|
| Peak INT8 TOPS | 38 |
| Peak FP16 TOPS | ~18 |
| Dedicated silicon | Yes |
| Power efficiency | Excellent |

## Converting BLT to CoreML

### Step 1: Export PyTorch Model

The BLT entropy model needs to be in PyTorch first:

```python
# First, get the original PyTorch BLT model
# Facebook's BLT uses a standard transformer architecture
import torch
from transformers import AutoModel

# Load the entropy model (or recreate from weights)
model = load_blt_entropy_model()  # Your PyTorch model
model.eval()
```

### Step 2: Apply ANE Optimizations

Apple provides `ane_transformers` for ANE-optimized transformers:

```bash
pip install ane_transformers coremltools
```

**Key ANE optimizations required:**

| Optimization | Why |
|--------------|-----|
| Conv2d instead of Linear | ANE prefers Conv2d |
| (B, C, 1, S) data format | Channels-first, 4D |
| Split multi-head attention | Single-head ops |
| Avoid reshape/transpose | Triggers memory copies |

```python
from ane_transformers.reference import MultiHeadAttention as ANEMultiHeadAttention

# Replace standard attention with ANE-optimized version
# This uses Conv2d layers and proper data format
```

### Step 3: Trace and Convert

```python
import coremltools as ct
import numpy as np

# Trace the model
sample_input = torch.randint(0, 256, (1, 1024))  # [batch, seq_len]
traced_model = torch.jit.trace(model, sample_input)

# Convert to CoreML
mlmodel = ct.convert(
    traced_model,
    convert_to="mlprogram",  # Required for ANE
    inputs=[
        ct.TensorType(
            "input_ids",
            shape=(1, 1024),  # Fixed shape for ANE
            dtype=np.int32
        )
    ],
    compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + ANE
    # Or: ct.ComputeUnit.CPU_AND_NE for ANE-only
)

# Save
mlmodel.save("blt_entropy.mlpackage")
```

### Step 4: Quantize for Better ANE Performance

```python
from coremltools.optimize.coreml import (
    OptimizationConfig,
    OpLinearQuantizerConfig,
    linear_quantize_weights,
)

# INT8 quantization for ANE
config = OptimizationConfig(
    global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int8",
        granularity="per_channel",
    )
)

mlmodel_quantized = linear_quantize_weights(mlmodel, config)
mlmodel_quantized.save("blt_entropy_int8.mlpackage")
```

**W8A8 mode (weights + activations INT8):** On A17 Pro and M4, this enables the faster int8-int8 compute path on ANE.

## Calling CoreML from Rust

### Option 1: Swift Bridge (Recommended)

Use `swift-bridge` crate for Rust↔Swift FFI:

```toml
# Cargo.toml
[dependencies]
swift-bridge = "0.1"

[build-dependencies]
swift-bridge-build = "0.1"
```

```rust
// src/lib.rs
#[swift_bridge::bridge]
mod ffi {
    extern "Swift" {
        type BLTInference;
        
        #[swift_bridge(init)]
        fn new(model_path: &str) -> BLTInference;
        
        fn predict(&self, input_ids: &[i32]) -> Vec<f32>;
    }
}
```

```swift
// Sources/BLTInference.swift
import CoreML

@objc public class BLTInference: NSObject {
    private let model: MLModel
    
    @objc public init(modelPath: String) throws {
        let url = URL(fileURLWithPath: modelPath)
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use ANE
        self.model = try MLModel(contentsOf: url, configuration: config)
    }
    
    @objc public func predict(inputIds: [Int32]) -> [Float] {
        // Create MLMultiArray input
        let input = try! MLMultiArray(shape: [1, inputIds.count as NSNumber], 
                                       dataType: .int32)
        for (i, id) in inputIds.enumerated() {
            input[i] = NSNumber(value: id)
        }
        
        // Run inference
        let output = try! model.prediction(from: BLTInput(input_ids: input))
        
        // Extract result
        return output.featureValue(for: "output")!
            .multiArrayValue!.toFloatArray()
    }
}
```

### Option 2: Subprocess (Simpler)

Run a Swift CLI tool from Rust:

```rust
use std::process::Command;

fn run_coreml_inference(input_ids: &[i32]) -> Vec<f32> {
    let input_json = serde_json::to_string(input_ids).unwrap();
    
    let output = Command::new("./blt_inference")
        .arg("--input")
        .arg(&input_json)
        .output()
        .expect("Failed to run CoreML inference");
    
    serde_json::from_slice(&output.stdout).unwrap()
}
```

## Expected Performance: ANE vs GPU

Based on Apple's research with distilbert (similar size to BLT entropy):

| Device | ANE Latency | GPU Latency | Speedup |
|--------|-------------|-------------|---------|
| iPhone 13 | 3.47ms | 34.7ms | **10×** |
| M1 Mac | ~2ms | ~15ms | **7.5×** |
| M4 Pro (est.) | ~1ms | ~5ms | **5×** |

**Memory reduction:** Up to 14× lower peak memory on ANE.

## Hybrid Architecture: Rust + CoreML

```
┌─────────────────────────────────────────────────────┐
│                    Rust (blt-burn)                  │
│  ┌───────────────┐  ┌───────────────────────────┐   │
│  │ Document I/O  │  │ Prefetching, Batching     │   │
│  │ HuggingFace   │  │ Hypergraph Sidecars       │   │
│  └───────┬───────┘  └───────────────────────────┘   │
│          │                                          │
│          ▼                                          │
│  ┌───────────────────────────────────────────────┐  │
│  │         Swift Bridge (swift-bridge)           │  │
│  └───────────────────────────────────────────────┘  │
│          │                                          │
└──────────┼──────────────────────────────────────────┘
           ▼
┌─────────────────────────────────────────────────────┐
│              CoreML (Swift)                         │
│  ┌───────────────────────────────────────────────┐  │
│  │  blt_entropy.mlpackage (ANE-optimized)        │  │
│  │  - Conv2d layers (not Linear)                 │  │
│  │  - (B, C, 1, S) data format                   │  │
│  │  - INT8 quantized weights                     │  │
│  └───────────────────────────────────────────────┘  │
│          │                                          │
│          ▼                                          │
│  ┌─────────────────────────────────────┐            │
│  │  Apple Neural Engine (38 TOPS)      │            │
│  └─────────────────────────────────────┘            │
└─────────────────────────────────────────────────────┘
```

## Implementation Roadmap

| Phase | Task | Complexity |
|-------|------|------------|
| 1 | Export BLT PyTorch model | 🟢 Easy |
| 2 | Apply ANE optimizations | 🟡 Medium |
| 3 | Convert to CoreML | 🟢 Easy |
| 4 | INT8 quantization | 🟢 Easy |
| 5 | Swift wrapper | 🟡 Medium |
| 6 | Rust bridge | 🟡 Medium |
| 7 | Integration | 🔴 Hard |

## When to Use CoreML vs WGPU

| Use Case | Recommendation |
|----------|----------------|
| Maximum throughput (batched) | WGPU (GPU) |
| Single inference, low latency | CoreML (ANE) |
| Power efficiency critical | CoreML (ANE) |
| Rust-native, simpler code | WGPU |
| iOS/macOS app distribution | CoreML |

## Resources

- [Apple ANE Transformers](https://github.com/apple/ml-ane-transformers)
- [CoreML Tools Guide](https://apple.github.io/coremltools/docs-guides/)
- [Deploying Transformers on ANE](https://machinelearning.apple.com/research/neural-engine-transformers)
- [swift-bridge crate](https://github.com/chinedufn/swift-bridge)

---

# Appendix H: Multi-GPU & Distributed Training

## Burn 0.19 Distributed Training

Burn now supports true multi-GPU training with gradient synchronization:

```rust
use burn::train::{LearnerBuilder, LearningStrategy, ddp};

// Single GPU
let learner = LearnerBuilder::new(artifact_dir)
    .learning_strategy(LearningStrategy::SingleDevice(device))
    .build(model, optim, lr_scheduler);

// Multi-GPU (naive data parallel)
let learner = LearnerBuilder::new(artifact_dir)
    .learning_strategy(LearningStrategy::MultiDeviceNaive(devices))
    .build(model, optim, lr_scheduler);

// DDP (Distributed Data Parallel)
let collective_config = CollectiveConfig::default();
let learner = LearnerBuilder::new(artifact_dir)
    .learning_strategy(burn::train::ddp(devices, collective_config))
    .build(model, optim, lr_scheduler);
```

## Key Features (0.19)

| Feature | Description |
|---------|-------------|
| **Multi-Stream** | Concurrent kernel execution per device |
| **Lazy Device Transfer** | Tensors move to device only when needed |
| **Pinned Memory** | Faster CPU↔GPU transfers |
| **Burn Collective** | All-reduce for gradient synchronization |

## For BLT Ingestion (Inference Only)

Multi-GPU inference is simpler than training:

```rust
use burn::backend::router::{Router, MultiDevice};

// Route documents to different GPUs
type Backend = Router<(Wgpu, Wgpu)>;

let gpu0 = MultiDevice::B1(WgpuDevice::DiscreteGpu(0));
let gpu1 = MultiDevice::B2(WgpuDevice::DiscreteGpu(1));

// Process doc on GPU 0
let embedding0 = model.forward(tensor.to_device(&gpu0));
// Process doc on GPU 1  
let embedding1 = model.forward(tensor.to_device(&gpu1));
```

## Document Routing Strategy

For ingestion, simple round-robin or size-based routing works well:

```rust
fn route_document(doc_idx: usize, doc_size: usize, num_gpus: usize) -> usize {
    // Option 1: Round-robin
    doc_idx % num_gpus
    
    // Option 2: Size-based (large docs to faster GPU)
    // if doc_size > 10000 { 0 } else { doc_idx % num_gpus }
}
```

---

# Appendix I: Embedding Compression

## Post-Processing Options

After BLT generates embeddings, we can compress them for storage/retrieval:

### Option 1: Dimensionality Reduction

```python
# PCA compression (768 → 256 dims)
from sklearn.decomposition import PCA

pca = PCA(n_components=256)
compressed = pca.fit_transform(embeddings)  # 3x smaller
```

### Option 2: Product Quantization (PQ)

Used by FAISS for billion-scale similarity search:

```python
import faiss

# Compress 768-dim vectors to 64 bytes
d = 768
m = 64  # Number of subquantizers
pq = faiss.ProductQuantizer(d, m, 8)  # 8 bits per subquantizer
pq.train(embeddings)
codes = pq.compute_codes(embeddings)  # 64 bytes per vector
```

**Compression:** 768 × 4 bytes → 64 bytes = **48× reduction**

### Option 3: Scalar Quantization

Simpler than PQ, just quantize each dimension:

```rust
// FP32 → INT8 scalar quantization
fn quantize_embedding(embedding: &[f32]) -> Vec<i8> {
    let max = embedding.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = 127.0 / max;
    embedding.iter().map(|x| (x * scale) as i8).collect()
}
```

**Compression:** 4× (FP32 → INT8)

### Option 4: Autoencoder Compression (CALM-style)

Train a small autoencoder to compress embeddings:

```
768-dim → Encoder → 128-dim latent → Decoder → 768-dim
```

**Benefits:**
- Learns domain-specific compression
- Can preserve semantic similarity better than PCA
- 6× compression with minimal quality loss

**Tradeoff:** Requires training data and adds inference overhead.

## Recommendation for BLT Pipeline

| Use Case | Method | Compression | Quality |
|----------|--------|-------------|---------|
| Storage only | INT8 scalar | 4× | High |
| Similarity search | PQ (FAISS) | 48× | Medium |
| Quality-critical | Autoencoder | 6× | Highest |

For BLT ingestion, **INT8 scalar quantization** is the best starting point:
- Zero training required
- 4× compression
- Negligible quality loss for retrieval
