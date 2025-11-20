# BLT-Burn Library API Reference

> **Documentation for the partial Rust implementation of ByteLatent Transformer (BLT) components, focused on entropy-based segmentation and sphere embedding support.**

⚠️ **Scope Note**: This is a specialized implementation extracting only the BLT components needed for hypersphere embeddings. For full BLT functionality, see the [original repository](https://github.com/facebookresearch/blt).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Modules](#core-modules)
3. [Tokenization System](#tokenization-system)
4. [Pre-Tokenization Framework](#pre-tokenization-framework)
5. [Model Architecture](#model-architecture)
6. [Entropy & Patching](#entropy--patching)
7. [Dataset Integration](#dataset-integration)
8. [Water-Filling Integration](#water-filling-integration)
9. [Usage Examples](#usage-examples)
10. [Configuration Reference](#configuration-reference)

---

## Architecture Overview

### High-Level Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data    │────▶│ Pre-Tokenize │────▶│ BLT Entropy  │────▶│   Sphere     │
│ (multimodal) │     │  (semantic)  │     │    Model     │     │ Water-Fill   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                            │                     │                     │
                            ▼                     ▼                     ▼
                     ByteSegments          Pre-norm Embeddings   Hypersphere Coords
                     + Metadata            + Prominence Scores   + Shell Assignments
```

### Key Design Principles

1. **Modality Agnostic**: Accepts any byte stream (text, images, audio, code, etc.)
2. **Pre-L2-Norm Signal Extraction**: Captures embedding norms *before* L2 normalization for prominence
3. **Entropy-Based Boundaries**: Uses model confidence (entropy) to determine natural segmentation
4. **Sphere-Ready Output**: Produces embeddings and prominence scores ready for hypersphere organization

---

## Core Modules

### Module Structure

```
blt-burn/
├── src/
│   ├── model.rs           # BLT transformer architecture
│   ├── tokenizer.rs       # Text tokenization (BLT, TikToken, SentencePiece)
│   ├── pretokenize.rs     # Multimodal pre-tokenization framework
│   ├── patcher.rs         # Entropy calculation & patch extraction
│   ├── dataset.rs         # FineWeb-Edu and dataset utilities
│   └── lib.rs             # Public API exports
├── docs/
│   ├── API_REFERENCE.md   # This file
│   └── PRENORM_SIGNAL_SUMMARY.md
└── scripts/
    ├── water_filling_integration.py
    ├── demo_prenorm_signal.py
    └── inspect_sphere_result.py
```

---

## Tokenization System

### BltTokenizer

Simple byte-level tokenizer with optional BPE delimiters.

```rust
use blt_burn::tokenizer::BltTokenizer;

// Basic usage
let tokenizer = BltTokenizer::new(
    true,  // add_bos
    true   // add_eos
);

let tokens = tokenizer.encode("Hello world");
// Output: [1, 76, 105, 112, 112, 115, 36, 91, 115, 118, 112, 104, 2]
// Format: [BOS, 'H'+4, 'e'+4, 'l'+4, ..., EOS]

let text = tokenizer.decode(&tokens);
```

**Constants:**
- `BOS_ID = 1`
- `EOS_ID = 2`
- `BPE_ID = 3`
- `OFFSET = 4` (byte offset for token IDs)
- `VOCAB_SIZE = 260` (256 bytes + 4 special tokens)

### TikTokenTokenizer

Wrapper for OpenAI's tiktoken tokenizer.

```rust
use blt_burn::tokenizer::TikTokenTokenizer;

let tokenizer = TikTokenTokenizer::new("gpt-4")?;
let tokens = tokenizer.encode("Hello world");
```

### SentencePieceTokenizer

Wrapper for SentencePiece models (e.g., Llama tokenizers).

```rust
use blt_burn::tokenizer::SentencePieceTokenizer;

let tokenizer = SentencePieceTokenizer::new("path/to/tokenizer.model")?;
let tokens = tokenizer.encode("Hello world");
```

### BPE Delimiter Mode

For research experiments with BPE boundaries:

```rust
use blt_burn::tokenizer::{BltTokenizer, TikTokenTokenizer, Tokenizer};

let bpe = Box::new(TikTokenTokenizer::new("gpt-4")?);
let tokenizer = BltTokenizer::new_with_bpe(true, true, bpe);
// Inserts BPE_ID markers at token boundaries
```

---

## Pre-Tokenization Framework

### Overview

The pre-tokenization system segments raw data into semantically meaningful byte chunks **before** entropy analysis. This provides deterministic structure that helps with sphere organization.

### Core Trait

```rust
pub trait ModalityPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>>;
    fn modality(&self) -> &str;
}

pub struct ByteSegment {
    pub bytes: Vec<u8>,
    pub label: Option<String>,
    pub metadata: Option<SegmentMetadata>,
}
```

### Text Pre-Tokenizer

```rust
use blt_burn::pretokenize::{TextPreTokenizer, ModalityPreTokenizer};

// Simple whitespace tokenization
let pretokenizer = TextPreTokenizer::new_simple()?;
let segments = pretokenizer.pre_tokenize(b"Hello world")?;

// From tokenizer file
let pretokenizer = TextPreTokenizer::from_file("tokenizer.json")?;
```

### Image Pre-Tokenizer

```rust
use blt_burn::pretokenize::{PreTokenizerType, ModalityPreTokenizer};

let pretokenizer = PreTokenizerType::Image {
    patch_size: 196,  // 14x14 at 1 byte/pixel
    stride: 196,
}.create()?;

let image_bytes = std::fs::read("image.raw")?;
let patches = pretokenizer.pre_tokenize(&image_bytes)?;
// Each patch is a ByteSegment with metadata
```

### Audio Pre-Tokenizer

```rust
let pretokenizer = PreTokenizerType::Audio {
    frame_size: 160,    // 10ms at 16kHz
    sample_rate: 16000,
}.create()?;

let audio = std::fs::read("audio.raw")?;
let frames = pretokenizer.pre_tokenize(&audio)?;
```

### Code Pre-Tokenizer

```rust
let pretokenizer = PreTokenizerType::Code {
    language: "rust".to_string(),
}.create()?;

let code = std::fs::read("main.rs")?;
let lines = pretokenizer.pre_tokenize(&code)?;
// Currently line-based; tree-sitter integration planned
```

---

## Model Architecture

### LMTransformer

The core BLT entropy model.

```rust
use blt_burn::model::{LMTransformerConfig, LMTransformer};
use burn::backend::wgpu::{Wgpu, WgpuDevice};

let config = LMTransformerConfig {
    dim: 768,
    n_layers: 14,
    n_heads: Some(12),
    n_kv_heads: None,  // Use n_heads (full MHA)
    head_dim: None,    // Computed: dim / n_heads
    ffn_dim_multiplier: Some(1.0),
    multiple_of: 256,
    norm_eps: 1e-5,
    rope_theta: 10000.0,
    max_seqlen: 8192,
    vocab_size: 260,
};

let device = WgpuDevice::default();
let model = config.init::<Wgpu>(&device);
```

### Model Output Structure

**Critical for sphere integration:**

```rust
pub struct ModelOutput<B: Backend> {
    pub logits: Tensor<B, 3>,              // [batch, seq_len, vocab]
    pub pre_norm_embeddings: Tensor<B, 3>, // [batch, seq_len, dim] - PRE L2 norm
    pub embedding_norms: Tensor<B, 2>,     // [batch, seq_len] - PROMINENCE
}
```

**Why pre-norm embeddings matter:**

- Post-norm embeddings have uniform L2 norm (≈1.0), losing signal
- Pre-norm embeddings preserve magnitude variance (17.5 to 1e13 observed)
- `embedding_norms` provides the "prominence" signal for water-filling

### Forward Pass

```rust
let tokens = Tensor::<Wgpu, 2, Int>::from_data([[1, 76, 105, 112, 112, 115]], &device);
let output = model.forward_with_embeddings(tokens);

// Extract for sphere processing
let embeddings = output.pre_norm_embeddings;  // DON'T use post-norm!
let prominence = output.embedding_norms;      // Water-filling input
```

---

## Entropy & Patching

### Entropy Calculation

```rust
use blt_burn::patcher::entropy;

let logits = output.logits;  // [batch, seq_len, vocab]
let entropies = entropy(logits);  // [batch, seq_len]
```

### Patch Boundary Detection

Uses entropy + monotonicity constraint:

```rust
use blt_burn::patcher::{
    patch_start_mask_from_entropy_with_monotonicity,
    patch_start_indices_cpu,
};

let threshold = 1.35;  // Typical value
let mask = patch_start_mask_from_entropy_with_monotonicity(
    entropies.clone(),
    threshold
);

let patch_indices = patch_start_indices_cpu(mask);
// patch_indices[0] = [0, 15, 42, 87, ...]  // Start positions
```

**Monotonicity constraint**: Prevents backwards jumps, enforcing left-to-right patch growth.

---

## Dataset Integration

### FineWeb-Edu

```rust
use blt_burn::dataset::FineWebEduDataset;

let dataset = FineWebEduDataset::new(
    "sample-10BT",  // subset
    "train",        // split
    "dataset_cache" // cache directory
)?;

for item in dataset.iter().take(100) {
    println!("Processing: {}", item.id.unwrap_or_default());
    // item.text contains the document
}
```

---

## Water-Filling Integration

### Output Format

The `ingest` binary produces `.safetensors` files with:

```python
{
    "embeddings": [batch, seq_len, 768],      # Pre-norm!
    "prominence": [batch, seq_len],           # L2 norms
    "patch_indices": [num_patches],           # Start positions
    "patch_mask": [batch, seq_len],           # Binary mask
}
```

### Python Integration

```python
import numpy as np
from safetensors.numpy import load_file

# Load BLT output
data = load_file("ingest_output/item_0.safetensors")
embeddings = data["embeddings"]    # Pre-norm embeddings
prominence = data["prominence"]    # For water-filling

# Apply water-filling (osmotic or THRML)
from water_filling_integration import osmotic_water_filling
sphere_coords, radii, shells = osmotic_water_filling(
    embeddings=embeddings,
    prominence_scores=prominence,
    target_shells=512,
    capacity_exponent=1.5
)

# Save sphere results
np.savez("sphere_results/item_0.npz",
    sphere_coords=sphere_coords,
    radii=radii,
    shells=shells,
    original_prominence=prominence
)
```

---

## Usage Examples

### Example 1: Process Text with BLT

```rust
use blt_burn::{
    model::{LMTransformerConfig, LMTransformer},
    tokenizer::BltTokenizer,
    patcher::entropy,
};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, Int};
use burn::record::{NamedMpkFileRecorder, HalfPrecisionSettings, Recorder};

fn main() -> anyhow::Result<()> {
    let device = WgpuDevice::default();
    
    // Load model
    let config = LMTransformerConfig::default();
    let model = config.init::<Wgpu>(&device);
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
    let model = model.load_record(
        recorder.load("blt_entropy_model.mpk".into(), &device)?
    );
    
    // Tokenize
    let tokenizer = BltTokenizer::new(true, true);
    let text = "The quick brown fox";
    let tokens: Vec<i32> = tokenizer.encode(text)
        .iter().map(|&t| t as i32).collect();
    
    // Forward pass
    let input = Tensor::<Wgpu, 1, Int>::from_ints(&tokens, &device)
        .reshape([1, tokens.len()]);
    let output = model.forward_with_embeddings(input);
    
    // Extract for sphere
    let embeddings = output.pre_norm_embeddings;  // [1, seq_len, 768]
    let prominence = output.embedding_norms;       // [1, seq_len]
    
    // Calculate entropy
    let entropies = entropy(output.logits);
    
    println!("Embeddings shape: {:?}", embeddings.dims());
    println!("Prominence shape: {:?}", prominence.dims());
    
    Ok(())
}
```

### Example 2: Multimodal Pipeline

```rust
use blt_burn::pretokenize::{PreTokenizerType, ModalityPreTokenizer};

fn process_image(image_path: &str) -> anyhow::Result<()> {
    // Pre-tokenize image into patches
    let pretokenizer = PreTokenizerType::Image {
        patch_size: 196,
        stride: 196,
    }.create()?;
    
    let image_bytes = std::fs::read(image_path)?;
    let patches = pretokenizer.pre_tokenize(&image_bytes)?;
    
    println!("Image split into {} patches", patches.len());
    
    // Each patch can now be processed by BLT
    for (i, patch) in patches.iter().enumerate() {
        println!("Patch {}: {} bytes, label: {:?}",
            i, patch.bytes.len(), patch.label);
    }
    
    Ok(())
}
```

---

## Configuration Reference

### Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dim` | 768 | Hidden dimension |
| `n_layers` | 14 | Number of transformer blocks |
| `n_heads` | 12 | Attention heads |
| `vocab_size` | 260 | 256 bytes + 4 special tokens |
| `max_seqlen` | 8192 | Maximum sequence length |
| `norm_eps` | 1e-5 | RMS norm epsilon |
| `rope_theta` | 10000.0 | RoPE base frequency |

### Entropy Thresholds

| Use Case | Threshold | Notes |
|----------|-----------|-------|
| Standard text | 1.35 | Default for FineWeb-Edu |
| Code | 1.5-2.0 | Higher for more granular patches |
| Noisy data | 1.0-1.2 | Lower for more boundaries |

### Water-Filling Parameters

| Parameter | Range | Recommended |
|-----------|-------|-------------|
| `target_shells` | 64-512 | 128-256 for most cases |
| `capacity_exponent` | 1.0-2.0 | 1.5 (empirically optimal) |
| `min_radius` | 32-128 | 64.0 |
| `max_radius` | 512-2048 | 1024.0 |

---

## Integration Checklist for Sphere Development

- [ ] Consume `.safetensors` files from `ingest_output/`
- [ ] Use `embeddings` field (pre-norm, **not** post-norm)
- [ ] Use `prominence` field for water-filling prominence scores
- [ ] Respect `patch_indices` for semantic boundaries
- [ ] Output `.npz` files with `sphere_coords`, `radii`, `shells`
- [ ] Maintain `original_prominence` for analysis
- [ ] Support multimodal inputs via `pretokenize` module

---

## Model Weights

### Hugging Face Repository

```bash
# Automatic download at build time
cargo build  # Downloads model during build
cargo run --bin ingest  # Uses cached model
```

**Repository**: `SashimiSaketoro/entropy_burn`  
**File**: `blt_entropy_model.mpk` (190MB, bf16 precision)

### Manual Download

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="SashimiSaketoro/entropy_burn",
    filename="blt_entropy_model.mpk",
    local_dir="blt-burn/"
)
```

---

## Performance Notes

- **Batch Processing**: Process in chunks of 1024 tokens for optimal GPU utilization
- **Memory**: ~2GB VRAM for inference, ~4GB for training
- **Throughput**: ~100-200 tokens/sec on M1 Mac, ~500-1000 on NVIDIA GPUs
- **Pre-tokenization**: Adds <5% overhead, enables better sphere organization

---

## Further Reading

- [Pre-L2-Norm Signal Extraction](./PRENORM_SIGNAL_SUMMARY.md)
- [Water-Filling Integration](../scripts/water_filling_integration.py)
- [Rust Burn Book](https://burn.dev/book/)

---

**Last Updated**: 2025-11-19  
**Version**: 0.1.0  
**Maintainer**: BLT-Burn Contributors
