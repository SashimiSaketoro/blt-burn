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
                     HypergraphSidecar     Pre-norm Embeddings   Hypersphere Coords
                     (Trunk/Branch/Leaf)   + Prominence Scores   + Shell Assignments
```

### Key Design Principles

1. **Modality Agnostic**: Accepts any byte stream (text, images, audio, code, etc.)
2. **Pre-L2-Norm Signal Extraction**: Captures embedding norms *before* L2 normalization for prominence
3. **Entropy-Based Boundaries**: Uses model confidence (entropy) to determine natural segmentation
4. **Hypergraph Topology**: Explicitly models the hierarchical relationship between File (Trunk), Modality (Branch), and Patch (Leaf).
5. **Sphere-Ready Output**: Produces embeddings and prominence scores ready for hypersphere organization

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
│   ├── sidecar.rs         # Hypergraph DTO and builder
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

The pre-tokenization system segments raw data into semantically meaningful byte chunks **before** entropy analysis. This provides deterministic structure that helps with sphere organization. The framework prioritizes pure-Rust implementations to avoid system dependencies.

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

### Automatic Format Detection

```rust
use blt_burn::pretokenize::detect_modality;

let data = std::fs::read("file.unknown")?;
let pt_type = detect_modality(&data);
let segments = pt_type.create()?.pre_tokenize(&data)?;
```

### Text Pre-Tokenizers

```rust
use blt_burn::pretokenize::{PreTokenizerType, ModalityPreTokenizer};

// Raw UTF-8 bytes (BLT-style)
let pretokenizer = PreTokenizerType::TextRaw.create()?;
let segments = pretokenizer.pre_tokenize(b"Hello world")?;

// HuggingFace tokenizer
let pretokenizer = PreTokenizerType::TextFromFile {
    path: "tokenizer.json".to_string(),
}.create()?;

// Simple whitespace tokenization
let pretokenizer = PreTokenizerType::TextSimple.create()?;
```

### Image Pre-Tokenizer

Decodes JPEG/PNG to RGB pixels, creates patches with adaptive entropy-based merging:

```rust
let pretokenizer = PreTokenizerType::Image {
    patch_size: 16,   // 16x16 pixels
    stride: 16,       // Non-overlapping
}.create()?;

let image_bytes = std::fs::read("image.jpg")?;
let patches = pretokenizer.pre_tokenize(&image_bytes)?;

// Patches include entropy metadata for adaptive processing
for patch in patches {
    if let Some(meta) = patch.metadata {
        let entropy = meta.extra["local_entropy"].as_f64().unwrap();
        // Low-entropy patches may have been merged
    }
}
```

### Audio Pre-Tokenizer

Currently supports WAV decoding to PCM frames:

```rust
let pretokenizer = PreTokenizerType::Audio {
    frame_size: 160,    // 10ms at 16kHz
    sample_rate: 16000,
}.create()?;

let wav_data = std::fs::read("audio.wav")?;
let frames = pretokenizer.pre_tokenize(&wav_data)?;
```

**Future**: Pure-Rust `symphonia` integration for MP3/OGG/MP4 support.

### Code Pre-Tokenizer

AST-aware segmentation using tree-sitter:

```rust
let pretokenizer = PreTokenizerType::Code {
    language: "rust".to_string(),  // or "python"
}.create()?;

let code = std::fs::read("main.rs")?;
let segments = pretokenizer.pre_tokenize(&code)?;

// Segments are semantic units: functions, structs, classes
for seg in segments {
    println!("Found: {}", seg.label.unwrap()); // e.g., "function_item"
}
```

### Planned Pre-Tokenizers (Stubs Available)

#### PDF Pre-Tokenizer
```rust
// Requires: pdf = "0.9" in Cargo.toml
let pretokenizer = PreTokenizerType::Pdf { 
    extract_text: true 
}.create()?;
// Currently returns error, implement with pdf crate
```

#### Video Pre-Tokenizer
```rust
let pretokenizer = PreTokenizerType::Video { 
    frame_rate: 30 
}.create()?;

// FFmpeg is required and automatically installed during build if missing
// Full video frame extraction with comprehensive codec support
```

**FFmpeg Integration**:
- **Automatic Installation**: FFmpeg is detected and installed automatically during build if missing
- **No User Interaction**: Installation happens transparently - no prompts required
- **Full Codec Support**: All major video codecs supported via FFmpeg

**Supported Codecs (via FFmpeg/video-rs)**:
- H.264 (all profiles: Baseline, Main, High)
- H.265/HEVC
- VP8, VP9
- AV1
- MPEG-4, MPEG-2
- And many more formats

**Installation**:
FFmpeg is automatically installed during `cargo build` if not found on your system.
The installation script supports:
- macOS (Homebrew)
- Ubuntu/Debian (apt)
- Fedora/RHEL/CentOS (dnf)
- Arch/Manjaro (pacman)
- Windows (winget or manual)

#### Binary Pre-Tokenizer
```rust
// Requires: goblin = "0.8" in Cargo.toml
let pretokenizer = PreTokenizerType::Binary.create()?;
// Parses ELF sections, PE segments, etc.
```

### Pure-Rust Philosophy

To maintain portability and ease of deployment, BLT-Burn prioritizes pure-Rust implementations:

- **Images**: `image` crate (JPEG, PNG, GIF, BMP, etc.)
- **Audio**: `hound` for WAV, planned `symphonia` for MP3/OGG/MP4
- **Documents**: `pdf` crate for PDF parsing
- **Binaries**: `goblin` for ELF/PE/Mach-O analysis
- **Code**: `tree-sitter` with language bindings

This approach minimizes system dependencies where possible. FFmpeg is required for video processing and is automatically installed during build if missing.

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

### Handling Large Multimodal Datasets (e.g., MINT-1T)

For massive datasets like MINT-1T (1 trillion tokens, mixed modalities), use Burn's dataset transforms:

```rust
use burn::data::dataset::transform::{
    ComposedDataset, PartialDataset, SamplerDataset, 
    WindowsDataset, ShuffledDataset
};

// Assume MINT-1T loaded from Parquet
struct Mint1TDataset { /* ... */ }
impl Dataset<MintItem> for Mint1TDataset { /* ... */ }

// 1. Compose multiple shards
let shard1 = Mint1TDataset::new("mint-1t-shard-001.parquet");
let shard2 = Mint1TDataset::new("mint-1t-shard-002.parquet");
let composed = ComposedDataset::new(vec![shard1, shard2]);

// 2. Take a subset to avoid OOM
let partial = PartialDataset::new(composed, 0..1_000_000);

// 3. Sample for balanced modalities
let sampled = SamplerDataset::new(partial, 10_000, true);

// 4. Shuffle for training
let shuffled = ShuffledDataset::new(sampled, 42);

// 5. Process with automatic format detection
for item in shuffled.iter() {
    // Detect and process each field
    if let Some(text) = item.text {
        let pt_type = detect_modality(&text);
        let segments = pt_type.create()?.pre_tokenize(&text)?;
    }
    if let Some(image) = item.image_bytes {
        let pt_type = detect_modality(&image);
        let segments = pt_type.create()?.pre_tokenize(&image)?;
    }
}
```

### Streaming Large Files

For files too large to fit in memory:

```rust
use std::io::{BufReader, Read};

fn process_large_file(path: &str) -> Result<()> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buffer = vec![0u8; 1024 * 1024]; // 1MB chunks
    
    while let Ok(n) = reader.read(&mut buffer) {
        if n == 0 { break; }
        
        let chunk = &buffer[..n];
        let pt_type = detect_modality(chunk);
        match pt_type.create()?.pre_tokenize(chunk) {
            Ok(segments) => process_segments(segments),
            Err(_) => continue, // Skip unparseable chunks
        }
    }
    Ok(())
}
```

---

## Water-Filling Integration

### Output Format

The `ingest` binary produces matched pairs of files for every input item:

1. **Tensor Data (`.safetensors`)**:
   - Header: Contains `metadata_file` key pointing to the JSON sidecar.
   - Tensors:
     ```python
     {
         "embeddings": [batch, seq_len, 768],      # Pre-norm!
         "prominence": [batch, seq_len],           # L2 norms
         "patch_indices": [num_patches],           # Start positions
         "patch_mask": [batch, seq_len],           # Binary mask
     }
     ```

2. **Hypergraph Sidecar (`.hypergraph.json`)**:
   - Replaces the old linear `.metadata.json`.
   - Contains rich semantic information structured as a Directed Hypergraph.
   - Implements **Skeleton (Topology) + Flesh (Data)** architecture.
   - Format:
     ```json
     {
       "nodes": [
         { "Trunk": { "source_hash": "...", "total_bytes": 1024 } },
         { "Branch": { "label": "video_stream", "modality": "video" } },
         { "Leaf": { "bytes": [], "label": "frame_0", "metadata": { ... } } }
       ],
       "edges": [
         { "label": "contains", "weight": 1.0 },
         { "label": "next", "weight": 1.0 }
       ],
       "topology": {
         "edges": [
           [0, [0, 1]], // Edge 0 connects Node 0 (Trunk) -> Node 1 (Branch)
           [0, [1, 2]]  // Edge 0 (contains) connects Node 1 -> Node 2 (Leaf)
         ]
       }
     }
     ```

### Python Integration

```python
import numpy as np
import json
from pathlib import Path
from safetensors.numpy import load_file

# 1. Load Tensor Data
tensor_path = Path("ingest_output/item_0.safetensors")
data = load_file(tensor_path)
embeddings = data["embeddings"]    # Pre-norm embeddings
prominence = data["prominence"]    # For water-filling

# 2. Load Hypergraph Sidecar
meta_filename = tensor_path.with_suffix(".hypergraph.json")
with open(meta_filename, 'r') as f:
    hypergraph = json.load(f)

nodes = hypergraph['nodes']
# Find Trunk/Branch info
trunk = next(n['Trunk'] for n in nodes if 'Trunk' in n)
print(f"Processing {trunk['source_hash']} content")

# 3. Apply water-filling (osmotic or THRML)
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
| `min_radius` | 10.0-64.0 | 32.0 (Hollow Core size) |
| `max_radius` | Unbounded | Determined by prominence |

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

**Last Updated**: 2025-11-20  
**Version**: 0.1.0  
**Maintainer**: BLT-Burn Contributors
