# BLT-Burn

> **Partial Rust implementation of ByteLatent Transformer (BLT) entropy model with Burn framework, specifically designed for hypersphere embedding pipelines.**

> **Note**: This is NOT a complete port of the original BLT repository. This implementation focuses specifically on entropy-based text segmentation and pre-norm signal extraction for hypersphere embeddings. Many features from the original BLT are intentionally omitted.

[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)

## What's New in v0.3

- **Entropy-Weighted Prominence**: Physics-inspired water-filling allocation (`--entropy-weighted` flag)
- **Entropy & Coherence Export**: Full signal metrics exported to safetensors
- **Coherence-Biased Retrieval**: Favor low-entropy, high-confidence segments over surface statistics

## What's New in v0.2

- **User-controlled FFmpeg**: Interactive CLI prompts instead of automatic installation
- **SQLite Hypergraph Storage**: Compact, random-access friendly sidecar format
- **Python 3.12+ Support**: Automatic virtual environment management
- **JAX Sharding Support**: Automatic dataset sharding for distributed processing
- **Improved Error Handling**: Graceful failures throughout the pipeline

## Overview

BLT-Burn is a specialized implementation of select BLT components, extracting only the entropy model and embedding functionality needed for hypersphere-based systems. It provides:

- **Pre-norm signal extraction** - Captures embedding magnitudes before L2 normalization for prominence detection
- **Entropy-based patching** - Uses model confidence to determine natural segmentation boundaries
- **Multimodal pre-tokenization** - Supports text, images, audio, and code
- **GPU acceleration** - Uses WGPU backend (Metal on macOS, CUDA/Vulkan on Linux)
- **bf16 model weights** - Uses half-precision weights for efficiency

## What This Is / What This Isn't

### ✅ What This Is
- A focused implementation of BLT's entropy model for text segmentation
- A tool for extracting pre-norm embeddings for hypersphere placement
- A preprocessing pipeline for the Sphere water-filling algorithms
- A high-performance Rust implementation of specific BLT components

### ❌ What This Isn't
- A complete port of the BLT repository
- A training framework for BLT models
- A general-purpose transformer library
- A replacement for the original BLT implementation

If you need the full BLT functionality, please refer to the [original repository](https://github.com/facebookresearch/blt).

## Known Limitations

- **Video Processing**: Requires FFmpeg to be installed (handled automatically by build script).

## Entropy-Weighted Prominence Quick Start

```bash
# 1. Run ingestion (exports entropy + coherence automatically)
cargo run --release --bin ingest -- --file input.txt --output-dir output/

# 2. Apply entropy-weighted water-filling
python scripts/water_filling_integration.py --input output/ --entropy-weighted

# 3. Test and validate
python scripts/test_entropy_weighted.py --input output/item_0.safetensors
```

See [docs/PRE_NORM_SIGNAL_EXTRACTION.md](docs/PRE_NORM_SIGNAL_EXTRACTION.md#entropy-weighted-prominence-allocation) for the full write-up.

## Quick Start

### Installation

```bash
git clone https://github.com/SashimiSaketoro/blt-burn.git
cd blt-burn
cargo build --release
```

### Basic Usage

```rust
use blt_burn::{
    model::{LMTransformerConfig, LMTransformer},
    tokenizer::BltTokenizer,
};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder};

let device = WgpuDevice::default();
let config = LMTransformerConfig {
    dim: 768,
    n_layers: 14,
    n_heads: Some(12),
    vocab_size: 260,
    // ... other config
};
let model = config.init::<Wgpu>(&device);

// Load weights (auto-downloads from HF if not present)
let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
let model = model.load_record(
    recorder.load("blt_entropy_model.mpk".into(), &device)?
);

// Extract pre-norm embeddings
let output = model.forward_with_embeddings(tokens);
let embeddings = output.pre_norm_embeddings;  // For sphere placement
let prominence = output.embedding_norms;      // For water-filling
```

### Download Model

The model is automatically downloaded at build time from HuggingFace:

```bash
# Build downloads the model automatically
cargo build --release

# Or run any binary that needs the model
cargo run --bin ingest
```

**Repository**: [SashimiSaketoro/entropy_burn](https://huggingface.co/SashimiSaketoro/entropy_burn)  
**File**: `blt_entropy_model.mpk` (bf16 weights)

## Features

- ✅ **Pre-L2-Norm Signal Extraction** - Preserves magnitude variance for prominence detection
- ✅ **Entropy-Based Patching** - Monotonic boundary detection using model confidence
- ✅ **Entropy-Weighted Allocation** - Physics-inspired allocation for coherence-biased retrieval
- ✅ **Multimodal Support** - Text, images, audio, code pre-tokenization
- ✅ **Pure-Rust Approach** - No system dependencies for core functionality
- ✅ **GPU Acceleration** - Automatic acceleration via WGPU
- ✅ **bf16 Weights** - Half-precision model weights
- ✅ **FineWeb-Edu Integration** - Built-in dataset utilities
- ✅ **Water-Filling Ready** - Output format optimized for hypersphere pipelines
- ✅ **Hypergraph Sidecar** - SQLite-based storage with explicit Trunk-Branch-Leaf topology alongside tensors
- ✅ **JAX-Compatible Sharding** - Automatic dataset sharding for distributed processing

### Multimodal Pre-Tokenization

BLT-Burn includes a comprehensive pre-tokenization system that handles diverse data types:

#### Modality Support Matrix

| Modality | Decode Backend | Entropy Patching | Status |
|----------|----------------|------------------|--------|
| **Text** | Raw Bytes / HF Tokenizer | ✅ Yes | **Stable** |
| **Image** | `image` crate (Rust) | ✅ Yes | **Beta** |
| **Audio** | `symphonia` (Rust) | ✅ Yes | **Beta** |
| **Code** | `tree-sitter` (Rust) | ✅ Yes | **Beta** |
| **Video** | FFmpeg (`video-rs`) | 🚧 Partial | **Alpha** (Interactive install) |
| **PDF** | `pdf` crate | ❌ TODO | Stub |
| **Binary** | `goblin` | ❌ TODO | Stub |

#### Detection & Routing
Automatic format detection based on magic bytes:
- JPEG (`FF D8`), PNG (`89 PNG`)
- PDF (`%PDF-`), MP4/Video (`ftyp`)
- WAV (`RIFF`), MP3 (`ID3` or sync bytes)
- ELF binaries (`7F ELF`), ZIP archives (`PK`)
- Code files (shebang, import statements)

### Pure-Rust Philosophy

To maintain portability and ease of deployment, BLT-Burn prioritizes pure-Rust implementations where possible:

- **Audio**: Uses `symphonia` for pure-Rust decoding (MP3, OGG, MP4, WAV, FLAC, etc.)
- **Images**: Uses `image` crate (pure-Rust JPEG/PNG/GIF support)
- **Documents**: `pdf` crate for PDF parsing (pure-Rust)
- **Binaries**: `goblin` for ELF/PE/Mach-O analysis (pure-Rust)
- **Video**: Requires FFmpeg (automatically installed during build if missing)

### Hugging Face datasets via Polars

- `cargo run --bin ingest -- --huggingface-dataset openai/gdpval --limit 1` now streams Hugging Face splits directly with Polars.
- The loader resolves `hf://datasets/...` URIs from `dataset_info.json`, downloads the referenced Parquet shards through `hf-hub`, and falls back to Arrow IPC or JSON automatically.
- Because it leans on the Polars plugin architecture (hashing/spatial ops), no Python runtime or `burn-dataset` shim is required—everything stays inside the Rust process. [Polars plugin overview](https://docs.pola.rs/user-guide/plugins/)

#### Reference caching & archive streaming

- Every ingestion run now performs a **prefetch pass**: it scans the entire Polars `DataFrame`, deduplicates all `images/`, `files/`, and `hf://` references, and hydrates them once into `<output>/.cache/<dataset-slug>/…` before per-row work begins.
- The cache is **persistent**—if you keep the output directory around, downstream training or navigation stages can reuse the already-downloaded assets without touching Hugging Face again.
- When a dataset only exposes pointers (e.g. TreeVGR’s `images/...` paths), the loader can **stream individual files out of remote archives** hosted under a different Hugging Face repo. The TreeVGR ingestion, for example, automatically pulls the matching image out of `lmms-lab/LLaVA-NeXT-Data/llava_next_raw_format_images_*.tar.gz` without requiring Python or pre-extraction of the entire tarball.
- Prefetch still happens lazily per dataset—files are fetched on demand, but once cached they’re treated as the authoritative copy for subsequent ingest runs or analysis tools.

### Video Processing with FFmpeg

BLT-Burn uses FFmpeg for comprehensive video codec support:

- **User-Controlled Installation**: Interactive prompt when FFmpeg is needed
- **Full Codec Support**: H.264, H.265/HEVC, VP8, VP9, AV1, MPEG-4, MPEG-2, and more
- **CLI Options**:
  - `--no-audio-video`: Disable audio/video support
  - `--auto-install-ffmpeg`: Non-interactive auto-install for CI/Docker
  - `--ffmpeg-path /path/to/ffmpeg`: Use custom FFmpeg binary

## End-to-End Example

Process a directory of files, extracting embeddings and generating the hypergraph sidecar:

```bash
# 1. Run ingest (interactive FFmpeg check)
cargo run --release --bin ingest -- \
  --input ./samples \
  --output ./out

# 1b. For large files, enable JAX sharding
cargo run --release --bin ingest -- \
  --text "Your large text content..." \
  --output ./out \
  --num-shards 4  # Creates 4 shards for distributed processing

# 2. Inspect the resulting topology
python scripts/inspect_sphere_result.py ./out/output.safetensors
```

### Hypergraph Sidecar (`.hypergraph.db`)

BLT-Burn produces a SQLite "sidecar" file alongside the tensor output. This preserves the **Trunk-Branch-Leaf** topology that flattens into the tensor.

Example structure (when exported as JSON with `--export-json`):
```json
{
  "nodes": [
    { "Trunk": { "source_hash": "a1b2c3...", "total_bytes": 50000 } },
    { "Branch": { "label": "text_content", "modality": "text" } },
    { "Leaf": { "bytes": [], "label": "text_token", "metadata": { "start_offset": 0, "end_offset": 5 } } }
  ],
  "edges": [
    { "label": "contains", "weight": 1.0 },
    { "label": "next", "weight": 1.0 }
  ],
  "topology": {
    "edges": [
      [0, [0, 1]],  // Edge 0 connects Node 0 -> Node 1
      [1, [1, 2]]   // Edge 1 connects Node 1 -> Node 2  
    ]
  }
}
```

## Documentation

- **[API Reference](docs/API_REFERENCE.md)** - Complete library documentation
- **[Pre-Norm Signal Guide](docs/PRENORM_SIGNAL_SUMMARY.md)** - Why pre-norm matters
- **[Signal Extraction Details](docs/PRE_NORM_SIGNAL_EXTRACTION.md)** - Technical deep-dive

## Project Structure

```
blt-burn/
├── src/
│   ├── model.rs          # BLT transformer architecture
│   ├── tokenizer.rs      # Text tokenization
│   ├── pretokenize.rs    # Multimodal pre-tokenization
│   ├── patcher.rs        # Entropy & patch extraction
│   ├── dataset.rs        # FineWeb-Edu utilities
│   └── bin/              # Binary executables
│       ├── ingest.rs     # Main ingestion pipeline
│       ├── pretokenize_demo.rs  # Pre-tokenization demo
│       └── test_tokenizer.rs    # Tokenizer testing
├── scripts/
│   ├── water_filling_integration.py  # Python sphere algorithms
│   ├── demo_prenorm_signal.py       # Pre-norm signal demo
│   ├── inspect_sphere_result.py     # Sphere result inspector
│   └── tune_entropy_threshold.py    # Find optimal entropy thresholds
├── docs/                 # Documentation
└── Cargo.toml
```

## Requirements

- Rust 1.70+
- macOS (for Metal) or Linux/Windows (CPU/WGPU)
- Python 3.9+ (for scripts)

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)** license.

### Attribution
This codebase is a partial Rust/Burn implementation of specific components from the ByteLatent Transformer (BLT) architecture.
- **Original Work**: [BLT (Meta Research)](https://github.com/facebookresearch/blt)
- **Original License**: CC-BY-NC 4.0
- **Scope**: This implementation includes only:
  - The entropy model for text segmentation
  - Embedding extraction (pre-L2-norm)
  - Basic tokenization
  - **NOT included**: Training code, full transformer capabilities, compression features, or other BLT functionality
- **Modifications**: 
  - Implemented select components in Rust using the Burn framework
  - Added multimodal pre-tokenization system
  - Added pre-norm signal extraction for hypersphere integration
  - Optimized for Metal acceleration (Apple Silicon)

**Commercial Use**: Commercial use of this software is **prohibited** under the terms of the CC-BY-NC 4.0 license, unless you obtain separate permission from the original rights holders (Meta) and the authors of this derivative work.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## API Quick Reference

Three functions you probably care about most:

- **`tokenizer::BltTokenizer::encode_bytes`**  
  _Turn raw bytes into entropy-model-ready tokens._

- **`model::LMTransformer::forward_with_embeddings`**  
  _Run the model and get both `pre_norm_embeddings` (for geometry) and `embedding_norms` (for prominence)._

- **`pretokenize::detect_modality`**  
  _Auto-detect content type (Image, Audio, Video, Code) from magic bytes._

**Version**: 0.2.0  
**Last Updated**: 2025-11-20

