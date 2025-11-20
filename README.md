# BLT-Burn

> **Partial Rust implementation of ByteLatent Transformer (BLT) entropy model with Burn framework, specifically designed for hypersphere embedding pipelines.**

⚠️ **Note**: This is NOT a complete port of the original BLT repository. This implementation focuses specifically on entropy-based text segmentation and pre-norm signal extraction for hypersphere embeddings. Many features from the original BLT are intentionally omitted.

[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)

## Overview

BLT-Burn is a specialized implementation of select BLT components, extracting only the entropy model and embedding functionality needed for hypersphere-based systems. It provides:

- **Pre-norm signal extraction** - Captures embedding magnitudes before L2 normalization for prominence detection
- **Entropy-based patching** - Uses model confidence to determine natural segmentation boundaries
- **Multimodal pre-tokenization** - Supports text, images, audio, and code
- **Metal acceleration** - Automatically uses Metal on macOS (M1/M2/M3/M4) via WGPU backend
- **bf16 precision** - Optimized model weights (190MB vs 380MB fp32)

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

- **Dataset Python Environment**: The burn-dataset loader creates its own Python 3.9 virtual environment, which may conflict with projects requiring Python 3.12+ syntax. Workaround: Use `--text` or `--file` flags for direct text input, or ensure your system has Python 3.9 available.

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
**File**: `blt_entropy_model.mpk` (190MB, bf16)

## Features

- ✅ **Pre-L2-Norm Signal Extraction** - Preserves magnitude variance for prominence detection
- ✅ **Entropy-Based Patching** - Monotonic boundary detection using model confidence
- ✅ **Multimodal Support** - Text, images, audio, code pre-tokenization
- ✅ **Pure-Rust Approach** - No system dependencies for core functionality
- ✅ **Metal Acceleration** - Automatic GPU acceleration on macOS
- ✅ **bf16 Precision** - 50% smaller model size with maintained accuracy
- ✅ **FineWeb-Edu Integration** - Built-in dataset utilities
- ✅ **Water-Filling Ready** - Output format optimized for hypersphere pipelines

### Multimodal Pre-Tokenization

BLT-Burn includes a comprehensive pre-tokenization system that handles diverse data types:

#### Currently Supported
- **Text**: Raw UTF-8 bytes (BLT-style), HuggingFace tokenizers, or simple whitespace
- **Images**: JPEG/PNG decoding to RGB pixels with adaptive entropy-based patch merging
- **Audio**: WAV decoding to PCM samples with frame-based segmentation
- **Code**: AST-aware segmentation using tree-sitter (Rust, Python)

#### Detection & Routing
Automatic format detection based on magic bytes:
- JPEG (`FF D8`), PNG (`89 PNG`)
- PDF (`%PDF-`), MP4/Video (`ftyp`)
- WAV (`RIFF`), MP3 (`ID3` or sync bytes)
- ELF binaries (`7F ELF`), ZIP archives (`PK`)
- Code files (shebang, import statements)

#### Planned Support (Stubs Available)
- **PDF**: Text/image extraction (requires `pdf` crate)
- **Video**: Frame extraction (requires `ffmpeg-next` or pure-Rust alternatives)
- **Binary**: ELF section parsing (requires `goblin` crate)

### Pure-Rust Philosophy

To maintain portability and ease of deployment, BLT-Burn prioritizes pure-Rust implementations:

- **Audio/Video**: Can use `symphonia` for pure-Rust decoding (MP3, OGG, MP4)
- **Images**: Uses `image` crate (pure-Rust JPEG/PNG/GIF support)
- **Documents**: `pdf` crate for PDF parsing (pure-Rust)
- **Binaries**: `goblin` for ELF/PE/Mach-O analysis (pure-Rust)

This approach avoids system dependencies like FFmpeg, making the library more portable and easier to build.

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
│   └── inspect_sphere_result.py     # Sphere result inspector
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

**Version**: 0.1.0  
**Last Updated**: 2025-11-20

