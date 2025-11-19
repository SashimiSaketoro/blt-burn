# BLT-Burn

> **Rust implementation of ByteLatent Transformer (BLT) entropy model with Burn framework, optimized for hypersphere embedding pipelines.**

[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)

## Overview

BLT-Burn is a complete Rust port of the BLT entropy model, designed for integration with hypersphere-based embedding systems. It provides:

- **Pre-norm signal extraction** - Captures embedding magnitudes before L2 normalization for prominence detection
- **Entropy-based patching** - Uses model confidence to determine natural segmentation boundaries
- **Multimodal pre-tokenization** - Supports text, images, audio, and code
- **Metal acceleration** - Automatically uses Metal on macOS (M1/M2/M3/M4) via WGPU backend
- **bf16 precision** - Optimized model weights (190MB vs 380MB fp32)

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

The model is available on HuggingFace:

```bash
# Automatic (via ingest binary)
cargo run --bin ingest

# Or manual download
python3 scripts/upload_model.py --file blt_entropy_model.mpk
```

**Repository**: [SashimiSaketoro/entropy_burn](https://huggingface.co/SashimiSaketoro/entropy_burn)  
**File**: `blt_entropy_model.mpk` (190MB, bf16)

## Features

- ✅ **Pre-L2-Norm Signal Extraction** - Preserves magnitude variance for prominence detection
- ✅ **Entropy-Based Patching** - Monotonic boundary detection using model confidence
- ✅ **Multimodal Support** - Text, images, audio, code pre-tokenization
- ✅ **Metal Acceleration** - Automatic GPU acceleration on macOS
- ✅ **bf16 Precision** - 50% smaller model size with maintained accuracy
- ✅ **FineWeb-Edu Integration** - Built-in dataset utilities
- ✅ **Water-Filling Ready** - Output format optimized for hypersphere pipelines

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
│       ├── convert.rs    # Safetensors → MPK converter
│       ├── ingest.rs     # Main ingestion pipeline
│       └── verify_weights.rs  # Weight verification
├── scripts/
│   ├── upload_model.py   # HF upload utility
│   └── water_filling_integration.py  # Python integration
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
This codebase is a Rust/Burn implementation of the ByteLatent Transformer (BLT) architecture.
- **Original Work**: [BLT (Meta Research)](https://github.com/facebookresearch/blt)
- **Original License**: CC-BY-NC 4.0
- **Modifications**: 
  - Complete rewrite in Rust using the Burn framework
  - Added multimodal pre-tokenization system
  - Added pre-norm signal extraction for hypersphere integration
  - Optimized for Metal acceleration (Apple Silicon)

**Commercial Use**: Commercial use of this software is **prohibited** under the terms of the CC-BY-NC 4.0 license, unless you obtain separate permission from the original rights holders (Meta) and the authors of this derivative work.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Version**: 0.1.0  
**Last Updated**: 2025-11-19

