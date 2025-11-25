# Project TODO List

This file tracks pending tasks and improvements.

## Core BLT Pipeline

### Completed
- [x] Implement RoPE (Rotary Position Embeddings) with θ=10000
- [x] Implement causal attention mask
- [x] Add SafeTensors model loading via `burn-import`
- [x] Create `blt_core.rs` with canonical BLT data structures
- [x] Implement chunked inference for long sequences
- [x] Fix HuggingFace loader to actually run inference
- [x] Add `--external-drive` flag for large dataset processing
- [x] Fix Python 3.12 compatibility in dataset importer
- [x] Add sequence edges ("next") to hypergraph sidecar
- [x] Inject coherence scores into leaf node metadata
- [x] **Fused CubeCL Kernels (v0.5)** - All core GPU ops now use optimized fused kernels:
  - Entropy: 1.56x faster (parallel reduction with `plane_sum`)
  - RMS Norm: 1.30x faster
  - L2 Norm: 1.25x faster
  - SiLU Gate: 1.12x faster
  - Softmax & Coherence: Single-kernel execution

### Pending
- [ ] Verify weight mapping from Facebook safetensors to Burn model
- [ ] Add batch processing support in `blt_core.rs`
- [ ] Benchmark inference speed vs. Facebook's PyTorch implementation

## Source Code (`src/`)

### `src/arrow_reader.rs`
- [ ] **Line 521**: Implement image path resolution from Hugging Face cache when reading image data.

## Scripts (`scripts/`)

### `scripts/tune_entropy_threshold.py`
- [ ] **Line 30**: Add `--max-patch-length` support when implemented in ingest binary.
- [ ] **Line 155**: Add `max-patch-length` support to CLI arguments when implemented in ingest binary.

## Documentation / Features (`README.md`)

### Feature Support
- [ ] **PDF Support**: Implement PDF processing using `pdf` crate.
- [ ] **Binary Support**: Implement binary file processing using `goblin`.
