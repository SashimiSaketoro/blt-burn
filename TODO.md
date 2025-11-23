# Project TODO List

This file tracks pending tasks and improvements found in the codebase.

## Source Code (`src/`)

### `src/model.rs`
- [ ] **Line 311**: Apply RoPE (Rotary Positional Embeddings) to query and key tensors.
- [ ] **Line 325**: Apply Causal Mask to attention scores.

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
