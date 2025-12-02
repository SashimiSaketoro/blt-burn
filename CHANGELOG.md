## What's New in v0.7

- **Centralized HuggingFace Path Resolution**: New `hf_resolver.rs` module provides unified path resolution with automatic download, caching, and zero-tensor fallbacks
- **Polars 0.46 Integration**: Dataset loading uses Polars 0.46 with improved `List` dtype handling for array columns
- **Global API Instance**: Single shared HuggingFace API instance reduces overhead and avoids rate limiting
- **CLI Enhancements**: New `--skip-missing-files` and `--hf-token` flags for fine-grained control over dataset loading
- **Zero Tensor Fallback**: Missing images return 224×224×3 zero tensors instead of placeholder text, preserving tensor shape without semantic noise

## What's New in v0.6

- **Modular Pre-Tokenization Architecture**: `pretokenize.rs` is split into `src/modalities/` with separate files per modality for cleaner organization
- **Multi-View PDF Processing**: `--multiview-pdf` emits raw bytes, extracted text, and rendered images as separate hypergraph branches with a shared `source_id`
- **Cross-View Hyperedges**: `same_source` edges connect views of the same document for cross-modal association learning
- **PDF Mode Selection**: `--pdf-mode` selects the extraction mode (`raw_only`, `text_only`, `image_only`)
- **Binary Pre-Tokenizer**: ELF/PE section extraction via `goblin` crate

## What's New in v0.5

- **Fused CubeCL Kernels (Default)**: Core GPU operations now use fused kernels that combine multiple steps into single passes:
  - Entropy: computes softmax and entropy in one kernel (parallel reduction with `plane_sum`)
  - RMS Norm: computes mean, normalization, and scaling in one pass
  - L2 Norm: computes squared sum and square root in one pass
  - Softmax: single-kernel implementation with parallel max/sum reduction
  - SiLU Gate: fuses sigmoid and multiply for FFN
  - Coherence Score: computes coherence from norms and entropy in a single pass
- **Configuration**: Fused kernels are enabled by default; no feature flags are required

## What's New in v0.4

- **Async Prefetching**: Background document loading with `--prefetch-buffer N` (default: 4)
- **INT8/INT4 Quantization**: Optional model quantization via `--quantize int8|int4`
- **Batch Statistics**: Document size distribution with `--batch-stats`
- **Persistent Memory**: CubeCL configuration for reduced GPU memory waste
- **Performance Optimizations**: Clone removal, length-sorted processing

## What's New in v0.3

- **RoPE Implementation**: Full Rotary Position Embeddings following Meta's BLT architecture (θ=10000)
- **Causal Attention Mask**: Proper autoregressive masking in self-attention
- **Direct SafeTensors Loading**: Load original Facebook `blt-entropy` weights directly via `burn-import`
- **Core BLT Module**: `blt_core.rs` with canonical data structures matching Facebook's implementation
- **External Drive Support**: `--external-drive` flag for processing large datasets on external storage


## What's New in v0.2

- **User-controlled FFmpeg**: Interactive CLI prompts instead of automatic installation
- **SQLite Hypergraph Storage**: Compact, random-access friendly sidecar format
- **Python 3.12+ Support**: Automatic virtual environment management
- **JAX Sharding Support**: Automatic dataset sharding for distributed processing
- **Improved Error Handling**: Graceful failures throughout the pipeline
