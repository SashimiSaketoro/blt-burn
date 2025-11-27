# BLT-Burn Documentation

Welcome to the BLT-Burn library documentation!

## Quick Start

**New to this library?** Start here:

1. **[API_REFERENCE.md](./API_REFERENCE.md)** - Library reference
   - Architecture overview
   - Complete API documentation  
   - Usage examples
   - Integration checklist

2. **[PRENORM_SIGNAL_SUMMARY.md](./PRENORM_SIGNAL_SUMMARY.md)** - Why pre-L2-norm matters
   - Signal extraction rationale
   - Empirical comparisons
   - Water-filling integration

## For Sphere Development

If you're working on sphere/water-filling integration:

### Required Reading
- [API_REFERENCE.md](./API_REFERENCE.md) - Start with "Water-Filling Integration" section
- [Integration Checklist](./API_REFERENCE.md#integration-checklist-for-sphere-development)

### Quick Reference

| Output Field | Shape | Use For |
|--------------|-------|---------|
| `embeddings` | `[1, seq, 768]` | Pre-L2-norm embeddings (sphere directions) |
| `prominence` | `[1, seq]` | L2 norms (density signal) |
| `entropies` | `[1, seq]` | Per-token entropy |
| `coherence_scores` | `[1, seq]` | Confidence-weighted prominence (prominence²/entropy) |
| `patch_indices` | `[num_patches]` | Patch boundary positions |
| `patch_lengths` | `[num_patches]` | Length of each patch |
| `tokens` | `[1, seq]` | Byte tokens (0-259) |
| `mask` | `[1, seq]` | Attention mask |

### Key Concepts
1. **Use `pre_norm_embeddings`** (not post-norm) from `ModelOutput`
2. **Use `embedding_norms`** for prominence/water-filling input  
3. **Pre-tokenize multimodal data** for better structure
4. **Respect `patch_indices`** for semantic boundaries

### Pipeline Overview
```
Raw Data → Pre-Tokenize → BLT Model → Extract Pre-Norm → Water-Fill → Sphere
   ↓            ↓              ↓             ↓              ↓          ↓
ByteSegments  Patches    ModelOutput  (embeddings,   (sphere_coords,
                                       prominence)    radii, shells)
```

## Documentation Index

| Document | Purpose | Audience |
|----------|---------|----------|
| [API_REFERENCE.md](./API_REFERENCE.md) | Complete library reference | All developers |
| [PRENORM_SIGNAL_SUMMARY.md](./PRENORM_SIGNAL_SUMMARY.md) | Signal extraction details | Sphere developers, researchers |
| [README.md](../README.md) | Project quick start | First-time users |

## Examples

### Process Text
```rust
use blt_burn::{model::LMTransformerConfig, tokenizer::BltTokenizer};
use burn::backend::wgpu::{Wgpu, WgpuDevice};

let device = WgpuDevice::default();
let config = LMTransformerConfig::default();
let model = config.init::<Wgpu>(&device);
// Load weights (see API_REFERENCE.md for details)

let tokenizer = BltTokenizer::new(true, true);
let output = model.forward_with_embeddings(tokens);

// For sphere: use these two fields
let embeddings = output.pre_norm_embeddings;  // ← Pre-norm!
let prominence = output.embedding_norms;      // ← For water-filling
```

### Pre-Tokenize Images
```rust
use blt_burn::pretokenize::{PreTokenizerType, ModalityPreTokenizer};

let pretokenizer = PreTokenizerType::Image {
    patch_size: 196,
    stride: 196,
}.create()?;

let patches = pretokenizer.pre_tokenize(&image_bytes)?;
// Each patch gets its own embedding + prominence score
```

## Repository Structure

```
blt-burn/
├── docs/              ← You are here
│   ├── API_REFERENCE.md
│   ├── PRENORM_SIGNAL_SUMMARY.md
│   └── README.md
├── src/
│   ├── model.rs       # BLT transformer
│   ├── tokenizer.rs   # Text tokenization
│   ├── hf_resolver.rs # HuggingFace path resolution
│   ├── modalities/    # Multimodal pre-tokenization
│   ├── patcher.rs     # Entropy & patch extraction
│   ├── dataset.rs     # FineWeb-Edu utilities
│   ├── generic_processor.rs  # Schema-inferred processing
│   └── polars_dataset_loader.rs  # Polars-based HF loading
├── scripts/
│   └── water_filling_integration.py  # Python sphere algorithms
└── README.md          # Project quick start
```

## Get Help

- **API Questions**: See [API_REFERENCE.md](./API_REFERENCE.md)
- **Signal Questions**: See [PRENORM_SIGNAL_SUMMARY.md](./PRENORM_SIGNAL_SUMMARY.md)
- **Setup Issues**: See root [README.md](../README.md)

---

**Last Updated**: 2025-11-26  
**Version**: 0.7.0
