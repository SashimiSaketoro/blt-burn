# blt-burn Implementation Guide

This guide documents the implementation details for BLT-Burn's multimodal ingestion pipeline, HuggingFace dataset integration, and related modules.

**System Requirements**:
- Rust 1.75+ (stable channel)
- Burn 0.19.1 (from crates.io)
- Polars 0.46 with features: `lazy`, `ipc`, `json`, `parquet`, `dtype-struct`
- For video: FFmpeg 3.4-8.0 (see README.md "Video Processing with FFmpeg" section)
- Testing: `cargo test --features fused-entropy` (add `video` if FFmpeg installed)

---

## Core Modules

### HuggingFace Path Resolution (`hf_resolver.rs`)

The `HfResolver` module provides centralized path resolution for HuggingFace datasets with automatic download, persistent caching, and graceful fallback handling.

#### Architecture

```rust
use blt_burn::hf_resolver::{HfResolver, HfResolverConfig};

// Configuration options
let config = HfResolverConfig::new("username/dataset", Path::new(".cache"))
    .with_skip_missing(false)    // Use fallbacks instead of skipping
    .with_token(Some("hf_xxx".to_string()))  // Optional auth token
    .with_repo_type(RepoType::Dataset);

let resolver = HfResolver::new(config);
```

#### Key Features

1. **Global API Instance**: Single `hf_hub::Api` instance shared across all resolutions, reducing overhead and avoiding rate limiting.

2. **Local Cache Priority**: Checks local cache before making network requests. Cache paths are sanitized to prevent directory traversal.

3. **URI Parsing**: Supports `hf://datasets/owner/name@revision/path` format with revision specifiers.

4. **HTTP Fallback**: Automatically retries via direct HTTP when `repo.get()` fails.

5. **Zero Tensor Fallback**: For images that cannot be resolved, returns 224×224×3 zero tensor (150,528 bytes) preserving tensor shape without semantic noise.

#### Public API

```rust
// Core resolution
pub fn resolve(&self, path: &str, is_image: bool) -> Result<Option<Vec<u8>>>;
pub fn resolve_hf_uri(&self, uri: &str, is_image: bool) -> Result<Option<Vec<u8>>>;
pub fn resolve_with_candidates(&self, path: &str, is_image: bool) -> Result<Option<Vec<u8>>>;

// URL resolution
pub fn resolve_url(url: &str, cache_dir: &Path) -> Result<Vec<u8>>;
pub fn resolve_url_with_fallback(url: &str, cache_dir: &Path, is_image: bool, skip_missing: bool) -> Result<Option<Vec<u8>>>;

// Helper functions
pub fn parse_hf_uri(uri: &str) -> Option<ParsedHfUri>;
pub fn encode_path_segments(path: &str) -> String;
pub fn sanitize_path(reference: &str) -> PathBuf;
pub fn looks_like_url(value: &str) -> bool;
pub fn filename_from_url(url: &str) -> String;
pub fn get_api() -> Result<&'static Api>;
```

---

The quantization module provides INT8 and INT4 model quantization using Burn's stable quantization API.

#### Supported Modes

| Mode | CLI Flag | Description |
|------|----------|-------------|
| None | `none` | No quantization (default) |
| INT8 Per-Tensor | `int8` | Symmetric INT8 quantization |
| INT8 Per-Channel | `int8-channel` or `q8c` | Channel-wise INT8 quantization |
| INT4 Block-32 | `int4` | 32-element block INT4 quantization |
| INT4 Block-64 | `int4-64` or `q4-64` | 64-element block INT4 quantization |

#### Usage

```rust
use blt_burn::quantization::{QuantConfig, quantize_model};

let config = QuantConfig::from_str("int8")?;
let quantized_model = quantize_model(model, config);
```

---

## Video Pre-Tokenization (`modalities/video.rs`)

Video processing extracts RGB frames using FFmpeg with 224×224 scaling.

#### Frame Extraction

```rust
use ffmpeg::software::scaling::context::Context as ScalerContext;
use ffmpeg::format::Pixel;

// Scale to 224x224 RGB24
let mut scaler = ScalerContext::get(
    frame.format(),
    frame.width(),
    frame.height(),
    Pixel::RGB24,
    TARGET_WIDTH,   // 224
    TARGET_HEIGHT,  // 224
    ffmpeg::software::scaling::flag::Flags::BILINEAR,
)?;

let mut rgb_frame = ffmpeg::frame::Video::empty();
scaler.run(&frame, &mut rgb_frame)?;
let raw_bytes: Vec<u8> = rgb_frame.data(0).to_vec();
```

#### Setup

```bash
# macOS
brew install ffmpeg pkg-config
source scripts/setup_ffmpeg_env.sh
cargo build --features video

# Linux (Ubuntu/Debian)
sudo apt install ffmpeg libavcodec-dev libavformat-dev libswscale-dev libavutil-dev pkg-config
cargo build --features video

# Windows
# Download FFmpeg from https://www.gyan.dev/ffmpeg/builds/
# Set FFMPEG_DIR environment variable
cargo build --features video
```

---

## Binary Pre-Tokenization (`modalities/binary.rs`)

Binary file processing extracts sections from ELF, PE (Windows), and Mach-O (macOS) executables using the `goblin` crate.

#### Supported Formats

| Format | Platform | Sections Extracted |
|--------|----------|-------------------|
| ELF | Linux/Unix | `.text`, `.data`, `.rodata`, etc. |
| PE | Windows | All named sections with valid data |
| Mach-O | macOS | Sections from all load commands (supports fat binaries) |

#### Section Metadata

Each extracted section includes:
- Section name and type
- Size and file offsets
- Virtual addresses (where applicable)

---

## Dependencies

### Cargo.toml Configuration

```toml
# Burn 0.19.1 - stable release from crates.io
burn = { version = "0.19.1", features = [
    "wgpu", "train", "sqlite-bundled", "store", "fusion", "autodiff", "autotune"
] }
burn-wgpu = "0.19.1"
burn-cubecl = { version = "0.19.1", optional = true }
burn-fusion = { version = "0.19.1", optional = true }
burn-import = "0.19.1"
cubecl = { version = "0.8.1", features = ["wgpu"], optional = true }

# Polars 0.46 for dataset loading
polars = { version = "0.46", features = ["lazy", "ipc", "json", "parquet", "dtype-struct"] }

# HuggingFace Hub
hf-hub = "0.3.2"

# Video (v8.0 supports FFmpeg 3.4 - 8.0)
ffmpeg-next = { version = "8.0", optional = true }
ffmpeg-sys-next = { version = "8.0", optional = true }

# CLI with environment variable support
clap = { version = "4.5", features = ["derive", "env"] }
```

---

## Dataset Processing Pipeline

### Generic Processor (`generic_processor.rs`)

The `GenericDatasetProcessor` provides schema-inferred processing for Polars DataFrames, automatically detecting modalities from column names and data types.

#### Features

- **Modality Inference**: Automatically detects text, image, audio, video, document, and binary columns
- **Reference Resolution**: Resolves file paths using `HfResolver` with caching
- **External Archive Support**: Streams files from remote tar.gz archives
- **Deduplication**: Tracks seen hashes to avoid duplicate processing

#### Integration with HfResolver

```rust
use crate::hf_resolver::{HfResolver, HfResolverConfig, sanitize_path, looks_like_url};

// URL resolution
if looks_like_url(target) {
    match HfResolver::resolve_url_with_fallback(target, cache_dir, is_image, false) {
        Ok(Some(bytes)) => { /* process bytes */ }
        Ok(None) => { /* skipped */ }
        Err(e) => { /* handle error */ }
    }
}

// HF URI resolution
if target.starts_with("hf://") {
    let config = HfResolverConfig::new(&dataset_name, cache_dir);
    let resolver = HfResolver::new(config);
    resolver.resolve_hf_uri(target, is_image)?;
}

// Dataset asset resolution with path candidates
let resolver = HfResolver::new(config);
resolver.resolve_with_candidates(path, is_image)?;
```

### Polars Dataset Loader (`polars_dataset_loader.rs`)

Pure-Rust HuggingFace dataset loading via Polars with support for JSON, Parquet, and Arrow IPC formats.

#### Features

- **Shared API Instance**: Uses `get_api()` from `hf_resolver` for efficient API reuse
- **Multiple Format Support**: Automatically detects and loads JSON, Parquet, Arrow IPC
- **URI Resolution**: Parses `hf://datasets/` URIs for parquet file discovery
- **Native List Dtype**: Proper handling of array columns using Polars List type

#### Usage

```rust
use blt_burn::polars_dataset_loader::load_hf_dataset;

let df = load_hf_dataset(
    "username/dataset",
    Some("train"),
    Some(1000),  // limit
)?;
```

---

## CLI Integration

### HuggingFace Dataset Loading

```bash
# Basic usage
cargo run --release --bin ingest -- \
    --huggingface-dataset HaochenWang/TreeVGR-SFT-35K \
    --limit 5

# With authentication (for private datasets)
cargo run --release --bin ingest -- \
    --huggingface-dataset private/dataset \
    --hf-token hf_xxx

# Or via environment variable
HF_TOKEN=hf_xxx cargo run --release --bin ingest -- \
    --huggingface-dataset private/dataset

# Strict mode (fail instead of fallback)
cargo run --release --bin ingest -- \
    --huggingface-dataset username/dataset \
    --skip-missing-files
```

---

## Testing

### Unit Tests

```bash
cargo test --features fused-entropy
```

### Integration Test

```bash
# Test with a real HF dataset
cargo run --release --bin ingest -- \
    --huggingface-dataset HaochenWang/TreeVGR-SFT-35K \
    --limit 5 \
    --output-dir ./test_output

# Verify output
python -c "
import safetensors
with safetensors.safe_open('./test_output/item_0.safetensors', framework='numpy') as f:
    emb = f.get_tensor('embeddings')
    print(f'Embeddings shape: {emb.shape}')
"
```

### Verification

```bash
# Verify no Api::new() calls outside hf_resolver.rs
grep -r "Api::new()" src/ --include="*.rs" | grep -v hf_resolver
# Should return no results
```
