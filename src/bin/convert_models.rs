//! Pre-convert entropy model for fast runtime loading.
//!
//! This binary converts SafeTensors models to Burn's optimized binary format (.mpk).
//! Pre-converted models load ~10x faster.
//!
//! **Memory Requirements:**
//! - Entropy model: ~2GB RAM
//!
//! # Usage
//!
//! ```bash
//! # Convert entropy model
//! cargo run --release --bin convert_models
//! ```
//!
//! After conversion, the file is saved alongside the original in the HF cache:
//! - `model.safetensors` → `model.burn.mpk`

use std::path::{Path, PathBuf};
use std::time::Instant;

use blt_burn::model::LMTransformerConfig;

// Use NdArray backend for conversion (CPU-only, more memory efficient than WGPU)
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "convert-models")]
#[command(about = "Pre-convert entropy model for fast runtime loading")]
struct Args {
    /// Force re-conversion even if .mpk files exist
    #[arg(long)]
    force: bool,
}

fn get_hf_cache() -> PathBuf {
    let hf_cache = std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
        .unwrap_or_else(|_| format!("{}/.cache/huggingface", std::env::var("HOME").unwrap()));
    PathBuf::from(hf_cache)
}

fn find_model_path(hf_cache: &Path, model_name: &str) -> Option<PathBuf> {
    let model_base = hf_cache.join(format!(
        "hub/models--{}/snapshots",
        model_name.replace('/', "--")
    ));
    if model_base.exists() {
        std::fs::read_dir(&model_base)
            .ok()
            .and_then(|entries| {
                entries
                    .filter_map(std::result::Result::ok)
                    .find(|e| e.path().is_dir())
                    .map(|e| e.path().join("model.safetensors"))
            })
            .filter(|p| p.exists())
    } else {
        None
    }
}

fn convert_entropy_model(
    device: &NdArrayDevice,
    safetensors_path: &PathBuf,
    force: bool,
) -> anyhow::Result<()> {
    type Backend = NdArray<f32>;

    let mpk_path = safetensors_path.with_file_name("model.burn.mpk");

    if mpk_path.exists() && !force {
        println!("✓ Entropy model already converted: {mpk_path:?}");
        return Ok(());
    }

    println!("📦 Converting entropy model...");
    println!("   Source: {safetensors_path:?}");
    println!("   Using CPU (NdArray) backend for memory-efficient conversion");

    let start = Instant::now();

    // BLT entropy model config (matches facebook/blt-entropy)
    let config = LMTransformerConfig {
        dim: 768,
        n_layers: 14,
        head_dim: None,
        n_heads: Some(12),
        n_kv_heads: None,
        ffn_dim_multiplier: Some(1.0),
        multiple_of: 256,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seqlen: 8192,
        vocab_size: 260,
    };
    let model = config.init::<Backend>(device);

    // Load from SafeTensors
    let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder.load(safetensors_path.clone().into(), device)?;
    let model = model.load_record(record);

    println!("   Loaded in {:.2}s", start.elapsed().as_secs_f64());

    // Save to Burn's binary format
    let save_start = Instant::now();
    let mpk_recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    mpk_recorder.record(model.into_record(), mpk_path.clone())?;

    println!("   Saved to {} in {:.2}s", mpk_path.display(), save_start.elapsed().as_secs_f64());
    println!("✅ Entropy model converted in {:.2}s", start.elapsed().as_secs_f64());

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════╗");
    println!("║       Entropy Model Pre-Conversion Utility             ║");
    println!("╚════════════════════════════════════════════════════════╝");

    // Use CPU backend for memory-efficient conversion
    let device = NdArrayDevice::default();
    let hf_cache = get_hf_cache();

    println!("\n🔍 HuggingFace cache: {}", hf_cache.display());

    // Convert entropy model
    if let Some(entropy_path) = find_model_path(&hf_cache, "facebook/blt-entropy") {
        convert_entropy_model(&device, &entropy_path, args.force)?;
    } else {
        println!("⚠️  Entropy model not found in HF cache");
        println!("   Run: hf download facebook/blt-entropy");
    }

    println!("\n✅ Conversion complete!");
    println!("   Entropy model will now load faster at runtime.");

    Ok(())
}
