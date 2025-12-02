//! Utility to peek at model weights for comparison.
//!
//! Compares weights between SafeTensors (original) and MPK (converted) formats.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::use_debug)] // Debug printing is intentional for weight inspection

use std::path::PathBuf;

use blt_burn::model::LMTransformerConfig;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;

fn main() -> anyhow::Result<()> {
    type Backend = NdArray<f32>;
    let device = NdArrayDevice::default();

    // Find entropy model paths
    let hf_cache = std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
        .unwrap_or_else(|_| format!("{}/.cache/huggingface", std::env::var("HOME").unwrap()));

    let base_path = PathBuf::from(&hf_cache).join("hub/models--facebook--blt-entropy/snapshots");

    let snapshot_dir = std::fs::read_dir(&base_path)?
        .filter_map(std::result::Result::ok)
        .find(|e| e.path().is_dir())
        .map(|e| e.path())
        .expect("No snapshot found");

    let safetensors_path = snapshot_dir.join("model.safetensors");
    let mpk_path = snapshot_dir.join("model.burn.mpk");

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              Weight Comparison: SafeTensors vs MPK             ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // BLT entropy model config
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

    // Load from SafeTensors
    println!("Loading from SafeTensors: {}", safetensors_path.display());
    let model_st = config.init::<Backend>(&device);
    let recorder_st = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
    let record_st = recorder_st.load(safetensors_path.clone().into(), &device)?;
    let model_st = model_st.load_record(record_st);

    // Load from MPK
    println!("Loading from MPK: {}", mpk_path.display());
    let model_mpk = config.init::<Backend>(&device);
    let recorder_mpk = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let record_mpk = recorder_mpk.load(mpk_path.clone(), &device)?;
    let model_mpk = model_mpk.load_record(record_mpk);

    println!();
    println!("=== Comparing layer 0 attention weights ===");
    println!();

    // Compare wq weights from layer 0
    let wq_st = model_st.layers.first().unwrap().attention.wq.weight.val();
    let wq_mpk = model_mpk.layers.first().unwrap().attention.wq.weight.val();

    let wq_st_data = wq_st.clone().into_data();
    let wq_mpk_data = wq_mpk.clone().into_data();

    println!("Layer 0 wq.weight:");
    println!("  Shape: {:?}", wq_st.dims());

    // Get first 5 values
    let st_vals: Vec<f32> = wq_st_data.to_vec().unwrap();
    let mpk_vals: Vec<f32> = wq_mpk_data.to_vec().unwrap();

    println!("  SafeTensors first 5: {:?}", &st_vals[..5]);
    println!("  MPK first 5:         {:?}", &mpk_vals[..5]);

    // Compare statistics
    let st_min = st_vals.iter().copied().fold(f32::INFINITY, f32::min);
    let st_max = st_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let st_mean: f32 = st_vals.iter().sum::<f32>() / st_vals.len() as f32;

    let mpk_min = mpk_vals.iter().copied().fold(f32::INFINITY, f32::min);
    let mpk_max = mpk_vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mpk_mean: f32 = mpk_vals.iter().sum::<f32>() / mpk_vals.len() as f32;

    println!();
    println!(
        "  SafeTensors: min={st_min:.6}, max={st_max:.6}, mean={st_mean:.6}"
    );
    println!(
        "  MPK:         min={mpk_min:.6}, max={mpk_max:.6}, mean={mpk_mean:.6}"
    );

    // Compute max absolute difference
    let max_diff = st_vals
        .iter()
        .zip(mpk_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let mean_diff: f32 = st_vals
        .iter()
        .zip(mpk_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / st_vals.len() as f32;

    println!();
    println!("  Max absolute diff:  {max_diff:.10}");
    println!("  Mean absolute diff: {mean_diff:.10}");

    if max_diff < 1e-6 {
        println!("  ✅ Weights match perfectly (diff < 1e-6)");
    } else if max_diff < 1e-3 {
        println!("  ✅ Weights match closely (diff < 1e-3)");
    } else {
        println!("  ⚠️  Weights have significant differences");
    }

    println!();
    println!("=== Comparing tok_embeddings ===");

    let emb_st = model_st.tok_embeddings.weight.val();
    let emb_mpk = model_mpk.tok_embeddings.weight.val();

    let emb_st_data = emb_st.clone().into_data();
    let emb_mpk_data = emb_mpk.clone().into_data();

    let st_vals: Vec<f32> = emb_st_data.to_vec().unwrap();
    let mpk_vals: Vec<f32> = emb_mpk_data.to_vec().unwrap();

    println!("  Shape: {:?}", emb_st.dims());
    println!("  SafeTensors first 5: {:?}", &st_vals[..5]);
    println!("  MPK first 5:         {:?}", &mpk_vals[..5]);

    let max_diff = st_vals
        .iter()
        .zip(mpk_vals.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("  Max absolute diff:  {max_diff:.10}");

    if max_diff < 1e-6 {
        println!("  ✅ Weights match perfectly (diff < 1e-6)");
    } else if max_diff < 1e-3 {
        println!("  ✅ Weights match closely (diff < 1e-3)");
    } else {
        println!("  ⚠️  Weights have significant differences");
    }

    Ok(())
}
