//! Model loading utilities for BLT entropy model.
//!
//! This module provides convenience functions for loading the BLT entropy model
//! from various sources (HuggingFace cache, local paths, etc.).
//!
//! # Example
//!
//! ```rust,ignore
//! use blt_burn::model_loader::load_entropy_model;
//! use blt_burn::quantization::QuantConfig;
//!
//! let device = WgpuDevice::default();
//!
//! // Load entropy model for patching
//! let entropy_model = load_entropy_model(&device, None, QuantConfig::Int8)?;
//! ```

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;

use crate::model::{LMTransformer, LMTransformerConfig};
use crate::quantization::{quantize_model, QuantConfig};

/// Default configuration for BLT entropy model (768-dim, 14 layers).
pub fn entropy_model_config() -> LMTransformerConfig {
    LMTransformerConfig {
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
    }
}

/// Resolve the HuggingFace cache directory.
pub fn hf_cache_dir() -> PathBuf {
    let hf_cache = std::env::var("HF_HOME")
        .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
        .unwrap_or_else(|_| {
            format!(
                "{}/.cache/huggingface",
                std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
            )
        });
    PathBuf::from(hf_cache)
}

/// Find a model in the HuggingFace cache by its model ID.
///
/// # Arguments
/// * `model_id` - Model identifier like "facebook/blt-entropy"
///
/// # Returns
/// Path to the snapshot directory if found
pub fn find_hf_model(model_id: &str) -> Option<PathBuf> {
    let cache_dir = hf_cache_dir();
    let model_dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots_dir = cache_dir
        .join("hub")
        .join(&model_dir_name)
        .join("snapshots");

    if !snapshots_dir.exists() {
        return None;
    }

    // Return the first snapshot directory found
    std::fs::read_dir(&snapshots_dir).ok().and_then(|entries| {
        entries
            .filter_map(|e| e.ok())
            .find(|e| e.path().is_dir())
            .map(|e| e.path())
    })
}

/// Find the BLT entropy model in HuggingFace cache.
pub fn find_entropy_model() -> Option<PathBuf> {
    find_hf_model("facebook/blt-entropy")
        .map(|p| p.join("model.safetensors"))
        .filter(|p| p.exists())
}

/// Load the BLT entropy model for patch boundary detection.
///
/// # Arguments
/// * `device` - WGPU device
/// * `model_path` - Optional explicit path; if None, searches HF cache
/// * `quant_config` - Quantization configuration
///
/// # Returns
/// Loaded and optionally quantized entropy model
pub fn load_entropy_model(
    device: &WgpuDevice,
    model_path: Option<PathBuf>,
    quant_config: QuantConfig,
) -> Result<LMTransformer<Wgpu>> {
    let config = entropy_model_config();
    let model = config.init::<Wgpu>(device);

    // Resolve model path
    let path = model_path.or_else(find_entropy_model).ok_or_else(|| {
        anyhow!(
            "BLT entropy model not found. Download with: \
             hf download facebook/blt-entropy"
        )
    })?;

    // Determine format from extension
    let use_mpk = path.extension().is_some_and(|e| e == "mpk");

    let model =
        if use_mpk {
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
            model.load_record(
                recorder
                    .load(path.clone(), device)
                    .map_err(|e| anyhow!("Failed to load MPK weights from {path:?}: {e}"))?,
            )
        } else {
            let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
            model.load_record(recorder.load(path.clone().into(), device).map_err(|e| {
                anyhow!("Failed to load SafeTensors weights from {path:?}: {e}")
            })?)
        };

    // Apply quantization if requested
    let model = if quant_config.is_enabled() {
        quantize_model(model, quant_config)
    } else {
        model
    };

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_cache_resolution() {
        let cache = hf_cache_dir();
        // Should return a valid path (may or may not exist)
        assert!(!cache.to_string_lossy().is_empty());
    }

    #[test]
    fn test_entropy_config() {
        let config = entropy_model_config();
        assert_eq!(config.dim, 768);
        assert_eq!(config.n_layers, 14);
        assert_eq!(config.vocab_size, 260);
    }
}
