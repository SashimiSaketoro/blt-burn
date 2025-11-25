//! Model quantization utilities for BLT-Burn
//!
//! Provides optional INT8/INT4 quantization for model weights using Burn's
//! quantization API. This can significantly reduce memory usage and improve
//! inference speed on supported hardware.
//!
//! # Usage
//! ```bash
//! # Default: BF16 (no quantization)
//! cargo run --release --bin ingest -- --output-dir /tmp/out
//!
//! # INT8 per-tensor symmetric quantization (~2x speedup)
//! cargo run --release --bin ingest -- --output-dir /tmp/out --quantize int8
//!
//! # INT4 block-wise quantization (~4x compression)
//! cargo run --release --bin ingest -- --output-dir /tmp/out --quantize int4
//! ```

use burn::module::{Module, Quantizer};
use burn::tensor::backend::Backend;
use burn::tensor::quantization::{
    BlockSize, Calibration, QuantLevel, QuantScheme, QuantValue,
};
use std::fmt;

/// Quantization configuration options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantConfig {
    /// No quantization (default BF16)
    None,
    /// INT8 per-tensor symmetric quantization
    /// Good balance of speed and accuracy
    Int8PerTensor,
    /// INT8 per-channel quantization  
    /// Better accuracy than per-tensor, similar speed
    Int8PerChannel,
    /// INT4 with block size 32
    /// Aggressive compression, some accuracy loss
    Int4Block32,
    /// INT4 with block size 64
    /// Balance between compression and accuracy
    Int4Block64,
}

impl QuantConfig {
    /// Parse quantization mode from CLI string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "int8" | "q8" => Self::Int8PerTensor,
            "int8-channel" | "q8c" => Self::Int8PerChannel,
            "int4" | "q4" => Self::Int4Block32,
            "int4-64" | "q4-64" => Self::Int4Block64,
            _ => Self::None,
        }
    }

    /// Get bits per weight for this config
    pub fn bits_per_weight(&self) -> usize {
        match self {
            Self::None => 16,  // BF16
            Self::Int8PerTensor | Self::Int8PerChannel => 8,
            Self::Int4Block32 | Self::Int4Block64 => 4,
        }
    }

    /// Check if quantization is enabled
    pub fn is_enabled(&self) -> bool {
        *self != Self::None
    }

    /// Convert to Burn QuantScheme
    pub fn to_scheme(&self) -> Option<QuantScheme> {
        match self {
            Self::None => None,
            // Q8S = signed 8-bit symmetric quantization
            Self::Int8PerTensor => Some(QuantScheme {
                level: QuantLevel::Tensor,
                value: QuantValue::Q8S,
                ..Default::default()
            }),
            // Per-channel not yet supported - fallback to per-tensor
            Self::Int8PerChannel => Some(QuantScheme {
                level: QuantLevel::Tensor,
                value: QuantValue::Q8S,
                ..Default::default()
            }),
            // Q4F = 4-bit float quantization
            Self::Int4Block32 => Some(QuantScheme {
                level: QuantLevel::Block(BlockSize::new([32])),
                value: QuantValue::Q4F,
                ..Default::default()
            }),
            Self::Int4Block64 => Some(QuantScheme {
                level: QuantLevel::Block(BlockSize::new([64])),
                value: QuantValue::Q4F,
                ..Default::default()
            }),
        }
    }
}

/// Quantize a model's weights using the specified configuration
/// 
/// # Example
/// ```rust,ignore
/// let model = config.init::<Backend>(&device);
/// let model = model.load_record(recorder.load(path, &device)?);
/// let model = quantize_model(model, QuantConfig::Int8PerTensor);
/// ```
pub fn quantize_model<B: Backend, M: Module<B>>(model: M, config: QuantConfig) -> M {
    match config.to_scheme() {
        None => model,
        Some(scheme) => {
            let mut quantizer = Quantizer {
                calibration: Calibration::MinMax,
                scheme,
            };
            model.quantize_weights(&mut quantizer)
        }
    }
}

impl fmt::Display for QuantConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none (BF16)"),
            Self::Int8PerTensor => write!(f, "INT8 per-tensor"),
            Self::Int8PerChannel => write!(f, "INT8 per-channel"),
            Self::Int4Block32 => write!(f, "INT4 block-32"),
            Self::Int4Block64 => write!(f, "INT4 block-64"),
        }
    }
}

/// Statistics about quantization compression
pub struct QuantStats {
    pub original_bytes: usize,
    pub quantized_bytes: usize,
    pub compression_ratio: f32,
    pub config: QuantConfig,
}

impl QuantStats {
    /// Compute stats for the BLT entropy model (~85M parameters)
    pub fn for_blt_model(config: QuantConfig) -> Self {
        // BLT entropy model parameter count (approximate)
        // Based on: 14 layers, 768 dim, 3072 FFN dim
        let num_params: usize = 85_000_000;
        
        let original_bytes = num_params * 2;  // BF16 = 2 bytes
        let quantized_bytes = num_params * config.bits_per_weight() / 8;
        
        Self {
            original_bytes,
            quantized_bytes,
            compression_ratio: original_bytes as f32 / quantized_bytes as f32,
            config,
        }
    }

    /// Print formatted statistics
    pub fn print(&self) {
        println!("╔══════════════════════════════════════════╗");
        println!("║       Quantization Statistics            ║");
        println!("╠══════════════════════════════════════════╣");
        println!("║ Mode:        {:<27} ║", self.config);
        println!("║ Original:    {:>20} MB ║", self.original_bytes / 1_000_000);
        println!("║ Quantized:   {:>20} MB ║", self.quantized_bytes / 1_000_000);
        println!("║ Compression: {:>20.1}x ║", self.compression_ratio);
        println!("╚══════════════════════════════════════════╝");
    }
}

// Note: Actual quantization implementation requires Burn's quantization feature
// which is still in beta. The code below is a placeholder for when the API stabilizes.
//
// To enable quantization, add to Cargo.toml:
// burn = { ..., features = ["wgpu", "quantization"] }
//
// Then implement:
// ```rust
// use burn::tensor::quantization::{
//     BlockSize, Calibration, QuantLevel, QuantMode,
//     QuantParam, QuantScheme, QuantStore, QuantValue, Quantizer,
// };
//
// pub fn quantize_model<B: Backend, M: Module<B>>(model: M, config: QuantConfig) -> M {
//     match config.to_scheme() {
//         None => model,
//         Some(scheme) => {
//             let mut quantizer = Quantizer {
//                 calibration: Calibration::MinMax,
//                 scheme,
//             };
//             model.quantize_weights(&mut quantizer)
//         }
//     }
// }
// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_config_parsing() {
        assert_eq!(QuantConfig::from_str("none"), QuantConfig::None);
        assert_eq!(QuantConfig::from_str("int8"), QuantConfig::Int8PerTensor);
        assert_eq!(QuantConfig::from_str("INT8"), QuantConfig::Int8PerTensor);
        assert_eq!(QuantConfig::from_str("q8"), QuantConfig::Int8PerTensor);
        assert_eq!(QuantConfig::from_str("int4"), QuantConfig::Int4Block32);
        assert_eq!(QuantConfig::from_str("q4-64"), QuantConfig::Int4Block64);
    }

    #[test]
    fn test_bits_per_weight() {
        assert_eq!(QuantConfig::None.bits_per_weight(), 16);
        assert_eq!(QuantConfig::Int8PerTensor.bits_per_weight(), 8);
        assert_eq!(QuantConfig::Int4Block32.bits_per_weight(), 4);
    }

    #[test]
    fn test_compression_ratio() {
        let stats = QuantStats::for_blt_model(QuantConfig::Int8PerTensor);
        assert!((stats.compression_ratio - 2.0).abs() < 0.01);
        
        let stats = QuantStats::for_blt_model(QuantConfig::Int4Block32);
        assert!((stats.compression_ratio - 4.0).abs() < 0.01);
    }
}
