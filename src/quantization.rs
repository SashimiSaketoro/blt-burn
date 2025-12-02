//! Model quantization utilities for BLT-Burn
//!
//! Provides comprehensive quantization options using Burn's quantization API.
//! This reduces memory usage and can improve inference speed on supported hardware.
//!
//! # Quantization Levels
//! - **none**: Keep original BF16 precision (~9GB for BLT-1B)
//! - **int8**: INT8 per-tensor (~4.5GB, minimal accuracy loss)
//! - **int8-bf16**: INT8 with BF16 scale parameters (better accuracy)
//! - **int4**: INT4 block-32 (~2.3GB, some accuracy loss)
//! - **int4-64**: INT4 block-64 (better accuracy than block-32)
//! - **int2**: INT2 block-32 (~1.1GB, experimental)
//! - **fp8-e5m2**: FP8 E5M2 format (good for inference)
//! - **fp8-e4m3**: FP8 E4M3 format (better dynamic range)
//!
//! # Usage
//! ```bash
//! # Default: BF16 (no quantization)
//! cargo run --release --bin ingest -- --output-dir /tmp/out
//!
//! # INT8 per-tensor with BF16 scale params (recommended)
//! cargo run --release --bin ingest -- --output-dir /tmp/out --quantize int8-bf16
//!
//! # INT4 block-wise quantization (aggressive compression)
//! cargo run --release --bin ingest -- --output-dir /tmp/out --quantize int4
//!
//! # FP8 E5M2 (8-bit floating point)
//! cargo run --release --bin ingest -- --output-dir /tmp/out --quantize fp8
//! ```

use burn::module::{Module, Quantizer};
use burn::tensor::backend::Backend;
use burn::tensor::quantization::{
    BlockSize, Calibration, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantValue,
};
use std::fmt;

/// Quantization configuration options
///
/// These map to Burn's QuantScheme with various combinations of:
/// - QuantValue: Q8S, Q4F, Q2F, E5M2, E4M3
/// - QuantLevel: Tensor, Block(size)
/// - QuantParam: F32, F16, BF16 (precision for scale/zero-point)
/// - QuantMode: Symmetric (default)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantConfig {
    /// No quantization (original BF16 precision)
    None,

    /// INT8 per-tensor symmetric quantization with F32 scale
    /// Good balance of speed and accuracy, ~2x compression
    Int8PerTensor,

    /// INT8 per-tensor with BF16 scale parameters
    /// Better preserves BF16 model characteristics
    Int8BF16Params,

    /// INT8 per-block(32) for finer granularity
    Int8Block32,

    /// INT4 with block size 32, F16 scale
    /// Aggressive compression (~4x), some accuracy loss
    Int4Block32,

    /// INT4 with block size 64
    /// Balance between compression and accuracy
    Int4Block64,

    /// INT2 with block size 32 (experimental)
    /// Maximum compression (~8x), significant accuracy loss
    Int2Block32,

    /// FP8 E5M2 format (5-bit exponent, 2-bit mantissa)
    /// Good for inference, preserves dynamic range
    Fp8E5M2,

    /// FP8 E4M3 format (4-bit exponent, 3-bit mantissa)  
    /// Better precision than E5M2, smaller range
    Fp8E4M3,
}

impl std::str::FromStr for QuantConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" | "bf16" => Ok(Self::None),
            "int8" | "q8" => Ok(Self::Int8PerTensor),
            "int8-bf16" | "q8-bf16" => Ok(Self::Int8BF16Params),
            "int8-block" | "q8-block" => Ok(Self::Int8Block32),
            "int4" | "q4" => Ok(Self::Int4Block32),
            "int4-64" | "q4-64" => Ok(Self::Int4Block64),
            "int2" | "q2" => Ok(Self::Int2Block32),
            "fp8" | "fp8-e5m2" | "e5m2" => Ok(Self::Fp8E5M2),
            "fp8-e4m3" | "e4m3" => Ok(Self::Fp8E4M3),
            _ => Err(format!(
                "Invalid quantization mode '{s}'. Valid options:\n\
                 - none, bf16: Original BF16 precision\n\
                 - int8, q8: INT8 per-tensor\n\
                 - int8-bf16: INT8 with BF16 scale params (recommended)\n\
                 - int8-block: INT8 per-block(32)\n\
                 - int4, q4: INT4 block-32\n\
                 - int4-64: INT4 block-64\n\
                 - int2, q2: INT2 block-32 (experimental)\n\
                 - fp8, e5m2: FP8 E5M2 format\n\
                 - e4m3: FP8 E4M3 format"
            )),
        }
    }
}

impl QuantConfig {
    /// Parse with default fallback (for CLI where unknown = None)
    pub fn parse_or_default(s: &str) -> Self {
        s.parse().unwrap_or(Self::None)
    }

    /// Get bits per weight for this config
    pub fn bits_per_weight(&self) -> usize {
        match self {
            Self::None => 16, // BF16
            Self::Int8PerTensor
            | Self::Int8BF16Params
            | Self::Int8Block32
            | Self::Fp8E5M2
            | Self::Fp8E4M3 => 8,
            Self::Int4Block32 | Self::Int4Block64 => 4,
            Self::Int2Block32 => 2,
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

            // INT8 per-tensor with F32 scale (default)
            Self::Int8PerTensor => Some(
                QuantScheme::default()
                    .with_mode(QuantMode::Symmetric)
                    .with_level(QuantLevel::Tensor)
                    .with_value(QuantValue::Q8S)
                    .with_param(QuantParam::F32),
            ),

            // INT8 per-tensor with F16 scale (good for BF16 models, WGPU compatible)
            // Note: WGPU doesn't support BF16 natively, so we use F16 for scale params
            Self::Int8BF16Params => Some(
                QuantScheme::default()
                    .with_mode(QuantMode::Symmetric)
                    .with_level(QuantLevel::Tensor)
                    .with_value(QuantValue::Q8S)
                    .with_param(QuantParam::F16),
            ),

            // INT8 per-block(32) with F16 scale
            Self::Int8Block32 => Some(
                QuantScheme::default()
                    .with_mode(QuantMode::Symmetric)
                    .with_level(QuantLevel::Block(BlockSize::new([32])))
                    .with_value(QuantValue::Q8S)
                    .with_param(QuantParam::F16),
            ),

            // INT4 block-32 with F16 scale
            Self::Int4Block32 => Some(
                QuantScheme::default()
                    .with_mode(QuantMode::Symmetric)
                    .with_level(QuantLevel::Block(BlockSize::new([32])))
                    .with_value(QuantValue::Q4F)
                    .with_param(QuantParam::F16),
            ),

            // INT4 block-64 with F16 scale
            Self::Int4Block64 => Some(
                QuantScheme::default()
                    .with_mode(QuantMode::Symmetric)
                    .with_level(QuantLevel::Block(BlockSize::new([64])))
                    .with_value(QuantValue::Q4F)
                    .with_param(QuantParam::F16),
            ),

            // INT2 block-32 (experimental)
            Self::Int2Block32 => Some(
                QuantScheme::default()
                    .with_mode(QuantMode::Symmetric)
                    .with_level(QuantLevel::Block(BlockSize::new([32])))
                    .with_value(QuantValue::Q2F)
                    .with_param(QuantParam::F16),
            ),

            // FP8 E5M2 per-tensor (good dynamic range)
            Self::Fp8E5M2 => Some(
                QuantScheme::default()
                    .with_mode(QuantMode::Symmetric)
                    .with_level(QuantLevel::Tensor)
                    .with_value(QuantValue::E5M2)
                    .with_param(QuantParam::F32),
            ),

            // FP8 E4M3 per-tensor (better precision)
            Self::Fp8E4M3 => Some(
                QuantScheme::default()
                    .with_mode(QuantMode::Symmetric)
                    .with_level(QuantLevel::Tensor)
                    .with_value(QuantValue::E4M3)
                    .with_param(QuantParam::F32),
            ),
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
            Self::Int8PerTensor => write!(f, "INT8 per-tensor (F32 scale)"),
            Self::Int8BF16Params => write!(f, "INT8 per-tensor (F16 scale)"),
            Self::Int8Block32 => write!(f, "INT8 block-32 (F16 scale)"),
            Self::Int4Block32 => write!(f, "INT4 block-32"),
            Self::Int4Block64 => write!(f, "INT4 block-64"),
            Self::Int2Block32 => write!(f, "INT2 block-32 (experimental)"),
            Self::Fp8E5M2 => write!(f, "FP8 E5M2"),
            Self::Fp8E4M3 => write!(f, "FP8 E4M3"),
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

        let original_bytes = num_params * 2; // BF16 = 2 bytes
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
        println!(
            "║ Original:    {:>20} MB ║",
            self.original_bytes / 1_000_000
        );
        println!(
            "║ Quantized:   {:>20} MB ║",
            self.quantized_bytes / 1_000_000
        );
        println!("║ Compression: {:>20.1}x ║", self.compression_ratio);
        println!("╚══════════════════════════════════════════╝");
    }
}

// =============================================================================
// Implementation Status (Burn 0.19.1)
// =============================================================================
// Quantization is now fully implemented using Burn's stable quantization API.
// The `quantize_model` function above uses:
//   - Quantizer with MinMax calibration
//   - Per-tensor symmetric INT8 (Q8S) for Int8PerTensor
//   - Block-wise INT4 (Q4F) with configurable block sizes
//
// Supported modes:
//   - Int8PerTensor: ~2x compression, minimal accuracy loss
//   - Int4Block32/64: ~4x compression, some accuracy loss
//
// Usage: Apply to any Burn Module after loading weights:
//   let quantized_model = quantize_model(model, QuantConfig::Int8PerTensor);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_config_parsing() {
        // Basic options - uses FromStr trait via .parse()
        assert_eq!("none".parse::<QuantConfig>().unwrap(), QuantConfig::None);
        assert_eq!("bf16".parse::<QuantConfig>().unwrap(), QuantConfig::None);
        assert_eq!("int8".parse::<QuantConfig>().unwrap(), QuantConfig::Int8PerTensor);
        assert_eq!("INT8".parse::<QuantConfig>().unwrap(), QuantConfig::Int8PerTensor);
        assert_eq!("q8".parse::<QuantConfig>().unwrap(), QuantConfig::Int8PerTensor);

        // BF16 params
        assert_eq!(
            "int8-bf16".parse::<QuantConfig>().unwrap(),
            QuantConfig::Int8BF16Params
        );
        assert_eq!(
            "q8-bf16".parse::<QuantConfig>().unwrap(),
            QuantConfig::Int8BF16Params
        );

        // Block quantization
        assert_eq!(
            "int8-block".parse::<QuantConfig>().unwrap(),
            QuantConfig::Int8Block32
        );
        assert_eq!("int4".parse::<QuantConfig>().unwrap(), QuantConfig::Int4Block32);
        assert_eq!("q4-64".parse::<QuantConfig>().unwrap(), QuantConfig::Int4Block64);
        assert_eq!("int2".parse::<QuantConfig>().unwrap(), QuantConfig::Int2Block32);

        // FP8 formats
        assert_eq!("fp8".parse::<QuantConfig>().unwrap(), QuantConfig::Fp8E5M2);
        assert_eq!("e5m2".parse::<QuantConfig>().unwrap(), QuantConfig::Fp8E5M2);
        assert_eq!("e4m3".parse::<QuantConfig>().unwrap(), QuantConfig::Fp8E4M3);

        // Test error case
        assert!("invalid".parse::<QuantConfig>().is_err());

        // Test parse_or_default
        assert_eq!(QuantConfig::parse_or_default("invalid"), QuantConfig::None);
    }

    #[test]
    fn test_bits_per_weight() {
        assert_eq!(QuantConfig::None.bits_per_weight(), 16);
        assert_eq!(QuantConfig::Int8PerTensor.bits_per_weight(), 8);
        assert_eq!(QuantConfig::Int8BF16Params.bits_per_weight(), 8);
        assert_eq!(QuantConfig::Fp8E5M2.bits_per_weight(), 8);
        assert_eq!(QuantConfig::Int4Block32.bits_per_weight(), 4);
        assert_eq!(QuantConfig::Int2Block32.bits_per_weight(), 2);
    }

    #[test]
    fn test_compression_ratio() {
        let stats = QuantStats::for_blt_model(QuantConfig::Int8PerTensor);
        assert!((stats.compression_ratio - 2.0).abs() < 0.01);

        let stats = QuantStats::for_blt_model(QuantConfig::Int4Block32);
        assert!((stats.compression_ratio - 4.0).abs() < 0.01);

        let stats = QuantStats::for_blt_model(QuantConfig::Int2Block32);
        assert!((stats.compression_ratio - 8.0).abs() < 0.01);
    }
}
