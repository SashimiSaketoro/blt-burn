//! Fused CubeCL operations for BLT.
//!
//! This module provides custom CubeCL kernels that fuse multiple operations
//! into single GPU kernels, reducing kernel launch overhead and memory traffic
//! compared to calling separate operations.
//!
//! # Available Fused Operations
//!
//! | Operation | Description |
//! |-----------|-------------|
//! | `fused_entropy` | Entropy calculation (softmax + log + sum) |
//! | `fused_rms_norm` | RMS normalization with scale |
//! | `fused_softmax` | Softmax over last dimension |
//! | `fused_l2_norm` | L2 norm of embeddings |
//! | `fused_silu_gate` | SiLU activation with gating (SwiGLU FFN) |
//! | `fused_coherence` | Coherence score computation |
//!
//! # Architecture
//!
//! The `FusedOpsBackend` trait is implemented for:
//! - `CubeBackend<R, F, I, BT>` - Direct kernel execution (fastest)
//! - `Fusion<B>` (e.g., Wgpu) - Fallback to reference implementation
//!
//! # Usage
//!
//! When using `CubeBackend` directly, all operations use fused kernels.
//! When using `Wgpu` (Fusion backend), auto-fusion handles element-wise ops.
//!
//! ```ignore
//! use burn_wgpu::{WgpuRuntime, CubeBackend};
//! 
//! // For maximum performance with fused kernels:
//! type FastBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
//!
//! // Standard usage (auto-fusion for element-wise ops):
//! type Backend = burn::backend::wgpu::Wgpu;
//! ```

mod backend;
mod coherence_kernel;
mod kernel;
mod l2_norm_kernel;
mod rms_norm_kernel;
mod silu_gate_kernel;
mod softmax_kernel;

use burn::tensor::{backend::Backend, ops::FloatTensor, Tensor, TensorPrimitive};

/// Unified backend trait for all fused operations.
///
/// This trait provides fused versions of common operations used in BLT.
/// Implementing this trait allows backends to use optimized kernels.
pub trait FusedOpsBackend: Backend {
    /// Fused entropy calculation.
    /// Input: logits [batch, seq_len, vocab_size]
    /// Output: entropy [batch, seq_len]
    fn fused_entropy(logits: FloatTensor<Self>, vocab_size: usize) -> FloatTensor<Self>;

    /// Fused RMS normalization with scale weights.
    /// Input: x [batch, seq, dim], weight [dim]
    /// Output: normalized [batch, seq, dim]
    fn fused_rms_norm(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        epsilon: f32,
        dim: usize,
    ) -> FloatTensor<Self>;

    /// Fused softmax over the last dimension.
    /// Input: x [*, softmax_dim]
    /// Output: softmax(x) [*, softmax_dim]
    fn fused_softmax(input: FloatTensor<Self>, dim_size: usize) -> FloatTensor<Self>;

    /// Fused L2 norm over the last dimension.
    /// Input: x [batch, seq, dim]
    /// Output: norms [batch, seq]
    fn fused_l2_norm(input: FloatTensor<Self>, dim: usize) -> FloatTensor<Self>;

    /// Fused SiLU activation with gating (SwiGLU pattern).
    /// gate_input: [batch, seq, hidden] (w1 output)
    /// up_input: [batch, seq, hidden] (w3 output)
    /// Output: silu(gate_input) * up_input
    fn fused_silu_gate(
        gate_input: FloatTensor<Self>,
        up_input: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Fused coherence score calculation.
    /// norms: [batch, seq] - L2 norms
    /// entropies: [batch, seq] - Entropy values
    /// Output: norms^2 / (entropies + epsilon)
    fn fused_coherence(
        norms: FloatTensor<Self>,
        entropies: FloatTensor<Self>,
        epsilon: f32,
    ) -> FloatTensor<Self>;
}

// Keep the original trait for backward compatibility
pub use FusedOpsBackend as FusedEntropyBackend;

// ============================================================================
// Public API Functions
// ============================================================================

/// Compute entropy using the fused CubeCL kernel.
pub fn fused_entropy<B: FusedOpsBackend>(logits: Tensor<B, 3>, vocab_size: usize) -> Tensor<B, 2> {
    let output = B::fused_entropy(logits.into_primitive().tensor(), vocab_size);
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Compute RMS normalization using the fused kernel.
pub fn fused_rms_norm<B: FusedOpsBackend>(
    input: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    epsilon: f32,
) -> Tensor<B, 3> {
    let [_, _, dim] = input.dims();
    let output = B::fused_rms_norm(
        input.into_primitive().tensor(),
        weight.into_primitive().tensor(),
        epsilon,
        dim,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Compute softmax using the fused kernel.
pub fn fused_softmax<B: FusedOpsBackend, const D: usize>(
    input: Tensor<B, D>,
    dim_size: usize,
) -> Tensor<B, D> {
    let output = B::fused_softmax(input.into_primitive().tensor(), dim_size);
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Compute L2 norm using the fused kernel.
pub fn fused_l2_norm<B: FusedOpsBackend>(input: Tensor<B, 3>) -> Tensor<B, 2> {
    let [_, _, dim] = input.dims();
    let output = B::fused_l2_norm(input.into_primitive().tensor(), dim);
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Compute SiLU + gating (SwiGLU) using the fused kernel.
pub fn fused_silu_gate<B: FusedOpsBackend>(
    gate_input: Tensor<B, 3>,
    up_input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let output = B::fused_silu_gate(
        gate_input.into_primitive().tensor(),
        up_input.into_primitive().tensor(),
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Compute coherence scores using the fused kernel.
pub fn fused_coherence<B: FusedOpsBackend>(
    norms: Tensor<B, 2>,
    entropies: Tensor<B, 2>,
    epsilon: f32,
) -> Tensor<B, 2> {
    let output = B::fused_coherence(
        norms.into_primitive().tensor(),
        entropies.into_primitive().tensor(),
        epsilon,
    );
    Tensor::from_primitive(TensorPrimitive::Float(output))
}
