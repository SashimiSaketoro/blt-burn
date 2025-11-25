//! Tests for all fused operations.
//!
//! These tests validate that fused CubeCL kernels produce numerically
//! equivalent results to reference implementations.
//!
//! Run with: `cargo test --features fused-entropy test_fused_ops`

#![cfg(feature = "fused-entropy")]

use burn::tensor::{activation::silu, backend::Backend, Distribution, Tensor};
use burn_wgpu::{Wgpu, WgpuDevice};

type TestBackend = Wgpu;

// ============================================================================
// RMS Norm Tests
// ============================================================================

/// Reference RMS norm implementation
fn rms_norm_reference<B: Backend>(x: Tensor<B, 3>, weight: Tensor<B, 1>, epsilon: f32) -> Tensor<B, 3> {
    let [_, _, dim] = x.dims();
    let squared = x.clone() * x.clone();
    let norm = squared.mean_dim(2).sqrt().add_scalar(epsilon as f64);
    let weight = weight.reshape([1, 1, dim]);
    x.div(norm) * weight
}

#[test]
fn test_fused_rms_norm_matches_reference() {
    let device = WgpuDevice::default();

    let x = Tensor::<TestBackend, 3>::random([2, 8, 64], Distribution::Normal(0.0, 1.0), &device);
    let weight = Tensor::<TestBackend, 1>::random([64], Distribution::Uniform(0.5, 1.5), &device);

    let reference = rms_norm_reference(x.clone(), weight.clone(), 1e-6);
    let fused = blt_burn::fused_ops::fused_rms_norm(x, weight, 1e-6);

    let ref_data = reference.into_data();
    let fused_data = fused.into_data();

    let ref_values: Vec<f32> = ref_data.as_slice().unwrap().to_vec();
    let fused_values: Vec<f32> = fused_data.as_slice().unwrap().to_vec();

    let mut max_diff = 0.0f32;
    for (r, f) in ref_values.iter().zip(fused_values.iter()) {
        max_diff = max_diff.max((r - f).abs());
    }

    assert!(
        max_diff < 1e-3,
        "RMS norm max diff {} exceeds tolerance",
        max_diff
    );
}

// ============================================================================
// L2 Norm Tests
// ============================================================================

/// Reference L2 norm implementation
fn l2_norm_reference<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 2> {
    let [bs, seq, _dim] = x.dims();
    let squared = x.clone() * x;
    squared.sum_dim(2).sqrt().reshape([bs, seq])
}

#[test]
fn test_fused_l2_norm_matches_reference() {
    let device = WgpuDevice::default();

    let x = Tensor::<TestBackend, 3>::random([2, 16, 128], Distribution::Normal(0.0, 1.0), &device);

    let reference = l2_norm_reference(x.clone());
    let fused = blt_burn::fused_ops::fused_l2_norm(x);

    let ref_data = reference.into_data();
    let fused_data = fused.into_data();

    let ref_values: Vec<f32> = ref_data.as_slice().unwrap().to_vec();
    let fused_values: Vec<f32> = fused_data.as_slice().unwrap().to_vec();

    let mut max_diff = 0.0f32;
    for (r, f) in ref_values.iter().zip(fused_values.iter()) {
        max_diff = max_diff.max((r - f).abs());
    }

    assert!(
        max_diff < 1e-4,
        "L2 norm max diff {} exceeds tolerance",
        max_diff
    );
}

// ============================================================================
// SiLU Gate Tests
// ============================================================================

/// Reference SiLU gate implementation (SwiGLU pattern)
fn silu_gate_reference<B: Backend>(gate: Tensor<B, 3>, up: Tensor<B, 3>) -> Tensor<B, 3> {
    silu(gate) * up
}

#[test]
fn test_fused_silu_gate_matches_reference() {
    let device = WgpuDevice::default();

    let gate = Tensor::<TestBackend, 3>::random([2, 8, 256], Distribution::Normal(0.0, 1.0), &device);
    let up = Tensor::<TestBackend, 3>::random([2, 8, 256], Distribution::Normal(0.0, 1.0), &device);

    let reference = silu_gate_reference(gate.clone(), up.clone());
    let fused = blt_burn::fused_ops::fused_silu_gate(gate, up);

    let ref_data = reference.into_data();
    let fused_data = fused.into_data();

    let ref_values: Vec<f32> = ref_data.as_slice().unwrap().to_vec();
    let fused_values: Vec<f32> = fused_data.as_slice().unwrap().to_vec();

    let mut max_diff = 0.0f32;
    for (r, f) in ref_values.iter().zip(fused_values.iter()) {
        max_diff = max_diff.max((r - f).abs());
    }

    assert!(
        max_diff < 1e-4,
        "SiLU gate max diff {} exceeds tolerance",
        max_diff
    );
}

// ============================================================================
// Coherence Tests
// ============================================================================

/// Reference coherence implementation
fn coherence_reference<B: Backend>(norms: Tensor<B, 2>, entropies: Tensor<B, 2>, epsilon: f32) -> Tensor<B, 2> {
    norms.clone().powf_scalar(2.0) / (entropies + epsilon as f64)
}

#[test]
fn test_fused_coherence_matches_reference() {
    let device = WgpuDevice::default();

    let norms = Tensor::<TestBackend, 2>::random([4, 32], Distribution::Uniform(0.1, 10.0), &device);
    let entropies = Tensor::<TestBackend, 2>::random([4, 32], Distribution::Uniform(0.1, 5.0), &device);

    let reference = coherence_reference(norms.clone(), entropies.clone(), 1e-6);
    let fused = blt_burn::fused_ops::fused_coherence(norms, entropies, 1e-6);

    let ref_data = reference.into_data();
    let fused_data = fused.into_data();

    let ref_values: Vec<f32> = ref_data.as_slice().unwrap().to_vec();
    let fused_values: Vec<f32> = fused_data.as_slice().unwrap().to_vec();

    let mut max_diff = 0.0f32;
    for (r, f) in ref_values.iter().zip(fused_values.iter()) {
        max_diff = max_diff.max((r - f).abs());
    }

    assert!(
        max_diff < 1e-4,
        "Coherence max diff {} exceeds tolerance",
        max_diff
    );
}

// ============================================================================
// Softmax Tests
// ============================================================================

/// Reference softmax implementation
fn softmax_reference<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    burn::tensor::activation::softmax(x, dim)
}

#[test]
fn test_fused_softmax_3d() {
    let device = WgpuDevice::default();

    let x = Tensor::<TestBackend, 3>::random([2, 8, 64], Distribution::Normal(0.0, 1.0), &device);

    let reference = softmax_reference(x.clone(), 2);
    let fused = blt_burn::fused_ops::fused_softmax(x, 64);

    let ref_data = reference.into_data();
    let fused_data = fused.into_data();

    let ref_values: Vec<f32> = ref_data.as_slice().unwrap().to_vec();
    let fused_values: Vec<f32> = fused_data.as_slice().unwrap().to_vec();

    let mut max_diff = 0.0f32;
    for (r, f) in ref_values.iter().zip(fused_values.iter()) {
        max_diff = max_diff.max((r - f).abs());
    }

    assert!(
        max_diff < 1e-4,
        "Softmax 3D max diff {} exceeds tolerance",
        max_diff
    );
}

#[test]
fn test_fused_softmax_attention_dims() {
    let device = WgpuDevice::default();

    // Attention scores: [batch, heads, seq_q, seq_k]
    let x = Tensor::<TestBackend, 4>::random([2, 8, 32, 32], Distribution::Normal(0.0, 1.0), &device);

    let reference = softmax_reference(x.clone(), 3);
    let fused = blt_burn::fused_ops::fused_softmax(x, 32);

    let ref_data = reference.into_data();
    let fused_data = fused.into_data();

    let ref_values: Vec<f32> = ref_data.as_slice().unwrap().to_vec();
    let fused_values: Vec<f32> = fused_data.as_slice().unwrap().to_vec();

    let mut max_diff = 0.0f32;
    for (r, f) in ref_values.iter().zip(fused_values.iter()) {
        max_diff = max_diff.max((r - f).abs());
    }

    assert!(
        max_diff < 1e-4,
        "Softmax attention max diff {} exceeds tolerance",
        max_diff
    );
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_fused_ops_batch_size_one() {
    let device = WgpuDevice::default();

    // Test all ops with batch size 1
    let x = Tensor::<TestBackend, 3>::random([1, 16, 64], Distribution::Normal(0.0, 1.0), &device);
    let weight = Tensor::<TestBackend, 1>::ones([64], &device);

    let _ = blt_burn::fused_ops::fused_rms_norm(x.clone(), weight, 1e-6);
    let _ = blt_burn::fused_ops::fused_l2_norm(x.clone());
    let _ = blt_burn::fused_ops::fused_softmax(x.clone(), 64);

    // SiLU gate
    let gate = Tensor::<TestBackend, 3>::random([1, 16, 128], Distribution::Normal(0.0, 1.0), &device);
    let up = Tensor::<TestBackend, 3>::random([1, 16, 128], Distribution::Normal(0.0, 1.0), &device);
    let _ = blt_burn::fused_ops::fused_silu_gate(gate, up);

    // Coherence
    let norms = Tensor::<TestBackend, 2>::random([1, 16], Distribution::Uniform(0.1, 10.0), &device);
    let entropies = Tensor::<TestBackend, 2>::random([1, 16], Distribution::Uniform(0.1, 5.0), &device);
    let _ = blt_burn::fused_ops::fused_coherence(norms, entropies, 1e-6);
}

#[test]
fn test_fused_ops_numerical_stability() {
    let device = WgpuDevice::default();

    // Test with extreme values
    let x = Tensor::<TestBackend, 3>::random([2, 8, 64], Distribution::Normal(50.0, 20.0), &device);

    let softmax_result = blt_burn::fused_ops::fused_softmax(x.clone(), 64);
    let softmax_data = softmax_result.into_data();
    let softmax_values: Vec<f32> = softmax_data.as_slice().unwrap().to_vec();

    for val in softmax_values.iter() {
        assert!(val.is_finite(), "Softmax produced non-finite value");
        assert!(*val >= 0.0 && *val <= 1.0, "Softmax value {} out of [0,1] range", val);
    }
}

