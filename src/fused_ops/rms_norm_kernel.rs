//! Fused RMS Normalization kernel.
//!
//! RMS Norm: x / sqrt(mean(x^2) + eps) * weight
//!
//! Two implementations:
//! - Simple: one thread per row
//! - Optimized: plane_sum for parallel reduction

use cubecl::{cube, prelude::*};

/// Simple fused RMS Norm kernel - one thread handles one (batch, seq) position.
#[cube(launch)]
pub fn fused_rms_norm_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<F>,
    output: &mut Tensor<F>,
    dim: u32,
) {
    let batch_idx = ABSOLUTE_POS_X;
    let seq_idx = ABSOLUTE_POS_Y;

    let batch_size = output.shape(0);
    let seq_len = output.shape(1);

    if batch_idx >= batch_size || seq_idx >= seq_len {
        terminate!();
    }

    let row_start = (batch_idx * seq_len + seq_idx) * dim;

    // Pass 1: Compute mean of squares
    let mut sum_sq = F::new(0.0);
    for i in 0..dim {
        let val = input[row_start + i];
        sum_sq = sum_sq + val * val;
    }
    let dim_float = F::cast_from(dim);
    let mean_sq = sum_sq / dim_float;

    // Compute RMS (with epsilon for stability)
    let eps = F::new(1e-6);
    let rms = F::sqrt(mean_sq + eps);
    let inv_rms = F::new(1.0) / rms;

    // Pass 2: Normalize and scale
    for i in 0..dim {
        let val = input[row_start + i];
        let w = weight[i];
        output[row_start + i] = val * inv_rms * w;
    }
}

/// Optimized RMS Norm kernel using plane_sum for parallel reduction.
/// Each plane cooperates to normalize one row.
#[cube(launch)]
pub fn fused_rms_norm_kernel_optimized<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<F>,
    output: &mut Tensor<F>,
    dim: u32,
) {
    // Each plane handles one row
    let row_idx = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let lane_idx = UNIT_POS_X;

    let batch_size = output.shape(0);
    let seq_len = output.shape(1);
    let total_rows = batch_size * seq_len;

    if row_idx >= total_rows {
        terminate!();
    }

    let row_start = row_idx * dim;
    let plane_size = CUBE_DIM_X;

    // Pass 1: Compute sum of squares using parallel reduction
    let mut local_sum_sq = F::new(0.0);
    let mut i = lane_idx;
    while i < dim {
        let val = input[row_start + i];
        local_sum_sq = local_sum_sq + val * val;
        i += plane_size;
    }
    let sum_sq = plane_sum(local_sum_sq);

    // Compute RMS
    let dim_float = F::cast_from(dim);
    let mean_sq = sum_sq / dim_float;
    let eps = F::new(1e-6);
    let rms = F::sqrt(mean_sq + eps);
    let inv_rms = F::new(1.0) / rms;

    // Pass 2: Normalize and scale (each lane handles its elements)
    let mut i = lane_idx;
    while i < dim {
        let val = input[row_start + i];
        let w = weight[i];
        output[row_start + i] = val * inv_rms * w;
        i += plane_size;
    }
}
