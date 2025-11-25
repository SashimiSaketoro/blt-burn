//! Fused L2 Norm kernel.
//!
//! L2 Norm: sqrt(sum(x^2)) over the last dimension
//!
//! Two implementations:
//! - Simple: one thread per row
//! - Optimized: plane_sum for parallel reduction

use cubecl::{cube, prelude::*};

/// Simple L2 Norm kernel - one thread per (batch, seq) position.
#[cube(launch)]
pub fn fused_l2_norm_kernel<F: Float>(
    input: &Tensor<F>,
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

    let mut sum_sq = F::new(0.0);
    for i in 0..dim {
        let val = input[row_start + i];
        sum_sq = sum_sq + val * val;
    }

    let out_idx = batch_idx * seq_len + seq_idx;
    output[out_idx] = F::sqrt(sum_sq);
}

/// Optimized L2 Norm kernel using plane_sum for parallel reduction.
#[cube(launch)]
pub fn fused_l2_norm_kernel_optimized<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    dim: u32,
) {
    // Each plane handles one row
    let row_idx = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let lane_idx = UNIT_POS_X;

    let total_rows = output.len();

    if row_idx >= total_rows {
        terminate!();
    }

    let row_start = row_idx * dim;
    let plane_size = CUBE_DIM_X;

    // Compute sum of squares using parallel reduction
    let mut local_sum_sq = F::new(0.0);
    let mut i = lane_idx;
    while i < dim {
        let val = input[row_start + i];
        local_sum_sq = local_sum_sq + val * val;
        i += plane_size;
    }
    let sum_sq = plane_sum(local_sum_sq);

    // Only lane 0 writes the result
    if lane_idx == 0 {
        output[row_idx] = F::sqrt(sum_sq);
    }
}

/// Fused squared L2 norm kernel (avoids sqrt).
#[cube(launch)]
pub fn fused_l2_norm_squared_kernel<F: Float>(
    input: &Tensor<F>,
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

    let mut sum_sq = F::new(0.0);
    for i in 0..dim {
        let val = input[row_start + i];
        sum_sq = sum_sq + val * val;
    }

    let out_idx = batch_idx * seq_len + seq_idx;
    output[out_idx] = sum_sq;
}
