//! CubeCL fused entropy kernel.
//!
//! This kernel computes entropy in a single pass by fusing:
//! 1. Find max for numerical stability
//! 2. Compute sum of exp(logits - max)
//! 3. Compute entropy = -sum(p * log(p))
//!
//! Two implementations:
//! - `fused_entropy_kernel`: Simple version, one thread per row
//! - `fused_entropy_kernel_optimized`: Uses plane_sum for parallel reduction

use cubecl::{cube, prelude::*};

/// Simple fused entropy kernel: one thread handles one (batch, seq) position.
/// Good for small vocab sizes or as a fallback.
#[cube(launch)]
pub fn fused_entropy_kernel<F: Float>(logits: &Tensor<F>, output: &mut Tensor<F>, vocab_size: u32) {
    let batch_idx = ABSOLUTE_POS_X;
    let seq_idx = ABSOLUTE_POS_Y;

    let batch_size = output.shape(0);
    let seq_len = output.shape(1);

    if batch_idx >= batch_size || seq_idx >= seq_len {
        terminate!();
    }

    let row_start = (batch_idx * seq_len + seq_idx) * vocab_size;

    // Pass 1: Find max
    let mut max_val = F::new(-65504.0);
    for i in 0..vocab_size {
        let val = logits[row_start + i];
        max_val = F::max(max_val, val);
    }

    // Pass 2: Sum of exp(logits - max)
    let mut exp_sum = F::new(0.0);
    for i in 0..vocab_size {
        exp_sum += F::exp(logits[row_start + i] - max_val);
    }
    let log_sum = F::log(exp_sum);

    // Pass 3: Entropy = -sum(p * log(p))
    let mut entropy = F::new(0.0);
    for i in 0..vocab_size {
        let log_p = logits[row_start + i] - max_val - log_sum;
        let p = F::exp(log_p);
        entropy -= p * log_p;
    }

    let out_idx = batch_idx * seq_len + seq_idx;
    output[out_idx] = entropy;
}

/// Optimized fused entropy kernel using plane_sum for parallel reduction.
///
/// Each plane (warp/subgroup) cooperates to process one row.
/// Uses SIMD-width parallel reductions for max, sum(exp), and entropy.
///
/// For vocab_size=260 with plane_size=32:
/// - Each thread in the plane handles ~8 elements
/// - 3 parallel reductions using plane_max/plane_sum
#[cube(launch)]
pub fn fused_entropy_kernel_optimized<F: Float>(
    logits: &Tensor<F>,
    output: &mut Tensor<F>,
    vocab_size: u32,
) {
    // Each plane handles one row
    let row_idx = CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y;
    let lane_idx = UNIT_POS_X; // Position within the plane (0..plane_size)

    let total_rows = output.len();

    if row_idx >= total_rows {
        terminate!();
    }

    let row_start = row_idx * vocab_size;
    let plane_size = CUBE_DIM_X;

    // Each lane processes elements: lane_idx, lane_idx + plane_size, lane_idx + 2*plane_size, ...
    // ========== Pass 1: Find max using parallel reduction ==========
    let mut local_max = F::new(-65504.0);
    let mut i = lane_idx;
    while i < vocab_size {
        let val = logits[row_start + i];
        local_max = F::max(local_max, val);
        i += plane_size;
    }
    // Reduce across the plane
    let max_val = plane_max(local_max);

    // ========== Pass 2: Sum of exp(logits - max) ==========
    let mut local_exp_sum = F::new(0.0);
    let mut i = lane_idx;
    while i < vocab_size {
        local_exp_sum += F::exp(logits[row_start + i] - max_val);
        i += plane_size;
    }
    let exp_sum = plane_sum(local_exp_sum);
    let log_sum = F::log(exp_sum);

    // ========== Pass 3: Entropy = -sum(p * log(p)) ==========
    let mut local_entropy = F::new(0.0);
    let mut i = lane_idx;
    while i < vocab_size {
        let log_p = logits[row_start + i] - max_val - log_sum;
        let p = F::exp(log_p);
        local_entropy -= p * log_p;
        i += plane_size;
    }
    let entropy = plane_sum(local_entropy);

    // Only lane 0 writes the result
    if lane_idx == 0 {
        output[row_idx] = entropy;
    }
}
