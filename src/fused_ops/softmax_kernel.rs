//! Fused Softmax kernel.
//!
//! Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
//!
//! This kernel fuses:
//! 1. Find max for numerical stability
//! 2. Subtract max and compute exp
//! 3. Sum the exponentials
//! 4. Divide each element by the sum
//!
//! Current: 5+ kernel launches
//! Fused: 1 kernel launch

use cubecl::{cube, prelude::*};

/// Fused Softmax kernel for the last dimension.
///
/// Each thread handles one row (all elements before the softmax dimension).
/// This is optimized for attention patterns where softmax is over seq_len.
///
/// # Type Parameters
/// * `F` - Float type (f16, f32, etc.)
///
/// # Arguments
/// * `input` - Input tensor, softmax computed over last dim
/// * `output` - Output tensor, same shape as input
/// * `softmax_dim_size` - Size of the dimension to softmax over
#[cube(launch)]
pub fn fused_softmax_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    softmax_dim_size: u32,
) {
    // Each thread handles one "row" (everything except the last dimension)
    let row_idx = ABSOLUTE_POS_X;

    // Total number of rows
    let total_elements = output.len();
    let num_rows = total_elements / softmax_dim_size;

    if row_idx >= num_rows {
        terminate!();
    }

    let row_start = row_idx * softmax_dim_size;

    // ========== Pass 1: Find max for numerical stability ==========
    let mut max_val = F::new(-65504.0);
    for i in 0..softmax_dim_size {
        let val = input[row_start + i];
        max_val = F::max(max_val, val);
    }

    // ========== Pass 2: Compute exp(x - max) and sum ==========
    let mut exp_sum = F::new(0.0);
    for i in 0..softmax_dim_size {
        let exp_val = F::exp(input[row_start + i] - max_val);
        output[row_start + i] = exp_val; // Store exp values temporarily
        exp_sum += exp_val;
    }

    // ========== Pass 3: Normalize by sum ==========
    let inv_sum = F::new(1.0) / exp_sum;
    for i in 0..softmax_dim_size {
        output[row_start + i] = output[row_start + i] * inv_sum;
    }
}

/// Fused Softmax kernel for 4D attention tensors.
///
/// Optimized for attention scores of shape [batch, heads, seq_q, seq_k]
/// where softmax is over the last dimension (seq_k).
///
/// Each thread handles one (batch, head, seq_q) position.
#[cube(launch)]
pub fn fused_softmax_attention_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    seq_k: u32,
) {
    let batch_idx = ABSOLUTE_POS_X;
    let head_idx = ABSOLUTE_POS_Y;
    let seq_q_idx = ABSOLUTE_POS_Z;

    let batch_size = output.shape(0);
    let n_heads = output.shape(1);
    let seq_q = output.shape(2);

    if batch_idx >= batch_size || head_idx >= n_heads || seq_q_idx >= seq_q {
        terminate!();
    }

    // Calculate row start: input is [batch, heads, seq_q, seq_k]
    let row_start = ((batch_idx * n_heads + head_idx) * seq_q + seq_q_idx) * seq_k;

    // ========== Pass 1: Find max ==========
    let mut max_val = F::new(-65504.0);
    for i in 0..seq_k {
        let val = input[row_start + i];
        max_val = F::max(max_val, val);
    }

    // ========== Pass 2: Compute exp(x - max) and sum ==========
    let mut exp_sum = F::new(0.0);
    for i in 0..seq_k {
        let exp_val = F::exp(input[row_start + i] - max_val);
        output[row_start + i] = exp_val;
        exp_sum += exp_val;
    }

    // ========== Pass 3: Normalize ==========
    let inv_sum = F::new(1.0) / exp_sum;
    for i in 0..seq_k {
        output[row_start + i] = output[row_start + i] * inv_sum;
    }
}
