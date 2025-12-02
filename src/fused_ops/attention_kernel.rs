//! Fused Scaled Dot-Product Attention kernel.
//!
//! Fuses: Q·K^T / sqrt(d) + mask → softmax → ·V
//!
//! This avoids materializing the full attention matrix in memory,
//! which is critical for long sequences.

use cubecl::{cube, prelude::*};

/// Fused scaled dot-product attention kernel.
///
/// Computes: softmax((Q·K^T) / sqrt(head_dim) + mask) · V
///
/// Input shapes:
/// - q: [batch, n_heads, seq_q, head_dim]
/// - k: [batch, n_heads, seq_kv, head_dim]
/// - v: [batch, n_heads, seq_kv, head_dim]
///
/// Output: [batch, n_heads, seq_q, head_dim]
#[cube(launch)]
pub fn fused_attention_kernel<F: Float>(
    q: &Tensor<F>,
    k: &Tensor<F>,
    v: &Tensor<F>,
    output: &mut Tensor<F>,
    head_dim: u32,
    seq_kv: u32,
    scale: F,
    use_causal_mask: u32,
) {
    // Each thread handles one output position: (batch, head, seq_q_pos)
    let batch_idx = ABSOLUTE_POS_Z;
    let head_idx = ABSOLUTE_POS_Y;
    let seq_q_idx = ABSOLUTE_POS_X;

    let batch_size = output.shape(0);
    let n_heads = output.shape(1);
    let seq_q = output.shape(2);

    if batch_idx >= batch_size || head_idx >= n_heads || seq_q_idx >= seq_q {
        terminate!();
    }

    // Compute attention scores and apply softmax for this query position
    // First pass: compute max for numerical stability
    let mut max_score = F::new(f32::NEG_INFINITY);

    for kv_idx in 0..seq_kv {
        // For causal mask: only attend to positions <= current position
        let should_attend = use_causal_mask == 0 || kv_idx <= seq_q_idx;

        if should_attend {
            // Compute Q·K^T for this position
            let mut dot = F::new(0.0);
            for d in 0..head_dim {
                let q_idx = batch_idx * n_heads * seq_q * head_dim
                    + head_idx * seq_q * head_dim
                    + seq_q_idx * head_dim
                    + d;
                let k_idx = batch_idx * n_heads * seq_kv * head_dim
                    + head_idx * seq_kv * head_dim
                    + kv_idx * head_dim
                    + d;
                dot += q[q_idx] * k[k_idx];
            }
            let score = dot * scale;
            max_score = F::max(max_score, score);
        }
    }

    // Second pass: compute exp(score - max) and sum
    let mut sum_exp = F::new(0.0);
    for kv_idx in 0..seq_kv {
        let should_attend = use_causal_mask == 0 || kv_idx <= seq_q_idx;

        if should_attend {
            let mut dot = F::new(0.0);
            for d in 0..head_dim {
                let q_idx = batch_idx * n_heads * seq_q * head_dim
                    + head_idx * seq_q * head_dim
                    + seq_q_idx * head_dim
                    + d;
                let k_idx = batch_idx * n_heads * seq_kv * head_dim
                    + head_idx * seq_kv * head_dim
                    + kv_idx * head_dim
                    + d;
                dot += q[q_idx] * k[k_idx];
            }
            let score = dot * scale;
            sum_exp += F::exp(score - max_score);
        }
    }

    // Third pass: compute weighted sum of V
    for d in 0..head_dim {
        let mut weighted_sum = F::new(0.0);

        for kv_idx in 0..seq_kv {
            let should_attend = use_causal_mask == 0 || kv_idx <= seq_q_idx;

            if should_attend {
                // Recompute attention weight
                let mut dot = F::new(0.0);
                for d2 in 0..head_dim {
                    let q_idx = batch_idx * n_heads * seq_q * head_dim
                        + head_idx * seq_q * head_dim
                        + seq_q_idx * head_dim
                        + d2;
                    let k_idx = batch_idx * n_heads * seq_kv * head_dim
                        + head_idx * seq_kv * head_dim
                        + kv_idx * head_dim
                        + d2;
                    dot += q[q_idx] * k[k_idx];
                }
                let score = dot * scale;
                let attn_weight = F::exp(score - max_score) / sum_exp;

                // Get V value
                let v_idx = batch_idx * n_heads * seq_kv * head_dim
                    + head_idx * seq_kv * head_dim
                    + kv_idx * head_dim
                    + d;
                weighted_sum += attn_weight * v[v_idx];
            }
        }

        let out_idx = batch_idx * n_heads * seq_q * head_dim
            + head_idx * seq_q * head_dim
            + seq_q_idx * head_dim
            + d;
        output[out_idx] = weighted_sum;
    }
}

/// Fused cross-attention kernel for byte→patch pooling.
///
/// Similar to self-attention but Q comes from patches, K/V from bytes.
/// No causal mask needed for cross-attention.
#[cube(launch)]
pub fn fused_cross_attention_kernel<F: Float>(
    q: &Tensor<F>,          // [batch, n_patches, n_heads, head_dim]
    k: &Tensor<F>,          // [batch, n_bytes, n_heads, head_dim]
    v: &Tensor<F>,          // [batch, n_bytes, n_heads, head_dim]
    output: &mut Tensor<F>, // [batch, n_patches, n_heads, head_dim]
    head_dim: u32,
    n_bytes: u32,
    scale: F,
) {
    let batch_idx = ABSOLUTE_POS_Z;
    let head_idx = ABSOLUTE_POS_Y;
    let patch_idx = ABSOLUTE_POS_X;

    let batch_size = output.shape(0);
    let n_patches = output.shape(1);
    let n_heads = output.shape(2);

    if batch_idx >= batch_size || patch_idx >= n_patches || head_idx >= n_heads {
        terminate!();
    }

    // Compute attention scores for this query (patch) position
    // First pass: find max for numerical stability
    let mut max_score = F::new(f32::NEG_INFINITY);

    for byte_idx in 0..n_bytes {
        let mut dot = F::new(0.0);
        for d in 0..head_dim {
            let q_idx = batch_idx * n_patches * n_heads * head_dim
                + patch_idx * n_heads * head_dim
                + head_idx * head_dim
                + d;
            let k_idx = batch_idx * n_bytes * n_heads * head_dim
                + byte_idx * n_heads * head_dim
                + head_idx * head_dim
                + d;
            dot += q[q_idx] * k[k_idx];
        }
        max_score = F::max(max_score, dot * scale);
    }

    // Second pass: compute sum of exp
    let mut sum_exp = F::new(0.0);
    for byte_idx in 0..n_bytes {
        let mut dot = F::new(0.0);
        for d in 0..head_dim {
            let q_idx = batch_idx * n_patches * n_heads * head_dim
                + patch_idx * n_heads * head_dim
                + head_idx * head_dim
                + d;
            let k_idx = batch_idx * n_bytes * n_heads * head_dim
                + byte_idx * n_heads * head_dim
                + head_idx * head_dim
                + d;
            dot += q[q_idx] * k[k_idx];
        }
        sum_exp += F::exp(dot * scale - max_score);
    }

    // Third pass: compute weighted V
    for d in 0..head_dim {
        let mut weighted_sum = F::new(0.0);

        for byte_idx in 0..n_bytes {
            let mut dot = F::new(0.0);
            for d2 in 0..head_dim {
                let q_idx = batch_idx * n_patches * n_heads * head_dim
                    + patch_idx * n_heads * head_dim
                    + head_idx * head_dim
                    + d2;
                let k_idx = batch_idx * n_bytes * n_heads * head_dim
                    + byte_idx * n_heads * head_dim
                    + head_idx * head_dim
                    + d2;
                dot += q[q_idx] * k[k_idx];
            }
            let attn_weight = F::exp(dot * scale - max_score) / sum_exp;

            let v_idx = batch_idx * n_bytes * n_heads * head_dim
                + byte_idx * n_heads * head_dim
                + head_idx * head_dim
                + d;
            weighted_sum += attn_weight * v[v_idx];
        }

        let out_idx = batch_idx * n_patches * n_heads * head_dim
            + patch_idx * n_heads * head_dim
            + head_idx * head_dim
            + d;
        output[out_idx] = weighted_sum;
    }
}
