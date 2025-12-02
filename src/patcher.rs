#[cfg(not(feature = "fused-entropy"))]
use burn::tensor::activation::log_softmax;
use burn::tensor::{backend::Backend, Int, Tensor};

/// Compute entropy from logits using the fused CubeCL kernel.
///
/// This version uses a single fused kernel that combines softmax, log,
/// multiplication, and reduction to reduce kernel launch overhead and memory traffic.
///
/// # Arguments
/// * `scores` - Input logits tensor of shape `[batch, seq_len, vocab_size]`
///
/// # Returns
/// Entropy tensor of shape `[batch, seq_len]`
#[cfg(feature = "fused-entropy")]
pub fn entropy<B>(scores: Tensor<B, 3>) -> Tensor<B, 2>
where
    B: crate::fused_ops::FusedEntropyBackend,
{
    let [_bs, _seq_len, vocab] = scores.dims();
    crate::fused_ops::fused_entropy(scores, vocab)
}

/// Compute entropy from logits using standard tensor operations.
///
/// This is the fallback implementation used when the `fused-entropy` feature
/// is not enabled. It uses Burn's standard tensor operations which benefit
/// from auto-fusion for element-wise operations but not for reductions.
///
/// # Arguments
/// * `scores` - Input logits tensor of shape `[batch, seq_len, vocab_size]`
///
/// # Returns
/// Entropy tensor of shape `[batch, seq_len]`
///
/// # Formula
/// `entropy = -sum(p * log(p))` where `p = softmax(scores)`
#[cfg(not(feature = "fused-entropy"))]
pub fn entropy<B: Backend>(scores: Tensor<B, 3>) -> Tensor<B, 2> {
    // scores: [bs, seq_len, vocab]
    // returns: [bs, seq_len]
    let [bs, seq_len, _vocab] = scores.dims();
    let log_probs = log_softmax(scores, 2);
    let probs = log_probs.clone().exp();

    // Numerical stability: log_softmax already handles -inf for zero probabilities,
    // but we multiply log_probs * probs, where probs can be exactly 0.0
    // When p=0, p*log(p) should be 0 (by L'Hôpital's rule: lim p->0 of p*log(p) = 0)
    // Since log_softmax outputs are all negative or 0, and exp() of very negative = ~0,
    // the multiplication naturally handles this case correctly (0 * -inf = 0 in IEEE754)
    let p_log_p = log_probs * probs;

    // sum_dim(2) on [bs, seq_len, vocab] -> [bs, seq_len, 1]
    // Then neg and reshape explicitly to [bs, seq_len]
    // This avoids squeeze() panicking on batch_size=1
    let entropy_values = p_log_p.sum_dim(2).neg().reshape([bs, seq_len]);

    // Note: burn's log_softmax is numerically stable and handles the case where
    // some probabilities are effectively zero (very negative log values).
    // The multiplication p*log(p) naturally goes to 0 as p->0 (L'Hôpital's rule)
    // so we don't need explicit NaN handling here.
    entropy_values
}

/// Compute patch start mask from entropy values with monotonicity detection.
///
/// Following the official BLT patcher, this function:
/// 1. Always marks positions 0 AND 1 as patch starts (first two bytes are separate patches)
/// 2. From position 2 onward, uses entropy delta threshold detection
///
/// This matches `find_entropy_patch_start_ids` in the official BLT repo which always
/// starts with `first_ids = torch.tensor([0, 1], ...)`.
///
/// # Arguments
/// * `entropies` - Entropy values `[batch, seq_len]`
/// * `threshold` - Delta threshold for detecting entropy spikes
///
/// # Returns
/// Boolean mask `[batch, seq_len]` where 1 = patch start
pub fn patch_start_mask_from_entropy_with_monotonicity<B: Backend>(
    entropies: Tensor<B, 2>,
    threshold: f64,
) -> Tensor<B, 2, Int> {
    let [bs, seq_len] = entropies.dims();

    if seq_len == 0 {
        return entropies.greater_elem(threshold).int();
    }

    if seq_len == 1 {
        // Only one byte - it's a patch start
        return Tensor::ones([bs, 1], &entropies.device());
    }

    if seq_len == 2 {
        // Two bytes - both are patch starts (official BLT behavior)
        return Tensor::ones([bs, 2], &entropies.device());
    }

    // Official BLT: first_ids = [0, 1], then entropy detection from position 2+
    // We mark positions 0 and 1 as True, then apply entropy detection to positions 2+

    // differences = entropies[:, 2:] - entropies[:, 1:-1]
    // (compare position i with position i-1, starting from position 2)
    let current = entropies.clone().slice([0..bs, 2..seq_len]);
    let prev = entropies.clone().slice([0..bs, 1..seq_len - 1]);
    let differences = current - prev;

    // condition = differences > threshold
    let condition = differences.greater_elem(threshold).int();

    // Construct full mask: [1, 1, condition...]
    // Position 0 = True (start of sequence)
    // Position 1 = True (official BLT always separates first two bytes)
    // Position 2+ = entropy-based detection
    let first_two = Tensor::ones([bs, 2], &entropies.device());
    Tensor::cat(vec![first_two, condition], 1)
}

/// Extract patch start indices from a mask tensor on CPU.
///
/// # Panics
/// Panics if the integer tensor cannot be converted to i32 slice (backend mismatch).
pub fn patch_start_indices_cpu<B: Backend>(patch_start_mask: Tensor<B, 2, Int>) -> Vec<Vec<usize>> {
    let [bs, seq_len] = patch_start_mask.dims();
    let mask_data = patch_start_mask.into_data();
    let mask_values = mask_data.as_slice::<i32>().unwrap(); // Assuming Int maps to i32

    let mut batch_indices = Vec::with_capacity(bs);

    for b in 0..bs {
        let mut indices = Vec::new();
        for i in 0..seq_len {
            if mask_values[b * seq_len + i] == 1 {
                indices.push(i);
            }
        }
        batch_indices.push(indices);
    }

    batch_indices
}

pub fn patch_lengths_from_start_indices(
    start_indices: &[Vec<usize>],
    seq_len: usize,
) -> Vec<Vec<usize>> {
    start_indices
        .iter()
        .map(|indices| {
            let mut lengths = Vec::new();
            for i in 0..indices.len() {
                let start = indices[i];
                let end = if i + 1 < indices.len() {
                    indices[i + 1]
                } else {
                    seq_len
                };
                lengths.push(end - start);
            }
            lengths
        })
        .collect()
}
