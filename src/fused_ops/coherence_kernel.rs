//! Fused Coherence Score kernel.
//!
//! Coherence: norms^2 / (entropies + epsilon)
//!
//! This kernel fuses:
//! 1. Square the norms
//! 2. Add epsilon to entropies
//! 3. Divide
//!
//! Current: 3 kernel launches
//! Fused: 1 kernel launch

use cubecl::{cube, prelude::*};

/// Default epsilon for coherence calculation (1e-6)
const COHERENCE_EPSILON: f32 = 1e-6;

/// Fused coherence score kernel.
///
/// Computes: norms^2 / (entropies + epsilon)
///
/// Both inputs must have the same shape [batch, seq].
/// Uses fixed epsilon of 1e-6 for numerical stability.
///
/// # Type Parameters
/// * `F` - Float type (f16, f32, etc.)
///
/// # Arguments
/// * `norms` - L2 norms tensor [batch, seq]
/// * `entropies` - Entropy tensor [batch, seq]
/// * `output` - Coherence scores [batch, seq]
#[cube(launch)]
pub fn fused_coherence_kernel<F: Float>(
    norms: &Tensor<F>,
    entropies: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let idx = ABSOLUTE_POS_X;

    if idx >= output.len() {
        terminate!();
    }

    let norm = norms[idx];
    let entropy = entropies[idx];

    // coherence = norm^2 / (entropy + eps)
    let norm_squared = norm * norm;
    let eps = F::new(COHERENCE_EPSILON);
    let denom = entropy + eps;
    output[idx] = norm_squared / denom;
}

/// Fused kernel that computes L2 norm squared and coherence in one pass.
///
/// This is more efficient when you need both the embedding norms and
/// coherence scores, as it avoids an intermediate tensor for norms.
///
/// Input: embeddings [batch, seq, dim]
/// Entropies: [batch, seq] (pre-computed)
/// Output: coherence [batch, seq]
#[cube(launch)]
pub fn fused_embedding_coherence_kernel<F: Float>(
    embeddings: &Tensor<F>,
    entropies: &Tensor<F>,
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

    // Calculate row start for embeddings
    let row_start = (batch_idx * seq_len + seq_idx) * dim;

    // Compute sum of squares (L2 norm squared)
    let mut sum_sq = F::new(0.0);
    for i in 0..dim {
        let val = embeddings[row_start + i];
        sum_sq += val * val;
    }

    // Get entropy and compute coherence
    let out_idx = batch_idx * seq_len + seq_idx;
    let entropy = entropies[out_idx];
    let eps = F::new(COHERENCE_EPSILON);
    let denom = entropy + eps;
    output[out_idx] = sum_sq / denom;
}
