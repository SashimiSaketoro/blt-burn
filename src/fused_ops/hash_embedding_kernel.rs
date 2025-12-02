//! CubeCL kernel for rolling polynomial hash computation.
//!
//! Uses float arithmetic to compute integer hashes - works because:
//! 1. Small primes (31, 37, 41) keep values in f32 exact integer range
//! 2. Modulo computed via: x - floor(x / vocab) * vocab
//! 3. Final cast to int for embedding lookup

use cubecl::{cube, prelude::*};

/// Float-based rolling polynomial hash kernel.
///
/// Computes hash indices for all 6 tables (3 primes × 2 ngram sizes).
/// Uses float arithmetic for GPU compatibility, then casts to int output.
///
/// # Layout
/// - Table 0: prime=31, ngram=2
/// - Table 1: prime=31, ngram=3
/// - Table 2: prime=37, ngram=2
/// - Table 3: prime=37, ngram=3
/// - Table 4: prime=41, ngram=2
/// - Table 5: prime=41, ngram=3
#[cube(launch)]
pub fn rolling_hash_float_kernel<F: Float>(
    bytes: &Tensor<F>,            // [batch, n_bytes] - bytes as floats
    hash_indices: &mut Tensor<F>, // [batch, n_bytes, 6] - output as floats (cast to int later)
    n_bytes: u32,
    hash_vocab: F, // 500002.0 for BLT-1B
) {
    let pos = ABSOLUTE_POS;
    let total_positions = hash_indices.len() / 6;

    if pos >= total_positions {
        terminate!();
    }

    let batch_idx = pos / n_bytes;
    let pos_idx = pos % n_bytes;

    // Precomputed prime powers (kept small for f32 exactness)
    // prime^0 = 1, prime^1 = prime, prime^2 = prime*prime
    let p0_pow0 = F::new(1.0); // 31^0
    let p0_pow1 = F::new(31.0); // 31^1
    let p0_pow2 = F::new(961.0); // 31^2

    let p1_pow0 = F::new(1.0); // 37^0
    let p1_pow1 = F::new(37.0); // 37^1
    let p1_pow2 = F::new(1369.0); // 37^2

    let p2_pow0 = F::new(1.0); // 41^0
    let p2_pow1 = F::new(41.0); // 41^1
    let p2_pow2 = F::new(1681.0); // 41^2

    // Gather bytes for positions [pos, pos-1, pos-2] with bounds check
    let b0 = bytes[batch_idx * n_bytes + pos_idx];

    let b1 = if pos_idx >= 1 {
        bytes[batch_idx * n_bytes + pos_idx - 1]
    } else {
        F::new(0.0)
    };

    let b2 = if pos_idx >= 2 {
        bytes[batch_idx * n_bytes + pos_idx - 2]
    } else {
        F::new(0.0)
    };

    // Table 0: prime=31, ngram=2 → hash = b0*1 + b1*31
    let h0 = b0 * p0_pow0 + b1 * p0_pow1;
    let h0_mod = h0 - F::floor(h0 / hash_vocab) * hash_vocab;

    // Table 1: prime=31, ngram=3 → hash = b0*1 + b1*31 + b2*961
    let h1 = b0 * p0_pow0 + b1 * p0_pow1 + b2 * p0_pow2;
    let h1_mod = h1 - F::floor(h1 / hash_vocab) * hash_vocab;

    // Table 2: prime=37, ngram=2
    let h2 = b0 * p1_pow0 + b1 * p1_pow1;
    let h2_mod = h2 - F::floor(h2 / hash_vocab) * hash_vocab;

    // Table 3: prime=37, ngram=3
    let h3 = b0 * p1_pow0 + b1 * p1_pow1 + b2 * p1_pow2;
    let h3_mod = h3 - F::floor(h3 / hash_vocab) * hash_vocab;

    // Table 4: prime=41, ngram=2
    let h4 = b0 * p2_pow0 + b1 * p2_pow1;
    let h4_mod = h4 - F::floor(h4 / hash_vocab) * hash_vocab;

    // Table 5: prime=41, ngram=3
    let h5 = b0 * p2_pow0 + b1 * p2_pow1 + b2 * p2_pow2;
    let h5_mod = h5 - F::floor(h5 / hash_vocab) * hash_vocab;

    // Write outputs
    let out_base = pos * 6;
    hash_indices[out_base] = F::abs(h0_mod);
    hash_indices[out_base + 1] = F::abs(h1_mod);
    hash_indices[out_base + 2] = F::abs(h2_mod);
    hash_indices[out_base + 3] = F::abs(h3_mod);
    hash_indices[out_base + 4] = F::abs(h4_mod);
    hash_indices[out_base + 5] = F::abs(h5_mod);
}
