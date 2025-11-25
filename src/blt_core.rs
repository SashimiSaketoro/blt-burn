//! Core BLT data structures and processing functions.
//!
//! This module provides the canonical BLT data flow matching Facebook's implementation:
//! text → bytes → tokens → entropy_model → logits → entropy → patches
//!
//! For hypersphere extensions (pre-norm embeddings, prominence, coherence), see `sphere_ext`.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Int, Tensor};
use serde::{Deserialize, Serialize};

use crate::model::LMTransformer;
use crate::patcher::{
    entropy, patch_lengths_from_start_indices, patch_start_indices_cpu,
    patch_start_mask_from_entropy_with_monotonicity,
};
use crate::tokenizer::{BltTokenizer, OFFSET};

/// Core BLT example matching Facebook's `BltExample` structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BltExample {
    /// Unique identifier for this sample
    pub sample_id: String,
    /// Original text (optional, may be omitted for binary data)
    pub text: Option<String>,
    /// Byte tokens: each byte value + OFFSET (range 4-259 for bytes, 0-3 for special)
    pub tokens: Vec<i32>,
    /// Per-token entropy from the entropy model
    pub entropies: Vec<f32>,
    /// Length of each patch (derived from entropy boundaries)
    pub patch_lengths: Vec<i32>,
    /// Attention mask (true = attend, false = ignore)
    pub mask: Vec<bool>,
}

/// Extended BLT example with pre-L2-norm signals for hypersphere placement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BltExampleWithEmbeddings {
    /// Core BLT data
    pub core: BltExample,
    /// Pre-L2-norm embeddings [seq_len, dim] flattened
    pub pre_norm_embeddings: Vec<f32>,
    /// Embedding dimension (typically 768)
    pub embedding_dim: usize,
    /// L2 norms of embeddings before normalization (prominence signal)
    pub prominence: Vec<f32>,
    /// Coherence scores: prominence² / entropy (confidence-weighted prominence)
    pub coherence_scores: Vec<f32>,
    /// Indices where patches start
    pub patch_indices: Vec<i32>,
}

/// Configuration for BLT processing
#[derive(Debug, Clone)]
pub struct BltConfig {
    /// Entropy threshold for patch boundary detection (default: 1.335)
    pub threshold: f64,
    /// Use monotonicity mode: diff > threshold instead of entropy > threshold
    pub monotonicity: bool,
    /// Maximum sequence length for chunked processing
    pub max_seq_len: usize,
    /// Overlap for chunked processing (context window)
    pub chunk_overlap: usize,
    /// Add BOS token
    pub add_bos: bool,
    /// Add EOS token
    pub add_eos: bool,
}

impl Default for BltConfig {
    fn default() -> Self {
        Self {
            threshold: 1.335, // Facebook's default
            monotonicity: true,
            max_seq_len: 1024,
            chunk_overlap: 512,
            add_bos: false,
            add_eos: false,
        }
    }
}

/// Process raw bytes through the BLT pipeline.
///
/// Returns core BLT data (tokens, entropies, patch_lengths) without embeddings.
pub fn process_bytes(
    data: &[u8],
    sample_id: &str,
    model: &LMTransformer<Wgpu>,
    device: &WgpuDevice,
    config: &BltConfig,
) -> BltExample {
    // 1. Tokenize: bytes → tokens with OFFSET
    let tokenizer = BltTokenizer::new(config.add_bos, config.add_eos);
    let text = String::from_utf8_lossy(data);
    let tokens_usize = tokenizer.encode(&text);
    let tokens: Vec<i32> = tokens_usize.iter().map(|&t| t as i32).collect();

    let total_tokens = tokens.len();

    if total_tokens == 0 {
        return BltExample {
            sample_id: sample_id.to_string(),
            text: Some(text.to_string()),
            tokens: vec![],
            entropies: vec![],
            patch_lengths: vec![],
            mask: vec![],
        };
    }

    // 2. Run entropy model with chunked processing
    let entropies = compute_entropies_chunked(&tokens, model, device, config);

    // 3. Compute patch boundaries
    let entropies_tensor =
        Tensor::<Wgpu, 1>::from_floats(entropies.as_slice(), device).reshape([1, total_tokens]);

    let mask_tensor =
        patch_start_mask_from_entropy_with_monotonicity(entropies_tensor, config.threshold);
    let patch_indices = patch_start_indices_cpu(mask_tensor);
    let patch_lengths_nested = patch_lengths_from_start_indices(&patch_indices, total_tokens);

    let patch_lengths: Vec<i32> = patch_lengths_nested
        .first()
        .map(|v| v.iter().map(|&x| x as i32).collect())
        .unwrap_or_default();

    // 4. Create attention mask (all true for now)
    let mask = vec![true; total_tokens];

    BltExample {
        sample_id: sample_id.to_string(),
        text: Some(text.to_string()),
        tokens,
        entropies,
        patch_lengths,
        mask,
    }
}

/// Process raw bytes and extract full embeddings for hypersphere placement.
///
/// Returns extended BLT data including pre-L2-norm embeddings and prominence scores.
pub fn process_bytes_with_embeddings(
    data: &[u8],
    sample_id: &str,
    model: &LMTransformer<Wgpu>,
    device: &WgpuDevice,
    config: &BltConfig,
) -> BltExampleWithEmbeddings {
    // 1. Tokenize
    let tokenizer = BltTokenizer::new(config.add_bos, config.add_eos);
    let text = String::from_utf8_lossy(data);
    let tokens_usize = tokenizer.encode(&text);
    let tokens: Vec<i32> = tokens_usize.iter().map(|&t| t as i32).collect();

    let total_tokens = tokens.len();

    if total_tokens == 0 {
        return BltExampleWithEmbeddings {
            core: BltExample {
                sample_id: sample_id.to_string(),
                text: Some(text.to_string()),
                tokens: vec![],
                entropies: vec![],
                patch_lengths: vec![],
                mask: vec![],
            },
            pre_norm_embeddings: vec![],
            embedding_dim: 768,
            prominence: vec![],
            coherence_scores: vec![],
            patch_indices: vec![],
        };
    }

    // 2. Run model with chunked processing, extracting all signals
    let (entropies, embeddings, norms) =
        compute_all_signals_chunked(&tokens, model, device, config);

    // 3. Compute patch boundaries
    let entropies_tensor =
        Tensor::<Wgpu, 1>::from_floats(entropies.as_slice(), device).reshape([1, total_tokens]);

    let mask_tensor =
        patch_start_mask_from_entropy_with_monotonicity(entropies_tensor, config.threshold);
    let patch_indices_nested = patch_start_indices_cpu(mask_tensor);
    let patch_lengths_nested = patch_lengths_from_start_indices(&patch_indices_nested, total_tokens);

    let patch_lengths: Vec<i32> = patch_lengths_nested
        .first()
        .map(|v| v.iter().map(|&x| x as i32).collect())
        .unwrap_or_default();

    let patch_indices: Vec<i32> = patch_indices_nested
        .first()
        .map(|v| v.iter().map(|&x| x as i32).collect())
        .unwrap_or_default();

    // 4. Compute coherence scores: prominence² / entropy
    let coherence_scores: Vec<f32> = norms
        .iter()
        .zip(entropies.iter())
        .map(|(&n, &e)| (n * n) / (e + 1e-6))
        .collect();

    // 5. Create attention mask
    let mask = vec![true; total_tokens];

    BltExampleWithEmbeddings {
        core: BltExample {
            sample_id: sample_id.to_string(),
            text: Some(text.to_string()),
            tokens,
            entropies,
            patch_lengths,
            mask,
        },
        pre_norm_embeddings: embeddings,
        embedding_dim: 768,
        prominence: norms,
        coherence_scores,
        patch_indices,
    }
}

/// Compute entropies using chunked processing for long sequences.
fn compute_entropies_chunked(
    tokens: &[i32],
    model: &LMTransformer<Wgpu>,
    device: &WgpuDevice,
    config: &BltConfig,
) -> Vec<f32> {
    let total_tokens = tokens.len();
    let chunk_size = config.max_seq_len;
    let stride = chunk_size - config.chunk_overlap;

    let mut all_entropies = Vec::with_capacity(total_tokens);
    let mut position = 0;

    while position < total_tokens {
        let start = if position == 0 {
            0
        } else {
            position.saturating_sub(config.chunk_overlap)
        };
        let end = (position + stride).min(total_tokens);
        let chunk = &tokens[start..end];
        let chunk_len = chunk.len();

        let skip_count = if position == 0 { 0 } else { config.chunk_overlap };

        let input =
            Tensor::<Wgpu, 1, Int>::from_ints(chunk, device).reshape([1, chunk_len]);

        let output = model.forward(input);
        let chunk_entropies = entropy(output);

        // Extract entropies to CPU
        let entropies_data = chunk_entropies.into_data();
        let entropies_slice: Vec<f32> = entropies_data.iter::<f32>().collect();

        // Only take the new tokens (skip overlap context)
        let valid_start = skip_count;
        let valid_end = chunk_len;

        for i in valid_start..valid_end {
            if all_entropies.len() < total_tokens {
                all_entropies.push(entropies_slice[i]);
            }
        }

        position = end;
    }

    // Ensure we have exactly total_tokens entropies
    all_entropies.truncate(total_tokens);
    while all_entropies.len() < total_tokens {
        all_entropies.push(0.0);
    }

    all_entropies
}

/// Compute all signals (entropies, embeddings, norms) using chunked processing.
fn compute_all_signals_chunked(
    tokens: &[i32],
    model: &LMTransformer<Wgpu>,
    device: &WgpuDevice,
    config: &BltConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let total_tokens = tokens.len();
    let embedding_dim = 768;
    let chunk_size = config.max_seq_len;
    let stride = chunk_size - config.chunk_overlap;

    let mut all_entropies = Vec::with_capacity(total_tokens);
    let mut all_embeddings = Vec::with_capacity(total_tokens * embedding_dim);
    let mut all_norms = Vec::with_capacity(total_tokens);

    let mut position = 0;

    while position < total_tokens {
        let start = if position == 0 {
            0
        } else {
            position.saturating_sub(config.chunk_overlap)
        };
        let end = (position + stride).min(total_tokens);
        let chunk = &tokens[start..end];
        let chunk_len = chunk.len();

        let skip_count = if position == 0 { 0 } else { config.chunk_overlap };

        let input =
            Tensor::<Wgpu, 1, Int>::from_ints(chunk, device).reshape([1, chunk_len]);

        let output = model.forward_with_embeddings(input);

        // Compute entropies from logits
        let chunk_entropies = entropy(output.logits);

        // Extract all data to CPU
        let entropies_data = chunk_entropies.into_data();
        let entropies_slice: Vec<f32> = entropies_data.iter::<f32>().collect();

        let embeddings_data = output.pre_norm_embeddings.into_data();
        let embeddings_slice: Vec<f32> = embeddings_data.iter::<f32>().collect();

        let norms_data = output.embedding_norms.into_data();
        let norms_slice: Vec<f32> = norms_data.iter::<f32>().collect();

        // Only take the new tokens (skip overlap context)
        let valid_start = skip_count;
        let valid_end = chunk_len;

        for i in valid_start..valid_end {
            if all_entropies.len() < total_tokens {
                all_entropies.push(entropies_slice[i]);
                all_norms.push(norms_slice[i]);

                // Copy embedding vector for this token
                let emb_start = i * embedding_dim;
                let emb_end = emb_start + embedding_dim;
                all_embeddings.extend_from_slice(&embeddings_slice[emb_start..emb_end]);
            }
        }

        position = end;
    }

    // Ensure correct sizes
    all_entropies.truncate(total_tokens);
    all_norms.truncate(total_tokens);
    all_embeddings.truncate(total_tokens * embedding_dim);

    while all_entropies.len() < total_tokens {
        all_entropies.push(0.0);
        all_norms.push(0.0);
        all_embeddings.extend(vec![0.0; embedding_dim]);
    }

    (all_entropies, all_embeddings, all_norms)
}

/// Convert tokens back to text (best-effort UTF-8 decoding)
pub fn tokens_to_text(tokens: &[i32]) -> String {
    let bytes: Vec<u8> = tokens
        .iter()
        .filter_map(|&t| {
            let t_usize = t as usize;
            if t_usize >= OFFSET {
                Some((t_usize - OFFSET) as u8)
            } else {
                None
            }
        })
        .collect();
    String::from_utf8_lossy(&bytes).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blt_config_default() {
        let config = BltConfig::default();
        assert!((config.threshold - 1.335).abs() < 0.001);
        assert!(config.monotonicity);
    }

    #[test]
    fn test_tokens_to_text() {
        let tokens: Vec<i32> = "Hello".bytes().map(|b| b as i32 + OFFSET as i32).collect();
        let text = tokens_to_text(&tokens);
        assert_eq!(text, "Hello");
    }
}
