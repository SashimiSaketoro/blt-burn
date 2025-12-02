//! Unified streaming pipeline for entropy model ingestion.
//!
//! This module implements a streaming pipeline that processes the entropy
//! model with unified chunking to ensure consistent context for patch
//! boundary detection.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                    ENTROPY MODEL PIPELINE                               │
//! │  CHUNK_SIZE = 4096, STRIDE = 2048 (50% overlap)                        │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  For each chunk:                                                        │
//! │    1. Entropy Model → byte-level entropies, norms                      │
//! │    2. Detect patches in "safe zone" (not in overlap)                   │
//! │    3. Aggregate prominence/entropy to patch-level                       │
//! │    4. Emit completed patches with 768-dim embeddings                    │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Why 50% Overlap?
//!
//! - Patches near chunk boundaries get processed in TWO chunks
//! - We keep embeddings from the chunk where the patch has MORE context
//! - Ensures every patch has at least 2048 bytes of preceding context

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Int, Tensor, Transaction};

use crate::model::LMTransformer;
use crate::patcher::{
    entropy, patch_start_indices_cpu, patch_start_mask_from_entropy_with_monotonicity,
};

/// Unified chunk size for both models (max context both can handle efficiently)
pub const CHUNK_SIZE: usize = 4096;

/// Stride = 50% overlap ensures patches near boundaries get full context
pub const STRIDE: usize = 2048;

/// Safe zone end - patches starting before this are confirmed in current chunk
pub const SAFE_ZONE_END: usize = CHUNK_SIZE - STRIDE; // 2048

/// Completed patch data ready for sphere optimization
#[derive(Debug, Clone)]
pub struct PatchData {
    /// Absolute byte offset where this patch starts
    pub start_offset: usize,
    /// Length of the patch in bytes
    pub length: usize,
    /// Patch embedding from entropy model [768]
    pub embedding: Vec<f32>,
    /// Mean prominence (L2 norm) for this patch
    pub prominence: f32,
    /// Mean entropy for this patch
    pub entropy: f32,
    /// Raw bytes of this patch
    pub bytes: Vec<u8>,
}

/// Result of processing a document through the entropy pipeline.
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Completed patches with embeddings
    pub patches: Vec<PatchData>,
    /// Raw byte-level entropies
    pub entropies: Vec<f32>,
    /// Raw byte-level norms (prominence)
    pub norms: Vec<f32>,
}

/// Buffer for streaming pipeline state
pub struct PipelineBuffer {
    /// Accumulated byte-level entropies
    entropies: Vec<f32>,
    /// Accumulated byte-level norms (prominence)
    norms: Vec<f32>,
    /// Accumulated raw bytes
    raw_bytes: Vec<u8>,
    /// Total bytes processed so far
    total_bytes_processed: usize,
    /// Current chunk index
    chunk_idx: usize,
    /// Next patch index to assign
    next_patch_idx: usize,
    /// Entropy threshold for patch detection
    threshold: f64,
}

impl PipelineBuffer {
    /// Create a new pipeline buffer
    pub fn new(threshold: f64) -> Self {
        Self {
            entropies: Vec::new(),
            norms: Vec::new(),
            raw_bytes: Vec::new(),
            total_bytes_processed: 0,
            chunk_idx: 0,
            // pending_patches reserved for future
            next_patch_idx: 0,
            threshold,
        }
    }

    /// Process a single chunk through entropy model.
    /// Returns completed patches that are ready for sphere optimization.
    ///
    /// # Panics
    /// Panics if the GPU transaction fails to return exactly 2 tensors (internal invariant).
    pub fn process_chunk(
        &mut self,
        chunk_bytes: &[u8],
        entropy_model: &LMTransformer<Wgpu>,
        blt_encoder: Option<()>,
        device: &WgpuDevice,
        is_last_chunk: bool,
    ) -> Result<Vec<PatchData>, anyhow::Error> {
        let chunk_len = chunk_bytes.len();
        let chunk_start_absolute = self.total_bytes_processed;

        // 1. Run entropy model on this chunk
        let tokens: Vec<i32> = chunk_bytes.iter().map(|&b| b as i32).collect();
        let input = Tensor::<Wgpu, 1, Int>::from_ints(&tokens[..], device).reshape([1, chunk_len]);

        let output = entropy_model.forward_with_embeddings(input);
        let chunk_entropies = entropy(output.logits);

        // Batch GPU->CPU transfers (single sync instead of 2)
        let [entropies_data, norms_data] = Transaction::default()
            .register(chunk_entropies)
            .register(output.embedding_norms)
            .execute()
            .try_into()
            .expect("Transaction should return 2 tensors");

        let chunk_entropies_f32: Vec<f32> = entropies_data.iter::<f32>().collect();
        let chunk_norms_f32: Vec<f32> = norms_data.iter::<f32>().collect();

        // 2. Determine which portion of this chunk is "new" (not overlap from previous)
        let new_data_start = if self.chunk_idx == 0 { 0 } else { STRIDE };
        let new_data_end = chunk_len;

        // For first chunk, accumulate everything
        // For subsequent chunks, only accumulate new data (after overlap)
        if self.chunk_idx == 0 {
            self.entropies.extend_from_slice(&chunk_entropies_f32);
            self.norms.extend_from_slice(&chunk_norms_f32);
            self.raw_bytes.extend_from_slice(chunk_bytes);
        } else {
            // Only add the new portion (skip the overlap that was already processed)
            if new_data_start < chunk_len {
                self.entropies
                    .extend_from_slice(&chunk_entropies_f32[new_data_start..]);
                self.norms
                    .extend_from_slice(&chunk_norms_f32[new_data_start..]);
                self.raw_bytes
                    .extend_from_slice(&chunk_bytes[new_data_start..]);
            }
        }

        // 3. Detect patch boundaries in accumulated entropies
        let accumulated_len = self.entropies.len();
        let entropies_tensor = Tensor::<Wgpu, 1>::from_floats(&self.entropies[..], device)
            .reshape([1, accumulated_len]);

        let mask =
            patch_start_mask_from_entropy_with_monotonicity(entropies_tensor, self.threshold);
        let patch_indices = patch_start_indices_cpu(mask);
        let patch_starts: Vec<usize> = if !patch_indices.is_empty() {
            patch_indices[0].clone()
        } else {
            vec![]
        };

        // 4. Determine which patches can be confirmed (in safe zone or last chunk)
        // Safe zone = patches that start before (chunk_start_absolute + SAFE_ZONE_END)
        // These patches have enough following context to be reliable
        let safe_zone_cutoff = if is_last_chunk {
            usize::MAX // Last chunk - confirm everything
        } else {
            chunk_start_absolute + SAFE_ZONE_END
        };

        // Find patches to confirm (those we haven't emitted yet and are in safe zone)
        let mut patches_to_emit: Vec<(usize, usize, usize)> = Vec::new(); // (patch_idx, start, end)

        for (i, &start) in patch_starts.iter().enumerate() {
            if start < safe_zone_cutoff {
                let end = if i + 1 < patch_starts.len() {
                    patch_starts[i + 1]
                } else {
                    accumulated_len
                };

                // Only emit if we haven't processed this patch yet
                if start >= self.total_bytes_processed.saturating_sub(STRIDE) || self.chunk_idx == 0
                {
                    patches_to_emit.push((self.next_patch_idx + i, start, end));
                }
            }
        }

        // 5. Generate embeddings for confirmed patches (entropy model 768-dim)
        let mut completed_patches = Vec::new();
        let _ = blt_encoder; // Unused, kept for API compatibility

        for (_, start, end) in &patches_to_emit {
            let actual_end = (*end).min(self.raw_bytes.len());
            let patch_bytes = self.raw_bytes[*start..actual_end].to_vec();
            let actual_length = actual_end - *start;

            // Entropy model provides 768-dim embeddings
            let embedding = vec![0.0; 768];
            let prominence = self.compute_mean_prominence(*start, actual_end);
            let entropy = self.compute_mean_entropy(*start, actual_end);

            completed_patches.push(PatchData {
                start_offset: *start,
                length: actual_length,
                embedding,
                prominence,
                entropy,
                bytes: patch_bytes,
            });
        }

        // Update state
        self.next_patch_idx += patches_to_emit.len();
        if !is_last_chunk {
            self.total_bytes_processed = chunk_start_absolute + new_data_end.saturating_sub(STRIDE);
        } else {
            self.total_bytes_processed = chunk_start_absolute + new_data_end;
        }
        self.chunk_idx += 1;

        Ok(completed_patches)
    }

    /// Compute mean prominence for a patch range
    fn compute_mean_prominence(&self, start: usize, end: usize) -> f32 {
        let end = end.min(self.norms.len());
        let start = start.min(end);
        if start >= end {
            return 0.0;
        }
        self.norms[start..end].iter().sum::<f32>() / (end - start) as f32
    }

    /// Compute mean entropy for a patch range
    fn compute_mean_entropy(&self, start: usize, end: usize) -> f32 {
        let end = end.min(self.entropies.len());
        let start = start.min(end);
        if start >= end {
            return 0.0;
        }
        self.entropies[start..end].iter().sum::<f32>() / (end - start) as f32
    }

    /// Get all accumulated byte-level entropies
    pub fn entropies(&self) -> &[f32] {
        &self.entropies
    }

    /// Get all accumulated byte-level norms
    pub fn norms(&self) -> &[f32] {
        &self.norms
    }

    /// Get all accumulated raw bytes
    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw_bytes
    }
}

/// Process a complete document through the unified pipeline.
///
/// This is a convenience function that handles chunking internally.
pub fn process_document(
    data: &[u8],
    entropy_model: &LMTransformer<Wgpu>,
    blt_encoder: Option<()>,
    device: &WgpuDevice,
    threshold: f64,
) -> Result<ProcessingResult, anyhow::Error> {
    let mut buffer = PipelineBuffer::new(threshold);
    let mut all_patches = Vec::new();

    let total_len = data.len();
    let mut position = 0;

    while position < total_len {
        // Determine chunk boundaries
        let _chunk_start = if position == 0 { 0 } else { position };
        let chunk_end = (position + CHUNK_SIZE).min(total_len);
        let is_last_chunk = chunk_end >= total_len;

        // For non-first chunks, include overlap from previous chunk
        let actual_start = if position > 0 {
            position.saturating_sub(STRIDE)
        } else {
            0
        };

        let chunk = &data[actual_start..chunk_end];

        let patches =
            buffer.process_chunk(chunk, entropy_model, blt_encoder, device, is_last_chunk)?;

        all_patches.extend(patches);

        if is_last_chunk {
            break;
        }

        position = chunk_end;
    }

    Ok(ProcessingResult {
        patches: all_patches,
        entropies: buffer.entropies().to_vec(),
        norms: buffer.norms().to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_constants() {
        assert_eq!(CHUNK_SIZE, 4096);
        assert_eq!(STRIDE, 2048);
        assert_eq!(SAFE_ZONE_END, 2048);
        assert_eq!(CHUNK_SIZE - STRIDE, SAFE_ZONE_END);
    }
}
