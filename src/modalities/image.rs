//! Image pre-tokenization with patch-based segmentation.
//!
//! Implements adaptive patch extraction with entropy-based merging:
//! 1. Extract fixed-size RGB patches from the image
//! 2. Compute local entropy for each patch
//! 3. Merge low-entropy (uniform) patches into super-patches
//!
//! This reduces the number of segments for homogeneous regions while
//! preserving detail in high-information areas.

use anyhow::Result;

use super::{ByteSegment, ModalityPreTokenizer, SegmentMetadata};

/// Image pre-tokenizer using patch-based segmentation on raw pixels.
///
/// Enhanced with adaptive entropy-based merging to reduce segment count
/// for uniform regions while preserving detail in complex areas.
pub struct ImagePreTokenizer {
    /// Size of each patch (width and height in pixels).
    patch_size: usize,
    /// Stride between patches (typically equals patch_size for non-overlapping).
    stride: usize,
}

impl ImagePreTokenizer {
    /// Create a new image pre-tokenizer.
    ///
    /// # Arguments
    /// * `patch_size` - Size of each patch in pixels (e.g., 16 for 16x16 patches)
    /// * `stride` - Stride between patches (use patch_size for non-overlapping)
    pub fn new(patch_size: usize, stride: usize) -> Self {
        Self { patch_size, stride }
    }

    /// Compute Shannon entropy of a byte slice.
    ///
    /// Higher entropy indicates more randomness/complexity.
    /// Lower entropy indicates uniformity (e.g., solid colors).
    fn compute_patch_entropy(bytes: &[u8]) -> f32 {
        let mut counts = [0usize; 256];
        for &b in bytes {
            counts[b as usize] += 1;
        }
        let total = bytes.len() as f32;
        if total == 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in &counts {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

impl ModalityPreTokenizer for ImagePreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        // Decode image from bytes (supports PNG, JPEG, GIF, etc.)
        let img = image::load_from_memory(data)
            .map_err(|e| anyhow::anyhow!("Failed to decode image: {}", e))?;

        // Convert to RGB8 for consistent processing
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();

        let mut initial_patches = Vec::new();

        // Step 1: Extract standard grid patches
        for y in (0..height).step_by(self.stride) {
            for x in (0..width).step_by(self.stride) {
                let mut patch_bytes = Vec::with_capacity(self.patch_size * self.patch_size * 3);

                for py in 0..self.patch_size as u32 {
                    for px in 0..self.patch_size as u32 {
                        let pixel = if x + px < width && y + py < height {
                            *rgb.get_pixel(x + px, y + py)
                        } else {
                            // Pad with black for edge patches
                            image::Rgb([0, 0, 0])
                        };
                        patch_bytes.extend_from_slice(&pixel.0);
                    }
                }

                let entropy = Self::compute_patch_entropy(&patch_bytes);

                initial_patches.push(ByteSegment {
                    bytes: patch_bytes,
                    label: Some("image_patch".to_string()),
                    metadata: Some(SegmentMetadata {
                        start_offset: (y * width + x) as usize * 3,
                        end_offset: ((y + self.patch_size as u32) * width
                            + x
                            + self.patch_size as u32) as usize
                            * 3,
                        confidence: 1.0,
                        extra: Some(serde_json::json!({
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "local_entropy": entropy,
                            "patch_size": self.patch_size
                        })),
                    }),
                });
            }
        }

        // Step 2: Adaptive Merging of low-entropy patches
        if initial_patches.is_empty() {
            return Ok(vec![]);
        }

        let mut merged = vec![initial_patches[0].clone()];
        const ENTROPY_MERGE_THRESHOLD: f32 = 1.5;

        for patch in initial_patches.into_iter().skip(1) {
            let last_meta = merged
                .last()
                .unwrap()
                .metadata
                .as_ref()
                .unwrap()
                .extra
                .as_ref()
                .unwrap();
            let last_entropy = last_meta["local_entropy"].as_f64().unwrap_or(0.0) as f32;

            let this_meta = patch.metadata.as_ref().unwrap().extra.as_ref().unwrap();
            let this_entropy = this_meta["local_entropy"].as_f64().unwrap_or(0.0) as f32;

            // Merge threshold: low entropy implies uniform region
            if (last_entropy + this_entropy) / 2.0 < ENTROPY_MERGE_THRESHOLD {
                let last_seg = merged.last_mut().unwrap();
                last_seg.bytes.extend(&patch.bytes);

                // Update metadata to reflect merged region
                if let Some(meta) = &mut last_seg.metadata {
                    meta.end_offset = patch.metadata.as_ref().unwrap().end_offset;

                    // Update extra to indicate this is a merged super-patch
                    if let Some(extra) = &mut meta.extra {
                        if let Some(obj) = extra.as_object_mut() {
                            let merge_count = obj
                                .get("merge_count")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(1)
                                + 1;
                            obj.insert(
                                "merge_count".to_string(),
                                serde_json::json!(merge_count),
                            );
                            obj.insert("is_super_patch".to_string(), serde_json::json!(true));
                        }
                    }
                }
            } else {
                merged.push(patch);
            }
        }

        Ok(merged)
    }

    fn modality(&self) -> &str {
        "image"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_calculation() {
        // All same bytes = low entropy
        let uniform: Vec<u8> = vec![128; 256];
        let entropy = ImagePreTokenizer::compute_patch_entropy(&uniform);
        assert!(entropy < 0.01, "Uniform data should have near-zero entropy");

        // Random-ish bytes = higher entropy
        let varied: Vec<u8> = (0..=255).collect();
        let entropy = ImagePreTokenizer::compute_patch_entropy(&varied);
        assert!(entropy > 7.0, "Varied data should have high entropy");
    }
}

