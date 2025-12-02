//! WebDataset output format for PyTorch streaming.
//!
//! Produces sharded .tar.gz archives compatible with the WebDataset loader:
//! ```python
//! import webdataset as wds
//! dataset = wds.WebDataset("output/shard_*.tar.gz").decode()
//! ```

use anyhow::Result;
use bytemuck;
use flate2::write::GzEncoder;
use flate2::Compression;
use safetensors::serialize;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use tar::{Builder, Header};

/// Writer for WebDataset sharded tar.gz archives.
///
/// Each shard contains multiple samples, where each sample consists of:
/// - `{sample_id}.safetensors`: embeddings, patch_lengths, entropies
/// - `{sample_id}.json`: metadata (hypergraph, source info, etc.)
pub struct WebDatasetWriter {
    output_dir: PathBuf,
    shard_size: usize,
    current_shard_idx: usize,
    current_sample_count: usize,
    total_samples: usize,
    builder: Option<Builder<GzEncoder<File>>>,
}

impl WebDatasetWriter {
    /// Create a new WebDataset writer.
    ///
    /// # Arguments
    /// * `output_dir` - Directory to write shards to
    /// * `shard_size` - Number of samples per shard
    pub fn new(output_dir: PathBuf, shard_size: usize) -> Result<Self> {
        std::fs::create_dir_all(&output_dir)?;

        let mut writer = Self {
            output_dir,
            shard_size,
            current_shard_idx: 0,
            current_sample_count: 0,
            total_samples: 0,
            builder: None,
        };

        writer.start_new_shard()?;
        Ok(writer)
    }

    /// Start a new shard file.
    fn start_new_shard(&mut self) -> Result<()> {
        let shard_path = self
            .output_dir
            .join(format!("shard_{:06}.tar.gz", self.current_shard_idx));
        let file = File::create(&shard_path)?;
        let encoder = GzEncoder::new(file, Compression::default());
        self.builder = Some(Builder::new(encoder));
        Ok(())
    }

    /// Finalize the current shard and prepare for next.
    fn finalize_current_shard(&mut self) -> Result<()> {
        if let Some(builder) = self.builder.take() {
            builder.into_inner()?.finish()?;
        }
        self.current_shard_idx += 1;
        self.current_sample_count = 0;
        Ok(())
    }

    /// Add a sample to the current shard.
    ///
    /// # Arguments
    /// * `sample_id` - Unique identifier for this sample
    /// * `bytes` - Raw byte content
    /// * `patch_lengths` - Patch boundary lengths
    /// * `entropies` - Per-token entropy values
    /// * `metadata` - Additional metadata as JSON
    pub fn add_sample(
        &mut self,
        sample_id: &str,
        bytes: &[u8],
        patch_lengths: &[i32],
        entropies: &[f32],
        metadata: &serde_json::Value,
    ) -> Result<()> {
        // Check if we need to start a new shard
        if self.current_sample_count >= self.shard_size {
            self.finalize_current_shard()?;
            self.start_new_shard()?;
        }

        // Create SafeTensors data (before borrowing builder)
        let st_data = Self::serialize_safetensors_data(bytes, patch_lengths, entropies)?;

        // Create JSON metadata
        let json_data = serde_json::to_vec_pretty(metadata)?;

        // Now borrow builder and append entries
        let builder = self
            .builder
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("WebDataset builder not initialized"))?;

        Self::append_tar_entry(builder, &format!("{sample_id}.safetensors"), &st_data)?;
        Self::append_tar_entry(builder, &format!("{sample_id}.json"), &json_data)?;

        self.current_sample_count += 1;
        self.total_samples += 1;

        Ok(())
    }

    /// Append a file entry to the tar archive (static method to avoid borrow issues).
    fn append_tar_entry(
        builder: &mut Builder<GzEncoder<File>>,
        name: &str,
        data: &[u8],
    ) -> Result<()> {
        let mut header = Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();

        builder.append_data(&mut header, name, data)?;
        Ok(())
    }

    /// Serialize data to SafeTensors format (static method to avoid borrow issues).
    fn serialize_safetensors_data(
        bytes: &[u8],
        patch_lengths: &[i32],
        entropies: &[f32],
    ) -> Result<Vec<u8>> {
        let num_patches = patch_lengths.len();

        let tensors: Vec<(&str, TensorView)> = vec![
            (
                "bytes",
                TensorView::new(Dtype::U8, vec![bytes.len()], bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to create bytes tensor: {e}"))?,
            ),
            (
                "patch_lengths",
                TensorView::new(
                    Dtype::I32,
                    vec![num_patches],
                    bytemuck::cast_slice(patch_lengths),
                )
                .map_err(|e| anyhow::anyhow!("Failed to create patch_lengths tensor: {e}"))?,
            ),
            (
                "entropies",
                TensorView::new(
                    Dtype::F32,
                    vec![entropies.len()],
                    bytemuck::cast_slice(entropies),
                )
                .map_err(|e| anyhow::anyhow!("Failed to create entropies tensor: {e}"))?,
            ),
        ];

        let mut metadata_map = HashMap::new();
        metadata_map.insert("format".to_string(), "blt_patches_v2".to_string());
        metadata_map.insert("num_patches".to_string(), num_patches.to_string());
        metadata_map.insert("total_bytes".to_string(), bytes.len().to_string());

        let serialized = serialize(tensors, &Some(metadata_map))?;
        Ok(serialized)
    }

    /// Finish writing and return statistics.
    ///
    /// Returns: (total_samples, total_shards)
    pub fn finish(mut self) -> Result<(usize, usize)> {
        // Finalize any remaining samples
        if self.current_sample_count > 0 {
            self.finalize_current_shard()?;
        }

        let total_shards = if self.total_samples > 0 {
            self.current_shard_idx
        } else {
            0
        };

        Ok((self.total_samples, total_shards))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_webdataset_writer_creates_shards() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = WebDatasetWriter::new(temp_dir.path().to_path_buf(), 2).unwrap();

        // Add 3 samples (should create 2 shards)
        for i in 0..3 {
            let metadata = serde_json::json!({
                "sample_id": i,
                "source": "test"
            });
            writer
                .add_sample(
                    &format!("{i:06}"),
                    b"test bytes",
                    &[5, 5],
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    &metadata,
                )
                .unwrap();
        }

        let (total_samples, total_shards) = writer.finish().unwrap();
        assert_eq!(total_samples, 3);
        assert_eq!(total_shards, 2);

        // Verify shard files exist
        assert!(temp_dir.path().join("shard_000000.tar.gz").exists());
        assert!(temp_dir.path().join("shard_000001.tar.gz").exists());
    }
}
