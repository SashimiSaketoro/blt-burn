use anyhow::Result;
use polars::prelude::DataFrame;
use std::path::Path;

use crate::generic_processor::GenericDatasetProcessor;

/// Common trait for dataset-specific item processors
pub trait DatasetItemProcessor {
    /// Process a dataset item and extract multimodal segments
    ///
    /// Takes the DataFrame and row index to allow efficient access to binary columns
    /// without full serialization/deserialization overhead.
    fn process_item(&self, df: &DataFrame, row_idx: usize) -> Result<Vec<ModalitySegment>>;

    /// Get dataset-specific metadata
    fn dataset_metadata(&self) -> DatasetMetadata;

    /// Download referenced files if needed
    fn download_references(
        &self,
        df: &DataFrame,
        row_idx: usize,
        cache_dir: &Path,
    ) -> Result<Vec<DownloadedFile>>;

    /// Prefetch all references for a dataset before per-item processing.
    /// Default implementation is a no-op.
    fn prefetch_references(&self, _df: &DataFrame, _cache_dir: &Path) -> Result<()> {
        Ok(())
    }
}

/// Represents a modality segment with its type and content
#[derive(Debug, Clone)]
pub struct ModalitySegment {
    pub modality_type: ModalityType,
    pub content: SegmentContent,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub enum ModalityType {
    Text,
    Image,
    Audio,
    Video,
    Document,
    Unknown(String),
}

#[derive(Debug, Clone)]
pub enum SegmentContent {
    /// Direct byte content
    Bytes(Vec<u8>),
    /// Path to a file (local or to be downloaded)
    FilePath(String),
    /// URL to download
    Url(String),
}

#[derive(Debug, Clone)]
pub struct DownloadedFile {
    pub url: String,
    pub local_path: std::path::PathBuf,
    pub file_type: String,
    pub size_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub requires_download: bool,
    pub modalities: Vec<ModalityType>,
}

/// Helper to detect dataset type and create appropriate processor
pub fn create_processor(dataset_name: &str) -> Result<Box<dyn DatasetItemProcessor + Send + Sync>> {
    Ok(Box::new(GenericDatasetProcessor::new(dataset_name)))
}
