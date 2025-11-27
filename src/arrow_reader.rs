/// Arrow file reader for accessing data from Hugging Face dataset Arrow files
///
/// Hugging Face datasets are stored as Arrow IPC files. This module provides
/// functionality to read image data and other binary content directly from
/// Arrow files when burn-dataset's SQLite conversion doesn't preserve it.
///
/// Arrow files are typically located at:
/// ~/.cache/huggingface/datasets/{dataset_name}/default/{version}/{hash}/{filename}.arrow
///
/// This uses Polars, a high-level Rust DataFrame library built on Arrow.
use anyhow::{Context, Result};
use polars::prelude::*;
use serde_json;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArrowError {
    #[error("Arrow file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("Failed to read Arrow file: {0}")]
    ReadError(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Row index out of bounds: {0}")]
    RowOutOfBounds(usize),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Locate the Hugging Face cache directory for the given dataset.
pub fn dataset_cache_dir(dataset_name: &str) -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .context("Could not find home directory")?;

    let cache_base = Path::new(&home)
        .join(".cache")
        .join("huggingface")
        .join("datasets");
    let cache_name = dataset_name.replace('/', "___").to_lowercase();
    let mut dataset_cache = cache_base.join(&cache_name);

    if !dataset_cache.exists() {
        if let Some(entry) = fs::read_dir(&cache_base)?.filter_map(|e| e.ok()).find(|e| {
            let name = e.file_name().to_string_lossy().to_lowercase();
            name.contains(&cache_name) || cache_name.contains(&name)
        }) {
            dataset_cache = entry.path();
        } else {
            return Err(ArrowError::FileNotFound(cache_base.join(&cache_name)).into());
        }
    }

    Ok(dataset_cache)
}

fn find_dataset_file_recursively(dir: &Path, split: &str, extension: &str) -> Option<PathBuf> {
    for entry in fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_dataset_file_recursively(&path, split, extension) {
                return Some(found);
            }
        } else if path
            .extension()
            .and_then(|s| s.to_str())
            .map(|ext| ext.eq_ignore_ascii_case(extension))
            .unwrap_or(false)
        {
            let filename = path.file_name()?.to_string_lossy().to_lowercase();
            if filename.contains(&split.to_lowercase()) {
                return Some(path);
            }
        }
    }
    None
}

fn find_named_file_recursively(dir: &Path, target: &str) -> Option<PathBuf> {
    for entry in fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_named_file_recursively(&path, target) {
                return Some(found);
            }
        } else if path.file_name().and_then(|n| n.to_str()) == Some(target) {
            return Some(path);
        }
    }
    None
}

/// Generic helper to find a dataset file with the requested extension.
pub fn find_dataset_file(dataset_name: &str, split: &str, extension: &str) -> Result<PathBuf> {
    let dataset_cache = dataset_cache_dir(dataset_name)?;
    find_dataset_file_recursively(&dataset_cache, split, extension).ok_or_else(|| {
        ArrowError::FileNotFound(dataset_cache.join(format!("*.{}", extension))).into()
    })
}

/// Find the Arrow file for a given dataset and split.
pub fn find_arrow_file(dataset_name: &str, split: &str) -> Result<PathBuf> {
    let path = find_dataset_file(dataset_name, split, "arrow")?;
    println!("  Using Arrow file {}", path.display());
    Ok(path)
}

/// Find dataset_info.json for the dataset.
pub fn find_dataset_info_file(dataset_name: &str) -> Result<PathBuf> {
    let dataset_cache = dataset_cache_dir(dataset_name)?;
    find_named_file_recursively(&dataset_cache, "dataset_info.json")
        .ok_or_else(|| ArrowError::FileNotFound(dataset_cache.join("dataset_info.json")).into())
}

/// Arrow file reader using Polars
pub struct ArrowReader {
    df: DataFrame,
}

impl ArrowReader {
    /// Open an Arrow file for reading using Polars
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let df = LazyFrame::scan_ipc(path.as_ref(), ScanArgsIpc::default())
            .map_err(|e| ArrowError::ReadError(format!("Failed to scan IPC file: {}", e)))?
            .collect()
            .map_err(|e| ArrowError::ReadError(format!("Failed to collect DataFrame: {}", e)))?;

        Ok(Self { df })
    }

    /// Get the total number of rows
    pub fn num_rows(&self) -> usize {
        self.df.height()
    }

    /// Get the schema (column names)
    pub fn column_names(&self) -> Vec<String> {
        self.df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Read a specific column value from a row
    pub fn read_column_value(&self, row_idx: usize, column_name: &str) -> Result<Vec<u8>> {
        if row_idx >= self.num_rows() {
            return Err(ArrowError::RowOutOfBounds(row_idx).into());
        }

        let column = self
            .df
            .column(column_name)
            .map_err(|_| ArrowError::ColumnNotFound(column_name.to_string()))?;

        // Get the underlying series from the column (Polars 0.46+)
        let series = column.as_materialized_series();

        // Extract value based on series dtype
        match series.dtype() {
            DataType::String => {
                let value = series
                    .str()
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?
                    .get(row_idx)
                    .ok_or_else(|| ArrowError::RowOutOfBounds(row_idx))?;
                Ok(value.as_bytes().to_vec())
            }
            DataType::Binary => {
                let value = series
                    .binary()
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?
                    .get(row_idx)
                    .ok_or_else(|| ArrowError::RowOutOfBounds(row_idx))?;
                Ok(value.to_vec())
            }
            DataType::List(_) => {
                let list_series = extract_list_series(column, row_idx)?;
                let json_value = list_to_json(&list_series)?;
                Ok(serde_json::to_vec(&json_value)
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?)
            }
            _ => {
                // Try to convert to string and then to bytes
                let value = series
                    .get(row_idx)
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?;
                let string_repr = format!("{:?}", value);
                Ok(string_repr.into_bytes())
            }
        }
    }

    /// Read image data from the images column for a specific row
    /// Returns a list of image paths or binary data
    pub fn read_images(&self, row_idx: usize) -> Result<Vec<ImageData>> {
        if row_idx >= self.num_rows() {
            return Err(ArrowError::RowOutOfBounds(row_idx).into());
        }

        let column = self
            .df
            .column("images")
            .map_err(|_| ArrowError::ColumnNotFound("images".to_string()))?;

        // Get the underlying series from the column (Polars 0.46+)
        let series = column.as_materialized_series();

        match series.dtype() {
            DataType::List(_) => {
                let list_series = extract_list_series(column, row_idx)?;
                return list_series_to_images(&list_series);
            }
            DataType::Binary => {
                let bytes = series
                    .binary()
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?
                    .get(row_idx)
                    .ok_or_else(|| ArrowError::RowOutOfBounds(row_idx))?;
                return Ok(vec![ImageData::Bytes(bytes.to_vec())]);
            }
            DataType::String => {
                let value = series
                    .str()
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?
                    .get(row_idx)
                    .ok_or_else(|| ArrowError::RowOutOfBounds(row_idx))?;

                if let Ok(paths) = serde_json::from_str::<Vec<String>>(value) {
                    return Ok(paths.into_iter().map(ImageData::Path).collect());
                }

                return Ok(vec![ImageData::Path(value.to_string())]);
            }
            _ => {}
        }

        // Fallback: read column as bytes and attempt JSON parsing
        let images_data = self.read_column_value(row_idx, "images")?;
        if let Ok(json_str) = String::from_utf8(images_data.clone()) {
            if let Ok(paths) = serde_json::from_str::<Vec<String>>(&json_str) {
                return Ok(paths.into_iter().map(ImageData::Path).collect());
            }
        }

        Ok(vec![ImageData::Bytes(images_data)])
    }
}

fn extract_list_series(column: &polars::frame::column::Column, row_idx: usize) -> Result<Series> {
    // Get the underlying series from the column (Polars 0.46+)
    let series = column.as_materialized_series();
    let list_chunked = series
        .list()
        .map_err(|e| ArrowError::ReadError(e.to_string()))?;
    let list_series = list_chunked
        .get_as_series(row_idx)
        .ok_or_else(|| ArrowError::RowOutOfBounds(row_idx))?;

    flatten_list_series(list_series)
}

fn flatten_list_series(series: Series) -> Result<Series> {
    let mut current = series;

    loop {
        match current.dtype() {
            DataType::List(_) => {
                let exploded = current
                    .list()
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?
                    .explode()
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?;
                current = exploded;
            }
            _ => break Ok(current),
        }
    }
}

fn list_series_to_images(list_series: &Series) -> Result<Vec<ImageData>> {
    match list_series.dtype() {
        DataType::String => {
            let string_series = list_series
                .str()
                .map_err(|e| ArrowError::ReadError(e.to_string()))?;
            let mut images = Vec::with_capacity(string_series.len());
            for value in string_series.into_iter().flatten() {
                images.push(ImageData::Path(value.to_string()));
            }
            Ok(images)
        }
        DataType::Binary => {
            let binary_series = list_series
                .binary()
                .map_err(|e| ArrowError::ReadError(e.to_string()))?;
            let mut images = Vec::with_capacity(binary_series.len());
            for value in binary_series.into_iter().flatten() {
                images.push(ImageData::Bytes(value.to_vec()));
            }
            Ok(images)
        }
        DataType::Struct(_) => {
            let struct_chunked = list_series
                .struct_()
                .map_err(|e| ArrowError::ReadError(e.to_string()))?;
            // In Polars 0.46+, use fields_as_series() instead of fields()
            let fields = struct_chunked.fields_as_series();
            let mut images = Vec::with_capacity(struct_chunked.len());

            for idx in 0..struct_chunked.len() {
                if let Some(path) = get_string_field(&fields, "path", idx)? {
                    images.push(ImageData::Path(path));
                    continue;
                }

                if let Some(data_uri) = get_string_field(&fields, "data", idx)? {
                    images.push(ImageData::Path(data_uri));
                    continue;
                }

                if let Some(bytes) = get_binary_field(&fields, "bytes", idx)? {
                    images.push(ImageData::Bytes(bytes));
                    continue;
                }

                let struct_json = struct_row_to_json(&fields, idx)?;
                images.push(ImageData::Bytes(
                    serde_json::to_vec(&struct_json)
                        .map_err(|e| ArrowError::ReadError(e.to_string()))?,
                ));
            }

            Ok(images)
        }
        DataType::List(_) => {
            let json_value = list_to_json(list_series)?;
            if let serde_json::Value::Array(items) = json_value {
                let mut images = Vec::with_capacity(items.len());
                for item in items {
                    if let Some(path) = item.get("path").and_then(|v| v.as_str()) {
                        images.push(ImageData::Path(path.to_string()));
                    } else if let Some(data_uri) = item.get("data").and_then(|v| v.as_str()) {
                        images.push(ImageData::Path(data_uri.to_string()));
                    } else {
                        let bytes = serde_json::to_vec(&item)
                            .map_err(|e| ArrowError::ReadError(e.to_string()))?;
                        images.push(ImageData::Bytes(bytes));
                    }
                }
                Ok(images)
            } else {
                Ok(vec![ImageData::Bytes(
                    serde_json::to_vec(&json_value)
                        .map_err(|e| ArrowError::ReadError(e.to_string()))?,
                )])
            }
        }
        _ => {
            let json_value = list_to_json(list_series)?;
            Ok(vec![ImageData::Bytes(
                serde_json::to_vec(&json_value)
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?,
            )])
        }
    }
}

fn get_string_field(fields: &[Series], name: &str, idx: usize) -> Result<Option<String>> {
    if let Some(field) = fields.iter().find(|s| s.name() == name) {
        let s_chunked = field
            .str()
            .map_err(|e| ArrowError::ReadError(e.to_string()))?;
        Ok(s_chunked.get(idx).map(|s| s.to_string()))
    } else {
        Ok(None)
    }
}

fn get_binary_field(fields: &[Series], name: &str, idx: usize) -> Result<Option<Vec<u8>>> {
    if let Some(field) = fields.iter().find(|s| s.name() == name) {
        let binary_chunked = field
            .binary()
            .map_err(|e| ArrowError::ReadError(e.to_string()))?;
        Ok(binary_chunked.get(idx).map(|bytes| bytes.to_vec()))
    } else {
        Ok(None)
    }
}

fn struct_row_to_json(fields: &[Series], idx: usize) -> Result<serde_json::Value> {
    let mut map = serde_json::Map::new();

    for field in fields {
        let name = field.name();
        let value = match field.dtype() {
            DataType::String => field
                .str()
                .map_err(|e| ArrowError::ReadError(e.to_string()))?
                .get(idx)
                .map(|s| serde_json::Value::String(s.to_string()))
                .unwrap_or(serde_json::Value::Null),
            DataType::Binary => {
                if let Some(bytes) = field
                    .binary()
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?
                    .get(idx)
                {
                    use base64::{engine::general_purpose, Engine as _};
                    let encoded = general_purpose::STANDARD.encode(bytes);
                    serde_json::Value::String(encoded)
                } else {
                    serde_json::Value::Null
                }
            }
            DataType::Int64 => field
                .i64()
                .map_err(|e| ArrowError::ReadError(e.to_string()))?
                .get(idx)
                .map(|v| serde_json::Value::Number(v.into()))
                .unwrap_or(serde_json::Value::Null),
            DataType::Float64 => field
                .f64()
                .map_err(|e| ArrowError::ReadError(e.to_string()))?
                .get(idx)
                .and_then(|v| serde_json::Number::from_f64(v))
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            DataType::Boolean => field
                .bool()
                .map_err(|e| ArrowError::ReadError(e.to_string()))?
                .get(idx)
                .map(serde_json::Value::Bool)
                .unwrap_or(serde_json::Value::Null),
            DataType::Struct(_) | DataType::List(_) => field
                .get(idx)
                .map(|v| serde_json::Value::String(format!("{:?}", v)))
                .map_err(|e| ArrowError::ReadError(e.to_string()))?,
            _ => serde_json::Value::String(
                field
                    .get(idx)
                    .map(|v| format!("{:?}", v))
                    .unwrap_or_else(|_| "null".to_string()),
            ),
        };

        map.insert(name.to_string(), value);
    }

    Ok(serde_json::Value::Object(map))
}

/// Convert a Polars Series (list) to JSON
fn list_to_json(list_value: &Series) -> Result<serde_json::Value> {
    let mut values = Vec::new();

    match list_value.dtype() {
        DataType::String => {
            let string_series = list_value
                .str()
                .map_err(|e| ArrowError::ReadError(e.to_string()))?;
            for i in 0..string_series.len() {
                if let Some(s) = string_series.get(i) {
                    values.push(serde_json::Value::String(s.to_string()));
                }
            }
        }
        DataType::Binary => {
            let binary_series = list_value
                .binary()
                .map_err(|e| ArrowError::ReadError(e.to_string()))?;
            for i in 0..binary_series.len() {
                if let Some(bytes) = binary_series.get(i) {
                    // Try to decode as string, otherwise base64 encode
                    if let Ok(s) = String::from_utf8(bytes.to_vec()) {
                        values.push(serde_json::Value::String(s));
                    } else {
                        // Base64 encode binary data
                        use base64::{engine::general_purpose, Engine as _};
                        let encoded = general_purpose::STANDARD.encode(bytes);
                        values.push(serde_json::Value::String(format!(
                            "data:image/png;base64,{}",
                            encoded
                        )));
                    }
                }
            }
        }
        _ => {
            // For other types, convert to string representation
            for i in 0..list_value.len() {
                let value = list_value
                    .get(i)
                    .map_err(|e| ArrowError::ReadError(e.to_string()))?;
                values.push(serde_json::Value::String(format!("{:?}", value)));
            }
        }
    }

    Ok(serde_json::Value::Array(values))
}

use crate::hf_resolver::{HfResolver, ZERO_IMAGE_SIZE};

/// Image data representation
#[derive(Debug, Clone)]
pub enum ImageData {
    /// Image stored as a file path
    Path(String),
    /// Image stored as raw bytes
    Bytes(Vec<u8>),
}

impl ImageData {
    /// Get the image bytes, reading from path if needed
    ///
    /// If a path cannot be resolved locally, returns a zero tensor (black image)
    /// to preserve shape without injecting semantic noise into embeddings.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.to_bytes_with_resolver(None)
    }

    /// Get the image bytes with optional HfResolver for remote paths
    ///
    /// # Arguments
    /// * `resolver` - Optional HfResolver for downloading from HuggingFace
    ///
    /// # Returns
    /// * Image bytes, or zero tensor if path cannot be resolved
    pub fn to_bytes_with_resolver(&self, resolver: Option<&HfResolver>) -> Result<Vec<u8>> {
        match self {
            ImageData::Path(path) => {
                // Try to read from local file system first
                if Path::new(path).exists() {
                    return std::fs::read(path)
                        .map_err(|e| anyhow::anyhow!("Failed to read image: {}", e));
                }

                // Try HfResolver if provided
                if let Some(resolver) = resolver {
                    match resolver.resolve(path, true)? {
                        Some(bytes) => return Ok(bytes),
                        None => {
                            // skip_missing mode - return zero tensor
                            return Ok(vec![0u8; ZERO_IMAGE_SIZE]);
                        }
                    }
                }

                // Fallback: return zero tensor instead of error
                // This preserves tensor shape without semantic noise
                eprintln!(
                    "Warning: Image path not found, using zero tensor: {}",
                    path
                );
                Ok(vec![0u8; ZERO_IMAGE_SIZE])
            }
            ImageData::Bytes(bytes) => Ok(bytes.clone()),
        }
    }
}

/// Convenience function to read image data from a dataset Arrow file
pub fn read_dataset_images(
    dataset_name: &str,
    split: &str,
    row_idx: usize,
) -> Result<Vec<ImageData>> {
    let arrow_file = find_arrow_file(dataset_name, split)?;
    let reader = ArrowReader::open(arrow_file)?;
    reader.read_images(row_idx)
}
