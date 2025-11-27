/// Polars-based Hugging Face dataset loader
///
/// This module provides a pure Rust implementation for loading Hugging Face datasets
/// using Polars and hf-hub, eliminating the need for Python-based burn-dataset.
///
/// Supports multiple formats:
/// - JSON files (e.g., TreeVGR-SFT-35K.json)
/// - Parquet files (auto-converted by Hugging Face)
/// - Arrow IPC files (from cache)
use crate::arrow_reader::{find_arrow_file, find_dataset_file, find_dataset_info_file};
use crate::hf_resolver::{get_api, parse_hf_uri, ParsedHfUri};
use anyhow::{Context, Result};
use hf_hub::{Repo, RepoType};
use polars::io::ipc::IpcReader;
use polars::prelude::*;
use serde::Deserialize;
use serde_json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetLoaderError {
    #[error("Dataset file not found: {0}")]
    FileNotFound(String),

    #[error("Failed to read dataset file: {0}")]
    ReadError(String),

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Hugging Face Hub error: {0}")]
    HubError(String),
}

/// Load a Hugging Face dataset using Polars
///
/// Automatically detects and loads JSON, Parquet, or Arrow IPC files.
/// Returns a Polars DataFrame with all dataset rows.
///
/// Uses lazy evaluation: keeps LazyFrame lazy as long as possible and only
/// collects when necessary. If `limit` is provided, uses `.slice()` to limit
/// rows before collecting, reducing memory usage.
pub fn load_hf_dataset(
    dataset_name: &str,
    split: Option<&str>,
    limit: Option<usize>,
) -> Result<DataFrame> {
    println!("Loading dataset {} with Polars...", dataset_name);

    // Try to find and load the dataset file
    // Priority: JSON > Parquet > Arrow IPC

    // 1. Try JSON file (e.g., TreeVGR-SFT-35K.json)
    match try_load_json(dataset_name, split, limit) {
        Ok(df) => {
            println!("✅ Loaded dataset from JSON file");
            return Ok(df);
        }
        Err(err) => {
            println!("  JSON path unavailable: {err}");
        }
    }

    // 2. Try Parquet files (auto-converted by HF)
    match try_load_parquet(dataset_name, split, limit) {
        Ok(df) => {
            println!("✅ Loaded dataset from Parquet file(s)");
            return Ok(df);
        }
        Err(err) => {
            println!("  Parquet path unavailable: {err}");
        }
    }

    // 3. Try Arrow IPC files (from cache)
    match try_load_arrow_ipc(dataset_name, split, limit) {
        Ok(df) => {
            println!("✅ Loaded dataset from Arrow IPC file");
            return Ok(df);
        }
        Err(err) => {
            println!("  Arrow IPC path unavailable: {err}");
        }
    }

    Err(DatasetLoaderError::FileNotFound(format!(
        "Could not find dataset files for {} in JSON, Parquet, or Arrow IPC format",
        dataset_name
    ))
    .into())
}

/// Try to load dataset from JSON file
///
/// Note: JSON loading is currently eager (loads entire file into memory).
/// For large JSON files, consider using Arrow IPC format instead.
fn try_load_json(
    dataset_name: &str,
    split: Option<&str>,
    limit: Option<usize>,
) -> Result<DataFrame> {
    // Try to find JSON file in Hugging Face cache or download it
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .context("Could not find home directory")?;

    // Common JSON file patterns
    let dataset_simple_name = dataset_name.split('/').last().unwrap_or(dataset_name);
    let json_patterns = vec![
        format!("{}.json", dataset_simple_name),
        format!("{}.json", dataset_name.replace('/', "-")),
        "data.json".to_string(),
        "train.json".to_string(),
    ];

    // Also try split-specific files
    let mut all_patterns = json_patterns;
    if let Some(s) = split {
        all_patterns.push(format!("{}.json", s));
        all_patterns.push(format!("{}-{}.json", dataset_simple_name, s));
    }

    // Try cache directory first
    let cache_base = Path::new(&home)
        .join(".cache")
        .join("huggingface")
        .join("datasets");
    let cache_name = dataset_name.replace('/', "___").to_lowercase();
    let dataset_cache = cache_base.join(&cache_name);

    // Look for JSON files in the dataset directory
    if dataset_cache.exists() {
        for pattern in &all_patterns {
            let json_path = dataset_cache.join(pattern);
            if json_path.exists() {
                println!("  Found JSON file: {}", json_path.display());
                return load_json_file(&json_path, limit);
            }
        }
    }

    if let Some(downloaded) = download_json_from_hub(dataset_name, &all_patterns)? {
        println!("  Downloaded JSON file: {}", downloaded.display());
        return load_json_file(&downloaded, limit);
    }

    Err(DatasetLoaderError::FileNotFound("No JSON file found in cache".to_string()).into())
}

/// Load a JSON file with Polars
///
/// Note: This is currently eager (loads entire file). For large files,
/// consider using Arrow IPC format which supports lazy loading.
fn load_json_file(path: &Path, limit: Option<usize>) -> Result<DataFrame> {
    // Read JSON file content
    let json_content = std::fs::read_to_string(path).context("Failed to read JSON file")?;

    let json_value: serde_json::Value =
        serde_json::from_str(&json_content).context("Failed to parse JSON")?;

    // Handle different JSON structures
    let items = match json_value {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Object(obj) => {
            // Could be a dataset with splits
            // Look for common split keys
            if let Some(train_data) = obj.get("train").and_then(|v| v.as_array()) {
                train_data.clone()
            } else if let Some(data) = obj.get("data").and_then(|v| v.as_array()) {
                data.clone()
            } else {
                // Single object - wrap in array
                vec![serde_json::Value::Object(obj)]
            }
        }
        _ => {
            return Err(DatasetLoaderError::ReadError(
                "JSON file is not an array or object".to_string(),
            )
            .into());
        }
    };

    // Apply limit if provided (before converting to DataFrame)
    let items_to_process = if let Some(limit_val) = limit {
        items.into_iter().take(limit_val).collect()
    } else {
        items
    };

    json_array_to_dataframe(&items_to_process)
}

fn download_json_from_hub(dataset_name: &str, candidates: &[String]) -> Result<Option<PathBuf>> {
    let api = get_api().map_err(|e| DatasetLoaderError::HubError(e.to_string()))?;
    let repo = api.repo(Repo::new(dataset_name.to_string(), RepoType::Dataset));

    for candidate in candidates {
        let mut attempts = Vec::with_capacity(2);
        attempts.push(candidate.clone());
        attempts.push(format!("data/{}", candidate));

        for attempt in attempts {
            match repo.get(&attempt) {
                Ok(path) => return Ok(Some(path)),
                Err(_) => continue,
            }
        }
    }

    Ok(None)
}

/// Convert JSON array to Polars DataFrame
fn json_array_to_dataframe(items: &[serde_json::Value]) -> Result<DataFrame> {
    if items.is_empty() {
        return Err(DatasetLoaderError::ReadError("Empty JSON array".to_string()).into());
    }

    // Get all unique keys from all items
    let mut all_keys = std::collections::HashSet::new();
    for item in items {
        if let Some(obj) = item.as_object() {
            all_keys.extend(obj.keys().cloned());
        }
    }

    // Build Columns for each column (Polars 0.46+ uses Column instead of Series)
    let mut columns_vec = Vec::new();

    for key in all_keys {
        let values: Vec<Option<serde_json::Value>> =
            items.iter().map(|item| item.get(&key).cloned()).collect();

        // Convert values to Series, then to Column (Polars 0.46+)
        let series = json_values_to_series(&key, &values)?;
        columns_vec.push(series.into_column());
    }

    DataFrame::new(columns_vec).map_err(|e| DatasetLoaderError::ReadError(e.to_string()).into())
}

/// Convert JSON values to a Polars Series
fn json_values_to_series(name: &str, values: &[Option<serde_json::Value>]) -> Result<Series> {
    // Determine the type from the first non-null value
    let first_value = values.iter().find_map(|v| v.as_ref());

    match first_value {
        Some(serde_json::Value::String(_)) => {
            let strings: Vec<Option<String>> = values
                .iter()
                .map(|v| {
                    v.as_ref()
                        .and_then(|val| val.as_str().map(|s| s.to_string()))
                })
                .collect();
            Ok(Series::new(name.into(), strings))
        }
        Some(serde_json::Value::Number(n)) => {
            if n.is_i64() {
                let ints: Vec<Option<i64>> = values
                    .iter()
                    .map(|v| v.as_ref().and_then(|val| val.as_i64()))
                    .collect();
                Ok(Series::new(name.into(), ints))
            } else if n.is_f64() {
                let floats: Vec<Option<f64>> = values
                    .iter()
                    .map(|v| v.as_ref().and_then(|val| val.as_f64()))
                    .collect();
                Ok(Series::new(name.into(), floats))
            } else {
                // Fallback to string
                let strings: Vec<Option<String>> = values
                    .iter()
                    .map(|v| v.as_ref().map(|val| val.to_string()))
                    .collect();
                Ok(Series::new(name.into(), strings))
            }
        }
        Some(serde_json::Value::Bool(_)) => {
            let bools: Vec<Option<bool>> = values
                .iter()
                .map(|v| v.as_ref().and_then(|val| val.as_bool()))
                .collect();
            Ok(Series::new(name.into(), bools))
        }
        Some(serde_json::Value::Array(first_arr)) => {
            // Determine inner element type from first array's first element
            let inner_type = first_arr.first();

            match inner_type {
                Some(serde_json::Value::String(_)) => {
                    // Array of strings - create proper list
                    let lists: Vec<Option<Series>> = values
                        .iter()
                        .map(|v| {
                            v.as_ref().and_then(|val| val.as_array()).map(|arr| {
                                let strings: Vec<Option<&str>> =
                                    arr.iter().map(|item| item.as_str()).collect();
                                Series::new("".into(), strings)
                            })
                        })
                        .collect();

                    // Convert to List series
                    let list_series: Vec<Option<Series>> = lists;
                    Ok(Series::new(name.into(), list_series))
                }
                Some(serde_json::Value::Number(n)) if n.is_f64() => {
                    // Array of floats
                    let lists: Vec<Option<Series>> = values
                        .iter()
                        .map(|v| {
                            v.as_ref().and_then(|val| val.as_array()).map(|arr| {
                                let floats: Vec<Option<f64>> =
                                    arr.iter().map(|item| item.as_f64()).collect();
                                Series::new("".into(), floats)
                            })
                        })
                        .collect();
                    Ok(Series::new(name.into(), lists))
                }
                Some(serde_json::Value::Number(_)) => {
                    // Array of integers
                    let lists: Vec<Option<Series>> = values
                        .iter()
                        .map(|v| {
                            v.as_ref().and_then(|val| val.as_array()).map(|arr| {
                                let ints: Vec<Option<i64>> =
                                    arr.iter().map(|item| item.as_i64()).collect();
                                Series::new("".into(), ints)
                            })
                        })
                        .collect();
                    Ok(Series::new(name.into(), lists))
                }
                _ => {
                    // Complex nested types: fallback to JSON strings
                    // This handles arrays of objects, nested arrays, etc.
                    let strings: Vec<Option<String>> = values
                        .iter()
                        .map(|v| {
                            v.as_ref()
                                .map(|val| serde_json::to_string(val).unwrap_or_default())
                        })
                        .collect();
                    Ok(Series::new(name.into(), strings))
                }
            }
        }
        Some(serde_json::Value::Object(_)) => {
            // For objects, serialize to JSON string
            let strings: Vec<Option<String>> = values
                .iter()
                .map(|v| {
                    v.as_ref()
                        .map(|val| serde_json::to_string(val).unwrap_or_default())
                })
                .collect();
            Ok(Series::new(name.into(), strings))
        }
        Some(serde_json::Value::Null) | None => {
            // All nulls - use string type as fallback
            let strings: Vec<Option<String>> = values.iter().map(|_| None).collect();
            Ok(Series::new(name.into(), strings))
        }
    }
}

/// Try to load the dataset from Hugging Face-hosted Parquet shards.
fn try_load_parquet(
    dataset_name: &str,
    split: Option<&str>,
    limit: Option<usize>,
) -> Result<DataFrame> {
    let split_name = split.unwrap_or("train");

    if let Ok(path) = find_dataset_file(dataset_name, split_name, "parquet") {
        println!("  Found local Parquet shard {}", path.display());
        return read_parquet_file(&path, limit);
    }

    if let Some(downloaded) = download_parquet_from_hub(dataset_name, split_name)? {
        println!("  Downloaded Parquet shard {}", downloaded.display());
        return read_parquet_file(&downloaded, limit);
    }

    Err(DatasetLoaderError::FileNotFound("No Parquet files found".to_string()).into())
}

/// Try to load dataset from Arrow IPC files (from cache)
///
/// Uses lazy evaluation: creates a LazyFrame and only collects when needed.
/// If `limit` is provided, uses `.slice()` to limit rows before collecting,
/// which is more memory-efficient than loading everything and then slicing.
fn try_load_arrow_ipc(
    dataset_name: &str,
    split: Option<&str>,
    limit: Option<usize>,
) -> Result<DataFrame> {
    let split_name = split.unwrap_or("train");
    let arrow_file = find_arrow_file(dataset_name, split_name)?;

    // Try lazy scan first, fall back to eager reader if IPC slice unsupported
    let mut df = match LazyFrame::scan_ipc(&arrow_file, ScanArgsIpc::default())
        .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
        .collect()
    {
        Ok(df) => df,
        Err(err) => {
            println!(
                "    Lazy IPC scan failed ({}), falling back to eager reader",
                err
            );
            let file = std::fs::File::open(&arrow_file)
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;
            IpcReader::new(file)
                .set_rechunk(true)
                .finish()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
        }
    };

    if let Some(limit_val) = limit {
        df = df.head(Some(limit_val));
    }

    Ok(df)
}

fn read_parquet_file(path: &Path, limit: Option<usize>) -> Result<DataFrame> {
    let mut df = match LazyFrame::scan_parquet(path, ScanArgsParquet::default())
        .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
        .collect()
    {
        Ok(df) => df,
        Err(err) => {
            println!(
                "    Lazy Parquet scan failed ({}), falling back to eager reader",
                err
            );
            let file = std::fs::File::open(path)
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;
            ParquetReader::new(file)
                .finish()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
        }
    };

    if let Some(limit_val) = limit {
        df = df.head(Some(limit_val));
    }

    Ok(df)
}

fn download_parquet_from_hub(dataset_name: &str, split: &str) -> Result<Option<PathBuf>> {
    let manifest = match load_dataset_info_manifest(dataset_name)? {
        Some(manifest) => manifest,
        None => return Ok(None),
    };

    let mut uris: Vec<_> = manifest
        .download_checksums
        .keys()
        .filter(|uri| uri.contains(split) && uri.ends_with(".parquet"))
        .filter_map(|uri| parse_hf_uri(uri))
        .collect();

    if uris.is_empty() {
        return Ok(None);
    }

    let api = get_api().map_err(|e| DatasetLoaderError::HubError(e.to_string()))?;

    for uri in uris.drain(..) {
        match download_hf_uri(api, &uri) {
            Ok(path) => return Ok(Some(path)),
            Err(err) => println!("    Failed to download {}: {}", uri.path, err),
        }
    }

    Ok(None)
}

#[derive(Debug, Deserialize)]
struct DatasetInfoManifest {
    #[serde(default)]
    download_checksums: HashMap<String, DownloadChecksum>,
}

#[derive(Debug, Deserialize)]
struct DownloadChecksum {
    #[serde(default)]
    _num_bytes: Option<u64>,
    #[serde(default)]
    _checksum: Option<String>,
}

fn load_dataset_info_manifest(dataset_name: &str) -> Result<Option<DatasetInfoManifest>> {
    if let Ok(path) = find_dataset_info_file(dataset_name) {
        if let Ok(manifest) = read_manifest_from_path(&path) {
            return Ok(Some(manifest));
        }
    }

    let api = get_api().map_err(|e| DatasetLoaderError::HubError(e.to_string()))?;
    let repo = api.repo(Repo::new(dataset_name.to_string(), RepoType::Dataset));
    match repo.get("dataset_info.json") {
        Ok(path) => read_manifest_from_path(&path).map(Some),
        Err(err) => {
            println!("    Unable to fetch dataset_info.json: {}", err);
            Ok(None)
        }
    }
}

fn read_manifest_from_path(path: &Path) -> Result<DatasetInfoManifest> {
    let contents =
        std::fs::read_to_string(path).map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;
    serde_json::from_str(&contents).map_err(|e| DatasetLoaderError::ReadError(e.to_string()).into())
}

/// Download a file from HuggingFace using a parsed URI
fn download_hf_uri(api: &hf_hub::api::sync::Api, uri: &ParsedHfUri) -> Result<PathBuf> {
    let repo = api.repo(Repo::with_revision(
        uri.repo_id.clone(),
        RepoType::Dataset,
        uri.revision.clone(),
    ));
    repo.get(&uri.path)
        .map_err(|e| DatasetLoaderError::HubError(e.to_string()).into())
}

/// Convert a Polars DataFrame row to serde_json::Value
/// This maintains compatibility with existing dataset processors
pub fn row_to_json_value(df: &DataFrame, row_idx: usize) -> Result<serde_json::Value> {
    if row_idx >= df.height() {
        return Err(DatasetLoaderError::ReadError(format!(
            "Row index {} out of bounds (dataset has {} rows)",
            row_idx,
            df.height()
        ))
        .into());
    }

    let mut obj = serde_json::Map::new();

    for col_name in df.get_column_names() {
        let series = df
            .column(col_name)
            .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;

        let value = series_value_to_json(series, row_idx)?;
        obj.insert(col_name.to_string(), value);
    }

    Ok(serde_json::Value::Object(obj))
}

/// Convert a Column value at a specific index to JSON
/// 
/// In Polars 0.46+, DataFrame.column() returns Column instead of Series.
/// This function handles the Column type and delegates to series_to_json.
fn series_value_to_json(column: &polars::frame::column::Column, idx: usize) -> Result<serde_json::Value> {
    // Get the underlying series from the column
    let series = column.as_materialized_series();
    series_to_json(series, idx)
}

/// Convert a Series value at a specific index to JSON (internal helper)
fn series_to_json(series: &Series, idx: usize) -> Result<serde_json::Value> {
    match series.dtype() {
        DataType::String => {
            let s = series
                .str()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
                .get(idx)
                .map(|s| s.to_string())
                .unwrap_or_default();
            Ok(serde_json::Value::String(s))
        }
        DataType::Int64 => {
            let v = series
                .i64()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
                .get(idx)
                .unwrap_or(0);
            Ok(serde_json::Value::Number(v.into()))
        }
        DataType::Float64 => {
            let v = series
                .f64()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
                .get(idx)
                .unwrap_or(0.0);
            Ok(serde_json::Value::Number(
                serde_json::Number::from_f64(v).unwrap_or(serde_json::Number::from(0)),
            ))
        }
        DataType::Boolean => {
            let v = series
                .bool()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
                .get(idx)
                .unwrap_or(false);
            Ok(serde_json::Value::Bool(v))
        }
        DataType::List(_) => {
            // For lists, properly extract the inner values
            // Strategy: slice the series to get just this row, then explode to get inner values
            // Get a single-row slice containing just the list at this index
            let row_slice = series.slice(idx as i64, 1);
            let row_list_series = row_slice
                .list()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;

            // Explode the list to get all inner values as a flat Series
            // This gives us all values from the list at this index
            let exploded = row_list_series
                .explode()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;

            // Now convert each value in the exploded series to JSON
            let mut json_array = Vec::new();
            for i in 0..exploded.len() {
                let value = series_to_json(&exploded, i)?;
                json_array.push(value);
            }

            Ok(serde_json::Value::Array(json_array))
        }
        DataType::Struct(_) => {
            // For structs, convert to JSON object using struct_() fields
            let struct_series = series
                .struct_()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;
            
            // In Polars 0.46+, iterate over fields using fields_as_series()
            let field_series = struct_series.fields_as_series();
            
            let mut map = serde_json::Map::new();
            for field in field_series.iter() {
                let field_name = field.name().to_string();
                let value = series_to_json(field, idx)?;
                map.insert(field_name, value);
            }

            Ok(serde_json::Value::Object(map))
        }
        _ => {
            // Fallback: convert to string
            let s = format!(
                "{:?}",
                series
                    .get(idx)
                    .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
            );
            Ok(serde_json::Value::String(s))
        }
    }
}

/// Extract a column as bytes (e.g. for images or binary data) without full JSON serialization
pub fn get_column_as_bytes(df: &DataFrame, row_idx: usize, column_name: &str) -> Result<Vec<u8>> {
    let column = df
        .column(column_name)
        .map_err(|_| DatasetLoaderError::ReadError(format!("Column {} not found", column_name)))?;

    // Get the underlying series from the column (Polars 0.46+)
    let series = column.as_materialized_series();

    if row_idx >= series.len() {
        return Err(
            DatasetLoaderError::ReadError(format!("Row index {} out of bounds", row_idx)).into(),
        );
    }

    match series.dtype() {
        DataType::Binary => {
            let binary_series = series
                .binary()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;
            match binary_series.get(row_idx) {
                Some(bytes) => Ok(bytes.to_vec()),
                None => Ok(Vec::new()),
            }
        }
        DataType::String => {
            let string_series = series
                .str()
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;
            match string_series.get(row_idx) {
                Some(s) => Ok(s.as_bytes().to_vec()),
                None => Ok(Vec::new()),
            }
        }
        DataType::List(_) => {
            // Handle List<Binary> or List<String> or other lists
            // We serialize the list content to JSON bytes as a fallback
            let json_val = series_value_to_json(column, row_idx)?;
            Ok(serde_json::to_vec(&json_val)?)
        }
        _ => {
            // Fallback: string representation
            let val_str = format!("{:?}", series.get(row_idx).unwrap());
            Ok(val_str.into_bytes())
        }
    }
}

/// Extract a text-friendly value from a column, normalizing to UTF-8 when possible
pub fn extract_text_value(
    df: &DataFrame,
    row_idx: usize,
    column_name: &str,
) -> Result<Option<String>> {
    let column = df
        .column(column_name)
        .map_err(|_| DatasetLoaderError::ReadError(format!("Column {} not found", column_name)))?;

    // Get the underlying series from the column (Polars 0.46+)
    let series = column.as_materialized_series();

    if row_idx >= series.len() {
        return Ok(None);
    }

    let value = match series.dtype() {
        DataType::String => series
            .str()
            .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
            .get(row_idx)
            .map(|s| s.to_string()),
        DataType::Binary => series
            .binary()
            .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
            .get(row_idx)
            .and_then(|bytes| String::from_utf8(bytes.to_vec()).ok()),
        DataType::Boolean => series
            .bool()
            .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
            .get(row_idx)
            .map(|b| b.to_string()),
        DataType::Int64 => series
            .i64()
            .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
            .get(row_idx)
            .map(|v| v.to_string()),
        DataType::Float64 => series
            .f64()
            .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
            .get(row_idx)
            .map(|v| v.to_string()),
        _ => {
            let value = series
                .get(row_idx)
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;
            Some(format!("{:?}", value))
        }
    };

    Ok(value)
}

/// Extract binary content from a column, falling back to UTF-8 encoding when needed
pub fn extract_binary_value(
    df: &DataFrame,
    row_idx: usize,
    column_name: &str,
) -> Result<Option<Vec<u8>>> {
    let column = df
        .column(column_name)
        .map_err(|_| DatasetLoaderError::ReadError(format!("Column {} not found", column_name)))?;

    // Get the underlying series from the column (Polars 0.46+)
    let series = column.as_materialized_series();

    if row_idx >= series.len() {
        return Ok(None);
    }

    let value = match series.dtype() {
        DataType::Binary => series
            .binary()
            .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
            .get(row_idx)
            .map(|bytes| bytes.to_vec()),
        DataType::String => series
            .str()
            .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?
            .get(row_idx)
            .map(|s| s.as_bytes().to_vec()),
        DataType::List(_) | DataType::Struct(_) => {
            let json = series_value_to_json(column, row_idx)?;
            let bytes = serde_json::to_vec(&json)?;
            Some(bytes)
        }
        _ => {
            let value = series
                .get(row_idx)
                .map_err(|e| DatasetLoaderError::ReadError(e.to_string()))?;
            Some(format!("{:?}", value).into_bytes())
        }
    };

    Ok(value)
}

/// Extract a list column as JSON, returning None if the column is not a list
pub fn extract_list_json(
    df: &DataFrame,
    row_idx: usize,
    column_name: &str,
) -> Result<Option<serde_json::Value>> {
    let column = df
        .column(column_name)
        .map_err(|_| DatasetLoaderError::ReadError(format!("Column {} not found", column_name)))?;

    if !matches!(column.dtype(), DataType::List(_)) {
        return Ok(None);
    }

    let value = series_value_to_json(column, row_idx)?;
    if value.is_array() {
        Ok(Some(value))
    } else {
        Ok(None)
    }
}

/// Extract a struct column as JSON, returning None if the column is not a struct
pub fn extract_struct_json(
    df: &DataFrame,
    row_idx: usize,
    column_name: &str,
) -> Result<Option<serde_json::Value>> {
    let column = df
        .column(column_name)
        .map_err(|_| DatasetLoaderError::ReadError(format!("Column {} not found", column_name)))?;

    if !matches!(column.dtype(), DataType::Struct(_)) {
        return Ok(None);
    }

    let value = series_value_to_json(column, row_idx)?;
    if value.is_object() {
        Ok(Some(value))
    } else {
        Ok(None)
    }
}
