//! Output format writers for BLT-Burn ingestion pipeline.
//!
//! Supports multiple output formats:
//! - SafeTensors (default): Individual .safetensors files with .hypergraph.db sidecars
//! - WebDataset: Sharded .tar.gz archives for PyTorch streaming

pub mod webdataset;

pub use webdataset::WebDatasetWriter;

