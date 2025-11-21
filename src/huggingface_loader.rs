use anyhow::Result;
use burn_wgpu::{Wgpu, WgpuDevice};
use polars::prelude::DataFrame;
use std::fs;
use std::path::Path;

use crate::dataset_helpers::{create_processor, ModalitySegment, SegmentContent};
use crate::model::LMTransformer;
use crate::polars_dataset_loader;
use crate::pretokenize::{ByteSegment, SegmentMetadata};
use crate::sidecar::{EdgeData, HypergraphBuilder, NodeData};
use crate::tokenizer::BltTokenizer;

/// Process a Hugging Face dataset with multimodal content
/// This function loads the dataset, processes each field through appropriate pre-tokenizers,
/// and returns processed segments ready for the main ingest pipeline
pub fn process_hf_dataset(
    dataset_name: &str,
    output_dir: &Path,
    _model: &LMTransformer<Wgpu>,
    _device: &WgpuDevice,
    _tokenizer: &BltTokenizer,
    _threshold: f64,
    subset: Option<&str>,
    limit: Option<usize>,
) -> Result<()> {
    println!("🤗 Loading dataset from Hugging Face: {}", dataset_name);

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    let processor = create_processor(dataset_name)?;
    let metadata = processor.dataset_metadata();

    println!("Dataset: {} v{}", metadata.name, metadata.version);
    println!("Description: {}", metadata.description);
    println!("Modalities: {:?}", metadata.modalities);

    // Load dataset via Polars (pure Rust)
    let df = polars_dataset_loader::load_hf_dataset(
        dataset_name,
        Some(subset.unwrap_or("train")),
        limit,
    )?;

    let num_entries = df.height();
    let process_limit = limit.unwrap_or(num_entries).min(num_entries);
    println!("Processing {} of {} entries...", process_limit, num_entries);

    // Create cache directory for downloads
    let cache_dir = output_dir.join(".cache");
    std::fs::create_dir_all(&cache_dir)?;

    // Prefetch all references once so subsequent per-row downloads hit the cache
    processor.prefetch_references(&df, &cache_dir)?;

    // Process each dataset item
    for row_idx in 0..process_limit {
        println!("\nProcessing entry {}/{}...", row_idx + 1, process_limit);

        let task_id = infer_task_id(&df, row_idx).unwrap_or_else(|| format!("item_{}", row_idx));

        println!("Processing: {}", task_id);

        // Step 1: Download any referenced files FIRST (before processing)
        // This ensures files are available when we process segments
        let downloaded = processor.download_references(&df, row_idx, &cache_dir)?;
        if !downloaded.is_empty() {
            println!("  Downloaded/resolved {} references", downloaded.len());
            for file in &downloaded {
                println!("    - {} [{}]", file.local_path.display(), file.file_type);
            }
        }

        // Step 2: Process item via processor (dataset-specific or generic)
        let segments = processor.process_item(&df, row_idx)?;
        println!("  Extracted {} segments", segments.len());

        // Step 3: Convert to ByteSegments and build hypergraph
        let (byte_segments, hypergraph) =
            convert_to_byte_segments(segments, &task_id, dataset_name, &cache_dir)?;

        // Save hypergraph metadata
        let metadata_path = output_dir.join(format!("{}.hypergraph.db", task_id));
        hypergraph.save_to_sqlite(&metadata_path)?;
        println!("✅ Saved hypergraph: {}", metadata_path.display());

        // In production, we would:
        // 1. Run each segment through appropriate pre-tokenizers
        // 2. Tokenize and process through the model
        // 3. Save embeddings to safetensors

        println!(
            "  Would process {} byte segments through model",
            byte_segments.len()
        );
        println!(
            "  Would save to: {}/{}.safetensors",
            output_dir.display(),
            task_id
        );
    }

    println!("\n✅ Dataset processing complete!");
    Ok(())
}

/// Convert ModalitySegments to ByteSegments and build hypergraph
fn convert_to_byte_segments(
    segments: Vec<ModalitySegment>,
    task_id: &str,
    dataset_name: &str,
    cache_dir: &Path,
) -> Result<(Vec<ByteSegment>, crate::sidecar::HypergraphSidecar)> {
    let mut builder = HypergraphBuilder::new();
    let mut byte_segments = Vec::new();

    // Create trunk node
    let trunk_idx = builder.add_node(NodeData::Trunk {
        source_hash: format!("{}:{}", dataset_name, task_id),
        total_bytes: 0, // Will be updated
    });

    // Group segments by modality
    let mut modality_groups: std::collections::HashMap<String, Vec<(usize, ModalitySegment)>> =
        std::collections::HashMap::new();

    for (idx, segment) in segments.into_iter().enumerate() {
        let modality_key = format!("{:?}", segment.modality_type);
        modality_groups
            .entry(modality_key)
            .or_insert_with(Vec::new)
            .push((idx, segment));
    }

    // Process each modality group
    let dataset_slug = dataset_name.replace('/', "_");

    for (modality_name, segments) in modality_groups {
        // Create branch node for this modality
        let branch_idx = builder.add_node(NodeData::Branch {
            label: modality_name.clone(),
            modality: modality_name.clone(),
        });

        // Connect branch to trunk
        builder.add_hyperedge(
            vec![trunk_idx, branch_idx],
            EdgeData {
                label: "contains".to_string(),
                weight: 1.0,
            },
        );

        // Process segments in this modality
        for (idx, segment) in segments {
            let bytes = match segment.content {
                SegmentContent::Bytes(b) => b,
                SegmentContent::FilePath(path) => {
                    // Try multiple locations:
                    // 1. Absolute path
                    // 2. Relative to cache directory
                    // 3. In cache directory with filename derived from path
                    let full_path = if path.starts_with('/') {
                        std::path::PathBuf::from(&path)
                    } else {
                        let cache_path = cache_dir.join(&path);
                        if cache_path.exists() {
                            cache_path
                        } else {
                            let dataset_path = cache_dir.join(&dataset_slug).join(&path);
                            if dataset_path.exists() {
                                dataset_path
                            } else {
                                let filename = path.split('/').last().unwrap_or(&path);
                                cache_dir.join(filename)
                            }
                        }
                    };

                    if full_path.exists() {
                        fs::read(&full_path)?
                    } else {
                        // If file doesn't exist, check if it's an image path from Arrow
                        // that we should have loaded but didn't
                        println!("    Warning: File not found: {}, using placeholder", path);
                        format!("[FILE: {}]", path).into_bytes()
                    }
                }
                SegmentContent::Url(url) => {
                    // Placeholder for URLs that haven't been downloaded
                    format!("[URL: {}]", url).into_bytes()
                }
            };

            let byte_segment = ByteSegment {
                bytes: bytes.clone(),
                label: Some(format!("{}_{}_{}", modality_name, idx, task_id)),
                metadata: Some(SegmentMetadata {
                    start_offset: 0,
                    end_offset: bytes.len(),
                    confidence: 1.0,
                    extra: segment.metadata,
                }),
            };

            // Add leaf node
            let leaf_idx = builder.add_node(NodeData::Leaf(byte_segment.clone()));

            // Connect branch to leaf
            builder.add_hyperedge(
                vec![branch_idx, leaf_idx],
                EdgeData {
                    label: "contains".to_string(),
                    weight: 1.0,
                },
            );

            byte_segments.push(byte_segment);
        }
    }

    // Build the hypergraph
    let hypergraph = builder.build();

    Ok((byte_segments, hypergraph))
}

fn infer_task_id(df: &DataFrame, row_idx: usize) -> Option<String> {
    for candidate in ["task_id", "id", "uuid", "name"] {
        if let Ok(series) = df.column(candidate) {
            if let Ok(val) = series.str() {
                if let Some(id) = val.get(row_idx) {
                    return Some(id.to_string());
                }
            }
        }
    }

    None
}
