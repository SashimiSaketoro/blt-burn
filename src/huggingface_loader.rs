use anyhow::Result;
use burn_wgpu::{Wgpu, WgpuDevice};
use polars::prelude::DataFrame;
use safetensors::serialize;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use crate::blt_core::{process_bytes_with_embeddings, BltConfig, BltExampleWithEmbeddings};
use crate::dataset_helpers::{create_processor, ModalitySegment, SegmentContent};
use crate::hf_resolver::{HfResolver, HfResolverConfig, ZERO_IMAGE_SIZE};
use crate::model::LMTransformer;
use crate::modalities::{ByteSegment, SegmentMetadata};
use crate::polars_dataset_loader;
use crate::sidecar::{EdgeData, HypergraphBuilder, HypergraphSidecar, NodeData};
use crate::tokenizer::BltTokenizer;

/// Process a Hugging Face dataset through the BLT entropy model.
///
/// This function:
/// 1. Loads the dataset via Polars (pure Rust)
/// 2. Processes each item through the entropy model
/// 3. Extracts pre-L2-norm embeddings and prominence scores
/// 4. Saves output as SafeTensors + hypergraph sidecar
pub fn process_hf_dataset(
    dataset_name: &str,
    output_dir: &Path,
    model: &LMTransformer<Wgpu>,
    device: &WgpuDevice,
    _tokenizer: &BltTokenizer,
    threshold: f64,
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

    // BLT processing config
    let blt_config = BltConfig {
        threshold,
        monotonicity: true,
        max_seq_len: 1024,
        chunk_overlap: 512,
        add_bos: false,
        add_eos: false,
    };

    // Process each dataset item
    for row_idx in 0..process_limit {
        println!("\nProcessing entry {}/{}...", row_idx + 1, process_limit);

        let task_id = infer_task_id(&df, row_idx).unwrap_or_else(|| format!("item_{}", row_idx));

        println!("Processing: {}", task_id);

        // Step 1: Download any referenced files FIRST (before processing)
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
        let (byte_segments, mut hypergraph) =
            convert_to_byte_segments(segments, &task_id, dataset_name, &cache_dir)?;

        // Step 4: Concatenate all byte segments for model processing
        let all_bytes: Vec<u8> = byte_segments.iter().flat_map(|s| s.bytes.clone()).collect();
        let total_bytes = all_bytes.len();

        if total_bytes == 0 {
            println!("  Skipping empty item");
            continue;
        }

        println!("  Running BLT inference on {} bytes...", total_bytes);

        // Step 5: Process through BLT model
        let blt_result = process_bytes_with_embeddings(
            &all_bytes,
            &task_id,
            model,
            device,
            &blt_config,
        );

        let total_tokens = blt_result.core.tokens.len();
        let num_patches = blt_result.core.patch_lengths.len();

        println!(
            "  Processed: {} tokens, {} patches",
            total_tokens, num_patches
        );

        // Update hypergraph trunk with actual byte count
        update_trunk_bytes(&mut hypergraph, total_bytes);

        // Inject coherence scores into leaf metadata
        inject_coherence_into_leaves(
            &mut hypergraph,
            &blt_result.coherence_scores,
            total_tokens,
        );

        // Step 6: Save hypergraph sidecar
        let metadata_path = output_dir.join(format!("{}.hypergraph.db", task_id));
        hypergraph.save_to_sqlite(&metadata_path)?;
        println!("  Saved hypergraph: {}", metadata_path.display());

        // Step 7: Save SafeTensors with all BLT outputs
        let safetensors_path = output_dir.join(format!("{}.safetensors", task_id));
        save_blt_safetensors(&safetensors_path, &blt_result, &task_id)?;
        println!("  Saved: {}", safetensors_path.display());
    }

    println!("\n✅ Dataset processing complete!");
    Ok(())
}

/// Save BLT results to SafeTensors format.
fn save_blt_safetensors(
    path: &Path,
    result: &BltExampleWithEmbeddings,
    task_id: &str,
) -> Result<()> {
    let total_tokens = result.core.tokens.len();
    let embedding_dim = result.embedding_dim;

    if total_tokens == 0 {
        return Ok(());
    }

    // Convert tokens to i32 for SafeTensors
    let tokens_i32: Vec<i32> = result.core.tokens.clone();

    // Convert mask to i32 (1 = attend, 0 = ignore)
    let mask_i32: Vec<i32> = result.core.mask.iter().map(|&b| if b { 1 } else { 0 }).collect();

    let tensors: Vec<(&str, TensorView)> = vec![
        (
            "tokens",
            TensorView::new(
                Dtype::I32,
                vec![1, total_tokens],
                bytemuck::cast_slice(&tokens_i32),
            )?,
        ),
        (
            "embeddings",
            TensorView::new(
                Dtype::F32,
                vec![1, total_tokens, embedding_dim],
                bytemuck::cast_slice(&result.pre_norm_embeddings),
            )?,
        ),
        (
            "prominence",
            TensorView::new(
                Dtype::F32,
                vec![1, total_tokens],
                bytemuck::cast_slice(&result.prominence),
            )?,
        ),
        (
            "entropies",
            TensorView::new(
                Dtype::F32,
                vec![1, total_tokens],
                bytemuck::cast_slice(&result.core.entropies),
            )?,
        ),
        (
            "coherence_scores",
            TensorView::new(
                Dtype::F32,
                vec![1, total_tokens],
                bytemuck::cast_slice(&result.coherence_scores),
            )?,
        ),
        (
            "patch_indices",
            TensorView::new(
                Dtype::I32,
                vec![result.patch_indices.len()],
                bytemuck::cast_slice(&result.patch_indices),
            )?,
        ),
        (
            "patch_lengths",
            TensorView::new(
                Dtype::I32,
                vec![result.core.patch_lengths.len()],
                bytemuck::cast_slice(&result.core.patch_lengths),
            )?,
        ),
        (
            "mask",
            TensorView::new(
                Dtype::I32,
                vec![1, total_tokens],
                bytemuck::cast_slice(&mask_i32),
            )?,
        ),
    ];

    let mut metadata_map = HashMap::new();
    metadata_map.insert("sample_id".to_string(), task_id.to_string());
    metadata_map.insert("embedding_dim".to_string(), embedding_dim.to_string());
    metadata_map.insert(
        "metadata_file".to_string(),
        format!("{}.hypergraph.db", task_id),
    );

    let serialized = serialize(tensors, &Some(metadata_map))?;
    let mut file = File::create(path)?;
    file.write_all(&serialized)?;

    Ok(())
}

/// Update the trunk node's total_bytes field in the hypergraph.
fn update_trunk_bytes(hypergraph: &mut HypergraphSidecar, total_bytes: usize) {
    for node in &mut hypergraph.nodes {
        if let NodeData::Trunk { total_bytes: ref mut tb, .. } = node {
            *tb = total_bytes;
            break;
        }
    }
}

/// Inject coherence scores into leaf node metadata.
/// Maps byte offsets to coherence scores from the BLT result.
fn inject_coherence_into_leaves(
    hypergraph: &mut HypergraphSidecar,
    coherence_scores: &[f32],
    total_tokens: usize,
) {
    for node in &mut hypergraph.nodes {
        if let NodeData::Leaf(ref mut segment) = node {
            if let Some(ref mut meta) = segment.metadata {
                let start = meta.start_offset.min(total_tokens);
                let end = meta.end_offset.min(total_tokens);
                
                if start < end && end <= coherence_scores.len() {
                    // Calculate mean coherence for this segment
                    let segment_coherence: f32 = coherence_scores[start..end]
                        .iter()
                        .sum::<f32>() / ((end - start).max(1) as f32);
                    
                    // Inject into metadata.extra
                    let mut extra = meta.extra.take().unwrap_or(serde_json::json!({}));
                    if let Some(obj) = extra.as_object_mut() {
                        obj.insert(
                            "coherence_score".to_string(),
                            serde_json::json!(segment_coherence),
                        );
                        // Also add mean entropy and prominence for this segment
                    }
                    meta.extra = Some(extra);
                }
            }
        }
    }
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
    
    // Track previous leaf for sequence edges
    let mut prev_leaf_idx: Option<hypergraph::VertexIndex> = None;
    let mut global_byte_offset = 0usize;

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
                        // Use HfResolver for missing files - downloads from HF or uses zero tensor
                        let is_image = HfResolver::is_image_path(&path);
                        let config = HfResolverConfig::new(dataset_name, cache_dir);
                        let resolver = HfResolver::new(config);

                        match resolver.resolve(&path, is_image)? {
                            Some(bytes) => bytes,
                            None => {
                                // Skip this segment if resolver returns None (skip_missing mode)
                                println!("    Skipping missing file: {}", path);
                                continue;
                            }
                        }
                    }
                }
                SegmentContent::Url(url) => {
                    // Try to resolve URL via HfResolver if it's an HF URL
                    if url.starts_with("hf://") || url.contains("huggingface.co") {
                        let path = url
                            .strip_prefix("hf://datasets/")
                            .or_else(|| url.strip_prefix("hf://"))
                            .unwrap_or(&url);
                        let is_image = HfResolver::is_image_path(path);
                        let config = HfResolverConfig::new(dataset_name, cache_dir);
                        let resolver = HfResolver::new(config);

                        match resolver.resolve(path, is_image)? {
                            Some(bytes) => bytes,
                            None => {
                                println!("    Skipping missing URL: {}", url);
                                continue;
                            }
                        }
                    } else {
                        // Non-HF URLs: use placeholder (could add reqwest download later)
                        println!("    Warning: Non-HF URL not downloaded: {}", url);
                        if HfResolver::is_image_path(&url) {
                            vec![0u8; ZERO_IMAGE_SIZE]
                        } else {
                            format!("[URL: {}]", url).into_bytes()
                        }
                    }
                }
            };

            let segment_len = bytes.len();
            let start_offset = global_byte_offset;
            let end_offset = global_byte_offset + segment_len;

            let byte_segment = ByteSegment {
                bytes: bytes.clone(),
                label: Some(format!("{}_{}_{}", modality_name, idx, task_id)),
                metadata: Some(SegmentMetadata {
                    start_offset,
                    end_offset,
                    confidence: 1.0,
                    extra: segment.metadata,
                }),
            };

            // Add leaf node
            let leaf_idx = builder.add_node(NodeData::Leaf(byte_segment.clone()));

            // Connect branch to leaf (containment)
            builder.add_hyperedge(
                vec![branch_idx, leaf_idx],
                EdgeData {
                    label: "contains".to_string(),
                    weight: 1.0,
                },
            );

            // Connect sequence: prev_leaf -> this_leaf ("next" edge)
            if let Some(prev) = prev_leaf_idx {
                builder.add_hyperedge(
                    vec![prev, leaf_idx],
                    EdgeData {
                        label: "next".to_string(),
                        weight: 1.0,
                    },
                );
            }

            prev_leaf_idx = Some(leaf_idx);
            global_byte_offset = end_offset;
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
