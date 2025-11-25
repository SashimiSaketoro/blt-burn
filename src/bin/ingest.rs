use blt_burn::{
    batching::BatchStats,
    built_info,
    dataset::FineWebEduDataset,
    ffmpeg::{self, FfmpegError},
    model::{LMTransformer, LMTransformerConfig},
    patcher::{patch_lengths_from_start_indices, patch_start_indices_cpu, patch_start_mask_from_entropy_with_monotonicity},
    prefetch::DocumentPrefetcher,
    pretokenize::detect_modality,
    quantization::{quantize_model, QuantConfig, QuantStats},
    sidecar::{EdgeData, HypergraphBuilder, HypergraphSidecar, NodeData},
    tokenizer::BltTokenizer,
};
use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    module::Module,
    record::{FullPrecisionSettings, HalfPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::Tensor,
};
use burn_import::safetensors::SafetensorsFileRecorder;
use clap::Parser;
use dialoguer::Select;
use hypergraph::VertexIndex;
use safetensors::serialize;
use safetensors::tensor::{Dtype, TensorView};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Build long version string with git hash and feature status.
fn long_version() -> &'static str {
    // Use Box::leak to create a static string at runtime
    let git_hash = built_info::GIT_COMMIT_HASH.unwrap_or("unknown");
    let git_short = if git_hash.len() >= 7 { &git_hash[..7] } else { git_hash };
    let fused_status = if cfg!(feature = "fused-entropy") { "enabled" } else { "disabled" };
    
    let version = format!(
        "{}\nGit commit: {}\nFused kernels: {}",
        env!("CARGO_PKG_VERSION"),
        git_short,
        fused_status,
    );
    
    Box::leak(version.into_boxed_str())
}

#[derive(Parser, Debug)]
#[command(author, version, long_version = long_version(), about, long_about = None)]
struct Args {
    /// Input text to patch (optional override)
    #[arg(short, long)]
    text: Option<String>,

    /// Path to input file (optional override)
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Dataset subset/configuration (e.g., "sample-10BT" for 10B token sample)
    #[arg(long, default_value = "sample-10BT")]
    subset: String,

    /// HuggingFace partition name (required by HF API, usually "train" for full corpus)
    /// Note: This is NOT a training split - we ingest the entire selected corpus
    #[arg(long, default_value = "train")]
    partition: String,

    /// Limit number of dataset items to process
    #[arg(long)]
    limit: Option<usize>,

    /// Path to model weights (Burn binary format)
    #[arg(short, long)]
    model_path: Option<PathBuf>,

    /// Entropy threshold for patch boundary detection (1.35 from BLT paper, 1.55 for larger chunks)
    #[arg(short = 'r', long, default_value_t = 1.35)]
    threshold: f64,

    /// Output directory for safetensors shards
    #[arg(short, long, default_value = "ingest_output")]
    output_dir: PathBuf,

    /// Base directory for burn's SQLite database
    /// Defaults to ~/.cache/huggingface/blt-burn (same drive as HF cache, respects symlinks)
    #[arg(long)]
    base_dir: Option<String>,

    /// Override HuggingFace cache directory (default: uses ~/.cache/huggingface symlink)
    /// Only set this if you want to override the system's default HF cache location
    #[arg(long)]
    hf_cache: Option<String>,

    /// External drive path for output files only
    /// HF downloads still use system cache (respects symlinks like ~/.cache/huggingface -> /Volumes/ai/)
    #[arg(long)]
    external_drive: Option<String>,

    /// Number of shards to split output into (for JAX distributed loading)
    #[arg(long)]
    num_shards: Option<usize>,

    /// Target shard size in tokens (default: 100k tokens per shard)
    #[arg(long, default_value_t = 100_000)]
    shard_size: usize,

    /// Disable audio/video pre-tokenization even if ffmpeg is available.
    #[arg(long)]
    no_audio_video: bool,

    /// Try to auto-install ffmpeg using scripts/install_ffmpeg.sh (non-interactive).
    #[arg(long)]
    auto_install_ffmpeg: bool,

    /// Explicit path to ffmpeg binary (overrides PATH lookup).
    #[arg(long)]
    ffmpeg_path: Option<PathBuf>,

    /// Export metadata as JSON in addition to SQLite (for debugging/compatibility)
    #[arg(long)]
    export_json_metadata: bool,

    /// Load dataset from Hugging Face (e.g., "openai/gdpval")
    #[arg(long)]
    huggingface_dataset: Option<String>,

    /// Hugging Face dataset subset to load
    #[arg(long)]
    hf_subset: Option<String>,

    /// Export hypergraph sidecar as JSON for debugging (in addition to SQLite)
    #[arg(long)]
    export_json: bool,

    /// Force loading model weights from MPK format instead of SafeTensors
    #[arg(long)]
    use_mpk: bool,

    /// Quantization mode for model weights: none (default), int8, int4
    /// INT8 provides ~2x speedup with minimal accuracy loss
    /// INT4 provides ~4x compression but may affect embedding quality
    #[arg(long, default_value = "none")]
    quantize: String,

    /// Print quantization statistics and exit
    #[arg(long)]
    quant_stats: bool,

    /// Number of documents to prefetch in background (default: 4)
    /// Higher values use more memory but reduce GPU stalls
    #[arg(long, default_value_t = 4)]
    prefetch_buffer: usize,

    /// Print document size distribution statistics
    #[arg(long)]
    batch_stats: bool,

    /// Enable CubeCL kernel profiling (prints kernel timings to stdout)
    #[arg(long)]
    profile: bool,

    /// Compute and save entropy histogram for debugging patch boundaries
    #[arg(long)]
    entropy_histogram: bool,

    /// Output path for entropy histogram JSON (default: entropy_histogram.json)
    #[arg(long, default_value = "entropy_histogram.json")]
    histogram_output: PathBuf,

    /// Number of bins for entropy histogram (default: 50)
    #[arg(long, default_value_t = 50)]
    histogram_bins: usize,
}

fn compute_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Compute standard deviation of a slice of f32 values.
fn compute_std(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
    variance.sqrt()
}

/// Compute percentile of a slice of f32 values.
fn percentile(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Compute entropy histogram with statistics for debugging patch boundaries.
fn compute_entropy_histogram(entropies: &[f32], num_bins: usize) -> serde_json::Value {
    if entropies.is_empty() {
        return serde_json::json!({
            "error": "No entropy values to compute histogram",
            "total_tokens": 0
        });
    }

    let min = entropies.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = entropies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    // Handle edge case where all values are the same
    let bin_width = if (max - min).abs() < f32::EPSILON {
        1.0
    } else {
        (max - min) / num_bins as f32
    };
    
    let mut bins = vec![0u64; num_bins];
    for &e in entropies {
        let idx = if bin_width > 0.0 {
            ((e - min) / bin_width).floor() as usize
        } else {
            0
        };
        bins[idx.min(num_bins - 1)] += 1;
    }
    
    let mean = entropies.iter().sum::<f32>() / entropies.len() as f32;
    let std = compute_std(entropies);
    
    serde_json::json!({
        "statistics": {
            "min": min,
            "max": max,
            "mean": mean,
            "std": std,
            "total_tokens": entropies.len(),
        },
        "percentiles": {
            "p25": percentile(entropies, 0.25),
            "p50": percentile(entropies, 0.50),
            "p75": percentile(entropies, 0.75),
            "p90": percentile(entropies, 0.90),
            "p95": percentile(entropies, 0.95),
            "p99": percentile(entropies, 0.99),
        },
        "histogram": {
            "num_bins": num_bins,
            "bin_width": bin_width,
            "bin_edges": (0..=num_bins).map(|i| min + i as f32 * bin_width).collect::<Vec<_>>(),
            "counts": bins,
        },
    })
}

/// Result: (ffmpeg_enabled, ffmpeg_path_if_any)
fn ensure_ffmpeg_interactive(args: &Args) -> Result<(bool, Option<PathBuf>), anyhow::Error> {
    // Explicit opt-out always wins.
    if args.no_audio_video {
        println!("⚠️  Audio/video pre-tokenization disabled via --no-audio-video");
        return Ok((false, None));
    }

    // Non-interactive auto-install mode for CI / Docker.
    if args.auto_install_ffmpeg {
        match ffmpeg::resolve_ffmpeg(args.ffmpeg_path.as_deref()) {
            Ok(path) => {
                println!("✅ ffmpeg detected at {:?}", path);
                return Ok((true, Some(path)));
            }
            Err(FfmpegError::NotFound) => {
                println!("⚠️  ffmpeg not found, attempting auto-install...");
                let script_path = Path::new("scripts/install_ffmpeg.sh");
                ffmpeg::try_install_ffmpeg(script_path)?;
                let path = ffmpeg::resolve_ffmpeg(args.ffmpeg_path.as_deref())?;
                println!("✅ ffmpeg installed and detected at {:?}", path);
                return Ok((true, Some(path)));
            }
            Err(e) => return Err(e.into()),
        }
    }

    // Interactive mode: normal user running the ingest binary.
    loop {
        match ffmpeg::resolve_ffmpeg(args.ffmpeg_path.as_deref()) {
            Ok(path) => {
                println!("✅ ffmpeg detected at {:?}", path);
                return Ok((true, Some(path)));
            }
            Err(FfmpegError::NotFound) => {
                println!("⚠️  ffmpeg not found (and required for audio/video pre-tokenization)");
            }
            Err(e) => {
                eprintln!("❌ Error checking for ffmpeg: {e}");
            }
        }

        let options = vec![
            "Install ffmpeg now (recommended)",
            "Continue without audio/video support",
            "Manually installed – retry detection",
            "Abort",
        ];

        let selection = Select::new()
            .with_prompt("What would you like to do?")
            .items(&options)
            .default(0)
            .interact()?;

        match selection {
            0 => {
                let script_path = Path::new("scripts/install_ffmpeg.sh");
                match ffmpeg::try_install_ffmpeg(script_path) {
                    Ok(_) => println!("✅ ffmpeg installation script completed"),
                    Err(e) => eprintln!("❌ ffmpeg installation failed: {e}"),
                }
            }
            1 => {
                println!("⚠️  Proceeding without audio/video pre-tokenization");
                return Ok((false, None));
            }
            2 => {
                // Just loop again – the resolve_ffmpeg() call at the top will re-check.
            }
            3 => {
                // CLI can decide to abort; library never will.
                std::process::exit(1);
            }
            _ => unreachable!(),
        }
    }
}

/// Process raw bytes through BLT entropy model to compute patch boundaries.
/// 
/// Returns:
/// - Raw bytes (unchanged input)
/// - Patch lengths (computed via entropy-based boundary detection)
/// - Per-byte entropies (for downstream Orch-OR/lasing mode)
/// - Hypergraph sidecar (metadata)
fn process_data(
    data: &[u8],
    model: &LMTransformer<Wgpu>,
    device: &WgpuDevice,
    threshold: f64,
) -> Result<
    (
        Vec<u8>,    // raw bytes (unchanged)
        Vec<i32>,   // patch_lengths
        Vec<f32>,   // entropies (per-byte, for downstream aggregation)
        HypergraphSidecar,
    ),
    anyhow::Error,
> {
    // 1. Compute Hash
    let source_hash = compute_hash(data);

    // 2. Pre-tokenize
    let pt_type = detect_modality(data);
    let pretokenizer = pt_type.create()?;
    let modality_name = pretokenizer.modality().to_string();

    let segments = pretokenizer.pre_tokenize(data)?;

    // 3. Flatten to model inputs
    let tokens_vec: Vec<i32> = segments
        .iter()
        .flat_map(|s| s.bytes.iter().map(|&b| b as i32))
        .collect();

    let total_tokens = tokens_vec.len();

    // --- Hypergraph Builder Initialization ---
    let mut builder = HypergraphBuilder::new();

    if total_tokens == 0 {
        // Even if empty, return a valid graph with just the Trunk
        let _trunk = builder.add_node(NodeData::Trunk {
            source_hash: source_hash.clone(),
            total_bytes: 0,
        });
        return Ok((
            Vec::<u8>::new(),    // raw bytes
            Vec::<i32>::new(),   // patch_lengths
            Vec::<f32>::new(),   // entropies
            builder.build(),
        ));
    }

    // A. Add Trunk (File Root)
    let trunk_idx = builder.add_node(NodeData::Trunk {
        source_hash: source_hash.clone(),
        total_bytes: data.len(),
    });

    // B. Add Branch (Modality)
    let branch_idx = builder.add_node(NodeData::Branch {
        label: modality_name.clone(),
        modality: modality_name.clone(),
    });

    // C. Connect Trunk -> Branch (Containment)
    builder.add_hyperedge(
        vec![trunk_idx, branch_idx],
        EdgeData {
            label: "contains".to_string(),
            weight: 1.0,
        },
    );

    // --- End Hypergraph Builder ---

    // BLT entropy model config: max_seqlen=8192, sliding_window=512
    // Use larger chunks to minimize "seam" artifacts at boundaries
    // Each token attends to previous 512 (local attention), but larger chunks
    // reduce discontinuities and allow better model state buildup
    let chunk_size = 4096;  // Half of max_seqlen for memory safety
    let context_overlap = 512;  // Match the sliding_window size
    let stride = chunk_size - context_overlap;  // New tokens per chunk

    let mut chunk_entropies_list = Vec::new();
    let mut chunk_norms_list = Vec::new();
    let mut chunk_embeddings_list = Vec::new();

    let mut position = 0;
    while position < tokens_vec.len() {
        // Determine chunk boundaries
        let start = if position == 0 {
            0
        } else {
            position - context_overlap
        };
        let end = (position + stride).min(tokens_vec.len());
        let chunk = &tokens_vec[start..end];
        let chunk_len = chunk.len();

        // Skip index to ignore low-context outputs at the beginning of non-first chunks
        let skip_count = if position == 0 { 0 } else { context_overlap };

        let input =
            Tensor::<Wgpu, 1, burn::tensor::Int>::from_ints(chunk, device).reshape([1, chunk_len]);

        let output = model.forward_with_embeddings(input);
        let chunk_entropies = blt_burn::patcher::entropy(output.logits);

        // Extract only the valid portion (skip context that was already processed)
        // Note: No .clone() needed here - tensors are consumed by slice and not used after
        if skip_count > 0 {
            let valid_entropies = chunk_entropies.slice([0..1, skip_count..chunk_len]);
            let valid_norms = output
                .embedding_norms
                .slice([0..1, skip_count..chunk_len]);
            let valid_embeddings = output
                .pre_norm_embeddings
                .slice([0..1, skip_count..chunk_len, 0..768]);

            chunk_entropies_list.push(valid_entropies);
            chunk_norms_list.push(valid_norms);
            chunk_embeddings_list.push(valid_embeddings);
        } else {
            // First chunk - use everything
            chunk_entropies_list.push(chunk_entropies);
            chunk_norms_list.push(output.embedding_norms);
            chunk_embeddings_list.push(output.pre_norm_embeddings);
        }

        // Move position forward by stride amount
        position = end;
    }

    let entropies = Tensor::cat(chunk_entropies_list, 1).reshape([1, total_tokens]);
    let norms = Tensor::cat(chunk_norms_list, 1).reshape([1, total_tokens]);
    let embeddings = Tensor::cat(chunk_embeddings_list, 1).reshape([1, total_tokens, 768]);

    // Compute coherence scores: pre_norm^2 / entropy
    // This combines prominence signal with inverse entropy as confidence weighting
    let coherence_scores = norms.clone().powf_scalar(2.0) / (entropies.clone() + 1e-6);

    let mask = patch_start_mask_from_entropy_with_monotonicity(entropies.clone(), threshold);
    let patch_indices = patch_start_indices_cpu(mask);

    let embeddings_data = embeddings.into_data();
    let norms_data = norms.into_data();
    let entropies_data = entropies.into_data();
    let coherence_data = coherence_scores.into_data();

    let _embeddings_f32: Vec<f32> = embeddings_data.iter::<f32>().collect();
    let _norms_f32: Vec<f32> = norms_data.iter::<f32>().collect();
    let entropies_f32: Vec<f32> = entropies_data.iter::<f32>().collect();
    let coherence_f32: Vec<f32> = coherence_data.iter::<f32>().collect();

    let mut patch_mask_vec = vec![0i32; total_tokens];
    let patch_indices_inner = if !patch_indices.is_empty() {
        &patch_indices[0]
    } else {
        &vec![]
    };

    for &idx in patch_indices_inner {
        if idx < total_tokens {
            patch_mask_vec[idx] = 1;
        }
    }

    let _patch_indices_i32: Vec<i32> = patch_indices_inner.iter().map(|&x| x as i32).collect();
    
    // Compute patch lengths from indices
    let patch_lengths = patch_lengths_from_start_indices(&patch_indices, total_tokens);
    let patch_lengths_i32: Vec<i32> = if !patch_lengths.is_empty() {
        patch_lengths[0].iter().map(|&x| x as i32).collect()
    } else {
        vec![]
    };

    // D. Build Patch Leaves from entropy boundaries (not pre-tokenized segments)
    // Each patch becomes a leaf node with coherence score
    let mut prev_leaf_idx: Option<VertexIndex> = None;
    let mut byte_offset = 0;

    for (patch_idx, &patch_len) in patch_lengths_i32.iter().enumerate() {
        let len = patch_len as usize;
        let start = byte_offset;
        let end = byte_offset + len;

        // Extract patch bytes
        let patch_bytes = if end <= data.len() {
            data[start..end].to_vec()
        } else {
            data[start..].to_vec()
        };

        // Calculate mean coherence and entropy for this patch
        let patch_coherence: f32 = if end <= coherence_f32.len() {
            coherence_f32[start..end].iter().sum::<f32>() / (len.max(1) as f32)
        } else {
            0.0
        };
        
        let patch_entropy: f32 = if end <= entropies_f32.len() {
            entropies_f32[start..end].iter().sum::<f32>() / (len.max(1) as f32)
        } else {
            0.0
        };

        // Create segment with patch metadata
        let seg = blt_burn::pretokenize::ByteSegment {
            bytes: patch_bytes,
            label: Some(format!("patch_{}", patch_idx)),
            metadata: Some(blt_burn::pretokenize::SegmentMetadata {
                start_offset: start,
                end_offset: end,
                confidence: 1.0,
                extra: Some(serde_json::json!({
                    "patch_index": patch_idx,
                    "patch_length": len,
                    "coherence_score": patch_coherence,
                    "mean_entropy": patch_entropy,
                })),
            }),
        };

        byte_offset = end;

        // Add Leaf Node (one per patch)
        let leaf_idx = builder.add_node(NodeData::Leaf(seg));

        // Connect Branch -> Leaf (Containment)
        builder.add_hyperedge(
            vec![branch_idx, leaf_idx],
            EdgeData {
                label: "contains".to_string(),
                weight: 1.0,
            },
        );

        // Connect Sequence (Prev Leaf -> This Leaf)
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
    }
    // --- End Hypergraph Builder ---

    // Return raw bytes, patch lengths, and entropies
    Ok((
        data.to_vec(),       // raw bytes (unchanged)
        patch_lengths_i32,   // length of each patch
        entropies_f32,       // per-byte entropies for downstream Orch-OR
        builder.build(),
    ))
}

fn save_metadata_sidecar(
    path: &Path,
    sidecar: &HypergraphSidecar,
    export_json: bool,
) -> anyhow::Result<()> {
    // Save as SQLite (primary format)
    let db_path = path.with_extension("hypergraph.db");
    sidecar.save_to_sqlite(&db_path)?;

    // Optionally export as JSON for debugging
    if export_json {
        let json_path = path.with_extension("hypergraph.json");
        sidecar.save_to_json(&json_path)?;
    }

    Ok(())
}

/// Save byte patches with entropies to safetensors format.
/// 
/// Output format:
/// - `bytes`: [total_bytes] - raw byte values (0-255) concatenated
/// - `patch_lengths`: [num_patches] - length of each patch
/// - `entropies`: [total_bytes] - per-byte entropy values for downstream Orch-OR
/// 
/// To reconstruct patches in Python:
/// ```python
/// patches = []
/// offset = 0
/// for length in patch_lengths:
///     patches.append(bytes(byte_data[offset:offset+length]))
///     offset += length
/// ```
/// 
/// To aggregate entropies to patch level:
/// ```python
/// patch_entropies = []
/// offset = 0
/// for length in patch_lengths:
///     patch_entropies.append(entropies[offset:offset+length].mean())
///     offset += length
/// ```
fn save_patch_safetensors(
    path: &Path,
    bytes: &[u8],
    patch_lengths: &[i32],
    entropies: &[f32],
    metadata_filename: Option<&str>,
) -> anyhow::Result<()> {
    if bytes.is_empty() {
        println!("Warning: Skipping save - no bytes to save");
        return Ok(());
    }

    let num_patches = patch_lengths.len();
    println!("  Saving {} bytes in {} patches", bytes.len(), num_patches);

    let tensors: Vec<(&str, TensorView)> = vec![
        (
            "bytes",
            TensorView::new(
                Dtype::U8,
                vec![bytes.len()],
                bytes,
            )
            .map_err(|e| anyhow::anyhow!("Failed to create bytes tensor: {}", e))?,
        ),
        (
            "patch_lengths",
            TensorView::new(
                Dtype::I32,
                vec![num_patches],
                bytemuck::cast_slice(patch_lengths),
            )
            .map_err(|e| anyhow::anyhow!("Failed to create patch_lengths tensor: {}", e))?,
        ),
        (
            "entropies",
            TensorView::new(
                Dtype::F32,
                vec![entropies.len()],
                bytemuck::cast_slice(entropies),
            )
            .map_err(|e| anyhow::anyhow!("Failed to create entropies tensor: {}", e))?,
        ),
    ];

    let mut metadata_map = std::collections::HashMap::new();
    metadata_map.insert("format".to_string(), "blt_patches_v2".to_string());
    metadata_map.insert("num_patches".to_string(), num_patches.to_string());
    metadata_map.insert("total_bytes".to_string(), bytes.len().to_string());
    if let Some(mf) = metadata_filename {
        metadata_map.insert("metadata_file".to_string(), mf.to_string());
    }

    let serialized = serialize(tensors, &Some(metadata_map))?;
    let mut file = File::create(path)?;
    file.write_all(&serialized)?;
    Ok(())
}

fn get_default_base_dir() -> String {
    // Use ~/.cache/huggingface/blt-burn as default (same drive as HF cache, respects symlinks)
    if let Ok(home) = std::env::var("HOME") {
        let hf_cache = PathBuf::from(&home).join(".cache/huggingface/blt-burn");
        return hf_cache.to_string_lossy().to_string();
    }
    // Fallback to local directory
    "dataset_cache".to_string()
}

fn main() -> anyhow::Result<()> {
    let mut args = Args::parse();

    // Enable CubeCL profiling if requested
    if args.profile {
        std::env::set_var("CUBECL_DEBUG_OPTION", "profile");
        std::env::set_var("CUBECL_DEBUG_LOG", "stdout");
        println!("🔬 Profiling enabled - kernel timings will be printed to stdout");
    }
    
    // Compute default base_dir if not provided (uses HF cache location)
    let base_dir = args.base_dir.clone().unwrap_or_else(get_default_base_dir);

    // Handle external drive override
    if let Some(ref external_drive) = args.external_drive {
        let external_path = PathBuf::from(external_drive);
        
        // Verify the external drive exists and is writable
        if !external_path.exists() {
            println!("📁 Creating external drive directory: {}", external_path.display());
            fs::create_dir_all(&external_path)?;
        }
        
        // Test write access
        let test_file = external_path.join(".blt_write_test");
        match fs::write(&test_file, "test") {
            Ok(_) => {
                fs::remove_file(&test_file).ok();
                println!("✅ External drive verified: {}", external_path.display());
            }
            Err(e) => {
                anyhow::bail!("Cannot write to external drive {}: {}", external_path.display(), e);
            }
        }
        
        // External drive is for OUTPUT only - HF downloads use system cache (respects symlinks)
        args.output_dir = external_path.join("output");
        println!("📂 HF Cache: system default (respects ~/.cache/huggingface symlink)");
        println!("📂 Base directory: {}", base_dir);
        println!("📂 Output directory: {}", args.output_dir.display());
    }
    
    // Show base directory location
    println!("📂 Base directory: {} (for burn SQLite DB)", base_dir);

    // Check FFmpeg availability
    let (_av_enabled, _ffmpeg_path) = ensure_ffmpeg_interactive(&args)?;

    let device = WgpuDevice::default();
    // With "fusion" feature enabled, Wgpu automatically uses kernel fusion
    type Backend = Wgpu;

    // Initialize Model
    let config = LMTransformerConfig {
        dim: 768,
        n_layers: 14,
        head_dim: None,
        n_heads: Some(12),
        n_kv_heads: None,
        ffn_dim_multiplier: Some(1.0),
        multiple_of: 256,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seqlen: 8192,
        vocab_size: 260,
    };
    let model = config.init::<Backend>(&device);

    // Determine model path: argument > build-time env var > default local
    let model_path = args
        .model_path
        .clone()
        .or_else(|| option_env!("BLT_MODEL_SAFETENSORS_PATH").map(PathBuf::from))
        .or_else(|| option_env!("BLT_MODEL_CACHE_PATH").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("model.safetensors"));

    println!("Loading weights from {:?}", model_path);

    // Determine format from path extension or --use-mpk flag
    let use_mpk = args.use_mpk
        || model_path
            .extension()
            .map(|e| e == "mpk")
            .unwrap_or(false);

    let model = if use_mpk {
        println!("Using MPK format (NamedMpkFileRecorder)");
        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
        model.load_record(
            recorder
                .load(model_path.into(), &device)
                .expect("Failed to load MPK weights"),
        )
    } else {
        println!("Using SafeTensors format (SafetensorsFileRecorder)");
        let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
        model.load_record(
            recorder
                .load(model_path.into(), &device)
                .expect("Failed to load SafeTensors weights"),
        )
    };

    // Parse quantization configuration
    let quant_config = QuantConfig::from_str(&args.quantize);
    
    // Handle --quant-stats flag: print stats and exit
    if args.quant_stats {
        QuantStats::for_blt_model(quant_config).print();
        return Ok(());
    }
    
    // Apply quantization if requested
    let model = if quant_config.is_enabled() {
        println!("🔢 Quantizing model to {}...", quant_config);
        let start = std::time::Instant::now();
        let quantized = quantize_model(model, quant_config);
        println!("   ✅ Quantization complete in {:?}", start.elapsed());
        quantized
    } else {
        model
    };

    // Create output directory
    fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");

    // Handle Hugging Face dataset loading
    if let Some(hf_dataset) = &args.huggingface_dataset {
        println!("🤗 Loading dataset from Hugging Face: {}", hf_dataset);

        // Create tokenizer
        let tokenizer = BltTokenizer::new(false, false); // No BOS/EOS tokens

        blt_burn::huggingface_loader::process_hf_dataset(
            hf_dataset,
            &args.output_dir,
            &model,
            &device,
            &tokenizer,
            args.threshold,
            args.hf_subset.as_deref(),
            args.limit,
        )?;

        return Ok(());
    }

    // Default to dataset mode unless text/file is explicitly provided
    if args.text.is_some() || args.file.is_some() {
        // Single file/text mode
        let (bytes, filename_prefix) = if let Some(t) = args.text {
            (t.into_bytes(), "manual_input".to_string())
        } else if let Some(f) = args.file {
            let b = std::fs::read(&f).expect("Failed to read input file");
            let name = f
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            (b, name)
        } else {
            panic!("Should not happen")
        };

        println!("Processing single input ({} bytes)...", bytes.len());
        match process_data(&bytes, &model, &device, args.threshold) {
            Ok((raw_bytes, patch_lengths, entropies, sidecar)) => {
                if !patch_lengths.is_empty() {
                    // Save metadata sidecar
                    let metadata_basename = filename_prefix.clone();
                    let metadata_path = args.output_dir.join(&metadata_basename);
                    save_metadata_sidecar(&metadata_path, &sidecar, args.export_json_metadata)?;
                    let metadata_db_name = format!("{}.hypergraph.db", metadata_basename);

                    // Save patches with entropies
                    let path = args
                        .output_dir
                        .join(format!("{}.safetensors", filename_prefix));
                    save_patch_safetensors(
                        &path,
                        &raw_bytes,
                        &patch_lengths,
                        &entropies,
                        Some(&metadata_db_name),
                    )?;
                    println!("✅ Saved: {} ({} patches)", path.display(), patch_lengths.len());

                    // Save entropy histogram if requested
                    if args.entropy_histogram && !entropies.is_empty() {
                        let histogram = compute_entropy_histogram(&entropies, args.histogram_bins);
                        let histogram_path = args.output_dir.join(&args.histogram_output);
                        let histogram_json = serde_json::to_string_pretty(&histogram)?;
                        std::fs::write(&histogram_path, histogram_json)?;
                        println!(
                            "📊 Entropy histogram saved to {} ({} tokens, {} bins)",
                            histogram_path.display(),
                            entropies.len(),
                            args.histogram_bins
                        );
                    }
                }
            }
            Err(e) => println!("Error processing input: {}", e),
        }
    } else {
        // Dataset ingestion mode (default) - processes entire corpus, NOT training splits
        println!(
            "📚 Ingesting FineWeb-Edu corpus: {} (full {} partition)...",
            args.subset, args.partition
        );
        if let Some(ref hf_cache) = args.hf_cache {
            println!("HF Cache override: {}", hf_cache);
        } else {
            println!("HF Cache: system default (respects symlinks)");
        }
        fs::create_dir_all(&base_dir).expect("Failed to create base directory");

        let dataset = FineWebEduDataset::new(&args.subset, &args.partition, &base_dir, args.hf_cache.as_deref())
            .expect("Failed to load dataset");

        println!("Dataset size: {}", dataset.len());
        let limit = args.limit.unwrap_or(dataset.len());

        // Collect document references into owned data for prefetcher
        // The prefetcher spawns a thread, so it needs 'static lifetime
        println!("📥 Loading document metadata...");
        let docs: Vec<(String, Vec<u8>)> = dataset
            .iter()
            .take(limit)
            .enumerate()
            .filter_map(|(i, item)| {
                let id_str = item.id.clone().unwrap_or_else(|| format!("{}", i));
                if item.text.is_empty() {
                    None
                } else {
                    Some((id_str, item.text.into_bytes()))
                }
            })
            .collect();
        
        println!("📋 Collected {} documents for processing", docs.len());

        // Use async prefetcher - loads next docs while GPU processes current
        // This overlaps I/O with compute per Burn's async execution model
        println!("🚀 Async prefetch enabled (buffer: {} docs)", args.prefetch_buffer);
        let prefetcher = DocumentPrefetcher::new(docs.into_iter(), args.prefetch_buffer);
        
        let mut batch_stats = BatchStats::default();
        let mut processed = 0;
        let start_time = std::time::Instant::now();

        // Accumulate all entropies for histogram if requested
        let mut all_entropies: Vec<f32> = Vec::new();

        for doc in prefetcher {
            processed += 1;
            batch_stats.add(&doc);
            
            // Progress logging
            if processed % 100 == 0 || processed <= 10 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let rate = processed as f64 / elapsed;
                println!(
                    "Processing {}/{} (ID: {}, {} bytes) [{:.1} docs/s]",
                    processed, limit, doc.id, doc.original_len, rate
                );
            }

            // Process document - each doc gets its own hypergraph sidecar
            // This preserves entropy context integrity (no cross-doc bleeding)
            match process_data(&doc.bytes, &model, &device, args.threshold) {
                Ok((raw_bytes, patch_lengths, entropies, sidecar)) => {
                    // Accumulate entropies for histogram
                    if args.entropy_histogram {
                        all_entropies.extend_from_slice(&entropies);
                    }

                    if !patch_lengths.is_empty() {
                        // Save metadata sidecar (hypergraph with patch nodes)
                        let metadata_basename = format!("item_{}", doc.id);
                        let metadata_path = args.output_dir.join(&metadata_basename);
                        save_metadata_sidecar(&metadata_path, &sidecar, args.export_json_metadata)?;
                        let metadata_db_name = format!("{}.hypergraph.db", metadata_basename);

                        // Save patches with entropies
                        let filename = format!("item_{}.safetensors", doc.id);
                        let path = args.output_dir.join(filename);
                        save_patch_safetensors(
                            &path,
                            &raw_bytes,
                            &patch_lengths,
                            &entropies,
                            Some(&metadata_db_name),
                        )?;
                    }
                }
                Err(e) => println!("Error processing item {}: {}", doc.id, e),
            }
        }

        // Print batch statistics if requested
        if args.batch_stats {
            batch_stats.print_summary();
        }

        // Save entropy histogram if requested
        if args.entropy_histogram && !all_entropies.is_empty() {
            let histogram = compute_entropy_histogram(&all_entropies, args.histogram_bins);
            let histogram_path = args.output_dir.join(&args.histogram_output);
            let histogram_json = serde_json::to_string_pretty(&histogram)?;
            std::fs::write(&histogram_path, histogram_json)?;
            println!(
                "📊 Entropy histogram saved to {} ({} tokens, {} bins)",
                histogram_path.display(),
                all_entropies.len(),
                args.histogram_bins
            );
        }
        
        let total_time = start_time.elapsed();
        println!(
            "📊 Processed {} docs in {:.1}s ({:.1} docs/s)",
            processed,
            total_time.as_secs_f64(),
            processed as f64 / total_time.as_secs_f64()
        );
    }

    println!("✅ Ingestion complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use blt_burn::pretokenize::{ByteSegment, SegmentMetadata};
    use serde_json::json;

    #[test]
    fn test_coherence_injection() {
        // Mock coherence scores for a 5-token segment
        let coherence_f32 = vec![0.8, 1.2, 0.9, 1.5, 1.1];
        let len = 5;
        let start = 0;
        let end = 5;

        // Simulate segment creation (mock bytes and metadata)
        let mut seg = ByteSegment {
            bytes: vec![0u8; len],
            label: None,
            metadata: Some(SegmentMetadata {
                start_offset: start,
                end_offset: end,
                confidence: 0.95,
                extra: Some(json!({ "test": "value" })),
            }),
        };

        // Manually inject coherence (as done in process_data)
        if end <= coherence_f32.len() {
            let segment_coherence: f32 = coherence_f32[start..end].iter().sum::<f32>() / (len as f32);
            let expected_coherence: f32 = 1.1; // (0.8+1.2+0.9+1.5+1.1)/5 = 1.1

            if let Some(meta) = &mut seg.metadata {
                let mut extra = meta.extra.clone().unwrap_or(json!({}));
                if let Some(obj) = extra.as_object_mut() {
                    obj.insert("coherence_score".to_string(), json!(segment_coherence));
                }
                meta.extra = Some(extra);
            }

            // Verify injection
            assert_eq!(seg.metadata.as_ref().unwrap().extra.as_ref().unwrap()["coherence_score"].as_f64().unwrap() as f32, expected_coherence);
            assert!(seg.metadata.as_ref().unwrap().extra.as_ref().unwrap().get("test").is_some()); // Ensure other fields preserved
        } else {
            panic!("End index out of bounds for coherence slice");
        }
    }

    #[test]
    fn test_coherence_without_existing_metadata() {
        // Mock coherence scores
        let coherence_f32 = vec![1.0, 2.0];
        let len = 2;
        let start = 0;
        let end = 2;
        let expected_coherence: f32 = 1.5;

        let mut seg = ByteSegment {
            bytes: vec![0u8; len],
            label: None,
            metadata: None, // No initial metadata
        };

        // Inject coherence
        if end <= coherence_f32.len() {
            let segment_coherence: f32 = coherence_f32[start..end].iter().sum::<f32>() / (len as f32);

            seg.metadata = Some(SegmentMetadata {
                start_offset: start,
                end_offset: end,
                confidence: 1.0,
                extra: Some(json!({
                    "coherence_score": segment_coherence
                })),
            });
        }

        // Verify creation and injection
        let meta = seg.metadata.unwrap();
        assert_eq!(meta.extra.unwrap()["coherence_score"].as_f64().unwrap() as f32, expected_coherence);
    }
}
