use blt_burn::{
    dataset::FineWebEduDataset,
    ffmpeg::{self, FfmpegError},
    model::{LMTransformer, LMTransformerConfig},
    patcher::{patch_start_indices_cpu, patch_start_mask_from_entropy_with_monotonicity},
    pretokenize::detect_modality,
    sidecar::{EdgeData, HypergraphBuilder, HypergraphSidecar, NodeData, ShardingInfo},
    tokenizer::BltTokenizer,
};
use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    module::Module,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::Tensor,
};
use clap::Parser;
use dialoguer::Select;
use hypergraph::VertexIndex;
use safetensors::serialize;
use safetensors::tensor::{Dtype, TensorView};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

const MAX_TOKENS_PER_FILE: usize = 100_000; // Split large files into parts to prevent RAM explosion

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input text to patch (optional override)
    #[arg(short, long)]
    text: Option<String>,

    /// Path to input file (optional override)
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Dataset subset
    #[arg(long, default_value = "sample-10BT")]
    subset: String,

    /// Dataset split
    #[arg(long, default_value = "train")]
    split: String,

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

    /// Cache directory for dataset download (default: dataset_cache)
    #[arg(long, default_value = "dataset_cache")]
    cache_dir: String,

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
}

fn compute_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
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

fn process_data(
    data: &[u8],
    model: &LMTransformer<Wgpu>,
    device: &WgpuDevice,
    threshold: f64,
) -> Result<
    (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<i32>,
        Vec<i32>,
        usize,
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
            Vec::<f32>::new(),
            Vec::<f32>::new(),
            Vec::<f32>::new(),
            Vec::<f32>::new(),
            Vec::<i32>::new(),
            Vec::<i32>::new(),
            0,
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

    let chunk_size = 1024;
    let stride = 512; // Process 512 new tokens per chunk, keeping 512 as context
    let context_size = chunk_size - stride; // 512 tokens of context

    let mut chunk_entropies_list = Vec::new();
    let mut chunk_norms_list = Vec::new();
    let mut chunk_embeddings_list = Vec::new();

    let mut position = 0;
    while position < tokens_vec.len() {
        // Determine chunk boundaries
        let start = if position == 0 {
            0
        } else {
            position - context_size
        };
        let end = (position + stride).min(tokens_vec.len());
        let chunk = &tokens_vec[start..end];
        let chunk_len = chunk.len();

        // Skip index to ignore low-context outputs at the beginning of non-first chunks
        let skip_count = if position == 0 { 0 } else { context_size };

        let input =
            Tensor::<Wgpu, 1, burn::tensor::Int>::from_ints(chunk, device).reshape([1, chunk_len]);

        let output = model.forward_with_embeddings(input);
        let chunk_entropies = blt_burn::patcher::entropy(output.logits);

        // Extract only the valid portion (skip context that was already processed)
        if skip_count > 0 {
            let valid_entropies = chunk_entropies.clone().slice([0..1, skip_count..chunk_len]);
            let valid_norms = output
                .embedding_norms
                .clone()
                .slice([0..1, skip_count..chunk_len]);
            let valid_embeddings =
                output
                    .pre_norm_embeddings
                    .clone()
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

    let embeddings_f32: Vec<f32> = embeddings_data.iter::<f32>().collect();
    let norms_f32: Vec<f32> = norms_data.iter::<f32>().collect();
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

    let patch_indices_i32: Vec<i32> = patch_indices_inner.iter().map(|&x| x as i32).collect();

    // D. Process Segments (Leaves) with Coherence Injection
    // Move Hypergraph construction here to access computed coherence scores
    let mut prev_leaf_idx: Option<VertexIndex> = None;
    let mut token_offset = 0;

    for segment in segments {
        let mut seg = segment.clone();
        let len = seg.bytes.len();
        let start = token_offset;
        let end = token_offset + len;

        // Calculate mean coherence for this segment
        if end <= coherence_f32.len() {
            let segment_coherence: f32 =
                coherence_f32[start..end].iter().sum::<f32>() / (len.max(1) as f32);

            // Inject into metadata
            if let Some(meta) = &mut seg.metadata {
                let mut extra = meta.extra.clone().unwrap_or(serde_json::json!({}));
                if let Some(obj) = extra.as_object_mut() {
                    obj.insert(
                        "coherence_score".to_string(),
                        serde_json::json!(segment_coherence),
                    );
                }
                meta.extra = Some(extra);
            } else {
                // Create metadata if missing
                seg.metadata = Some(blt_burn::pretokenize::SegmentMetadata {
                    start_offset: start,
                    end_offset: end,
                    confidence: 1.0,
                    extra: Some(serde_json::json!({
                        "coherence_score": segment_coherence
                    })),
                });
            }
        }
        token_offset = end;

        // Add Leaf Node
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

    Ok((
        embeddings_f32,
        norms_f32,
        entropies_f32,
        coherence_f32,
        patch_indices_i32,
        patch_mask_vec,
        total_tokens,
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

fn save_sharded_output(
    base_path: &Path,
    embeddings_f32: &[f32],
    norms_f32: &[f32],
    entropies_f32: &[f32],
    coherence_f32: &[f32],
    patch_indices_i32: &[i32],
    patch_mask_vec: &[i32],
    total_tokens: usize,
    sidecar: HypergraphSidecar,
    num_shards: usize,
    export_json: bool,
) -> anyhow::Result<()> {
    if num_shards == 1 {
        // Single shard - use regular save
        save_safetensors(
            base_path,
            embeddings_f32,
            norms_f32,
            entropies_f32,
            coherence_f32,
            patch_indices_i32,
            patch_mask_vec,
            total_tokens,
            None,
        )?;
        save_metadata_sidecar(&base_path.with_extension(""), &sidecar, export_json)?;
        return Ok(());
    }

    // Calculate tokens per shard
    let tokens_per_shard = (total_tokens + num_shards - 1) / num_shards;
    let embedding_dim = 768;

    for shard_idx in 0..num_shards {
        let start_token = shard_idx * tokens_per_shard;
        let end_token = ((shard_idx + 1) * tokens_per_shard).min(total_tokens);
        let shard_tokens = end_token - start_token;

        if shard_tokens == 0 {
            continue;
        }

        // Extract shard data
        let emb_start = start_token * embedding_dim;
        let emb_end = end_token * embedding_dim;

        // Create sharding info
        let sharding_info = ShardingInfo {
            global_shape: vec![1, total_tokens, embedding_dim],
            shard_index: shard_idx,
            num_shards,
            process_index: Some(shard_idx % num_shards), // Simple round-robin assignment
            axis: 1,                                     // Sharding along sequence dimension
        };

        // Update sidecar with sharding info
        let mut shard_sidecar = sidecar.clone();
        shard_sidecar.sharding = Some(sharding_info);

        // Generate shard filename
        let shard_name = format!(
            "{}_shard_{}_of_{}",
            base_path.file_stem().unwrap_or_default().to_string_lossy(),
            shard_idx,
            num_shards
        );
        let shard_path = base_path
            .with_file_name(&shard_name)
            .with_extension("safetensors");

        // Save shard
        save_safetensors(
            &shard_path,
            &embeddings_f32[emb_start..emb_end],
            &norms_f32[start_token..end_token],
            &entropies_f32[start_token..end_token],
            &coherence_f32[start_token..end_token],
            patch_indices_i32, // Keep all patch indices for reference
            &patch_mask_vec[start_token..end_token],
            shard_tokens,
            Some(&format!("{}.hypergraph.db", shard_name)),
        )?;

        // Save shard metadata
        let shard_base = base_path.with_file_name(&shard_name);
        save_metadata_sidecar(&shard_base, &shard_sidecar, export_json)?;
    }

    Ok(())
}

fn save_safetensors(
    path: &Path,
    embeddings_f32: &[f32],
    norms_f32: &[f32],
    entropies_f32: &[f32],
    coherence_f32: &[f32],
    patch_indices_i32: &[i32],
    patch_mask_vec: &[i32],
    total_tokens: usize,
    metadata_filename: Option<&str>,
) -> anyhow::Result<()> {
    // Safety rail: Don't create empty files
    if total_tokens == 0 {
        println!("Warning: Skipping save_safetensors - no tokens to save");
        return Ok(());
    }

    let tensors: Vec<(&str, TensorView)> = vec![
        (
            "embeddings",
            TensorView::new(
                Dtype::F32,
                vec![1, total_tokens, 768],
                bytemuck::cast_slice(embeddings_f32),
            )
            .map_err(|e| anyhow::anyhow!("Failed to create embeddings tensor view: {}", e))?,
        ),
        (
            "prominence",
            TensorView::new(
                Dtype::F32,
                vec![1, total_tokens],
                bytemuck::cast_slice(norms_f32),
            )
            .map_err(|e| anyhow::anyhow!("Failed to create prominence tensor view: {}", e))?,
        ),
        (
            "entropies",
            TensorView::new(
                Dtype::F32,
                vec![1, total_tokens],
                bytemuck::cast_slice(entropies_f32),
            )
            .map_err(|e| anyhow::anyhow!("Failed to create entropies tensor view: {}", e))?,
        ),
        (
            "coherence_scores",
            TensorView::new(
                Dtype::F32,
                vec![1, total_tokens],
                bytemuck::cast_slice(coherence_f32),
            )
            .map_err(|e| anyhow::anyhow!("Failed to create coherence_scores tensor view: {}", e))?,
        ),
        (
            "patch_indices",
            TensorView::new(
                Dtype::I32,
                vec![patch_indices_i32.len()],
                bytemuck::cast_slice(patch_indices_i32),
            )
            .map_err(|e| anyhow::anyhow!("Failed to create patch_indices tensor view: {}", e))?,
        ),
        (
            "patch_mask",
            TensorView::new(
                Dtype::I32,
                vec![1, total_tokens],
                bytemuck::cast_slice(patch_mask_vec),
            )
            .map_err(|e| anyhow::anyhow!("Failed to create patch_mask tensor view: {}", e))?,
        ),
    ];

    let mut metadata_map = std::collections::HashMap::new();
    if let Some(mf) = metadata_filename {
        metadata_map.insert("metadata_file".to_string(), mf.to_string());
    }

    let serialized = serialize(tensors, &Some(metadata_map))?;
    let mut file = File::create(path)?;
    file.write_all(&serialized)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Check FFmpeg availability
    let (_av_enabled, _ffmpeg_path) = ensure_ffmpeg_interactive(&args)?;

    let device = WgpuDevice::default();
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
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();

    // Determine model path: argument > build-time env var > default local
    let model_path = args
        .model_path
        .or_else(|| option_env!("BLT_MODEL_CACHE_PATH").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("blt_entropy_model.mpk"));

    println!("Loading weights from {:?}", model_path);
    let model = model.load_record(
        recorder
            .load(model_path.into(), &device)
            .expect("Failed to load weights"),
    );

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
            Ok((emb, norms, entropies, coherence, idxs, mask, count, sidecar)) => {
                // Determine sharding strategy
                let num_shards = if let Some(n) = args.num_shards {
                    n
                } else if count > args.shard_size {
                    // Auto-shard based on size
                    (count + args.shard_size - 1) / args.shard_size
                } else {
                    1
                };

                if num_shards > 1 {
                    println!(
                        "📦 Sharding output into {} shards (JAX-compatible)...",
                        num_shards
                    );
                    save_sharded_output(
                        &args.output_dir.join(&filename_prefix),
                        &emb,
                        &norms,
                        &entropies,
                        &coherence,
                        &idxs,
                        &mask,
                        count,
                        sidecar,
                        num_shards,
                        args.export_json_metadata,
                    )?;
                    println!(
                        "✅ Saved {} shards with pattern: {}_shard_*_of_{}.safetensors",
                        num_shards, filename_prefix, num_shards
                    );
                } else {
                    // Single output
                    let metadata_basename = filename_prefix.clone();
                    let metadata_path = args.output_dir.join(&metadata_basename);
                    save_metadata_sidecar(&metadata_path, &sidecar, args.export_json_metadata)?;
                    let metadata_db_name = format!("{}.hypergraph.db", metadata_basename);

                    let path = args
                        .output_dir
                        .join(format!("{}.safetensors", filename_prefix));
                    save_safetensors(
                        &path,
                        &emb,
                        &norms,
                        &entropies,
                        &coherence,
                        &idxs,
                        &mask,
                        count,
                        Some(&metadata_db_name),
                    )?;
                    println!("✅ Saved: {}", path.display());
                }
            }
            Err(e) => println!("Error processing input: {}", e),
        }
    } else {
        // Dataset mode (default)
        println!(
            "Loading FineWeb-Edu dataset ({}/{})...",
            args.subset, args.split
        );
        println!("Cache directory: {}", args.cache_dir);
        fs::create_dir_all(&args.cache_dir).expect("Failed to create cache directory");

        let dataset = FineWebEduDataset::new(&args.subset, &args.split, &args.cache_dir)
            .expect("Failed to load dataset");

        println!("Dataset size: {}", dataset.len());
        let limit = args.limit.unwrap_or(dataset.len());

        for (i, item) in dataset.iter().take(limit).enumerate() {
            let id_str = item.id.clone().unwrap_or_else(|| format!("{}", i));
            println!("Processing item {}/{} (ID: {})...", i + 1, limit, id_str);

            // Skip empty
            if item.text.is_empty() {
                continue;
            }

            let bytes = item.text.into_bytes();
            match process_data(&bytes, &model, &device, args.threshold) {
                Ok((emb, norms, entropies, coherence, idxs, mask, count, sidecar)) => {
                    if count > 0 {
                        // Save metadata sidecar (one per item)
                        let metadata_basename = format!("item_{}", id_str);
                        let metadata_path = args.output_dir.join(&metadata_basename);
                        save_metadata_sidecar(&metadata_path, &sidecar, args.export_json_metadata)?;
                        let metadata_db_name = format!("item_{}.hypergraph.db", id_str);

                        // Split into parts if too large to prevent RAM issues
                        let parts = (count + MAX_TOKENS_PER_FILE - 1) / MAX_TOKENS_PER_FILE;

                        if parts == 1 {
                            // Single file
                            let filename = format!("item_{}.safetensors", id_str);
                            let path = args.output_dir.join(filename);
                            save_safetensors(
                                &path,
                                &emb,
                                &norms,
                                &entropies,
                                &coherence,
                                &idxs,
                                &mask,
                                count,
                                Some(&metadata_db_name),
                            )?;
                        } else {
                            // Split into multiple parts
                            for part in 0..parts {
                                let start = part * MAX_TOKENS_PER_FILE;
                                let end = ((part + 1) * MAX_TOKENS_PER_FILE).min(count);
                                let part_tokens = end - start;

                                // Extract ranges for this part
                                let emb_start = start * 768;
                                let emb_end = end * 768;

                                let filename = format!("item_{}_part_{}.safetensors", id_str, part);
                                let path = args.output_dir.join(filename);

                                // Create slices for this part
                                save_safetensors(
                                    &path,
                                    &emb[emb_start..emb_end],
                                    &norms[start..end],
                                    &entropies[start..end],
                                    &coherence[start..end],
                                    &idxs, // Keep all patch indices for context
                                    &mask[start..end],
                                    part_tokens,
                                    Some(&metadata_db_name),
                                )?;
                            }
                        }
                    }
                }
                Err(e) => println!("Error processing item {}: {}", id_str, e),
            }
        }
    }

    println!("✅ Ingestion complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use blt_burn::pretokenize::{Segment, SegmentMetadata};
    use serde_json::json;

    #[test]
    fn test_coherence_injection() {
        // Mock coherence scores for a 5-token segment
        let coherence_f32 = vec![0.8, 1.2, 0.9, 1.5, 1.1];
        let len = 5;
        let start = 0;
        let end = 5;

        // Simulate segment creation (mock bytes and metadata)
        let mut seg = Segment {
            bytes: vec![0u8; len],
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
            let expected_coherence = 1.1; // (0.8+1.2+0.9+1.5+1.1)/5 = 1.1

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

        let mut seg = Segment {
            bytes: vec![0u8; len],
            metadata: None, // No initial metadata
        };

        // Inject coherence
        if end <= coherence_f32.len() {
            let segment_coherence: f32 = coherence_f32[start..end].iter().sum::<f32>() / (len as f32);
            let expected_coherence = 1.5;

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
