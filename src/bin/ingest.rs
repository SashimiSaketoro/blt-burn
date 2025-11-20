use blt_burn::{
    model::{LMTransformerConfig, LMTransformer},
    tokenizer::BltTokenizer,
    patcher::{patch_start_mask_from_entropy_with_monotonicity, patch_start_indices_cpu},
    dataset::FineWebEduDataset,
    pretokenize::{detect_modality, ByteSegment, SegmentMetadata},
};
use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    tensor::Tensor,
    record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder},
    module::Module,
};
use clap::Parser;
use std::path::{Path, PathBuf};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::serialize;
use std::fs::{self, File};
use std::io::Write;
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

const MAX_TOKENS_PER_FILE: usize = 100_000; // Split large files into parts to prevent RAM explosion

#[derive(Debug, Serialize, Deserialize)]
struct MetadataSidecar {
    source_hash: String,
    total_bytes: usize,
    modality: String,
    segments: Vec<ByteSegment>,
}

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
    
    /// Entropy threshold
    #[arg(short = 'r', long, default_value_t = 1.35)]
    threshold: f64,

    /// Output directory for safetensors shards
    #[arg(short, long, default_value = "ingest_output")]
    output_dir: PathBuf,

    /// Cache directory for dataset download (default: dataset_cache)
    #[arg(long, default_value = "dataset_cache")]
    cache_dir: String,
}

fn compute_hash(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

fn process_data(
    data: &[u8],
    model: &LMTransformer<Wgpu>,
    device: &WgpuDevice,
    threshold: f64,
) -> Result<(Vec<f32>, Vec<f32>, Vec<i32>, Vec<i32>, usize, MetadataSidecar), anyhow::Error> {
    // 1. Compute Hash
    let source_hash = compute_hash(data);

    // 2. Pre-tokenize
    let pt_type = detect_modality(data);
    let pretokenizer = pt_type.create()?;
    let modality_name = pretokenizer.modality().to_string();
    
    let segments = pretokenizer.pre_tokenize(data)?;
    
    // 3. Flatten to model inputs
    let tokens_vec: Vec<i32> = segments.iter()
        .flat_map(|s| s.bytes.iter().map(|&b| b as i32))
        .collect();
        
    let total_tokens = tokens_vec.len();
    
    if total_tokens == 0 {
        return Ok((vec![], vec![], vec![], vec![], 0, MetadataSidecar {
            source_hash,
            total_bytes: 0,
            modality: modality_name,
            segments: vec![],
        }));
    }

    let chunk_size = 1024;
    let stride = 512; // Process 512 new tokens per chunk, keeping 512 as context
    let context_size = chunk_size - stride; // 512 tokens of context
    
    let mut chunk_entropies_list = Vec::new();
    let mut chunk_norms_list = Vec::new();
    let mut chunk_embeddings_list = Vec::new();
    
    let mut position = 0;
    while position < tokens_vec.len() {
        // Determine chunk boundaries
        let start = if position == 0 { 0 } else { position - context_size };
        let end = (position + stride).min(tokens_vec.len());
        let chunk = &tokens_vec[start..end];
        let chunk_len = chunk.len();
        
        // Skip index to ignore low-context outputs at the beginning of non-first chunks
        let skip_count = if position == 0 { 0 } else { context_size };
        
        let input = Tensor::<Wgpu, 1, burn::tensor::Int>::from_ints(
            chunk,
            device,
        ).reshape([1, chunk_len]);

        let output = model.forward_with_embeddings(input);
        let chunk_entropies = blt_burn::patcher::entropy(output.logits);
        
        // Extract only the valid portion (skip context that was already processed)
        if skip_count > 0 {
            let valid_entropies = chunk_entropies.clone().slice([0..1, skip_count..chunk_len]);
            let valid_norms = output.embedding_norms.clone().slice([0..1, skip_count..chunk_len]);
            let valid_embeddings = output.pre_norm_embeddings.clone().slice([0..1, skip_count..chunk_len, 0..768]);
            
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
    
    let mask = patch_start_mask_from_entropy_with_monotonicity(entropies.clone(), threshold);
    let patch_indices = patch_start_indices_cpu(mask);
    
    let embeddings_data = embeddings.into_data();
    let norms_data = norms.into_data();
    
    let embeddings_f32: Vec<f32> = embeddings_data.iter::<f32>().collect();
    let norms_f32: Vec<f32> = norms_data.iter::<f32>().collect();
    
    let mut patch_mask_vec = vec![0i32; total_tokens];
    let patch_indices_inner = if !patch_indices.is_empty() { &patch_indices[0] } else { &vec![] };
    
    for &idx in patch_indices_inner {
        if idx < total_tokens {
            patch_mask_vec[idx] = 1;
        }
    }
    
    let patch_indices_i32: Vec<i32> = patch_indices_inner.iter().map(|&x| x as i32).collect();
    
    let sidecar = MetadataSidecar {
        source_hash,
        total_bytes: data.len(),
        modality: modality_name,
        segments,
    };
    
    Ok((embeddings_f32, norms_f32, patch_indices_i32, patch_mask_vec, total_tokens, sidecar))
}


fn save_metadata_sidecar(path: &Path, sidecar: &MetadataSidecar) {
    let file = File::create(path).expect("Failed to create metadata sidecar file");
    serde_json::to_writer_pretty(file, sidecar).expect("Failed to write metadata sidecar");
}

fn save_safetensors(
    path: &Path, 
    embeddings_f32: &[f32], 
    norms_f32: &[f32], 
    patch_indices_i32: &[i32], 
    patch_mask_vec: &[i32], 
    total_tokens: usize,
    metadata_filename: Option<&str>
) {
    // Safety rail: Don't create empty files
    if total_tokens == 0 {
        println!("Warning: Skipping save_safetensors - no tokens to save");
        return;
    }
    
    let tensors: Vec<(&str, TensorView)> = vec![
        ("embeddings", TensorView::new(
            Dtype::F32,
            vec![1, total_tokens, 768],
            bytemuck::cast_slice(embeddings_f32)
        ).unwrap()),
        ("prominence", TensorView::new(
            Dtype::F32,
            vec![1, total_tokens],
            bytemuck::cast_slice(norms_f32)
        ).unwrap()),
        ("patch_indices", TensorView::new(
            Dtype::I32,
            vec![patch_indices_i32.len()],
            bytemuck::cast_slice(patch_indices_i32)
        ).unwrap()),
        ("patch_mask", TensorView::new(
            Dtype::I32,
            vec![1, total_tokens],
            bytemuck::cast_slice(patch_mask_vec)
        ).unwrap()),
    ];

    let mut metadata_map = std::collections::HashMap::new();
    if let Some(mf) = metadata_filename {
        metadata_map.insert("metadata_file".to_string(), mf.to_string());
    }

    let serialized = serialize(tensors, &Some(metadata_map)).expect("Serialization failed");
    let mut file = File::create(path).expect("Failed to create output file");
    file.write_all(&serialized).expect("Failed to write to file");
}

fn main() {
    let args = Args::parse();
    
    let device = WgpuDevice::default();
    type Backend = Wgpu;
    
    // Initialize Model
    let config = LMTransformerConfig {
        dim: 768, n_layers: 14, head_dim: None, n_heads: Some(12), n_kv_heads: None,
        ffn_dim_multiplier: Some(1.0), multiple_of: 256, norm_eps: 1e-5,
        rope_theta: 10000.0, max_seqlen: 8192, vocab_size: 260,
    };
    let model = config.init::<Backend>(&device);
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();

    // Determine model path: argument > build-time env var > default local
    let model_path = args.model_path.or_else(|| {
        option_env!("BLT_MODEL_CACHE_PATH").map(PathBuf::from)
    }).unwrap_or_else(|| PathBuf::from("blt_entropy_model.mpk"));

    println!("Loading weights from {:?}", model_path);
    let model = model.load_record(recorder.load(model_path.into(), &device).expect("Failed to load weights"));
    
    // Create output directory
    fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");
    
    // Default to dataset mode unless text/file is explicitly provided
    if args.text.is_some() || args.file.is_some() {
        // Single file/text mode
        let (bytes, filename_prefix) = if let Some(t) = args.text { 
            (t.into_bytes(), "manual_input".to_string())
        } else if let Some(f) = args.file {
            let b = std::fs::read(&f).expect("Failed to read input file");
            let name = f.file_stem().unwrap().to_string_lossy().to_string();
            (b, name)
        } else { panic!("Should not happen") };

        println!("Processing single input ({} bytes)...", bytes.len());
        match process_data(&bytes, &model, &device, args.threshold) {
            Ok((emb, norms, idxs, mask, count, sidecar)) => {
                let metadata_filename = format!("{}.metadata.json", filename_prefix);
                let metadata_path = args.output_dir.join(&metadata_filename);
                save_metadata_sidecar(&metadata_path, &sidecar);

                let path = args.output_dir.join(format!("{}.safetensors", filename_prefix));
                save_safetensors(&path, &emb, &norms, &idxs, &mask, count, Some(&metadata_filename));
            },
            Err(e) => println!("Error processing input: {}", e),
        }
    } else {
        // Dataset mode (default)
        println!("Loading FineWeb-Edu dataset ({}/{})...", args.subset, args.split);
        println!("Cache directory: {}", args.cache_dir);
        fs::create_dir_all(&args.cache_dir).expect("Failed to create cache directory");
        
        let dataset = FineWebEduDataset::new(&args.subset, &args.split, &args.cache_dir).expect("Failed to load dataset");
        
        println!("Dataset size: {}", dataset.len());
        let limit = args.limit.unwrap_or(dataset.len());
        
        for (i, item) in dataset.iter().take(limit).enumerate() {
            let id_str = item.id.clone().unwrap_or_else(|| format!("{}", i));
            println!("Processing item {}/{} (ID: {})...", i + 1, limit, id_str);
            
            // Skip empty
            if item.text.is_empty() { continue; }
            
            let bytes = item.text.into_bytes();
            match process_data(&bytes, &model, &device, args.threshold) {
                Ok((emb, norms, idxs, mask, count, sidecar)) => {
                    if count > 0 {
                        // Save metadata sidecar (one per item)
                        let metadata_filename = format!("item_{}.metadata.json", id_str);
                        let metadata_path = args.output_dir.join(&metadata_filename);
                        save_metadata_sidecar(&metadata_path, &sidecar);

                        // Split into parts if too large to prevent RAM issues
                        let parts = (count + MAX_TOKENS_PER_FILE - 1) / MAX_TOKENS_PER_FILE;
                        
                        if parts == 1 {
                            // Single file
                            let filename = format!("item_{}.safetensors", id_str);
                            let path = args.output_dir.join(filename);
                            save_safetensors(&path, &emb, &norms, &idxs, &mask, count, Some(&metadata_filename));
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
                                    &idxs,  // Keep all patch indices for context
                                    &mask[start..end], 
                                    part_tokens,
                                    Some(&metadata_filename)
                                );
                            }
                        }
                    }
                },
                Err(e) => println!("Error processing item {}: {}", id_str, e),
            }
        }
    }
    
    println!("✅ Ingestion complete!");
}
