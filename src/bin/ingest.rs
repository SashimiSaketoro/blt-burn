use blt_burn::{
    model::{LMTransformerConfig, LMTransformer},
    tokenizer::BltTokenizer,
    patcher::{patch_start_mask_from_entropy_with_monotonicity, patch_start_indices_cpu},
    dataset::FineWebEduDataset,
};
use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    tensor::{Tensor, ElementConversion},
    record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder},
    module::Module,
};
use clap::Parser;
use std::path::{Path, PathBuf};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::serialize;
use std::fs::{self, File};
use std::io::Write;
use hf_hub::{api::sync::Api, Repo, RepoType};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input text to patch (optional override)
    #[arg(short, long)]
    text: Option<String>,

    /// Path to input file (optional override)
    #[arg(short, long)]
    file: Option<PathBuf>,
    
    /// Use FineWeb-Edu dataset
    #[arg(long, default_value_t = false)]
    dataset: bool,
    
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
    #[arg(short, long, default_value = "blt_entropy_model.mpk")]
    model_path: PathBuf,
    
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

fn process_text(
    text: &str,
    model: &LMTransformer<Wgpu>,
    tokenizer: &BltTokenizer,
    device: &WgpuDevice,
    threshold: f64,
) -> (Vec<f32>, Vec<f32>, Vec<i32>, Vec<i32>, usize) {
    let tokens = tokenizer.encode(text);
    let tokens_vec: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let total_tokens = tokens_vec.len();
    
    if total_tokens == 0 {
        return (vec![], vec![], vec![], vec![], 0);
    }

    let chunk_size = 1024;
    let mut chunk_entropies_list = Vec::new();
    let mut chunk_norms_list = Vec::new();
    let mut chunk_embeddings_list = Vec::new();
    
    for chunk in tokens_vec.chunks(chunk_size) {
        let chunk_len = chunk.len();
        let input = Tensor::<Wgpu, 1, burn::tensor::Int>::from_ints(
            chunk,
            device,
        ).reshape([1, chunk_len]);

        let output = model.forward_with_embeddings(input);
        let chunk_entropies = blt_burn::patcher::entropy(output.logits);
        
        chunk_entropies_list.push(chunk_entropies);
        chunk_norms_list.push(output.embedding_norms);
        chunk_embeddings_list.push(output.pre_norm_embeddings);
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
    
    (embeddings_f32, norms_f32, patch_indices_i32, patch_mask_vec, total_tokens)
}

fn save_safetensors(
    path: &Path, 
    embeddings_f32: &[f32], 
    norms_f32: &[f32], 
    patch_indices_i32: &[i32], 
    patch_mask_vec: &[i32], 
    total_tokens: usize
) {
    let tensors: Vec<(&str, TensorView)> = vec![
        ("embeddings", TensorView::new(
            Dtype::F32,
            vec![1, total_tokens, 768],
            unsafe { std::slice::from_raw_parts(embeddings_f32.as_ptr() as *const u8, embeddings_f32.len() * 4) }
        ).unwrap()),
        ("prominence", TensorView::new(
            Dtype::F32,
            vec![1, total_tokens],
            unsafe { std::slice::from_raw_parts(norms_f32.as_ptr() as *const u8, norms_f32.len() * 4) }
        ).unwrap()),
        ("patch_indices", TensorView::new(
            Dtype::I32,
            vec![patch_indices_i32.len()],
            unsafe { std::slice::from_raw_parts(patch_indices_i32.as_ptr() as *const u8, patch_indices_i32.len() * 4) }
        ).unwrap()),
        ("patch_mask", TensorView::new(
            Dtype::I32,
            vec![1, total_tokens],
            unsafe { std::slice::from_raw_parts(patch_mask_vec.as_ptr() as *const u8, patch_mask_vec.len() * 4) }
        ).unwrap()),
    ];

    let serialized = serialize(tensors, &None).expect("Serialization failed");
    let mut file = File::create(path).expect("Failed to create output file");
    file.write_all(&serialized).expect("Failed to write to file");
}

fn main() {
    let args = Args::parse();
    
    let device = WgpuDevice::default();
    type Backend = Wgpu;
    let tokenizer = BltTokenizer::new(true, true);
    
    // Initialize Model
    let config = LMTransformerConfig {
        dim: 768, n_layers: 14, head_dim: None, n_heads: Some(12), n_kv_heads: None,
        ffn_dim_multiplier: Some(1.0), multiple_of: 256, norm_eps: 1e-5,
        rope_theta: 10000.0, max_seqlen: 8192, vocab_size: 260,
    };
    let model = config.init::<Backend>(&device);
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();

    // Check if model exists, if not download from HF
    let model_path = if args.model_path.exists() {
        println!("Found local model at {:?}", args.model_path);
        args.model_path
    } else {
        println!("Model not found at {:?}, downloading from Hugging Face...", args.model_path);
        let api = Api::new().expect("Failed to create HF API");
        let repo = api.repo(Repo::new("SashimiSaketoro/entropy_burn".to_string(), RepoType::Model));
        let path = repo.get("blt_entropy_model.mpk").expect("Failed to download model");
        println!("Downloaded model to {:?}", path);
        path
    };

    let model = model.load_record(recorder.load(model_path.into(), &device).expect("Failed to load weights"));
    
    // Create output directory
    fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");
    
    if args.text.is_some() || args.file.is_some() {
        // Single file/text mode
        let text = if let Some(t) = args.text { t } else if let Some(f) = args.file {
            let bytes = std::fs::read(f).expect("Failed to read input file");
            String::from_utf8_lossy(&bytes).to_string()
        } else { panic!("Should not happen") };

        println!("Processing single text input...");
        let (emb, norms, idxs, mask, count) = process_text(&text, &model, &tokenizer, &device, args.threshold);
        let path = args.output_dir.join("manual_input.safetensors");
        save_safetensors(&path, &emb, &norms, &idxs, &mask, count);
    } else {
        // Dataset mode
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
            
            let (emb, norms, idxs, mask, count) = process_text(&item.text, &model, &tokenizer, &device, args.threshold);
            
            if count > 0 {
                let filename = format!("item_{}.safetensors", id_str);
                let path = args.output_dir.join(filename);
                save_safetensors(&path, &emb, &norms, &idxs, &mask, count);
            }
        }
    }
    
    println!("✅ Ingestion complete!");
}
