// Clippy pedantic allows for binary
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]

use blt_burn::{
    model::LMTransformerConfig,
    patcher::{patch_start_indices_cpu, patch_start_mask_from_entropy_with_monotonicity},
    tokenizer::BltTokenizer,
};
use burn::{
    backend::wgpu::{Wgpu, WgpuDevice},
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::Tensor,
};
use burn_import::safetensors::SafetensorsFileRecorder;
use clap::Parser;
use safetensors::serialize;
use safetensors::tensor::{Dtype, TensorView};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input text to patch
    #[arg(short, long)]
    text: Option<String>,

    /// Path to input file
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Path to model weights (.safetensors or .mpk format)
    #[arg(short, long)]
    model_path: Option<PathBuf>,

    /// Force using .mpk format instead of safetensors
    #[arg(long, default_value_t = false)]
    use_mpk: bool,

    /// Entropy threshold
    #[arg(short = 'r', long, default_value_t = 1.35)]
    threshold: f64,

    /// Output path for .safetensors file containing embeddings and metadata
    #[arg(short, long, default_value = "blt_output.safetensors")]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();

    // Determine model path and format
    // Priority: CLI arg > safetensors env var > mpk env var > default
    let (model_path, use_safetensors) = if let Some(path) = args.model_path {
        let is_safetensors = path
            .extension()
            .map(|e| e == "safetensors")
            .unwrap_or(false);
        (path, is_safetensors && !args.use_mpk)
    } else if let Some(path) = option_env!("BLT_MODEL_SAFETENSORS_PATH") {
        (PathBuf::from(path), true)
    } else if let Some(path) = option_env!("BLT_MODEL_CACHE_PATH") {
        (PathBuf::from(path), false)
    } else {
        // Default to looking for safetensors in HF cache
        let home = std::env::var("HOME").unwrap_or_default();
        let hf_path = PathBuf::from(&home)
            .join(".cache/huggingface/hub/models--facebook--blt-entropy/snapshots");

        let mut found_path: Option<PathBuf> = None;
        if hf_path.exists() {
            if let Ok(entries) = std::fs::read_dir(&hf_path) {
                for entry in entries.flatten() {
                    let model_path = entry.path().join("model.safetensors");
                    if model_path.exists() {
                        println!("Found Facebook BLT entropy model in HF cache");
                        found_path = Some(model_path);
                        break;
                    }
                }
            }
        }

        if let Some(path) = found_path {
            (path, true)
        } else {
            // Fallback to mpk
            (PathBuf::from("blt_entropy_model.mpk"), false)
        }
    };

    let text = if let Some(t) = args.text {
        t
    } else if let Some(f) = args.file {
        let bytes = std::fs::read(f).expect("Failed to read input file");
        String::from_utf8_lossy(&bytes).to_string()
    } else {
        panic!("Must provide either --text or --file");
    };

    // Initialize device
    let device = WgpuDevice::default();
    type Backend = Wgpu;

    // Initialize Tokenizer
    let tokenizer = BltTokenizer::new(true, true);
    let tokens = tokenizer.encode(&text);
    println!("Tokens: {}", tokens.len());

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

    // Load weights from safetensors or mpk file
    println!(
        "Loading weights from {} (format: {})",
        model_path.display(),
        if use_safetensors {
            "safetensors"
        } else {
            "mpk"
        }
    );

    let model = if use_safetensors {
        let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
        model.load_record(
            recorder
                .load(model_path.clone().into(), &device)
                .expect("Failed to load safetensors weights"),
        )
    } else {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        model.load_record(
            recorder
                .load(model_path.clone(), &device)
                .expect("Failed to load mpk weights"),
        )
    };

    println!("Model initialized and weights loaded (with RoPE and causal mask).");

    // Run Model in Chunks with Overlapping Windows
    let chunk_size = 1024; // Full context window size
    let stride = 512; // Process 512 new tokens per chunk, keeping 512 as context
    let context_size = chunk_size - stride; // 512 tokens of context

    let mut chunk_entropies_list = Vec::new();
    let mut chunk_norms_list = Vec::new();
    let mut chunk_embeddings_list = Vec::new();

    let tokens_vec: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let total_tokens = tokens_vec.len();

    println!(
        "Processing {total_tokens} tokens with overlapping windows (chunk_size={chunk_size}, stride={stride})..."
    );

    let mut position = 0;
    let mut chunk_idx = 0;
    while position < tokens_vec.len() {
        if chunk_idx % 5 == 0 {
            let progress = (position as f32 / total_tokens as f32 * 100.0) as u32;
            println!(
                "Processing position {position}/{total_tokens} ({progress}%)"
            );
        }

        // Determine chunk boundaries with context overlap
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

        let input = Tensor::<Backend, 1, burn::tensor::Int>::from_ints(chunk, &device)
            .reshape([1, chunk_len]);

        // Note: Use forward_with_embeddings to get density signals
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
        chunk_idx += 1;
    }

    println!("Aggregating results...");

    // Concatenate all chunks on device
    let entropies = Tensor::cat(chunk_entropies_list, 1).reshape([1, total_tokens]); // [1, seq_len]
    let norms = Tensor::cat(chunk_norms_list, 1).reshape([1, total_tokens]); // [1, seq_len]
    let embeddings = Tensor::cat(chunk_embeddings_list, 1).reshape([1, total_tokens, 768]); // [1, seq_len, dim]

    // Patching
    let mask = patch_start_mask_from_entropy_with_monotonicity(entropies.clone(), args.threshold);
    let patch_indices = patch_start_indices_cpu(mask); // Vec<Vec<usize>>

    // Prepare data for export
    // Move to CPU/Data for serialization
    let embeddings_data = embeddings.into_data();
    let norms_data = norms.into_data();
    let _patch_indices_flat: Vec<i32> = patch_indices[0].iter().map(|&x| x as i32).collect();

    // Convert to flattened f32 vectors for safetensors
    let embeddings_f32: Vec<f32> = embeddings_data.iter::<f32>().collect();
    let norms_f32: Vec<f32> = norms_data.iter::<f32>().collect();

    // Create patch mask tensor (1 where patch starts, 0 otherwise)
    // This is easier for JAX to consume than indices
    let mut patch_mask_vec = vec![0i32; total_tokens];
    for &idx in &patch_indices[0] {
        if idx < total_tokens {
            patch_mask_vec[idx] = 1;
        }
    }

    // Create patch indices tensor
    // Safetensors doesn't support U64/usize, using I32
    let patch_indices_i32: Vec<i32> = patch_indices[0].iter().map(|&x| x as i32).collect();

    println!("Exporting to {}...", args.output.display());
    println!("  Embeddings: [{}, {}]", total_tokens, 768);
    println!("  Norms: [{total_tokens}]");
    println!("  Patches: {}", patch_indices[0].len());

    // SAFETENSORS EXPORT
    let tensors: Vec<(&str, TensorView)> = vec![
        (
            "embeddings",
            TensorView::new(
                Dtype::F32,
                vec![1, total_tokens, 768],
                bytemuck::cast_slice(&embeddings_f32),
            )
            .unwrap(),
        ),
        (
            "prominence",
            TensorView::new(
                Dtype::F32,
                vec![1, total_tokens],
                bytemuck::cast_slice(&norms_f32),
            )
            .unwrap(),
        ),
        (
            "patch_indices",
            TensorView::new(
                Dtype::I32,
                vec![patch_indices_i32.len()],
                bytemuck::cast_slice(&patch_indices_i32),
            )
            .unwrap(),
        ),
        (
            "patch_mask",
            TensorView::new(
                Dtype::I32,
                vec![1, total_tokens],
                bytemuck::cast_slice(&patch_mask_vec),
            )
            .unwrap(),
        ),
    ];

    let serialized = serialize(tensors, &None).expect("Serialization failed");
    let mut file = File::create(args.output).expect("Failed to create output file");
    file.write_all(&serialized)
        .expect("Failed to write to file");

    println!("✅ Done! Preprocessing complete.");
}
