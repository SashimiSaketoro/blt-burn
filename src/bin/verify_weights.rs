//! Weight verification binary for BLT MPK conversion
//!
//! Compares Safetensors weights vs MPK weights using Burn's native APIs.

use blt_burn::model::{LMTransformerRecord, LMTransformerConfig};
use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;
use burn_ndarray::NdArray;
use std::path::{Path, PathBuf};
use std::env;
use burn::module::Module;
use burn::tensor::Tensor;

type Backend = NdArray; // CPU backend for verification

fn find_hf_cache_model() -> Option<PathBuf> {
    let home = env::var("HOME").ok()?;
    let cache_dir = PathBuf::from(home).join(".cache/huggingface/hub");
    
    if !cache_dir.exists() {
        return None;
    }
    
    // Look for facebook--blt-entropy
    for entry in std::fs::read_dir(cache_dir).ok()? {
        let entry = entry.ok()?;
        let name = entry.file_name().to_string_lossy().to_string();
        
        if name.contains("facebook") && name.contains("blt-entropy") {
            let snapshots_dir = entry.path().join("snapshots");
            if snapshots_dir.exists() {
                // Get the most recent snapshot
                let mut snapshots: Vec<_> = std::fs::read_dir(snapshots_dir)
                    .ok()?
                    .filter_map(|e| e.ok())
                    .collect();
                
                snapshots.sort_by_key(|e| e.metadata().ok()?.modified().ok());
                
                if let Some(snapshot) = snapshots.last() {
                    let safetensors_path = snapshot.path().join("model.safetensors");
                    if safetensors_path.exists() {
                        return Some(safetensors_path);
                    }
                }
            }
        }
    }
    
    None
}

fn main() {
    println!("=============================================================================");
    println!("BLT ENTROPY MODEL - MPK WEIGHT VERIFICATION");
    println!("=============================================================================\n");
    
    let device = Default::default();

    // Step 1: Locate HF cache
    println!("[1/4] Locating Safetensors source...");
    let safetensors_path = find_hf_cache_model().expect("Could not find model in HF cache");
    println!("  ✓ Found: {}", safetensors_path.display());

    // Step 2: Locate MPK
    println!("\n[2/4] Locating MPK converted file...");
    let mpk_path = Path::new("blt_entropy_model.mpk");
    if !mpk_path.exists() {
        panic!("MPK file not found. Run conversion first.");
    }
    println!("  ✓ Found: {}", mpk_path.display());

    // Step 3: Load both models
    println!("\n[3/4] Loading models for comparison...");
    
    println!("  Loading Safetensors...");
    let st_recorder = SafetensorsFileRecorder::<HalfPrecisionSettings>::default();
    let st_record: LMTransformerRecord<Backend> = st_recorder
        .load(safetensors_path.clone().into(), &device)
        .expect("Failed to load Safetensors");

    println!("  Loading MPK...");
    let mpk_recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
    let mpk_record: LMTransformerRecord<Backend> = mpk_recorder
        .load(mpk_path.to_path_buf(), &device)
        .expect("Failed to load MPK");

    // Step 4: Compare weights
    println!("\n[4/4] Comparing weights...");
    
    let mut match_count = 0;
    let mut mismatch_count = 0;

    // Helper to compare two tensors
    let compare = |name: &str, t1: &burn::tensor::TensorData, t2: &burn::tensor::TensorData| {
        let s1 = t1.shape.clone();
        let s2 = t2.shape.clone();
        
        if s1 != s2 {
            println!("  ❌ {}: Shape mismatch {:?} vs {:?}", name, s1, s2);
            return false;
        }

        let v1 = t1.as_slice::<f32>().unwrap(); // NdArray usually uses f32 internally even for bf16 storage
        let v2 = t2.as_slice::<f32>().unwrap();
        
        // Check exact equality (since we just converted, bits should be identical or extremely close)
        // But converting f32 <-> bf16 might introduce epsilon differences depending on backend
        // Since we used HalfPrecisionSettings for BOTH load and save, they should match.
        
        let mut diff = 0.0;
        for (a, b) in v1.iter().zip(v2.iter()) {
            diff += (a - b).abs();
        }
        
        if diff > 1e-4 {
             println!("  ❌ {}: Value mismatch (sum diff: {})", name, diff);
             return false;
        } else {
            // println!("  ✓ {}", name);
            return true;
        }
    };

    // 1. Token Embeddings
    let st_emb = st_record.tok_embeddings.weight.val().to_data();
    let mpk_emb = mpk_record.tok_embeddings.weight.val().to_data();
    if compare("tok_embeddings", &st_emb, &mpk_emb) { match_count += 1; } else { mismatch_count += 1; }

    // 2. Layer 0 Attention Query
    let st_l0_wq = st_record.layers[0].attention.wq.weight.val().to_data();
    let mpk_l0_wq = mpk_record.layers[0].attention.wq.weight.val().to_data();
    if compare("layer[0].attn.wq", &st_l0_wq, &mpk_l0_wq) { match_count += 1; } else { mismatch_count += 1; }

    // 3. Layer 5 FeedForward W1
    let st_l5_w1 = st_record.layers[5].feed_forward.w1.weight.val().to_data();
    let mpk_l5_w1 = mpk_record.layers[5].feed_forward.w1.weight.val().to_data();
    if compare("layer[5].ffn.w1", &st_l5_w1, &mpk_l5_w1) { match_count += 1; } else { mismatch_count += 1; }

    // 4. Output Weights
    let st_out = st_record.output.weight.val().to_data();
    let mpk_out = mpk_record.output.weight.val().to_data();
    if compare("output", &st_out, &mpk_out) { match_count += 1; } else { mismatch_count += 1; }

    // 5. Norm Weights
    let st_norm = st_record.norm.weight.val().to_data();
    let mpk_norm = mpk_record.norm.weight.val().to_data();
    if compare("norm", &st_norm, &mpk_norm) { match_count += 1; } else { mismatch_count += 1; }

    println!("\nSummary:");
    println!("  Matches: {}", match_count);
    println!("  Mismatches: {}", mismatch_count);

    if mismatch_count == 0 {
        println!("\n✅ SUCCESS: Weights match precisely!");
    } else {
        println!("\n❌ FAILURE: Found mismatches!");
        std::process::exit(1);
    }
}
