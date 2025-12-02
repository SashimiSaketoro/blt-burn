//! Integration tests for blt_core module.
//!
//! These tests verify the end-to-end BLT data flow:
//! text → bytes → tokens → entropy_model → logits → entropy → patches

use blt_burn::blt_core::{process_bytes_with_embeddings, BltConfig, BltExample};
use blt_burn::tokenizer::OFFSET;

#[test]
fn test_blt_config_default() {
    let config = BltConfig::default();
    assert!((config.threshold - 1.335).abs() < 0.001);
    assert!(config.monotonicity);
    assert_eq!(config.max_seq_len, 1024);
    assert_eq!(config.chunk_overlap, 512);
}

#[test]
fn test_blt_example_serialization() {
    let example = BltExample {
        sample_id: "test_001".to_string(),
        text: Some("Hello, world!".to_string()),
        tokens: vec![72 + OFFSET as i32, 101 + OFFSET as i32], // "He"
        entropies: vec![1.5, 2.0],
        patch_lengths: vec![2],
        mask: vec![true, true],
    };

    // Test JSON serialization
    let json = serde_json::to_string(&example).expect("JSON serialization failed");
    let parsed: BltExample = serde_json::from_str(&json).expect("JSON deserialization failed");

    assert_eq!(parsed.sample_id, example.sample_id);
    assert_eq!(parsed.tokens, example.tokens);
    assert_eq!(parsed.entropies.len(), example.entropies.len());
}

#[test]
fn test_token_encoding() {
    use blt_burn::tokenizer::BltTokenizer;

    let tokenizer = BltTokenizer::new(false, false);
    let tokens = tokenizer.encode("ABC");

    // 'A' = 65, 'B' = 66, 'C' = 67
    // With OFFSET = 4: [69, 70, 71]
    assert_eq!(tokens.len(), 3);
    assert_eq!(tokens[0], 65 + OFFSET);
    assert_eq!(tokens[1], 66 + OFFSET);
    assert_eq!(tokens[2], 67 + OFFSET);
}

#[test]
fn test_token_decoding() {
    use blt_burn::blt_core::tokens_to_text;

    let tokens: Vec<i32> = "Hello".bytes().map(|b| b as i32 + OFFSET as i32).collect();
    let text = tokens_to_text(&tokens);
    assert_eq!(text, "Hello");
}

// Note: Full model inference tests require GPU and loaded weights.
// Run with: cargo test --features integration -- --ignored
#[test]
#[ignore]
fn test_full_inference_with_model() {
    use blt_burn::model::LMTransformerConfig;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, Recorder};
    use burn_import::safetensors::SafetensorsFileRecorder;

    let device = WgpuDevice::default();

    // Initialize model
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

    let model = config.init::<Wgpu>(&device);

    // Try to load weights from env var or default path
    let model_path = std::env::var("BLT_MODEL_SAFETENSORS_PATH")
        .unwrap_or_else(|_| "model.safetensors".to_string());

    let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
    let model = model.load_record(
        recorder
            .load(model_path.into(), &device)
            .expect("Failed to load model weights"),
    );

    // Process a simple text
    let blt_config = BltConfig::default();
    let result = process_bytes_with_embeddings(
        b"The quick brown fox jumps over the lazy dog.",
        "test_sample",
        &model,
        &device,
        &blt_config,
    );

    // Verify outputs
    assert!(!result.core.tokens.is_empty());
    assert_eq!(result.core.tokens.len(), result.core.entropies.len());
    assert_eq!(result.core.tokens.len(), result.prominence.len());
    assert_eq!(result.core.tokens.len(), result.coherence_scores.len());
    assert_eq!(
        result.pre_norm_embeddings.len(),
        result.core.tokens.len() * result.embedding_dim
    );

    // Verify patch boundaries are computed
    assert!(!result.core.patch_lengths.is_empty());
    let total_from_patches: i32 = result.core.patch_lengths.iter().sum();
    assert_eq!(total_from_patches as usize, result.core.tokens.len());

    println!("✅ Full inference test passed!");
    println!("   Tokens: {}", result.core.tokens.len());
    println!("   Patches: {}", result.core.patch_lengths.len());
    println!(
        "   Avg entropy: {:.3}",
        result.core.entropies.iter().sum::<f32>() / result.core.entropies.len() as f32
    );
}
