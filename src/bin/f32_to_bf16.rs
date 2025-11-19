// Simple Python script to convert safetensors to bf16 MPK using burn-import
// 
// According to Burn docs, SafetensorsFileRecorder is in burn_import::safetensors
// but it's not exposed in the current burn-import crate.
//
// WORKAROUND: Load existing f32 MPK and save as bf16

use blt_burn::model::LMTransformerRecord;
use burn::record::{FullPrecisionSettings, HalfPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::backend::wgpu::{Wgpu, WgpuDevice};

type Backend = Wgpu;

fn main() -> anyhow::Result<()> {
    let device = WgpuDevice::default();
    
    println!("📥 Loading existing f32 model...");
    
    // Download from HuggingFace if not present  
    let model_path = "blt_entropy_model_f32.mpk";
    if !std::path::Path::new(model_path).exists() {
        println!("❌ f32 model not found");
        println!("   Please ensure blt_entropy_model.mpk exists");
        return Err(anyhow::anyhow!("Model file not found"));
    }
    
    // Load f32 MPK
    let f32_recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let record: LMTransformerRecord<Backend> = f32_recorder
        .load(model_path.into(), &device)?;
    
    println!("✓ Loaded f32 model");
    println!("💾 Converting to bf16...");
    
    // Save as bf16 MPK
    let bf16_recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
    bf16_recorder.record(record, "blt_entropy_model.mpk".into())?;
    
    println!("✅ Conversion complete!");
    println!("   Output: blt_entropy_model.mpk (bf16 precision)");
    
    // Show file sizes
    let f32_size = std::fs::metadata(model_path)?.len() as f64 / 1024.0 / 1024.0;
    let bf16_size = std::fs::metadata("blt_entropy_model.mpk")?.len() as f64 / 1024.0 / 1024.0;
    
    println!("\n📊 Size comparison:");
    println!(" f32:  {:.1} MB", f32_size);
    println!("  bf16: {:.1} MB", bf16_size);
    println!("   Savings: {:.1} MB ({:.0}%)", f32_size - bf16_size, (1.0 - bf16_size/f32_size) * 100.0);
    
    Ok(())
}
