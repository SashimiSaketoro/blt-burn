use blt_burn::model::LMTransformerRecord;
use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;
use burn_ndarray::NdArray;
use clap::Parser;
use std::path::PathBuf;
use hf_hub::{api::sync::Api, Repo, RepoType};

type Backend = NdArray;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to input safetensors model. If not provided, attempts to find in HF cache.
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// HuggingFace model ID to look for in cache
    #[arg(long, default_value = "facebook/blt-entropy")]
    model_id: String,

    /// Path to output mpk model
    #[arg(short, long, default_value = "blt_entropy_model.mpk")]
    output: PathBuf,
}

fn find_in_cache(model_id: &str) -> anyhow::Result<PathBuf> {
    println!("🔍 Looking for {} in HuggingFace cache...", model_id);
    
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
    
    // Try to get the file path (this will download if not cached, or just return path if cached)
    let path = repo.get("model.safetensors")?;
    
    println!("✓ Found at: {:?}", path);
    Ok(path)
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Default::default(); // NdArray device (CPU)

    // Determine input path
    let input_path = match args.input {
        Some(p) => p,
        None => {
            match find_in_cache(&args.model_id) {
                Ok(p) => p,
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Could not find model in cache and no input path provided. Error: {}", e
                    ));
                }
            }
        }
    };

    println!("📥 Loading weights from {:?}", input_path);

    // Load from Safetensors directly into bf16 (HalfPrecisionSettings)
    // Pass path directly - SafetensorsFileRecorder typically takes PathBuf as LoadArgs
    let recorder = SafetensorsFileRecorder::<HalfPrecisionSettings>::default();
    let record: LMTransformerRecord<Backend> = recorder
        .load(input_path.clone().into(), &device)
        .expect("Should decode state successfully");

    println!("💾 Saving to {:?}", args.output);
    
    // Save to Burn's binary format with bf16 precision
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::default();
    recorder
        .record(record, args.output.clone())
        .expect("Failed to save model record");
        
    println!("✅ Conversion complete!");
    
    // Verify output size
    if let Ok(metadata) = std::fs::metadata(&args.output) {
        println!("   Output size: {:.1} MB", metadata.len() as f64 / 1024.0 / 1024.0);
    }

    Ok(())
}
