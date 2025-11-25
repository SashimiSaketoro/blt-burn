use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;

fn find_project_venv() -> Option<PathBuf> {
    // Try current directory (project root)
    let current = std::env::current_dir().ok()?;
    let venv = current.join(".venv");
    if venv.exists() {
        return Some(venv);
    }
    
    // Try one level up
    if let Some(parent) = current.parent() {
        let venv = parent.join(".venv");
        if venv.exists() {
            return Some(venv);
        }
    }
    
    None
}

fn find_hf_cache_model() -> Option<PathBuf> {
    // Look for the original Facebook BLT entropy model in HF cache
    let home = std::env::var("HOME").ok()?;
    let cache_path = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--facebook--blt-entropy/snapshots");
    
    if cache_path.exists() {
        // Find the first snapshot directory
        if let Ok(entries) = std::fs::read_dir(&cache_path) {
            for entry in entries.flatten() {
                let model_path = entry.path().join("model.safetensors");
                if model_path.exists() {
                    return Some(model_path);
                }
            }
        }
    }
    None
}

fn main() -> Result<()> {
    // Only rerun if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");

    // Generate build info (git hash, build time, etc.)
    built::write_built_file().expect("Failed to write built.rs for version info");

    // 1. Find and configure Python venv for burn-dataset
    if let Some(venv) = find_project_venv() {
        let venv_bin = venv.join("bin");
        let venv_python = venv_bin.join("python3");
        
        if venv_python.exists() {
            println!("cargo:warning=Found project .venv at: {}", venv.display());
            println!("cargo:rustc-env=BLT_PYTHON_VENV_BIN={}", venv_bin.display());
            println!("cargo:rustc-env=BLT_PYTHON_PATH={}", venv_python.display());
        } else {
            println!("cargo:warning=Found .venv but python3 not found at: {}", venv_python.display());
        }
    } else {
        println!("cargo:warning=No project .venv found, burn-dataset will use system Python or create its own venv");
    }

    // 2. First check for original Facebook model in HF cache (preferred - has correct weights)
    if let Some(safetensors_path) = find_hf_cache_model() {
        println!("cargo:warning=Found original Facebook BLT entropy model at: {}", safetensors_path.display());
        println!("cargo:rustc-env=BLT_MODEL_SAFETENSORS_PATH={}", safetensors_path.display());
        println!("cargo:rustc-env=BLT_MODEL_FORMAT=safetensors");
        return Ok(());
    }

    // 3. Try to download from HuggingFace (fallback to converted .mpk if available)
    let repo_id = "facebook/blt-entropy";
    let filename = "model.safetensors";

    println!(
        "cargo:warning=Checking/Downloading model {} from {}...",
        filename, repo_id
    );

    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    match repo.get(filename) {
        Ok(path) => {
            println!("cargo:warning=Model available at: {:?}", path);
            println!("cargo:rustc-env=BLT_MODEL_SAFETENSORS_PATH={}", path.display());
            println!("cargo:rustc-env=BLT_MODEL_FORMAT=safetensors");
        }
        Err(e) => {
            println!("cargo:warning=Could not download model: {}. Will need to provide model path at runtime.", e);
        }
    }

    Ok(())
}
