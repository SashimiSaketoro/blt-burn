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

fn main() -> Result<()> {
    // Only rerun if build.rs changes, otherwise we might spam HF checks (though they are cached)
    println!("cargo:rerun-if-changed=build.rs");

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

    // 2. Define the model repository and file
    let repo_id = "SashimiSaketoro/entropy_burn";
    let filename = "blt_entropy_model.mpk";

    println!(
        "cargo:warning=Checking/Downloading model {} from {}...",
        filename, repo_id
    );

    // 3. Initialize HF API (Synchronous for build script)
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    // 4. Get the file (downloads if not in cache, validates checksum)
    let path = repo.get(filename)?;

    // 5. Inform the user and set environment variable
    println!("cargo:warning=Model available at: {:?}", path);
    println!("cargo:rustc-env=BLT_MODEL_CACHE_PATH={}", path.display());

    Ok(())
}
