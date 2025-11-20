use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};

fn main() -> Result<()> {
    // Only rerun if build.rs changes, otherwise we might spam HF checks (though they are cached)
    println!("cargo:rerun-if-changed=build.rs");

    // 1. Define the model repository and file
    let repo_id = "SashimiSaketoro/entropy_burn";
    let filename = "blt_entropy_model.mpk";

    println!("cargo:warning=Checking/Downloading model {} from {}...", filename, repo_id);

    // 2. Initialize HF API (Synchronous for build script)
    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    // 3. Get the file (downloads if not in cache, validates checksum)
    let path = repo.get(filename)?;

    // 4. Inform the user and set environment variable
    println!("cargo:warning=Model available at: {:?}", path);
    println!("cargo:rustc-env=BLT_MODEL_CACHE_PATH={}", path.display());

    Ok(())
}

