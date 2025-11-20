use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::process::Command;
use std::path::PathBuf;
use std::fs;
use std::env;

fn main() -> Result<()> {
    // Only rerun if build.rs changes, otherwise we might spam HF checks (though they are cached)
    println!("cargo:rerun-if-changed=build.rs");

    // 1. Check for FFmpeg and install if missing
    check_and_install_ffmpeg()?;

    // 2. Define the model repository and file
    let repo_id = "SashimiSaketoro/entropy_burn";
    let filename = "blt_entropy_model.mpk";

    println!("cargo:warning=Checking/Downloading model {} from {}...", filename, repo_id);

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

fn check_and_install_ffmpeg() -> Result<()> {
    // Check if FFmpeg is available
    let ffmpeg_available = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    // On macOS, also check for pkg-config (required by ffmpeg-sys-next)
    #[cfg(target_os = "macos")]
    let pkg_config_available = Command::new("pkg-config")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);
    
    #[cfg(not(target_os = "macos"))]
    let pkg_config_available = true; // Not required on other platforms

    let needs_installation = !ffmpeg_available || !pkg_config_available;

    if needs_installation {
        if !ffmpeg_available {
            println!("cargo:warning=FFmpeg not found. Installing automatically...");
        }
        #[cfg(target_os = "macos")]
        if !pkg_config_available {
            println!("cargo:warning=pkg-config not found. Installing automatically...");
        }
        
        // Get the script path relative to the project root
        let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
        let script_path = PathBuf::from(manifest_dir)
            .join("scripts")
            .join("install_ffmpeg.sh");
        
        // Make script executable (Unix-like systems)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script_path, perms)?;
        }
        
        // Run the installation script
        let status = Command::new("bash")
            .arg(&script_path)
            .status()?;
        
        if !status.success() {
            return Err(anyhow::anyhow!("FFmpeg installation failed. Please install FFmpeg manually."));
        }
        
        println!("cargo:warning=FFmpeg and dependencies installed successfully!");
        
        // Verify installation
        let verified = Command::new("ffmpeg")
            .arg("-version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        
        if !verified {
            return Err(anyhow::anyhow!("FFmpeg installation completed but verification failed. Please ensure FFmpeg is in your PATH."));
        }
    } else {
        println!("cargo:warning=FFmpeg detected and ready");
    }

    Ok(())
}

