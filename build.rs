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

    // 2. Check/Setup Python 3.12+ Environment (for burn-dataset)
    setup_python_env()?;

    // 3. Define the model repository and file
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

fn setup_python_env() -> Result<()> {
    // Check if .venv exists in the project root
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")?;
    let venv_path = PathBuf::from(&manifest_dir).join(".venv");

    if venv_path.exists() {
        println!("cargo:warning=Found existing Python environment at {:?}", venv_path);
        // Ideally we should check version here too, but assuming existing is correct if present
        return Ok(());
    }

    println!("cargo:warning=Python .venv not found. Attempting to create one with Python 3.12+...");

    // Check for python3 and version
    let python_candidates = ["python3", "python"];
    let mut chosen_python = None;

    for py in python_candidates {
        let output = Command::new(py).arg("--version").output();
        if let Ok(out) = output {
            if out.status.success() {
                let version_str = String::from_utf8_lossy(&out.stdout);
                // Output like "Python 3.12.1"
                if let Some(ver) = version_str.strip_prefix("Python ") {
                    let parts: Vec<&str> = ver.trim().split('.').collect();
                    if parts.len() >= 2 {
                        if let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                            if major == 3 && minor >= 12 {
                                chosen_python = Some(py);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    let python = chosen_python.ok_or_else(|| {
        anyhow::anyhow!("Could not find Python 3.12+ (checked 'python3', 'python'). Please install Python 3.12 or higher.")
    })?;

    println!("cargo:warning=Using {} to create virtual environment...", python);

    // Create venv
    let status = Command::new(python)
        .args(&["-m", "venv", ".venv"])
        .current_dir(&manifest_dir)
        .status()?;

    if !status.success() {
        return Err(anyhow::anyhow!("Failed to create virtual environment."));
    }

    println!("cargo:warning=Virtual environment created at .venv");
    
    // Pre-install burn-dataset requirements to ensure it uses this environment happy path
    // Usually: huggingface_hub
    // We use the venv's pip
    #[cfg(unix)]
    let pip_path = venv_path.join("bin").join("pip");
    #[cfg(windows)]
    let pip_path = venv_path.join("Scripts").join("pip.exe");

    if pip_path.exists() {
        println!("cargo:warning=Installing base dependencies (huggingface_hub)...");
        let _ = Command::new(pip_path)
            .args(&["install", "huggingface_hub", "numpy"])
            .status(); // Ignore failure, burn-dataset might handle it or user can fix
    }

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

