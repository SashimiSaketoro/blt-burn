use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;

/// Check if FFmpeg development libraries are available (required for video feature)
#[cfg(feature = "video")]
fn check_ffmpeg_dev() {
    use std::process::Command;
    // On macOS, check for Homebrew FFmpeg and provide setup instructions
    #[cfg(target_os = "macos")]
    {
        let homebrew_prefixes = [
            "/opt/homebrew", // Apple Silicon
            "/usr/local",    // Intel Mac
        ];

        let mut ffmpeg_found = false;
        for prefix in &homebrew_prefixes {
            let ffmpeg_dir = format!("{}/opt/ffmpeg", prefix);
            let ffmpeg_include = format!("{}/include", ffmpeg_dir);

            if std::path::Path::new(&ffmpeg_include).exists() {
                ffmpeg_found = true;
                println!("cargo:warning=✅ Found FFmpeg at: {}", ffmpeg_dir);

                // Check if FFMPEG_DIR is set
                if std::env::var("FFMPEG_DIR").is_err() {
                    println!("cargo:warning=");
                    println!("cargo:warning=⚠️  FFMPEG_DIR not set. Build may fail!");
                    println!("cargo:warning=Run this command before building:");
                    println!("cargo:warning=");
                    println!("cargo:warning=  export FFMPEG_DIR=\"{}\"", ffmpeg_dir);
                    println!("cargo:warning=  export PKG_CONFIG_PATH=\"{}/lib/pkgconfig:$PKG_CONFIG_PATH\"", ffmpeg_dir);
                    println!("cargo:warning=");
                    println!("cargo:warning=Or add to ~/.zshrc for permanent setup.");
                }
                break;
            }
        }

        if !ffmpeg_found {
            println!("cargo:warning=❌ FFmpeg not found in Homebrew locations");
            println!("cargo:warning=Install with: brew install ffmpeg pkg-config");
        }
    }

    // Check if pkg-config can find libavcodec (indicates dev headers are installed)
    let pkg_config_check = Command::new("pkg-config")
        .args(["--exists", "libavcodec"])
        .status();

    match pkg_config_check {
        Ok(status) if status.success() => {
            println!("cargo:warning=✅ FFmpeg development libraries found via pkg-config");
        }
        _ => {
            // pkg-config failed, try to check if ffmpeg binary exists at least
            let ffmpeg_check = Command::new("ffmpeg").arg("-version").output();

            match ffmpeg_check {
                Ok(output) if output.status.success() => {
                    println!("cargo:warning=⚠️  FFmpeg binary found, but pkg-config cannot find dev headers");

                    #[cfg(target_os = "macos")]
                    {
                        println!("cargo:warning=On macOS, try setting environment variables:");
                        println!("cargo:warning=  export PKG_CONFIG_PATH=\"$(brew --prefix ffmpeg)/lib/pkgconfig:$PKG_CONFIG_PATH\"");
                        println!("cargo:warning=  export FFMPEG_DIR=\"$(brew --prefix ffmpeg)\"");
                        println!("cargo:warning=Or reinstall: brew reinstall ffmpeg pkg-config");
                    }

                    #[cfg(target_os = "linux")]
                    {
                        println!("cargo:warning=Install FFmpeg development headers:");
                        println!("cargo:warning=  Ubuntu/Debian: sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavutil-dev pkg-config");
                        println!("cargo:warning=  Fedora: sudo dnf install ffmpeg-devel");
                        println!("cargo:warning=  Arch: sudo pacman -S ffmpeg");
                    }
                }
                _ => {
                    println!(
                        "cargo:warning=❌ FFmpeg NOT FOUND - video feature will fail to compile!"
                    );
                    println!("cargo:warning=Install FFmpeg with development headers:");

                    #[cfg(target_os = "macos")]
                    println!("cargo:warning=  brew install ffmpeg pkg-config");

                    #[cfg(target_os = "linux")]
                    {
                        println!("cargo:warning=  Ubuntu/Debian: sudo apt install ffmpeg libavcodec-dev libavformat-dev libswscale-dev libavutil-dev pkg-config");
                        println!("cargo:warning=  Fedora: sudo dnf install ffmpeg ffmpeg-devel");
                        println!("cargo:warning=  Arch: sudo pacman -S ffmpeg");
                    }

                    #[cfg(target_os = "windows")]
                    {
                        println!(
                            "cargo:warning=  Download from: https://www.gyan.dev/ffmpeg/builds/"
                        );
                        println!("cargo:warning=  Extract and set FFMPEG_DIR environment variable");
                    }

                    println!(
                        "cargo:warning=Or disable video feature: cargo build --no-default-features"
                    );
                }
            }
        }
    }
}

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
    let cache_path =
        PathBuf::from(home).join(".cache/huggingface/hub/models--facebook--blt-entropy/snapshots");

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

    // 0. Check FFmpeg availability when video feature is enabled
    #[cfg(feature = "video")]
    check_ffmpeg_dev();

    // 1. Find and configure Python venv for burn-dataset
    if let Some(venv) = find_project_venv() {
        let venv_bin = venv.join("bin");
        let venv_python = venv_bin.join("python3");

        if venv_python.exists() {
            println!("cargo:warning=Found project .venv at: {}", venv.display());
            println!("cargo:rustc-env=BLT_PYTHON_VENV_BIN={}", venv_bin.display());
            println!("cargo:rustc-env=BLT_PYTHON_PATH={}", venv_python.display());
        } else {
            println!(
                "cargo:warning=Found .venv but python3 not found at: {}",
                venv_python.display()
            );
        }
    } else {
        println!("cargo:warning=No project .venv found, burn-dataset will use system Python or create its own venv");
    }

    // 2. First check for original Facebook model in HF cache (preferred - has correct weights)
    if let Some(safetensors_path) = find_hf_cache_model() {
        println!(
            "cargo:warning=Found original Facebook BLT entropy model at: {}",
            safetensors_path.display()
        );
        println!(
            "cargo:rustc-env=BLT_MODEL_SAFETENSORS_PATH={}",
            safetensors_path.display()
        );
        println!("cargo:rustc-env=BLT_MODEL_FORMAT=safetensors");
        return Ok(());
    }

    // 3. Try to download from HuggingFace (fallback to converted .mpk if available)
    let repo_id = "facebook/blt-entropy";
    let filename = "model.safetensors";

    println!(
        "cargo:warning=Checking/Downloading model {filename} from {repo_id}..."
    );

    let api = Api::new()?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    match repo.get(filename) {
        Ok(path) => {
            println!("cargo:warning=Model available at: {}", path.display());
            println!(
                "cargo:rustc-env=BLT_MODEL_SAFETENSORS_PATH={}",
                path.display()
            );
            println!("cargo:rustc-env=BLT_MODEL_FORMAT=safetensors");
        }
        Err(e) => {
            println!("cargo:warning=Could not download model: {e}. Will need to provide model path at runtime.");
        }
    }

    Ok(())
}
