use std::path::{Path, PathBuf};
use std::process::Command;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FfmpegError {
    #[error("ffmpeg binary not found")]
    NotFound,

    #[error("ffmpeg installation script failed")]
    InstallFailed,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Check if `ffmpeg` is available on PATH.
pub fn ffmpeg_available_on_path() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

/// Check if `ffmpeg` exists and is runnable at a specific path.
pub fn ffmpeg_available_at(path: &Path) -> bool {
    Command::new(path).arg("-version").output().is_ok()
}

/// Try to resolve a usable ffmpeg binary given an optional override.
/// Returns the concrete `PathBuf` if available.
pub fn resolve_ffmpeg(custom_path: Option<&Path>) -> Result<PathBuf, FfmpegError> {
    if let Some(p) = custom_path {
        if ffmpeg_available_at(p) {
            return Ok(p.to_path_buf());
        } else {
            return Err(FfmpegError::NotFound);
        }
    }

    if ffmpeg_available_on_path() {
        Ok(PathBuf::from("ffmpeg"))
    } else {
        Err(FfmpegError::NotFound)
    }
}

/// Attempt to install ffmpeg via your shell script.
/// This does **not** prompt or exit; it just runs the script.
pub fn try_install_ffmpeg(script_path: &Path) -> Result<(), FfmpegError> {
    let status = Command::new("sh").arg(script_path).status()?;

    if status.success() {
        Ok(())
    } else {
        Err(FfmpegError::InstallFailed)
    }
}
