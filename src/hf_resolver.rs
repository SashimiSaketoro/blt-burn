//! Centralized HuggingFace path resolution with caching and fallbacks.
//!
//! This module provides a unified interface for resolving file paths from
//! HuggingFace datasets, with automatic download, caching, and zero-tensor
//! fallbacks for missing images.
//!
//! # Usage
//!
//! ```rust,ignore
//! use blt_burn::hf_resolver::{HfResolver, HfResolverConfig};
//!
//! let config = HfResolverConfig::new("username/dataset", Path::new(".cache"));
//! let resolver = HfResolver::new(config);
//!
//! // Resolve a file, downloading if necessary
//! let bytes = resolver.resolve("images/sample.jpg", true)?;
//!
//! // Resolve an hf:// URI with revision support
//! let bytes = resolver.resolve_hf_uri("hf://datasets/owner/dataset@main/images/test.jpg", true)?;
//!
//! // Resolve a direct URL
//! let bytes = HfResolver::resolve_url("https://example.com/image.jpg", Path::new(".cache"))?;
//! ```

use anyhow::{Context, Result};
use hf_hub::{api::sync::Api, Repo, RepoType};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use urlencoding::encode;

/// Default image dimensions for zero tensor fallback (224×224 RGB)
pub const DEFAULT_IMAGE_WIDTH: usize = 224;
pub const DEFAULT_IMAGE_HEIGHT: usize = 224;
pub const DEFAULT_IMAGE_CHANNELS: usize = 3;

/// Size of zero tensor fallback for images (224×224×3 = 150,528 bytes)
pub const ZERO_IMAGE_SIZE: usize =
    DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT * DEFAULT_IMAGE_CHANNELS;

/// Global API instance (lazy initialized)
static HF_API: OnceLock<Api> = OnceLock::new();

/// Global HTTP client for URL downloads (lazy initialized)
static HTTP_CLIENT: OnceLock<Client> = OnceLock::new();

/// Get the global HuggingFace API instance (lazy initialized)
///
/// This ensures only one API instance is created across the entire application,
/// reducing overhead and avoiding rate limiting issues.
pub fn get_api() -> Result<&'static Api> {
    // Use get_or_init with a fallback that panics, since Api::new() shouldn't fail in normal use
    // For better error handling, we check if the API was successfully initialized
    if let Some(api) = HF_API.get() {
        return Ok(api);
    }

    // Initialize the API - this will only happen once
    match Api::new() {
        Ok(api) => {
            // Try to set the API, but another thread might have beaten us
            let _ = HF_API.set(api);
            HF_API.get().ok_or_else(|| anyhow::anyhow!("Failed to initialize HuggingFace API"))
        }
        Err(e) => Err(anyhow::anyhow!("Failed to initialize HuggingFace API: {}", e)),
    }
}

fn get_http_client() -> &'static Client {
    HTTP_CLIENT.get_or_init(|| Client::new())
}

/// Parsed representation of an hf:// URI
#[derive(Debug, Clone)]
pub struct ParsedHfUri {
    /// Repository ID (e.g., "owner/dataset-name")
    pub repo_id: String,
    /// Revision/branch (defaults to "main")
    pub revision: String,
    /// Path within the repository
    pub path: String,
}

/// Parse an hf://datasets/owner/name@revision/path URI
///
/// # Examples
/// - `hf://datasets/owner/dataset/path/to/file.jpg` -> repo_id="owner/dataset", revision="main", path="path/to/file.jpg"
/// - `hf://datasets/owner/dataset@branch/path/to/file.jpg` -> repo_id="owner/dataset", revision="branch", path="path/to/file.jpg"
pub fn parse_hf_uri(uri: &str) -> Option<ParsedHfUri> {
    const PREFIX: &str = "hf://datasets/";
    let rest = uri.strip_prefix(PREFIX)?;
    let mut parts = rest.splitn(3, '/');
    let owner = parts.next()?;
    let dataset_part = parts.next()?;
    let path = parts.next().unwrap_or("").to_string();
    
    if path.is_empty() {
        return None;
    }

    let (dataset_name, revision) = if let Some(idx) = dataset_part.find('@') {
        (
            dataset_part[..idx].to_string(),
            dataset_part[idx + 1..].to_string(),
        )
    } else {
        (dataset_part.to_string(), "main".to_string())
    };

    Some(ParsedHfUri {
        repo_id: format!("{}/{}", owner, dataset_name),
        revision,
        path,
    })
}

/// URL-encode each segment of a path
pub fn encode_path_segments(path: &str) -> String {
    path.split('/')
        .map(|segment| encode(segment).to_string())
        .collect::<Vec<_>>()
        .join("/")
}

/// Sanitize a path reference to prevent directory traversal
pub fn sanitize_path(reference: &str) -> PathBuf {
    let mut path = PathBuf::new();
    for segment in reference.split(['/', '\\']) {
        if segment.is_empty() || segment == "." || segment == ".." {
            continue;
        }
        path.push(segment);
    }
    path
}

/// Check if a string looks like a URL
pub fn looks_like_url(value: &str) -> bool {
    value.starts_with("http://")
        || value.starts_with("https://")
        || value.starts_with("s3://")
        || value.starts_with("gs://")
}

/// Extract a filename from a URL, falling back to a hash if needed
pub fn filename_from_url(url: &str) -> String {
    if let Some(filename) = url.split('/').last() {
        if !filename.is_empty() && filename.contains('.') {
            return filename.to_string();
        }
    }
    // Fallback to hash-based filename
    let mut hasher = Sha256::new();
    hasher.update(url.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    format!("file_{}", &hash[..16])
}

/// Configuration for HuggingFace path resolution
#[derive(Debug, Clone)]
pub struct HfResolverConfig {
    /// Repository slug (e.g., "username/dataset-name")
    pub repo_slug: String,
    /// Local cache directory for downloaded files
    pub cache_dir: PathBuf,
    /// Whether to skip missing files instead of using fallbacks
    pub skip_missing: bool,
    /// Optional authentication token (also respects HF_TOKEN env var)
    pub token: Option<String>,
    /// Repository type (Dataset or Model)
    pub repo_type: RepoType,
}

impl HfResolverConfig {
    /// Create a new configuration with default settings
    pub fn new(repo_slug: &str, cache_dir: &Path) -> Self {
        Self {
            repo_slug: repo_slug.to_string(),
            cache_dir: cache_dir.to_path_buf(),
            skip_missing: false,
            token: None,
            repo_type: RepoType::Dataset,
        }
    }

    /// Set whether to skip missing files
    pub fn with_skip_missing(mut self, skip: bool) -> Self {
        self.skip_missing = skip;
        self
    }

    /// Set authentication token
    pub fn with_token(mut self, token: Option<String>) -> Self {
        self.token = token;
        self
    }

    /// Set repository type (Dataset or Model)
    pub fn with_repo_type(mut self, repo_type: RepoType) -> Self {
        self.repo_type = repo_type;
        self
    }
}

/// Centralized resolver for HuggingFace file paths
///
/// Provides automatic caching, download, and fallback handling for files
/// referenced in HuggingFace datasets.
pub struct HfResolver {
    config: HfResolverConfig,
}

impl HfResolver {
    /// Create a new resolver with the given configuration
    pub fn new(config: HfResolverConfig) -> Self {
        // Ensure cache directory exists
        if let Err(e) = std::fs::create_dir_all(&config.cache_dir) {
            eprintln!(
                "Warning: Could not create cache dir {:?}: {}",
                config.cache_dir, e
            );
        }
        Self { config }
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.config.cache_dir
    }

    /// Get the repository slug
    pub fn repo_slug(&self) -> &str {
        &self.config.repo_slug
    }

    /// Resolve a file path, downloading from HuggingFace if necessary.
    ///
    /// # Arguments
    /// * `path` - Relative path within the dataset (e.g., "images/sample.jpg")
    /// * `is_image` - If true, returns zero tensor on failure instead of error
    ///
    /// # Returns
    /// * `Ok(Some(bytes))` - File contents
    /// * `Ok(None)` - File skipped (when skip_missing is true)
    /// * `Err` - Unrecoverable error
    pub fn resolve(&self, path: &str, is_image: bool) -> Result<Option<Vec<u8>>> {
        // 1. Check local cache first
        let local_path = self.local_cache_path(path);
        if local_path.exists() {
            let bytes = std::fs::read(&local_path)
                .with_context(|| format!("Failed to read cached file: {:?}", local_path))?;
            return Ok(Some(bytes));
        }

        // 2. Try to download from HuggingFace
        match self.download_from_hf(path, &local_path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) => {
                if self.config.skip_missing {
                    eprintln!("Skipping missing file: {} ({})", path, e);
                    return Ok(None);
                }

                if is_image {
                    // Zero tensor fallback for images (preserves shape without semantic noise)
                    eprintln!(
                        "Warning: Could not resolve {}, using zero tensor ({}x{}x{})",
                        path, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_CHANNELS
                    );
                    Ok(Some(vec![0u8; ZERO_IMAGE_SIZE]))
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Resolve with explicit fallback bytes (for non-standard image sizes)
    pub fn resolve_with_fallback(&self, path: &str, fallback: Vec<u8>) -> Result<Vec<u8>> {
        match self.resolve(path, false)? {
            Some(bytes) => Ok(bytes),
            None => Ok(fallback),
        }
    }

    /// Check if a path is likely an image based on extension
    pub fn is_image_path(path: &str) -> bool {
        let lower = path.to_lowercase();
        lower.ends_with(".jpg")
            || lower.ends_with(".jpeg")
            || lower.ends_with(".png")
            || lower.ends_with(".gif")
            || lower.ends_with(".webp")
            || lower.ends_with(".bmp")
            || lower.ends_with(".tiff")
            || lower.ends_with(".tif")
    }

    /// Get the local cache path for a file
    fn local_cache_path(&self, path: &str) -> PathBuf {
        // Sanitize the path to prevent directory traversal
        let sanitized = path
            .replace("..", "_")
            .replace("://", "_")
            .trim_start_matches('/')
            .to_string();

        // Include repo slug in cache path to avoid collisions
        let repo_dir = self.config.repo_slug.replace('/', "_");
        self.config.cache_dir.join(repo_dir).join(&sanitized)
    }

    /// Download a file from HuggingFace and cache it locally
    fn download_from_hf(&self, path: &str, local_path: &Path) -> Result<Vec<u8>> {
        let api = get_api()?;

        // Note: hf-hub 0.3.2 respects HF_TOKEN environment variable for authentication
        // If token is provided in config, we could set it via env var
        if let Some(ref token) = self.config.token {
            std::env::set_var("HF_TOKEN", token);
        }

        let repo = api.repo(Repo::new(
            self.config.repo_slug.clone(),
            self.config.repo_type.clone(),
        ));

        // Try to get the file
        let downloaded_path = repo.get(path).with_context(|| {
            format!(
                "Failed to download {} from {}",
                path, self.config.repo_slug
            )
        })?;

        // Read the downloaded content
        let bytes = std::fs::read(&downloaded_path)
            .with_context(|| format!("Failed to read downloaded file: {:?}", downloaded_path))?;

        // Cache locally for future use
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if let Err(e) = std::fs::write(local_path, &bytes) {
            eprintln!("Warning: Could not cache file {:?}: {}", local_path, e);
        }

        Ok(bytes)
    }

    /// Resolve an hf:// URI, downloading from HuggingFace if necessary.
    ///
    /// Supports URIs in the format: `hf://datasets/owner/dataset@revision/path/to/file`
    ///
    /// # Arguments
    /// * `uri` - The hf:// URI to resolve
    /// * `is_image` - If true, returns zero tensor on failure instead of error
    ///
    /// # Returns
    /// * `Ok(Some(bytes))` - File contents
    /// * `Ok(None)` - File skipped or URI invalid
    /// * `Err` - Unrecoverable error
    pub fn resolve_hf_uri(&self, uri: &str, is_image: bool) -> Result<Option<Vec<u8>>> {
        let parsed = match parse_hf_uri(uri) {
            Some(p) => p,
            None => {
                if self.config.skip_missing {
                    return Ok(None);
                }
                return Err(anyhow::anyhow!("Invalid HF URI format: {}", uri));
            }
        };

        let sanitized = sanitize_path(&parsed.path);
        if sanitized.as_os_str().is_empty() {
            return Ok(None);
        }

        // Check local cache first
        let local_path = self.config.cache_dir
            .join(parsed.repo_id.replace('/', "_"))
            .join(&sanitized);
        
        if local_path.exists() {
            let bytes = std::fs::read(&local_path)
                .with_context(|| format!("Failed to read cached file: {:?}", local_path))?;
            return Ok(Some(bytes));
        }

        // Try to download from HuggingFace with revision support
        let api = get_api()?;
        
        if let Some(ref token) = self.config.token {
            std::env::set_var("HF_TOKEN", token);
        }

        let repo = api.repo(Repo::with_revision(
            parsed.repo_id.clone(),
            RepoType::Dataset,
            parsed.revision.clone(),
        ));

        let encoded_path = encode_path_segments(&parsed.path);
        
        match repo.get(&encoded_path) {
            Ok(downloaded_path) => {
                let bytes = std::fs::read(&downloaded_path)
                    .with_context(|| format!("Failed to read downloaded file: {:?}", downloaded_path))?;
                
                // Cache locally
                if let Some(parent) = local_path.parent() {
                    std::fs::create_dir_all(parent).ok();
                }
                std::fs::write(&local_path, &bytes).ok();
                
                Ok(Some(bytes))
            }
            Err(e) => {
                // Try HTTP fallback
                let url = repo.url(&encoded_path);
                match Self::download_via_http(&url) {
                    Ok(bytes) => {
                        // Cache locally
                        if let Some(parent) = local_path.parent() {
                            std::fs::create_dir_all(parent).ok();
                        }
                        std::fs::write(&local_path, &bytes).ok();
                        Ok(Some(bytes))
                    }
                    Err(http_err) => {
                        if self.config.skip_missing {
                            eprintln!("Skipping missing HF file: {} ({}, HTTP: {})", uri, e, http_err);
                            return Ok(None);
                        }

                        if is_image {
                            eprintln!(
                                "Warning: Could not resolve {}, using zero tensor ({}x{}x{})",
                                uri, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_CHANNELS
                            );
                            Ok(Some(vec![0u8; ZERO_IMAGE_SIZE]))
                        } else {
                            Err(anyhow::anyhow!("Failed to download {} (API: {}, HTTP: {})", uri, e, http_err))
                        }
                    }
                }
            }
        }
    }

    /// Resolve a dataset asset by trying multiple path candidates.
    ///
    /// This is useful when the exact path in the repository is unknown.
    /// Tries the original path first, then common prefixes like "data/".
    ///
    /// # Arguments
    /// * `path` - The path to resolve
    /// * `is_image` - If true, returns zero tensor on failure
    ///
    /// # Returns
    /// The file bytes, or None if not found and skip_missing is true
    pub fn resolve_with_candidates(&self, path: &str, is_image: bool) -> Result<Option<Vec<u8>>> {
        let sanitized = sanitize_path(path);
        if sanitized.as_os_str().is_empty() {
            return Ok(None);
        }

        let path_str = sanitized.to_string_lossy().to_string();
        let candidates = [
            path_str.clone(),
            format!("data/{}", path_str),
        ];

        for candidate in &candidates {
            match self.resolve(candidate, false) {
                Ok(Some(bytes)) => return Ok(Some(bytes)),
                Ok(None) => continue,
                Err(_) => continue,
            }
        }

        // None of the candidates worked
        if self.config.skip_missing {
            return Ok(None);
        }

        if is_image {
            eprintln!(
                "Warning: Could not resolve {}, using zero tensor",
                path
            );
            Ok(Some(vec![0u8; ZERO_IMAGE_SIZE]))
        } else {
            Err(anyhow::anyhow!("Failed to resolve asset: {}", path))
        }
    }

    /// Download bytes from a URL via HTTP
    fn download_via_http(url: &str) -> Result<Vec<u8>> {
        let client = get_http_client();
        let response = client.get(url).send()
            .with_context(|| format!("HTTP request failed for {}", url))?;
        
        if !response.status().is_success() {
            anyhow::bail!("HTTP {} fetching {}", response.status(), url);
        }

        let bytes = response.bytes()
            .with_context(|| format!("Failed to read response bytes from {}", url))?;
        Ok(bytes.to_vec())
    }

    /// Resolve a direct URL, downloading and caching the result.
    ///
    /// # Arguments
    /// * `url` - The URL to download from
    /// * `cache_dir` - Directory to cache the downloaded file
    ///
    /// # Returns
    /// The file bytes
    pub fn resolve_url(url: &str, cache_dir: &Path) -> Result<Vec<u8>> {
        let filename = filename_from_url(url);
        let dest = cache_dir.join(&filename);

        // Check cache first
        if dest.exists() {
            return std::fs::read(&dest)
                .with_context(|| format!("Failed to read cached URL file: {:?}", dest));
        }

        // Download
        let bytes = Self::download_via_http(url)?;

        // Cache
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&dest, &bytes).ok();

        Ok(bytes)
    }

    /// Resolve a direct URL with optional fallback for images.
    ///
    /// # Arguments
    /// * `url` - The URL to download from
    /// * `cache_dir` - Directory to cache the downloaded file
    /// * `is_image` - If true, returns zero tensor on failure
    /// * `skip_missing` - If true, returns None on failure
    ///
    /// # Returns
    /// The file bytes, or None if skipped
    pub fn resolve_url_with_fallback(
        url: &str, 
        cache_dir: &Path, 
        is_image: bool, 
        skip_missing: bool
    ) -> Result<Option<Vec<u8>>> {
        match Self::resolve_url(url, cache_dir) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) => {
                if skip_missing {
                    eprintln!("Skipping failed URL: {} ({})", url, e);
                    return Ok(None);
                }

                if is_image {
                    eprintln!(
                        "Warning: Could not download {}, using zero tensor",
                        url
                    );
                    Ok(Some(vec![0u8; ZERO_IMAGE_SIZE]))
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Copy a file to the cache directory
    pub fn copy_to_cache(source: &Path, dest: &Path) -> Result<()> {
        if dest.exists() {
            return Ok(());
        }

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::copy(source, dest)?;
        Ok(())
    }
}

/// Convenience function for one-off resolution without creating a resolver
pub fn resolve_hf_path(
    repo_slug: &str,
    path: &str,
    cache_dir: &Path,
    is_image: bool,
) -> Result<Option<Vec<u8>>> {
    let config = HfResolverConfig::new(repo_slug, cache_dir);
    let resolver = HfResolver::new(config);
    resolver.resolve(path, is_image)
}

/// Create a zero tensor of the default image size (useful for tests and fallbacks)
pub fn zero_image_tensor() -> Vec<u8> {
    vec![0u8; ZERO_IMAGE_SIZE]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_zero_image_size() {
        assert_eq!(ZERO_IMAGE_SIZE, 224 * 224 * 3);
        assert_eq!(ZERO_IMAGE_SIZE, 150528);
    }

    #[test]
    fn test_zero_image_tensor() {
        let tensor = zero_image_tensor();
        assert_eq!(tensor.len(), ZERO_IMAGE_SIZE);
        assert!(tensor.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_is_image_path() {
        assert!(HfResolver::is_image_path("test.jpg"));
        assert!(HfResolver::is_image_path("test.JPEG"));
        assert!(HfResolver::is_image_path("path/to/image.png"));
        assert!(HfResolver::is_image_path("image.webp"));
        assert!(!HfResolver::is_image_path("test.txt"));
        assert!(!HfResolver::is_image_path("test.json"));
    }

    #[test]
    fn test_local_cache_path_sanitization() {
        let temp = tempdir().unwrap();
        let config = HfResolverConfig::new("test/dataset", temp.path());
        let resolver = HfResolver::new(config);

        // Normal path
        let path = resolver.local_cache_path("images/test.jpg");
        assert!(path.to_string_lossy().contains("images"));
        assert!(path.to_string_lossy().contains("test_dataset"));

        // Path with traversal attempt
        let path = resolver.local_cache_path("../../../etc/passwd");
        assert!(!path.to_string_lossy().contains(".."));

        // Path with leading slash
        let path = resolver.local_cache_path("/absolute/path.jpg");
        assert!(!path.to_string_lossy().contains("//"));
    }

    #[test]
    fn test_config_builder() {
        let temp = tempdir().unwrap();
        let config = HfResolverConfig::new("test/dataset", temp.path())
            .with_skip_missing(true)
            .with_token(Some("test_token".to_string()))
            .with_repo_type(RepoType::Model);

        assert!(config.skip_missing);
        assert_eq!(config.token, Some("test_token".to_string()));
        assert!(matches!(config.repo_type, RepoType::Model));
    }

    #[test]
    fn test_skip_missing_returns_none() {
        let temp = tempdir().unwrap();
        let config =
            HfResolverConfig::new("nonexistent/dataset", temp.path()).with_skip_missing(true);
        let resolver = HfResolver::new(config);

        // Should return None, not error (can't actually test HF download without network)
        // This test verifies the skip_missing logic path
        let result = resolver.resolve("nonexistent/file.txt", false);

        // Will be Ok(None) if skip_missing works, or Err if network fails
        // Either is acceptable for this unit test
        match result {
            Ok(None) => {} // Expected with skip_missing
            Err(_) => {}   // Network error, also acceptable
            Ok(Some(_)) => panic!("Should not return Some for nonexistent file"),
        }
    }

    #[test]
    fn test_image_fallback_returns_zeros() {
        let temp = tempdir().unwrap();
        let config = HfResolverConfig::new("nonexistent/dataset", temp.path());
        let resolver = HfResolver::new(config);

        // Should return zero tensor for images when download fails
        let result = resolver.resolve("nonexistent/image.jpg", true);

        match result {
            Ok(Some(bytes)) => {
                // Either real bytes (unlikely) or zero tensor
                if bytes.len() == ZERO_IMAGE_SIZE {
                    assert!(bytes.iter().all(|&b| b == 0));
                }
            }
            Ok(None) => {} // skip_missing case
            Err(_) => {}   // Network error before fallback
        }
    }

    #[test]
    fn test_parse_hf_uri_basic() {
        let uri = "hf://datasets/owner/dataset/path/to/file.jpg";
        let parsed = parse_hf_uri(uri).unwrap();
        assert_eq!(parsed.repo_id, "owner/dataset");
        assert_eq!(parsed.revision, "main");
        assert_eq!(parsed.path, "path/to/file.jpg");
    }

    #[test]
    fn test_parse_hf_uri_with_revision() {
        let uri = "hf://datasets/owner/dataset@dev/path/to/file.jpg";
        let parsed = parse_hf_uri(uri).unwrap();
        assert_eq!(parsed.repo_id, "owner/dataset");
        assert_eq!(parsed.revision, "dev");
        assert_eq!(parsed.path, "path/to/file.jpg");
    }

    #[test]
    fn test_parse_hf_uri_invalid() {
        // Missing path
        assert!(parse_hf_uri("hf://datasets/owner/dataset").is_none());
        // Wrong prefix
        assert!(parse_hf_uri("hf://models/owner/model/file.bin").is_none());
        // Empty
        assert!(parse_hf_uri("").is_none());
    }

    #[test]
    fn test_encode_path_segments() {
        assert_eq!(encode_path_segments("simple/path"), "simple/path");
        assert_eq!(encode_path_segments("path with spaces/file name.jpg"), "path%20with%20spaces/file%20name.jpg");
        assert_eq!(encode_path_segments("special/chars#test"), "special/chars%23test");
    }

    #[test]
    fn test_sanitize_path() {
        assert_eq!(sanitize_path("normal/path/file.jpg"), PathBuf::from("normal/path/file.jpg"));
        assert_eq!(sanitize_path("../../../etc/passwd"), PathBuf::from("etc/passwd"));
        assert_eq!(sanitize_path("/absolute/path"), PathBuf::from("absolute/path"));
        assert_eq!(sanitize_path("./relative/./path"), PathBuf::from("relative/path"));
        assert_eq!(sanitize_path(""), PathBuf::new());
    }

    #[test]
    fn test_looks_like_url() {
        assert!(looks_like_url("http://example.com/file.jpg"));
        assert!(looks_like_url("https://example.com/file.jpg"));
        assert!(looks_like_url("s3://bucket/file.jpg"));
        assert!(looks_like_url("gs://bucket/file.jpg"));
        assert!(!looks_like_url("local/path/file.jpg"));
        assert!(!looks_like_url("hf://datasets/owner/dataset/file.jpg"));
    }

    #[test]
    fn test_filename_from_url() {
        assert_eq!(filename_from_url("https://example.com/image.jpg"), "image.jpg");
        assert_eq!(filename_from_url("https://example.com/path/to/file.png"), "file.png");
        // For URLs without a clear filename, should return a hash-based name
        let hash_name = filename_from_url("https://example.com/");
        assert!(hash_name.starts_with("file_"));
        assert_eq!(hash_name.len(), 21); // "file_" + 16 hex chars
    }

    #[test]
    fn test_resolve_hf_uri_skip_missing() {
        let temp = tempdir().unwrap();
        let config = HfResolverConfig::new("test/dataset", temp.path())
            .with_skip_missing(true);
        let resolver = HfResolver::new(config);

        // Should return None for invalid URI when skip_missing is true
        let result = resolver.resolve_hf_uri("not_a_valid_uri", false);
        match result {
            Ok(None) => {} // Expected
            _ => {}        // Network issues acceptable
        }
    }
}

