use burn::data::dataset::source::huggingface::HuggingfaceDatasetLoader;
use burn::data::dataset::{Dataset, SqliteDataset};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FineWebItem {
    pub text: String,
    pub id: Option<String>, 
    pub url: Option<String>,
    // Allow other fields to be missing or ignored
}

pub struct FineWebEduDataset {
    dataset: SqliteDataset<FineWebItem>,
}

impl FineWebEduDataset {
    /// Create dataset loader. If split is None, loads all available data (all splits combined).
    pub fn new(subset: &str, split: Option<&str>, cache_dir: &str) -> Result<Self, anyhow::Error> {
        // Find parent project's .venv at runtime (walk up directory tree)
        fn find_parent_venv() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
            let mut current = std::env::current_exe().ok()?;
            // Start from binary location, walk up to find .venv
            loop {
                current.pop(); // Remove filename, then directory
                let venv = current.join(".venv");
                let venv_bin = venv.join("bin");
                let venv_python = venv_bin.join("python3");
                
                if venv_python.exists() {
                    return Some((venv_bin, venv_python));
                }
                
                // Stop at filesystem root
                if !current.pop() {
                    break;
                }
            }
            None
        }
        
        // Try build-time venv first, then runtime search
        let (venv_bin, python_path) = if let (Some(venv_bin), Some(python_path)) = (
            option_env!("BLT_PYTHON_VENV_BIN").map(std::path::PathBuf::from),
            option_env!("BLT_PYTHON_PATH").map(std::path::PathBuf::from),
        ) {
            (venv_bin, python_path)
        } else if let Some((bin, python)) = find_parent_venv() {
            println!("Found parent project .venv at runtime: {}", python.display());
            (bin, python)
        } else {
            // Fallback: try current working directory
            let cwd = std::env::current_dir()
                .map_err(|e| anyhow::anyhow!("Failed to get current directory: {}", e))?;
            let venv = cwd.join(".venv");
            let venv_bin = venv.join("bin");
            let venv_python = venv_bin.join("python3");
            if venv_python.exists() {
                println!("Found .venv in current directory: {}", venv_python.display());
                (venv_bin, venv_python)
            } else {
                // No venv found, will use system Python
                println!("No .venv found, using system Python");
                return Err(anyhow::anyhow!("No Python venv found. Please run from TheSphere-JAX project root or ensure .venv exists."));
            }
        };
        
        // Prepend venv/bin to PATH so subprocesses use the venv's Python
        // CRITICAL: Set this BEFORE burn-dataset tries to use Python
        let current_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{}:{}", venv_bin.display(), current_path);
        std::env::set_var("PATH", &new_path);
        std::env::set_var("PYTHON", python_path.display().to_string());
        println!("Using Python from: {} (PATH updated)", python_path.display());
        
        // Verify Python version
        let python_ver = std::process::Command::new(&python_path)
            .arg("--version")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok());
        if let Some(ver) = python_ver {
            println!("Python version: {}", ver.trim());
        }
        
        // Pass Python path explicitly to burn-dataset to prevent venv creation
        let mut loader = HuggingfaceDatasetLoader::new("HuggingFaceFW/fineweb-edu")
            .with_subset(subset)
            .with_huggingface_cache_dir(cache_dir)
            .with_base_dir(cache_dir) // Also set base dir for burn db
            .with_use_python_venv(false); // Don't create venv, use existing one from PATH
        
        // If split is None, default to "train" which contains all the data
        // (We'll do our own train/val/test splits later for navigation training)
        let split_name = split.unwrap_or("train");
        if split.is_none() {
            println!("Note: No --split specified, loading 'train' split (all available data).");
            println!("      You'll split this data later for navigation training (GraphMERT-style).");
        }
        let dataset = loader.dataset(split_name)?;

        Ok(Self { dataset })
    }

    pub fn iter(&self) -> impl Iterator<Item = FineWebItem> + '_ {
        self.dataset.iter()
    }
    
    pub fn len(&self) -> usize {
        self.dataset.len()
    }
}

/*
/// Example: Handling MINT-1T with Burn Dataset Transforms
/// MINT-1T is massive and interleaved. Use composable transforms to handle it efficiently.
///
/// Requires: burn = { features = ["dataset"] }
///
/// use burn::data::dataset::transform::{ComposedDataset, PartialDataset, SamplerDataset, WindowsDataset, ShuffledDataset};
///
/// struct MintItem { text: Option<String>, image_bytes: Option<Vec<u8>>, url: String }
/// struct Mint1TDataset { ... } // Implements Dataset<MintItem>
///
/// fn create_pipeline() {
///     let mint = Mint1TDataset::new("mint-1t-parquet-shard");
///
///     // 1. Shard: Process a subset to avoid OOM or distributed processing
///     let partial = PartialDataset::new(mint, 0..1_000_000);
///
///     // 2. Sample: Balance modalities (e.g. oversample images if rare)
///     let sampled = SamplerDataset::new(partial, 10_000, true); // 10k items with replacement
///
///     // 3. Shuffle: Randomize order
///     let shuffled = ShuffledDataset::new(sampled, 42);
///
///     // 4. Window: Sliding window over bytes (if flattening to pure byte stream)
///     // Note: This usually requires mapping to a flat byte representation first
///     // let windowed = WindowsDataset::new(shuffled, 1024, 512);
///
///     for item in shuffled.iter() {
///         // Use pre-tokenizer on specific fields
///         if let Some(img) = item.image_bytes {
///             let pt = detect_modality(&img);
///             let segments = pt.create()?.pre_tokenize(&img)?;
///         }
///     }
/// }
*/
