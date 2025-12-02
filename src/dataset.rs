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
    /// Create a new FineWebEduDataset for corpus ingestion.
    ///
    /// # Arguments
    /// * `subset` - Dataset subset/configuration (e.g., "sample-10BT")
    /// * `partition` - HuggingFace partition name (usually "train" for full corpus - NOT a training split)
    /// * `base_dir` - Where burn stores its SQLite database
    /// * `hf_cache` - Optional HF cache override. If None, uses system default (respects symlinks)
    pub fn new(
        subset: &str,
        partition: &str,
        base_dir: &str,
        hf_cache: Option<&str>,
    ) -> Result<Self, anyhow::Error> {
        // Use build-time detected venv if available
        if let (Some(venv_bin), Some(python_path)) = (
            option_env!("BLT_PYTHON_VENV_BIN"),
            option_env!("BLT_PYTHON_PATH"),
        ) {
            // Prepend venv/bin to PATH so subprocesses use the venv's Python
            let current_path = std::env::var("PATH").unwrap_or_default();
            std::env::set_var("PATH", format!("{venv_bin}:{current_path}"));
            std::env::set_var("PYTHON", python_path);
        }

        // By default, let HuggingFace use its system cache (respects symlinks like ~/.cache/huggingface -> /Volumes/ai/)
        // Only override if user explicitly provides --hf-cache
        let mut loader = HuggingfaceDatasetLoader::new("HuggingFaceFW/fineweb-edu")
            .with_subset(subset)
            .with_base_dir(base_dir) // Where burn stores the SQLite DB
            .with_use_python_venv(false); // Don't create venv, use existing one from PATH

        // Only set HF cache if explicitly overridden
        if let Some(cache_path) = hf_cache {
            loader = loader.with_huggingface_cache_dir(cache_path);
        }

        let dataset = loader.dataset(partition)?;

        Ok(Self { dataset })
    }

    pub fn iter(&self) -> impl Iterator<Item = FineWebItem> + '_ {
        self.dataset.iter()
    }

    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
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
