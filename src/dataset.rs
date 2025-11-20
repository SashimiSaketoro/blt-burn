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
    pub fn new(subset: &str, split: &str, cache_dir: &str) -> Result<Self, anyhow::Error> {
        let dataset = HuggingfaceDatasetLoader::new("HuggingFaceFW/fineweb-edu")
            .with_subset(subset)
            .with_huggingface_cache_dir(cache_dir)
            .with_base_dir(cache_dir) // Also set base dir for burn db
            .dataset(split)?;

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
