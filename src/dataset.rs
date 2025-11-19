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
