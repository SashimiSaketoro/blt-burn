//! Smart batching utilities for BLT ingestion pipeline
//!
//! # Important Constraint: Hypergraph Sidecar Integrity
//! 
//! Each document MUST be processed individually through the BLT model because:
//! 1. Entropy is calculated with sliding window context
//! 2. Mixing documents would corrupt entropy boundaries
//! 3. Each document gets its own hypergraph sidecar
//!
//! Therefore, "smart batching" here means:
//! - **Length sorting**: Process similar-sized docs together to reduce GPU idle time
//! - **Bucket batching**: Group docs into size buckets for more predictable processing
//! - **Transaction batching**: Use Burn's Transaction API for multi-tensor reads
//!
//! We explicitly DO NOT concatenate multiple documents into a single model forward pass.

use crate::prefetch::PrefetchedDoc;

/// Size bucket for grouping documents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SizeBucket {
    /// < 512 bytes - very small documents
    Tiny,
    /// 512 - 2048 bytes - small documents  
    Small,
    /// 2048 - 8192 bytes - medium documents
    Medium,
    /// 8192 - 32768 bytes - large documents
    Large,
    /// > 32768 bytes - very large documents
    Huge,
}

impl SizeBucket {
    /// Classify a document by its byte length
    pub fn from_len(len: usize) -> Self {
        match len {
            0..=511 => Self::Tiny,
            512..=2047 => Self::Small,
            2048..=8191 => Self::Medium,
            8192..=32767 => Self::Large,
            _ => Self::Huge,
        }
    }
    
    /// Get the bucket's name for logging
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tiny => "tiny (<512B)",
            Self::Small => "small (512B-2KB)",
            Self::Medium => "medium (2KB-8KB)",
            Self::Large => "large (8KB-32KB)",
            Self::Huge => "huge (>32KB)",
        }
    }
}

/// Statistics about document batching
#[derive(Debug, Default)]
pub struct BatchStats {
    pub tiny_count: usize,
    pub small_count: usize,
    pub medium_count: usize,
    pub large_count: usize,
    pub huge_count: usize,
    pub total_bytes: usize,
}

impl BatchStats {
    /// Update stats with a new document
    pub fn add(&mut self, doc: &PrefetchedDoc) {
        self.total_bytes += doc.original_len;
        match SizeBucket::from_len(doc.original_len) {
            SizeBucket::Tiny => self.tiny_count += 1,
            SizeBucket::Small => self.small_count += 1,
            SizeBucket::Medium => self.medium_count += 1,
            SizeBucket::Large => self.large_count += 1,
            SizeBucket::Huge => self.huge_count += 1,
        }
    }
    
    /// Print a summary of batch statistics
    pub fn print_summary(&self) {
        let total = self.tiny_count + self.small_count + self.medium_count 
                  + self.large_count + self.huge_count;
        println!("📊 Document size distribution:");
        println!("   Tiny (<512B):     {} ({:.1}%)", 
                 self.tiny_count, 100.0 * self.tiny_count as f64 / total.max(1) as f64);
        println!("   Small (512B-2KB): {} ({:.1}%)", 
                 self.small_count, 100.0 * self.small_count as f64 / total.max(1) as f64);
        println!("   Medium (2KB-8KB): {} ({:.1}%)", 
                 self.medium_count, 100.0 * self.medium_count as f64 / total.max(1) as f64);
        println!("   Large (8KB-32KB): {} ({:.1}%)", 
                 self.large_count, 100.0 * self.large_count as f64 / total.max(1) as f64);
        println!("   Huge (>32KB):     {} ({:.1}%)", 
                 self.huge_count, 100.0 * self.huge_count as f64 / total.max(1) as f64);
        println!("   Total: {} docs, {} bytes", total, self.total_bytes);
    }
}

/// Sort documents by length for more efficient processing
/// 
/// Processing similar-sized documents together:
/// 1. Reduces variance in GPU kernel launch overhead
/// 2. Makes memory allocation patterns more predictable
/// 3. Allows autotune cache to be more effective
/// 
/// # Note
/// This collects documents into memory for sorting. For very large datasets,
/// consider using bucket-based streaming instead.
pub fn sort_by_length(mut docs: Vec<PrefetchedDoc>) -> Vec<PrefetchedDoc> {
    docs.sort_by_key(|d| d.original_len);
    docs
}

/// Sort documents by length in descending order (longest first)
/// 
/// Processing longest documents first can be beneficial for:
/// 1. Failing fast if a document is too large
/// 2. Front-loading the most expensive work
/// 3. Better progress estimation
pub fn sort_by_length_desc(mut docs: Vec<PrefetchedDoc>) -> Vec<PrefetchedDoc> {
    docs.sort_by_key(|d| std::cmp::Reverse(d.original_len));
    docs
}

/// Group documents into size buckets
/// 
/// Returns documents organized by bucket for batch processing.
/// Each bucket can be processed with similar GPU kernel configurations.
pub fn bucket_by_size(docs: Vec<PrefetchedDoc>) -> std::collections::HashMap<SizeBucket, Vec<PrefetchedDoc>> {
    let mut buckets = std::collections::HashMap::new();
    for doc in docs {
        let bucket = SizeBucket::from_len(doc.original_len);
        buckets.entry(bucket).or_insert_with(Vec::new).push(doc);
    }
    buckets
}

/// Streaming length-sorted iterator
/// 
/// For large datasets where we can't load all documents into memory,
/// this provides a sliding window approach that maintains approximate
/// length ordering within windows.
pub struct LengthSortedWindow {
    window: Vec<PrefetchedDoc>,
    window_size: usize,
}

impl LengthSortedWindow {
    /// Create a new length-sorted window
    /// 
    /// # Arguments
    /// * `window_size` - Number of documents to buffer for sorting (default: 100)
    pub fn new(window_size: usize) -> Self {
        Self {
            window: Vec::with_capacity(window_size),
            window_size,
        }
    }
    
    /// Add a document to the window
    /// 
    /// Returns sorted documents if window is full, otherwise None
    pub fn add(&mut self, doc: PrefetchedDoc) -> Option<Vec<PrefetchedDoc>> {
        self.window.push(doc);
        
        if self.window.len() >= self.window_size {
            let mut result = std::mem::take(&mut self.window);
            result.sort_by_key(|d| d.original_len);
            self.window = Vec::with_capacity(self.window_size);
            Some(result)
        } else {
            None
        }
    }
    
    /// Flush remaining documents (call at end of iteration)
    pub fn flush(&mut self) -> Vec<PrefetchedDoc> {
        let mut result = std::mem::take(&mut self.window);
        result.sort_by_key(|d| d.original_len);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_doc(id: &str, len: usize) -> PrefetchedDoc {
        PrefetchedDoc {
            id: id.to_string(),
            bytes: vec![0u8; len],
            original_len: len,
        }
    }
    
    #[test]
    fn test_size_bucket_classification() {
        assert_eq!(SizeBucket::from_len(100), SizeBucket::Tiny);
        assert_eq!(SizeBucket::from_len(511), SizeBucket::Tiny);
        assert_eq!(SizeBucket::from_len(512), SizeBucket::Small);
        assert_eq!(SizeBucket::from_len(2047), SizeBucket::Small);
        assert_eq!(SizeBucket::from_len(2048), SizeBucket::Medium);
        assert_eq!(SizeBucket::from_len(8191), SizeBucket::Medium);
        assert_eq!(SizeBucket::from_len(8192), SizeBucket::Large);
        assert_eq!(SizeBucket::from_len(32767), SizeBucket::Large);
        assert_eq!(SizeBucket::from_len(32768), SizeBucket::Huge);
        assert_eq!(SizeBucket::from_len(100000), SizeBucket::Huge);
    }
    
    #[test]
    fn test_sort_by_length() {
        let docs = vec![
            make_doc("c", 3000),
            make_doc("a", 100),
            make_doc("b", 1000),
        ];
        
        let sorted = sort_by_length(docs);
        assert_eq!(sorted[0].id, "a");
        assert_eq!(sorted[1].id, "b");
        assert_eq!(sorted[2].id, "c");
    }
    
    #[test]
    fn test_bucket_by_size() {
        let docs = vec![
            make_doc("tiny", 100),
            make_doc("small", 1000),
            make_doc("medium", 4000),
            make_doc("tiny2", 200),
        ];
        
        let buckets = bucket_by_size(docs);
        assert_eq!(buckets.get(&SizeBucket::Tiny).unwrap().len(), 2);
        assert_eq!(buckets.get(&SizeBucket::Small).unwrap().len(), 1);
        assert_eq!(buckets.get(&SizeBucket::Medium).unwrap().len(), 1);
    }
    
    #[test]
    fn test_length_sorted_window() {
        let mut window = LengthSortedWindow::new(3);
        
        // Add docs one by one
        assert!(window.add(make_doc("c", 3000)).is_none());
        assert!(window.add(make_doc("a", 100)).is_none());
        
        // Third doc triggers flush
        let batch = window.add(make_doc("b", 500)).unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].id, "a");  // 100 bytes
        assert_eq!(batch[1].id, "b");  // 500 bytes
        assert_eq!(batch[2].id, "c");  // 3000 bytes
        
        // Final flush
        window.add(make_doc("d", 200));
        let remaining = window.flush();
        assert_eq!(remaining.len(), 1);
    }
    
    #[test]
    fn test_batch_stats() {
        let mut stats = BatchStats::default();
        stats.add(&make_doc("a", 100));   // tiny
        stats.add(&make_doc("b", 1000));  // small
        stats.add(&make_doc("c", 4000));  // medium
        stats.add(&make_doc("d", 50));    // tiny
        
        assert_eq!(stats.tiny_count, 2);
        assert_eq!(stats.small_count, 1);
        assert_eq!(stats.medium_count, 1);
        assert_eq!(stats.total_bytes, 5150);
    }
}
