//! Async document prefetching for BLT ingestion pipeline
//!
//! Uses a background thread to load documents while GPU processes current batch.
//! Based on Burn's async execution model where different threads get different
//! execution queues, allowing prefetch to not block GPU compute.
//!
//! # Key Insight from Burn Docs
//! > "CubeCL-based backends assign different execution queues for different threads,
//! > meaning that syncing a thread shouldn't impact the throughput of another thread."

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::{self, JoinHandle};

/// A document ready for processing
#[derive(Debug, Clone)]
pub struct PrefetchedDoc {
    /// Unique identifier (e.g., dataset item ID)
    pub id: String,
    /// Raw document bytes
    pub bytes: Vec<u8>,
    /// Original byte length (for sorting/batching metrics)
    pub original_len: usize,
}

/// Background document prefetcher using bounded channel
///
/// # Architecture
/// ```text
/// [Dataset Iterator] --spawn--> [Background Thread]
///                                      |
///                               [Bounded Channel (N slots)]
///                                      |
///                               [Main Thread: GPU Processing]
/// ```
///
/// The bounded channel provides natural backpressure:
/// - If GPU is slow, prefetcher blocks waiting for channel space
/// - If prefetch is slow, GPU processes all buffered docs then waits
pub struct DocumentPrefetcher {
    receiver: Receiver<Option<PrefetchedDoc>>,
    handle: Option<JoinHandle<()>>,
    buffer_size: usize,
    docs_prefetched: usize,
}

impl DocumentPrefetcher {
    /// Create a new prefetcher with specified buffer size
    ///
    /// # Arguments
    /// * `iter` - Iterator yielding (id, bytes) pairs
    /// * `buffer_size` - Number of documents to buffer ahead (default: 4)
    ///
    /// # Example
    /// ```rust,ignore
    /// let docs = dataset.iter().map(|item| (item.id, item.text.into_bytes()));
    /// let prefetcher = DocumentPrefetcher::new(docs, 4);
    ///
    /// while let Some(doc) = prefetcher.next() {
    ///     // GPU processes doc.bytes while next doc loads in background
    ///     let result = process_data(&doc.bytes, &model, &device, threshold)?;
    /// }
    /// ```
    pub fn new<I>(iter: I, buffer_size: usize) -> Self
    where
        I: Iterator<Item = (String, Vec<u8>)> + Send + 'static,
    {
        // Bounded channel provides backpressure
        let (tx, rx) = sync_channel::<Option<PrefetchedDoc>>(buffer_size);

        let handle = thread::spawn(move || {
            Self::prefetch_worker(iter, tx);
        });

        Self {
            receiver: rx,
            handle: Some(handle),
            buffer_size,
            docs_prefetched: 0,
        }
    }

    /// Worker function running in background thread.
    /// Takes ownership of `tx` as it's moved into the spawned thread.
    #[allow(clippy::needless_pass_by_value)] // Thread worker needs owned SyncSender
    fn prefetch_worker<I>(iter: I, tx: SyncSender<Option<PrefetchedDoc>>)
    where
        I: Iterator<Item = (String, Vec<u8>)>,
    {
        for (id, bytes) in iter {
            let original_len = bytes.len();
            let doc = PrefetchedDoc {
                id,
                bytes,
                original_len,
            };

            // send() blocks if channel is full (backpressure)
            if tx.send(Some(doc)).is_err() {
                // Receiver dropped, stop prefetching
                break;
            }
        }

        // Signal completion
        let _ = tx.send(None);
    }

    /// Get next prefetched document
    ///
    /// Returns `None` when all documents have been processed.
    /// Blocks if no document is ready yet.
    pub fn recv(&mut self) -> Option<PrefetchedDoc> {
        match self.receiver.recv() {
            Ok(Some(doc)) => {
                self.docs_prefetched += 1;
                Some(doc)
            }
            Ok(None) | Err(_) => None, // Iterator exhausted or channel closed
        }
    }

    /// Get the configured buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get count of documents prefetched so far
    pub fn docs_prefetched(&self) -> usize {
        self.docs_prefetched
    }
}

impl Drop for DocumentPrefetcher {
    fn drop(&mut self) {
        // Wait for worker thread to finish
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Iterator adapter for DocumentPrefetcher
impl Iterator for DocumentPrefetcher {
    type Item = PrefetchedDoc;

    fn next(&mut self) -> Option<Self::Item> {
        self.recv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetcher_basic() {
        let docs = vec![
            ("doc1".to_string(), b"hello".to_vec()),
            ("doc2".to_string(), b"world".to_vec()),
            ("doc3".to_string(), b"test".to_vec()),
        ];

        let mut prefetcher = DocumentPrefetcher::new(docs.into_iter(), 2);

        let d1 = prefetcher.next().unwrap();
        assert_eq!(d1.id, "doc1");
        assert_eq!(d1.bytes, b"hello");

        let d2 = prefetcher.next().unwrap();
        assert_eq!(d2.id, "doc2");

        let d3 = prefetcher.next().unwrap();
        assert_eq!(d3.id, "doc3");

        assert!(prefetcher.next().is_none());
        assert_eq!(prefetcher.docs_prefetched(), 3);
    }

    #[test]
    fn test_prefetcher_empty() {
        let docs: Vec<(String, Vec<u8>)> = vec![];
        let mut prefetcher = DocumentPrefetcher::new(docs.into_iter(), 4);
        assert!(prefetcher.next().is_none());
    }

    #[test]
    fn test_prefetcher_as_iterator() {
        let docs = vec![
            ("a".to_string(), vec![1, 2, 3]),
            ("b".to_string(), vec![4, 5, 6]),
        ];

        let prefetcher = DocumentPrefetcher::new(docs.into_iter(), 2);
        let collected: Vec<_> = prefetcher.collect();

        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].id, "a");
        assert_eq!(collected[1].id, "b");
    }
}
