//! Double-buffered corpus writer for fast local ingestion with async external sync.
//!
//! Uses two local SQLite buffers (A/B) that swap when full. A background thread
//! continuously syncs completed buffers to the external drive's main corpus.db.
//!
//! This achieves local SSD throughput (~100+ KB/s) while writing to slow external
//! drives (~5 KB/s) in the background.
//!
//! # Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    INGESTION THREAD                         │
//! │  ┌──────────┐    swap     ┌──────────┐                     │
//! │  │ Buffer A │ ◄────────► │ Buffer B │  (local SSD: /tmp)   │
//! │  │ (active) │             │ (syncing)│                     │
//! │  └──────────┘             └──────────┘                     │
//! └─────────────────────────────────────────────────────────────┘
//!                                   │
//!                                   │ async batch ATTACH + INSERT
//!                                   ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  EXTERNAL DRIVE                             │
//! │  ┌────────────────────────────────────────────────────┐    │
//! │  │              corpus.db (append-only)               │    │
//! │  └────────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use anyhow::Result;
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Configuration for the double-buffer system
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Max documents per buffer before swap (default: 50)
    pub docs_per_buffer: usize,
    /// Local temp directory for buffers (default: /tmp/sphere_buffers)
    pub local_buffer_dir: PathBuf,
    /// Embedding dimension (must match corpus)
    pub embed_dim: usize,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            docs_per_buffer: 50,
            local_buffer_dir: PathBuf::from("/tmp/sphere_buffers"),
            embed_dim: 2048,
        }
    }
}

/// Message sent to the background sync thread
enum SyncMessage {
    /// A buffer is full and ready to sync
    BufferReady {
        buffer_path: PathBuf,
        buffer_id: usize,
    },
    /// Shutdown the sync thread
    Shutdown,
}

/// Double-buffered corpus writer
///
/// Wraps a CorpusDB with double-buffering for external drive performance.
/// The active buffer writes to local SSD, while background thread syncs
/// completed buffers to the external drive using ATTACH DATABASE.
pub struct BufferedCorpusWriter {
    /// Configuration
    config: BufferConfig,
    /// Current active buffer (A=0, B=1)
    active_buffer: usize,
    /// Local buffer connections (direct SQLite for simplicity)
    buffer_conns: [Option<Connection>; 2],
    /// Buffer paths
    buffer_paths: [PathBuf; 2],
    /// Documents in current buffer
    docs_in_buffer: usize,
    /// Channel to send sync requests
    sync_tx: Sender<SyncMessage>,
    /// Background sync thread handle
    sync_thread: Option<JoinHandle<()>>,
    /// Shared counter for synced documents
    synced_docs: Arc<AtomicUsize>,
    /// Flag indicating if sync thread is busy
    sync_busy: Arc<AtomicBool>,
    /// Total documents written
    total_docs: usize,
}

impl BufferedCorpusWriter {
    /// Create a new double-buffered writer
    pub fn new(external_db_path: &Path, config: BufferConfig) -> Result<Self> {
        // Create local buffer directory
        std::fs::create_dir_all(&config.local_buffer_dir)?;

        let buffer_a_path = config.local_buffer_dir.join("buffer_a.db");
        let buffer_b_path = config.local_buffer_dir.join("buffer_b.db");

        // Clean any existing buffers
        let _ = std::fs::remove_file(&buffer_a_path);
        let _ = std::fs::remove_file(&buffer_b_path);

        // Create buffer A with schema
        let buffer_a = create_buffer_db(&buffer_a_path, config.embed_dim)?;

        // Create channel for sync messages
        let (sync_tx, sync_rx) = mpsc::channel();

        // Shared state
        let synced_docs = Arc::new(AtomicUsize::new(0));
        let sync_busy = Arc::new(AtomicBool::new(false));

        // Spawn background sync thread
        let external_path = external_db_path.to_path_buf();
        let synced_docs_clone = synced_docs.clone();
        let sync_busy_clone = sync_busy.clone();
        let embed_dim = config.embed_dim;

        let sync_thread = thread::spawn(move || {
            sync_worker(
                sync_rx,
                external_path,
                synced_docs_clone,
                sync_busy_clone,
                embed_dim,
            );
        });

        Ok(Self {
            config,
            active_buffer: 0,
            buffer_conns: [Some(buffer_a), None],
            buffer_paths: [buffer_a_path, buffer_b_path],
            docs_in_buffer: 0,
            sync_tx,
            sync_thread: Some(sync_thread),
            synced_docs,
            sync_busy,
            total_docs: 0,
        })
    }

    /// Get the active buffer connection.
    ///
    /// # Panics
    /// Panics if the active buffer connection is not initialized (internal invariant).
    pub fn active_conn(&self) -> &Connection {
        self.buffer_conns[self.active_buffer]
            .as_ref()
            .expect("Active buffer should exist")
    }

    /// Get mutable active buffer connection.
    ///
    /// # Panics
    /// Panics if the active buffer connection is not initialized (internal invariant).
    pub fn active_conn_mut(&mut self) -> &mut Connection {
        self.buffer_conns[self.active_buffer]
            .as_mut()
            .expect("Active buffer should exist")
    }

    /// Increment document count and check for swap
    pub fn document_completed(&mut self) -> Result<()> {
        self.docs_in_buffer += 1;
        self.total_docs += 1;

        if self.docs_in_buffer >= self.config.docs_per_buffer {
            self.swap_buffers()?;
        }
        Ok(())
    }

    /// Swap active buffer and queue the full one for sync
    fn swap_buffers(&mut self) -> Result<()> {
        let full_buffer_id = self.active_buffer;
        let full_buffer_path = self.buffer_paths[full_buffer_id].clone();

        // Close the full buffer connection
        if let Some(conn) = self.buffer_conns[full_buffer_id].take() {
            conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
            drop(conn); // Close connection
        }

        // Switch to other buffer
        self.active_buffer = 1 - self.active_buffer;
        let new_buffer_path = &self.buffer_paths[self.active_buffer];

        // Wait if the other buffer is still being synced
        while self.sync_busy.load(Ordering::SeqCst) {
            println!("   ⏳ Waiting for sync to complete...");
            thread::sleep(std::time::Duration::from_millis(100));
        }

        // Clean and recreate the new active buffer
        let _ = std::fs::remove_file(new_buffer_path);
        self.buffer_conns[self.active_buffer] =
            Some(create_buffer_db(new_buffer_path, self.config.embed_dim)?);

        // Queue the full buffer for background sync
        self.sync_tx.send(SyncMessage::BufferReady {
            buffer_path: full_buffer_path,
            buffer_id: full_buffer_id,
        })?;

        println!(
            "   🔄 Swapped to buffer {} ({} docs queued for sync)",
            if self.active_buffer == 0 { "A" } else { "B" },
            self.docs_in_buffer
        );

        self.docs_in_buffer = 0;
        Ok(())
    }

    /// Get sync progress: (synced_docs, total_docs)
    pub fn sync_progress(&self) -> (usize, usize) {
        (self.synced_docs.load(Ordering::SeqCst), self.total_docs)
    }

    // ============================================================
    // CorpusDB-compatible methods for seamless integration
    // ============================================================

    /// Begin processing a document
    pub fn begin_document(&self, doc_id: &str, byte_count: usize) -> Result<()> {
        self.active_conn().execute(
            "INSERT OR REPLACE INTO documents (doc_id, status, byte_count, created_at)
             VALUES (?1, 'processing', ?2, datetime('now'))",
            params![doc_id, byte_count as i64],
        )?;
        Ok(())
    }

    /// Insert embeddings for a document
    pub fn insert_embeddings(
        &self,
        doc_id: &str,
        embeddings: &[f32],
        prominence: &[f32],
        entropies: &[f32],
        embed_dim: usize,
    ) -> Result<usize> {
        let n_patches = embeddings.len() / embed_dim;
        if n_patches == 0 {
            return Ok(0);
        }

        let conn = self.active_conn();
        let mut stmt = conn.prepare_cached(
            "INSERT INTO embeddings (doc_id, patch_idx, embedding, prominence, entropy)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )?;

        for i in 0..n_patches {
            let start = i * embed_dim;
            let end = start + embed_dim;
            let emb_slice = &embeddings[start..end];
            let emb_bytes: Vec<u8> = emb_slice.iter().flat_map(|f| f.to_le_bytes()).collect();
            let prom = prominence.get(i).copied();
            let ent = entropies.get(i).copied();
            stmt.execute(params![doc_id, i as i64, emb_bytes, prom, ent])?;
        }

        Ok(n_patches)
    }

    /// Insert patch bytes for a document
    pub fn insert_patch_bytes(
        &self,
        doc_id: &str,
        raw_bytes: &[u8],
        patch_lengths: &[usize],
    ) -> Result<()> {
        // Store as i32 to match CorpusDB format
        let lengths_blob: Vec<u8> = patch_lengths
            .iter()
            .flat_map(|&len| (len as i32).to_le_bytes())
            .collect();

        self.active_conn().execute(
            "INSERT INTO patch_bytes (doc_id, raw_bytes, patch_lengths) VALUES (?1, ?2, ?3)",
            params![doc_id, raw_bytes, lengths_blob],
        )?;
        Ok(())
    }

    /// Insert hypergraph sidecar for a document
    pub fn insert_hypergraph(
        &self,
        doc_id: &str,
        sidecar: &crate::sidecar::HypergraphSidecar,
    ) -> Result<()> {
        use crate::sidecar::{HypergraphEdge, NodeData};

        let conn = self.active_conn();

        // Insert nodes
        {
            let mut node_stmt = conn.prepare_cached(
                "INSERT INTO hypergraph_nodes (doc_id, node_idx, data) VALUES (?1, ?2, ?3)",
            )?;
            let mut meta_stmt = conn.prepare_cached(
                "INSERT OR REPLACE INTO patch_metadata (doc_id, node_idx, patch_index, prominence) 
                 VALUES (?1, ?2, ?3, ?4)",
            )?;

            for (idx, node) in sidecar.nodes.iter().enumerate() {
                let blob = bincode::serde::encode_to_vec(node, bincode::config::standard())?;
                node_stmt.execute(params![doc_id, idx as i64, blob])?;

                // Extract patch metadata from Leaf nodes
                if let NodeData::Leaf(seg) = node {
                    if let Some(ref meta) = seg.metadata {
                        if let Some(ref extra) = meta.extra {
                            let patch_index: u64 = extra
                                .get("patch_index")
                                .and_then(|v| v.as_u64().or_else(|| v.as_i64().map(|i| i as u64)))
                                .unwrap_or(idx as u64);
                            let prominence = extra
                                .get("prominence")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(1.0);

                            meta_stmt.execute(params![
                                doc_id,
                                idx as i64,
                                patch_index as i64,
                                prominence
                            ])?;
                        }
                    }
                }
            }
        }

        // Insert edges
        {
            let mut stmt = conn.prepare_cached(
                "INSERT INTO hypergraph_edges (doc_id, vertices, data) VALUES (?1, ?2, ?3)",
            )?;
            for (edge_data_idx, vertex_indices) in &sidecar.topology.edges {
                if *edge_data_idx < sidecar.edges.len() {
                    let edge_data = &sidecar.edges[*edge_data_idx];
                    let edge = HypergraphEdge {
                        vertices: vertex_indices.iter().map(|&v| v as u64).collect(),
                        label: edge_data.label.clone(),
                        weight: edge_data.weight,
                    };
                    let vertices_blob =
                        bincode::serde::encode_to_vec(&edge.vertices, bincode::config::standard())?;
                    let data_blob =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
                    stmt.execute(params![doc_id, vertices_blob, data_blob])?;
                }
            }
        }

        Ok(())
    }

    /// Complete a document and trigger buffer swap check
    pub fn complete_document(&mut self, doc_id: &str, patch_count: usize) -> Result<()> {
        self.active_conn().execute(
            "UPDATE documents SET status = 'complete', patch_count = ?2, completed_at = datetime('now')
             WHERE doc_id = ?1",
            params![doc_id, patch_count as i64],
        )?;

        self.document_completed()?;
        Ok(())
    }

    /// Print progress (delegates to sync progress)
    pub fn print_progress(&self, processed: usize, total: usize) -> Result<()> {
        let (synced, written) = self.sync_progress();
        let pending = written.saturating_sub(synced);
        println!(
            "📊 Progress: {processed}/{total} | Written: {written} | Synced: {synced} | Pending: {pending}"
        );
        Ok(())
    }

    /// Flush remaining buffer and wait for all syncs to complete.
    ///
    /// # Panics
    /// Panics if the background sync thread panicked during operation.
    pub fn finish(mut self) -> Result<()> {
        // Flush current buffer if it has data
        if self.docs_in_buffer > 0 {
            self.swap_buffers()?;
        }

        // Signal shutdown
        self.sync_tx.send(SyncMessage::Shutdown)?;

        // Wait for sync thread to finish
        if let Some(handle) = self.sync_thread.take() {
            handle.join().expect("Sync thread panicked");
        }

        println!(
            "✅ All {} documents synced to external drive",
            self.total_docs
        );
        Ok(())
    }
}

/// Create a buffer database with the corpus schema
fn create_buffer_db(path: &Path, embed_dim: usize) -> Result<Connection> {
    let conn = Connection::open(path)?;

    // Use WAL mode for better write performance
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA cache_size = -64000;  -- 64MB cache
         PRAGMA temp_store = MEMORY;",
    )?;

    // Create same schema as CorpusDB
    conn.execute_batch(&format!(
        "CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_dim', '{embed_dim}');
        INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '3');
        
        CREATE TABLE IF NOT EXISTS documents (
            doc_id      TEXT PRIMARY KEY,
            text        TEXT,
            source      TEXT,
            status      TEXT DEFAULT 'pending',
            byte_count  INTEGER,
            patch_count INTEGER,
            created_at  TEXT,
            completed_at TEXT
        );
        
        CREATE TABLE IF NOT EXISTS embeddings (
            id         INTEGER PRIMARY KEY,
            doc_id     TEXT NOT NULL,
            patch_idx  INTEGER NOT NULL,
            embedding  BLOB NOT NULL,
            prominence REAL,
            entropy    REAL
        );
        
        CREATE TABLE IF NOT EXISTS hypergraph_nodes (
            id       INTEGER PRIMARY KEY,
            doc_id   TEXT NOT NULL,
            node_idx INTEGER NOT NULL,
            data     BLOB NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS hypergraph_edges (
            id       INTEGER PRIMARY KEY,
            doc_id   TEXT NOT NULL,
            vertices BLOB NOT NULL,
            data     BLOB NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS patch_bytes (
            id            INTEGER PRIMARY KEY,
            doc_id        TEXT NOT NULL,
            raw_bytes     BLOB NOT NULL,
            patch_lengths BLOB NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS patch_metadata (
            id          INTEGER PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            node_idx    INTEGER NOT NULL,
            patch_index INTEGER NOT NULL,
            prominence  REAL NOT NULL DEFAULT 1.0,
            UNIQUE(doc_id, node_idx)
        );
        
        CREATE INDEX IF NOT EXISTS idx_emb_doc ON embeddings(doc_id);
        CREATE INDEX IF NOT EXISTS idx_nodes_doc ON hypergraph_nodes(doc_id);
        CREATE INDEX IF NOT EXISTS idx_edges_doc ON hypergraph_edges(doc_id);
        CREATE INDEX IF NOT EXISTS idx_patch_bytes_doc ON patch_bytes(doc_id);
        CREATE INDEX IF NOT EXISTS idx_patch_meta_doc ON patch_metadata(doc_id);
        CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(status);"
    ))?;

    Ok(conn)
}

/// Background worker that syncs buffers to external drive using ATTACH DATABASE.
/// Takes ownership of all parameters as they're moved into the spawned thread.
#[allow(clippy::needless_pass_by_value)] // Thread worker needs owned values
fn sync_worker(
    rx: Receiver<SyncMessage>,
    external_db_path: PathBuf,
    synced_docs: Arc<AtomicUsize>,
    sync_busy: Arc<AtomicBool>,
    embed_dim: usize,
) {
    // Create external database if it doesn't exist
    if !external_db_path.exists() {
        if let Err(e) = create_buffer_db(&external_db_path, embed_dim) {
            eprintln!("Failed to create external DB: {e}");
            return;
        }
    }

    loop {
        match rx.recv() {
            Ok(SyncMessage::BufferReady {
                buffer_path,
                buffer_id,
            }) => {
                sync_busy.store(true, Ordering::SeqCst);

                let start = std::time::Instant::now();
                match sync_buffer_via_attach(&buffer_path, &external_db_path) {
                    Ok(doc_count) => {
                        synced_docs.fetch_add(doc_count, Ordering::SeqCst);
                        println!(
                            "   📤 Synced buffer {} ({} docs) in {:.2}s",
                            if buffer_id == 0 { "A" } else { "B" },
                            doc_count,
                            start.elapsed().as_secs_f64()
                        );
                    }
                    Err(e) => {
                        eprintln!("   ❌ Sync error: {e}");
                    }
                }

                // Clean up the synced buffer file
                let _ = std::fs::remove_file(&buffer_path);
                let _ = std::fs::remove_file(buffer_path.with_extension("db-wal"));
                let _ = std::fs::remove_file(buffer_path.with_extension("db-shm"));

                sync_busy.store(false, Ordering::SeqCst);
            }
            Ok(SyncMessage::Shutdown) => {
                // Final checkpoint on external
                if let Ok(conn) = Connection::open(&external_db_path) {
                    let _ = conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
                }
                break;
            }
            Err(_) => break, // Channel closed
        }
    }
}

/// Sync buffer to external using ATTACH DATABASE for efficient bulk transfer
fn sync_buffer_via_attach(buffer_path: &Path, external_path: &Path) -> Result<usize> {
    let conn = Connection::open(external_path)?;

    // Attach the buffer database
    conn.execute(
        "ATTACH DATABASE ?1 AS buffer",
        params![buffer_path.to_string_lossy().to_string()],
    )?;

    // Count documents to sync
    let doc_count: usize = conn.query_row(
        "SELECT COUNT(*) FROM buffer.documents WHERE status = 'complete'",
        [],
        |row| row.get(0),
    )?;

    // Bulk copy all tables using INSERT ... SELECT (much faster than row-by-row)
    conn.execute_batch(
        "BEGIN TRANSACTION;
        
        -- Documents
        INSERT OR REPLACE INTO documents 
        SELECT * FROM buffer.documents WHERE status = 'complete';
        
        -- Embeddings
        INSERT INTO embeddings (doc_id, patch_idx, embedding, prominence, entropy)
        SELECT doc_id, patch_idx, embedding, prominence, entropy 
        FROM buffer.embeddings 
        WHERE doc_id IN (SELECT doc_id FROM buffer.documents WHERE status = 'complete');
        
        -- Hypergraph nodes
        INSERT INTO hypergraph_nodes (doc_id, node_idx, data)
        SELECT doc_id, node_idx, data 
        FROM buffer.hypergraph_nodes
        WHERE doc_id IN (SELECT doc_id FROM buffer.documents WHERE status = 'complete');
        
        -- Hypergraph edges
        INSERT INTO hypergraph_edges (doc_id, vertices, data)
        SELECT doc_id, vertices, data 
        FROM buffer.hypergraph_edges
        WHERE doc_id IN (SELECT doc_id FROM buffer.documents WHERE status = 'complete');
        
        -- Patch bytes
        INSERT INTO patch_bytes (doc_id, raw_bytes, patch_lengths)
        SELECT doc_id, raw_bytes, patch_lengths 
        FROM buffer.patch_bytes
        WHERE doc_id IN (SELECT doc_id FROM buffer.documents WHERE status = 'complete');
        
        -- Patch metadata
        INSERT OR REPLACE INTO patch_metadata (doc_id, node_idx, patch_index, prominence)
        SELECT doc_id, node_idx, patch_index, prominence 
        FROM buffer.patch_metadata
        WHERE doc_id IN (SELECT doc_id FROM buffer.documents WHERE status = 'complete');
        
        COMMIT;
        
        DETACH DATABASE buffer;",
    )?;

    Ok(doc_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_buffer_creation() {
        let temp = tempdir().unwrap();
        let path = temp.path().join("test.db");
        let conn = create_buffer_db(&path, 768).unwrap();

        let dim: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'embed_dim'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(dim, "768");
    }

    #[test]
    fn test_attach_sync() {
        let temp = tempdir().unwrap();
        let buffer_path = temp.path().join("buffer.db");
        let external_path = temp.path().join("external.db");

        // Create buffer with some data
        let buffer = create_buffer_db(&buffer_path, 768).unwrap();
        buffer.execute(
            "INSERT INTO documents (doc_id, text, source, status) VALUES ('doc1', 'test', 'src', 'complete')",
            []
        ).unwrap();
        buffer
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .unwrap();
        drop(buffer);

        // Create external
        create_buffer_db(&external_path, 768).unwrap();

        // Sync
        let count = sync_buffer_via_attach(&buffer_path, &external_path).unwrap();
        assert_eq!(count, 1);

        // Verify
        let ext = Connection::open(&external_path).unwrap();
        let doc_count: usize = ext
            .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
            .unwrap();
        assert_eq!(doc_count, 1);
    }
}
