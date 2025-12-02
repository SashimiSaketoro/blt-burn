//! Unified Corpus Database for Web-Scale Ingestion
//!
//! This module provides a single SQLite database that stores all corpus data:
//! - Embeddings (patch-level, stored as BLOBs)
//! - Raw bytes and patch structure (for BLT v3 export)
//! - Hypergraph nodes and edges
//! - Processing checkpoints for crash recovery
//!
//! ## Design Goals
//!
//! 1. **Streaming writes**: Data is written as it's processed, never held in memory
//! 2. **Crash recovery**: Resume from last successful document on restart
//! 3. **Single file output**: No explosion of per-document files
//! 4. **ACID guarantees**: SQLite transactions protect against corruption
//! 5. **Web-scale ready**: Tested for multi-TB datasets running for weeks
//!
//! ## BLT v3 SafeTensors Export
//!
//! The `export_safetensors()` method produces a file compatible with
//! `thrml-sphere::load_blt_safetensors()`:
//!
//! | Tensor | Shape | Dtype | Description |
//! |--------|-------|-------|-------------|
//! | `embeddings` | [N, D] | F32 | Patch embeddings |
//! | `prominence` | [N] | F32 | Importance scores |
//! | `patch_entropies` | [N] | F32 | Per-patch entropy |
//! | `bytes` | [total_bytes] | U8 | Concatenated raw bytes |
//! | `patch_lengths` | [N] | I32 | Length of each patch |
//!
//! ## Hypergraph Sidecar
//!
//! The `export_hypergraph_sidecar()` method produces a `.hypergraph.db` file
//! compatible with `thrml-sphere::load_hypergraph_from_sqlite()`.

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use std::path::Path;
use std::time::Instant;

use crate::sidecar::{HypergraphEdge, HypergraphSidecar};

/// Flattened representation of a ROOTS tree node for database storage.
///
/// The tree is serialized as a flat list of nodes with parent/child references.
/// This allows efficient storage and reconstruction of the hierarchical structure.
#[derive(Debug, Clone)]
pub struct RootsTreeNode {
    /// Unique node ID (0 = root)
    pub node_id: usize,
    /// Parent node ID (None for root)
    pub parent_id: Option<usize>,
    /// True if this is a leaf (partition), false if internal (signpost)
    pub is_leaf: bool,
    /// Partition ID (only set for leaf nodes)
    pub partition_id: Option<usize>,
    /// Left child node ID (only set for internal nodes)
    pub left_child: Option<usize>,
    /// Right child node ID (only set for internal nodes)
    pub right_child: Option<usize>,
    /// Centroid embedding (aggregate for internal, partition centroid for leaf)
    pub centroid: Vec<f32>,
    /// Total points in this subtree
    pub point_count: usize,
    /// Radius range (min, max) for this subtree
    pub radius_range: (f32, f32),
    /// Prominence range (min, max) for this subtree
    pub prom_range: (f32, f32),
}

/// Document processing status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocStatus {
    Processing,
    Complete,
    Failed,
}

/// A single embedding record from the database.
#[derive(Debug, Clone)]
pub struct EmbeddingRecord {
    /// Global patch ID
    pub id: i64,
    /// Raw embedding bytes (to be decoded by caller)
    pub embedding: Vec<u8>,
    /// Prominence score
    pub prominence: f32,
    /// Entropy value
    pub entropy: f32,
}

/// Spherical harmonics basis configuration stored in database.
#[derive(Debug, Clone)]
pub struct HarmonicsBasisRecord {
    /// Band limit L
    pub band_limit: usize,
    /// Grid theta dimension
    pub n_theta: usize,
    /// Grid phi dimension  
    pub n_phi: usize,
    /// Precomputed basis values
    pub basis_values: Vec<f64>,
}

/// Unified corpus database for web-scale ingestion
pub struct CorpusDB {
    conn: Connection,
    embed_dim: usize,
    batch_size: usize,
    pending_docs: usize,
    start_time: Instant,
}

impl CorpusDB {
    /// Read embed_dim from an existing database without opening it fully.
    /// Returns None if the database doesn't exist or doesn't have embed_dim set.
    pub fn read_embed_dim(path: &Path) -> Option<usize> {
        let conn = Connection::open(path).ok()?;
        let dim_str: String = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'embed_dim'",
                [],
                |row| row.get(0),
            )
            .ok()?;
        dim_str.parse().ok()
    }

    /// Open an existing corpus database, reading embed_dim from metadata.
    /// Fails if the database doesn't exist or doesn't have embed_dim set.
    pub fn open_existing(path: &Path) -> Result<Self> {
        let embed_dim = Self::read_embed_dim(path)
            .ok_or_else(|| anyhow::anyhow!("Cannot read embed_dim from {path:?}"))?;
        Self::open(path, embed_dim)
    }

    /// Open or create a corpus database
    ///
    /// If the database exists, it will be opened for append.
    /// If it doesn't exist, it will be created with the schema.
    pub fn open(path: &Path, embed_dim: usize) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open corpus DB at {path:?}"))?;

        // Enable WAL mode for concurrent reads during writes
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -64000;  -- 64MB cache
             PRAGMA temp_store = MEMORY;",
        )?;

        // Create schema if needed
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                doc_id       TEXT PRIMARY KEY,
                status       TEXT NOT NULL DEFAULT 'processing',
                byte_count   INTEGER,
                patch_count  INTEGER,
                created_at   TEXT DEFAULT (datetime('now')),
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id           INTEGER PRIMARY KEY,
                doc_id       TEXT NOT NULL,
                patch_idx    INTEGER NOT NULL,
                embedding    BLOB NOT NULL,
                prominence   REAL,
                entropy      REAL,
                -- SphereStage: spherical coordinates (added by sphere optimization)
                theta        REAL,
                phi          REAL,
                r            REAL,
                -- RootsStage: partition assignment (added by ROOTS indexing)
                partition_id INTEGER
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

            -- Raw bytes and patch structure (for BLT v3 SafeTensors export)
            CREATE TABLE IF NOT EXISTS patch_bytes (
                id            INTEGER PRIMARY KEY,
                doc_id        TEXT NOT NULL,
                raw_bytes     BLOB NOT NULL,
                patch_lengths BLOB NOT NULL
            );

            -- RootsStage: partition centroids (leaves of the tree)
            CREATE TABLE IF NOT EXISTS partitions (
                id         INTEGER PRIMARY KEY,
                centroid   BLOB NOT NULL,      -- [embed_dim] floats
                n_members  INTEGER NOT NULL DEFAULT 0
            );
            
            -- RootsStage: hierarchical tree structure (mutable - rebuilt after sphere optimization)
            CREATE TABLE IF NOT EXISTS roots_tree (
                node_id      INTEGER PRIMARY KEY,
                parent_id    INTEGER,              -- NULL for root node
                is_leaf      BOOLEAN NOT NULL,
                partition_id INTEGER,              -- Only set for leaf nodes
                left_child   INTEGER,              -- Only set for internal nodes
                right_child  INTEGER,              -- Only set for internal nodes
                centroid     BLOB NOT NULL,        -- Aggregate centroid
                point_count  INTEGER NOT NULL,
                radius_min   REAL,
                radius_max   REAL,
                prom_min     REAL,
                prom_max     REAL,
                FOREIGN KEY (partition_id) REFERENCES partitions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_roots_tree_parent ON roots_tree(parent_id);
            
            -- RootsStage: tree metadata (version tracking for staleness detection)
            CREATE TABLE IF NOT EXISTS roots_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            
            -- HarmonicsStage: precomputed spherical harmonics basis (corpus-level)
            CREATE TABLE IF NOT EXISTS harmonics_basis (
                id           INTEGER PRIMARY KEY,
                band_limit   INTEGER NOT NULL,
                n_theta      INTEGER NOT NULL,
                n_phi        INTEGER NOT NULL,
                basis_values BLOB NOT NULL     -- serialized f64 array
            );
            
            -- Pipeline stage tracking (for resume/audit)
            CREATE TABLE IF NOT EXISTS pipeline_stages (
                stage_name   TEXT PRIMARY KEY,
                status       TEXT NOT NULL DEFAULT 'pending',  -- pending, running, complete, failed
                started_at   TEXT,
                completed_at TEXT,
                config_json  TEXT                              -- stage config for reproducibility
            );

            -- Indices for efficient queries
            -- Direct patch metadata for sphere-optimized export (avoids bincode decode issues)
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
            CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(status);",
        )?;

        // Store/verify embed_dim
        let stored_dim: Option<String> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'embed_dim'",
                [],
                |row| row.get(0),
            )
            .ok();

        if let Some(dim_str) = stored_dim {
            let stored: usize = dim_str.parse().unwrap_or(0);
            if stored != embed_dim && stored > 0 {
                anyhow::bail!(
                    "Embedding dimension mismatch: DB has {stored}, but encoder produces {embed_dim}"
                );
            }
        } else {
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('embed_dim', ?1)",
                params![embed_dim.to_string()],
            )?;
        }

        // Store schema version
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '2')",
            [],
        )?;

        Ok(Self {
            conn,
            embed_dim,
            batch_size: 100, // Commit every N docs
            pending_docs: 0,
            start_time: Instant::now(),
        })
    }

    /// Check if a document has already been processed
    pub fn is_doc_complete(&self, doc_id: &str) -> Result<bool> {
        let status: Option<String> = self
            .conn
            .query_row(
                "SELECT status FROM documents WHERE doc_id = ?1",
                params![doc_id],
                |row| row.get(0),
            )
            .ok();

        Ok(status.is_some_and(|s| s == "complete"))
    }

    /// Get count of completed documents (for progress reporting)
    pub fn completed_count(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE status = 'complete'",
            [],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Get total patch count across all documents
    pub fn total_patches(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Begin processing a document (marks it as 'processing')
    pub fn begin_document(&mut self, doc_id: &str, byte_count: usize) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, status, byte_count, created_at)
             VALUES (?1, 'processing', ?2, datetime('now'))",
            params![doc_id, byte_count as i64],
        )?;
        Ok(())
    }

    /// Insert embeddings for a document
    pub fn insert_embeddings(
        &mut self,
        doc_id: &str,
        embeddings: &[f32],
        prominence: &[f32],
        entropies: &[f32],
    ) -> Result<usize> {
        let n_patches = embeddings.len() / self.embed_dim;
        if n_patches == 0 {
            return Ok(0);
        }

        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO embeddings (doc_id, patch_idx, embedding, prominence, entropy)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
            )?;

            for i in 0..n_patches {
                let start = i * self.embed_dim;
                let end = start + self.embed_dim;
                let emb_slice = &embeddings[start..end];

                // Convert f32 slice to bytes
                let emb_bytes: Vec<u8> = emb_slice.iter().flat_map(|f| f.to_le_bytes()).collect();

                let prom = prominence.get(i).copied();
                let ent = entropies.get(i).copied();

                stmt.execute(params![doc_id, i as i64, emb_bytes, prom, ent])?;
            }
        }
        tx.commit()?;

        Ok(n_patches)
    }

    /// Insert raw bytes and patch lengths for BLT v3 export
    ///
    /// This stores the original document bytes and patch structure needed for
    /// substring coupling in ROOTS navigation.
    pub fn insert_patch_bytes(
        &mut self,
        doc_id: &str,
        raw_bytes: &[u8],
        patch_lengths: &[i32],
    ) -> Result<()> {
        // Encode patch_lengths as bytes
        let lengths_bytes: Vec<u8> = patch_lengths.iter().flat_map(|l| l.to_le_bytes()).collect();

        self.conn.execute(
            "INSERT INTO patch_bytes (doc_id, raw_bytes, patch_lengths) VALUES (?1, ?2, ?3)",
            params![doc_id, raw_bytes, lengths_bytes],
        )?;

        Ok(())
    }

    /// Insert hypergraph sidecar for a document
    pub fn insert_hypergraph(&mut self, doc_id: &str, sidecar: &HypergraphSidecar) -> Result<()> {
        use crate::sidecar::NodeData;
        let tx = self.conn.transaction()?;

        // Insert nodes and extract patch metadata
        {
            let mut node_stmt = tx.prepare_cached(
                "INSERT INTO hypergraph_nodes (doc_id, node_idx, data) VALUES (?1, ?2, ?3)",
            )?;
            let mut meta_stmt = tx.prepare_cached(
                "INSERT OR REPLACE INTO patch_metadata (doc_id, node_idx, patch_index, prominence) 
                 VALUES (?1, ?2, ?3, ?4)",
            )?;

            for (idx, node) in sidecar.nodes.iter().enumerate() {
                let blob = bincode::serde::encode_to_vec(node, bincode::config::standard())?;
                node_stmt.execute(params![doc_id, idx as i64, blob])?;

                // Extract patch metadata from Leaf nodes for sphere-optimized export
                if let NodeData::Leaf(seg) = node {
                    if let Some(ref meta) = seg.metadata {
                        if let Some(ref extra) = meta.extra {
                            // Extract patch_index and prominence from JSON
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
            let mut stmt = tx.prepare_cached(
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

        tx.commit()?;
        Ok(())
    }

    /// Mark document as complete
    pub fn complete_document(&mut self, doc_id: &str, patch_count: usize) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET status = 'complete', patch_count = ?2, completed_at = datetime('now')
             WHERE doc_id = ?1",
            params![doc_id, patch_count as i64],
        )?;

        self.pending_docs += 1;

        // Checkpoint WAL periodically to bound its size
        if self.pending_docs >= self.batch_size {
            self.checkpoint()?;
            self.pending_docs = 0;
        }

        Ok(())
    }

    /// Mark document as failed
    pub fn fail_document(&mut self, doc_id: &str, error: &str) -> Result<()> {
        // Truncate error message if too long
        let error_truncated = if error.len() > 500 {
            &error[..500]
        } else {
            error
        };

        self.conn.execute(
            "UPDATE documents SET status = 'failed', completed_at = datetime('now')
             WHERE doc_id = ?1",
            params![doc_id],
        )?;

        // Store error in meta for debugging
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![format!("error:{}", doc_id), error_truncated],
        )?;

        Ok(())
    }

    /// Checkpoint WAL to main database file
    pub fn checkpoint(&mut self) -> Result<()> {
        self.conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE);")?;
        Ok(())
    }

    /// Get direct access to the SQLite connection (for advanced operations)
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Get processing statistics
    pub fn stats(&self) -> Result<CorpusStats> {
        let total_docs: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;

        let complete_docs: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE status = 'complete'",
            [],
            |row| row.get(0),
        )?;

        let failed_docs: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE status = 'failed'",
            [],
            |row| row.get(0),
        )?;

        let total_patches: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))?;

        let total_bytes: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(byte_count), 0) FROM documents WHERE status = 'complete'",
            [],
            |row| row.get(0),
        )?;

        Ok(CorpusStats {
            total_docs: total_docs as usize,
            complete_docs: complete_docs as usize,
            failed_docs: failed_docs as usize,
            total_patches: total_patches as usize,
            total_bytes: total_bytes as usize,
            elapsed_secs: self.start_time.elapsed().as_secs_f64(),
        })
    }

    /// Print progress summary
    pub fn print_progress(&self, current: usize, total: usize) -> Result<()> {
        let stats = self.stats()?;
        let rate = stats.complete_docs as f64 / stats.elapsed_secs.max(0.001);
        let eta_secs = if rate > 0.0 {
            (total - current) as f64 / rate
        } else {
            0.0
        };

        let eta_str = if eta_secs > 3600.0 {
            format!("{:.1}h", eta_secs / 3600.0)
        } else if eta_secs > 60.0 {
            format!("{:.1}m", eta_secs / 60.0)
        } else {
            format!("{eta_secs:.0}s")
        };

        println!(
            "📊 Progress: {}/{} docs ({:.1}%) | {} patches | {:.1} docs/s | ETA: {}",
            stats.complete_docs,
            total,
            100.0 * stats.complete_docs as f64 / total.max(1) as f64,
            stats.total_patches,
            rate,
            eta_str
        );

        Ok(())
    }

    /// Export to BLT v3 SafeTensors format for thrml-sphere
    ///
    /// Produces a file with full BLT v3 format:
    /// - `embeddings`: [N, D] f32 - patch embeddings
    /// - `prominence`: [N] f32 - prominence scores
    /// - `patch_entropies`: [N] f32 - per-patch entropy values
    /// - `bytes`: [total_bytes] u8 - concatenated raw bytes
    /// - `patch_lengths`: [N] i32 - length of each patch
    ///
    /// This format is compatible with thrml-sphere's `load_blt_safetensors()`.
    pub fn export_safetensors(&self, output_path: &Path) -> Result<usize> {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let total_patches = self.total_patches()?;
        if total_patches == 0 {
            anyhow::bail!("No patches to export");
        }

        println!(
            "📦 Exporting {total_patches} patches to BLT v3 SafeTensors..."
        );

        // Pre-allocate vectors for embeddings
        let mut all_embeddings: Vec<f32> = Vec::with_capacity(total_patches * self.embed_dim);
        let mut all_prominence: Vec<f32> = Vec::with_capacity(total_patches);
        let mut all_entropies: Vec<f32> = Vec::with_capacity(total_patches);

        // Query all embeddings ordered by doc_id, patch_idx
        let mut stmt = self.conn.prepare(
            "SELECT embedding, prominence, entropy 
             FROM embeddings 
             ORDER BY doc_id, patch_idx",
        )?;

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let emb_blob: Vec<u8> = row.get(0)?;
            let prom: Option<f64> = row.get(1)?;
            let ent: Option<f64> = row.get(2)?;

            // Decode embedding bytes to f32
            for chunk in emb_blob.chunks_exact(4) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                all_embeddings.push(val);
            }

            all_prominence.push(prom.unwrap_or(1.0) as f32);
            all_entropies.push(ent.unwrap_or(0.0) as f32);
        }

        let n_patches = all_prominence.len();

        // Query raw bytes and patch lengths
        let mut all_bytes: Vec<u8> = Vec::new();
        let mut all_patch_lengths: Vec<i32> = Vec::new();

        let mut bytes_stmt = self
            .conn
            .prepare("SELECT raw_bytes, patch_lengths FROM patch_bytes ORDER BY doc_id")?;

        let mut bytes_rows = bytes_stmt.query([])?;
        while let Some(row) = bytes_rows.next()? {
            let raw_bytes: Vec<u8> = row.get(0)?;
            let lengths_blob: Vec<u8> = row.get(1)?;

            all_bytes.extend_from_slice(&raw_bytes);

            // Decode patch_lengths from bytes
            for chunk in lengths_blob.chunks_exact(4) {
                let len = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                all_patch_lengths.push(len);
            }
        }

        // Build SafeTensors
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();

        // Embeddings [N, D]
        let emb_shape = vec![n_patches, self.embed_dim];
        tensors.insert(
            "embeddings".to_string(),
            TensorView::new(Dtype::F32, emb_shape, bytemuck::cast_slice(&all_embeddings))?,
        );

        // Prominence [N]
        tensors.insert(
            "prominence".to_string(),
            TensorView::new(
                Dtype::F32,
                vec![n_patches],
                bytemuck::cast_slice(&all_prominence),
            )?,
        );

        // Patch entropies [N] (thrml-sphere looks for "patch_entropies")
        tensors.insert(
            "patch_entropies".to_string(),
            TensorView::new(
                Dtype::F32,
                vec![n_patches],
                bytemuck::cast_slice(&all_entropies),
            )?,
        );

        // Raw bytes [total_bytes] - for substring coupling in ROOTS
        if !all_bytes.is_empty() {
            tensors.insert(
                "bytes".to_string(),
                TensorView::new(Dtype::U8, vec![all_bytes.len()], &all_bytes)?,
            );

            // Patch lengths [N]
            tensors.insert(
                "patch_lengths".to_string(),
                TensorView::new(
                    Dtype::I32,
                    vec![all_patch_lengths.len()],
                    bytemuck::cast_slice(&all_patch_lengths),
                )?,
            );
        }

        // Serialize with metadata
        let metadata: HashMap<String, String> = [
            ("format".to_string(), "blt_patches_v3".to_string()),
            ("embed_dim".to_string(), self.embed_dim.to_string()),
            ("n_patches".to_string(), n_patches.to_string()),
            ("total_bytes".to_string(), all_bytes.len().to_string()),
        ]
        .into_iter()
        .collect();

        let serialized = safetensors::serialize(&tensors, &Some(metadata))?;
        std::fs::write(output_path, serialized)?;

        println!(
            "✅ Exported {} patches ({} bytes) to {}",
            n_patches,
            all_bytes.len(),
            output_path.display()
        );

        Ok(n_patches)
    }

    /// Export hypergraph data to a sidecar SQLite file for thrml-sphere
    ///
    /// Creates a `.hypergraph.db` file in the format expected by thrml-sphere's
    /// `load_hypergraph_from_sqlite()` function. The sidecar consolidates all
    /// per-document hypergraph nodes and edges into a single corpus-level file.
    ///
    /// **Important:** This function remaps per-document node IDs and patch indices
    /// to global corpus-level indices, ensuring spring physics work correctly
    /// across document boundaries.
    pub fn export_hypergraph_sidecar(&self, output_path: &Path) -> Result<usize> {
        use crate::sidecar::HypergraphEdge;
        use rusqlite::Connection;
        use std::collections::HashMap;

        // Create output database
        let mut out_conn = Connection::open(output_path)?;

        out_conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            
            -- Full node data (for debugging/future use)
            CREATE TABLE IF NOT EXISTS nodes (
                id   INTEGER PRIMARY KEY,
                data BLOB NOT NULL
            );
            
            -- Sphere-optimized patch lookup (7x smaller, no decode needed)
            -- Maps node_id -> patch_idx with prominence for EBM coherence
            CREATE TABLE IF NOT EXISTS sphere_patches (
                node_id    INTEGER PRIMARY KEY,
                patch_idx  INTEGER NOT NULL,
                prominence REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sphere_patch ON sphere_patches(patch_idx);
            
            -- Full edge data (for debugging/future use)
            CREATE TABLE IF NOT EXISTS hyperedges (
                id        INTEGER PRIMARY KEY,
                vertices  BLOB NOT NULL,
                data      BLOB NOT NULL
            );
            
            -- Sphere-optimized edges (already remapped to patch indices)
            -- Direct load without node_id->patch_idx mapping
            CREATE TABLE IF NOT EXISTS sphere_edges (
                id         INTEGER PRIMARY KEY,
                src_patch  INTEGER NOT NULL,
                dst_patch  INTEGER NOT NULL,
                weight     REAL NOT NULL,
                label      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sphere_edge_label ON sphere_edges(label);",
        )?;

        // Insert schema version (v3 = sphere-optimized tables)
        out_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '3')",
            [],
        )?;

        let tx = out_conn.transaction()?;

        // Get ordered list of documents with their patch counts
        let mut doc_patch_counts: Vec<(String, i64)> = Vec::new();
        {
            let mut stmt = self.conn.prepare(
                "SELECT doc_id, patch_count FROM documents WHERE status = 'complete' ORDER BY doc_id"
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let doc_id: String = row.get(0)?;
                let patch_count: i64 = row.get(1)?;
                doc_patch_counts.push((doc_id, patch_count));
            }
        }

        // Build global offset maps
        // For each document, track: (global_node_offset, global_patch_offset)
        let mut doc_offsets: HashMap<String, (usize, usize)> = HashMap::new();
        let mut global_node_offset = 0usize;
        let mut global_patch_offset = 0usize;

        for (doc_id, patch_count) in &doc_patch_counts {
            // Count nodes for this document
            let node_count: i64 = self.conn.query_row(
                "SELECT COUNT(*) FROM hypergraph_nodes WHERE doc_id = ?1",
                params![doc_id],
                |row| row.get(0),
            )?;

            doc_offsets.insert(doc_id.clone(), (global_node_offset, global_patch_offset));
            global_node_offset += node_count as usize;
            global_patch_offset += *patch_count as usize;
        }

        let mut total_nodes = 0usize;
        let mut total_edges = 0usize;
        let mut total_sphere_patches = 0usize;

        // Build node_id -> patch_idx mapping for edge remapping
        let mut node_to_patch: HashMap<usize, usize> = HashMap::new();

        // Process nodes: copy blobs directly (no decode needed)
        // Use patch_metadata table for sphere_patches (populated during ingest)
        {
            let mut read_stmt = self.conn.prepare(
                "SELECT doc_id, node_idx, data FROM hypergraph_nodes ORDER BY doc_id, node_idx",
            )?;
            let mut write_stmt = tx.prepare("INSERT INTO nodes (data) VALUES (?1)")?;

            let mut rows = read_stmt.query([])?;
            while let Some(row) = rows.next()? {
                let data: Vec<u8> = row.get(2)?;
                write_stmt.execute(params![data])?;
                total_nodes += 1;
            }
        }

        // Populate sphere_patches from patch_metadata table (avoids bincode decode)
        {
            let mut sphere_stmt = tx.prepare(
                "INSERT INTO sphere_patches (node_id, patch_idx, prominence) VALUES (?1, ?2, ?3)",
            )?;

            let mut read_stmt = self.conn.prepare(
                "SELECT pm.doc_id, pm.node_idx, pm.patch_index, pm.prominence
                 FROM patch_metadata pm
                 ORDER BY pm.doc_id, pm.node_idx",
            )?;

            // Track global node ID as we iterate
            let mut last_doc_id: Option<String> = None;
            let mut doc_node_start = 0usize;

            let mut rows = read_stmt.query([])?;
            while let Some(row) = rows.next()? {
                let doc_id: String = row.get(0)?;
                let node_idx: i64 = row.get(1)?;
                let local_patch_idx: i64 = row.get(2)?;
                let prominence: f64 = row.get(3)?;

                // Get offsets for this document
                let (node_offset, patch_offset) =
                    doc_offsets.get(&doc_id).copied().unwrap_or((0, 0));

                // Track document transitions
                if last_doc_id.as_ref() != Some(&doc_id) {
                    doc_node_start = node_offset;
                    last_doc_id = Some(doc_id.clone());
                }

                // Calculate global indices
                let global_node_id = doc_node_start + node_idx as usize;
                let global_patch_idx = patch_offset + local_patch_idx as usize;

                sphere_stmt.execute(params![
                    global_node_id as i64,
                    global_patch_idx as i64,
                    prominence
                ])?;
                total_sphere_patches += 1;

                // Track for edge remapping
                node_to_patch.insert(global_node_id, global_patch_idx);
            }
        }

        // Process edges document by document, remapping vertex IDs
        let mut total_sphere_edges = 0usize;
        {
            let mut read_stmt = self
                .conn
                .prepare("SELECT doc_id, data FROM hypergraph_edges ORDER BY doc_id")?;
            let mut write_stmt =
                tx.prepare("INSERT INTO hyperedges (vertices, data) VALUES (?1, ?2)")?;
            let mut sphere_edge_stmt = tx.prepare(
                "INSERT INTO sphere_edges (src_patch, dst_patch, weight, label) VALUES (?1, ?2, ?3, ?4)"
            )?;

            let mut rows = read_stmt.query([])?;
            while let Some(row) = rows.next()? {
                let doc_id: String = row.get(0)?;
                let data: Vec<u8> = row.get(1)?;

                let (node_offset, _) = doc_offsets.get(&doc_id).copied().unwrap_or((0, 0));

                // Decode edge, remap vertices, re-encode
                if let Ok((mut edge, _)) = bincode::serde::decode_from_slice::<HypergraphEdge, _>(
                    &data,
                    bincode::config::standard(),
                ) {
                    // Remap vertex IDs to global node IDs
                    for v in &mut edge.vertices {
                        *v += node_offset as u64;
                    }

                    // Re-encode vertices and full edge for legacy table
                    let vertices_blob =
                        bincode::serde::encode_to_vec(&edge.vertices, bincode::config::standard())?;
                    let data_blob =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
                    write_stmt.execute(params![vertices_blob, data_blob])?;
                    total_edges += 1;

                    // Insert into sphere_edges with patch indices (for sphere-optimized loading)
                    // Only process 2-vertex edges (pairwise connections)
                    if edge.vertices.len() == 2 {
                        let global_node_a = edge.vertices[0] as usize;
                        let global_node_b = edge.vertices[1] as usize;

                        // Map node IDs to patch indices (only Leaf nodes have patches)
                        if let (Some(&patch_a), Some(&patch_b)) = (
                            node_to_patch.get(&global_node_a),
                            node_to_patch.get(&global_node_b),
                        ) {
                            sphere_edge_stmt.execute(params![
                                patch_a as i64,
                                patch_b as i64,
                                edge.weight as f64,
                                &edge.label
                            ])?;
                            total_sphere_edges += 1;
                        }
                    }
                }
            }
        }

        tx.commit()?;

        println!(
            "✅ Exported hypergraph sidecar: {total_nodes} nodes, {total_edges} edges → {}",
            output_path.display()
        );
        println!(
            "   Sphere-optimized: {total_sphere_patches} patches, {total_sphere_edges} edges (schema v3)"
        );

        Ok(total_nodes)
    }

    /// Finalize the database (final checkpoint, optimize)
    pub fn finalize(&mut self) -> Result<CorpusStats> {
        // Final WAL checkpoint
        self.conn
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;

        // Optimize
        self.conn.execute_batch("PRAGMA optimize;")?;

        self.stats()
    }

    // =========================================================================
    // Pipeline Stage Methods (streaming batch read/write)
    // =========================================================================

    /// Mark a pipeline stage as started
    pub fn begin_stage(&mut self, stage_name: &str, config_json: Option<&str>) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO pipeline_stages (stage_name, status, started_at, config_json)
             VALUES (?1, 'running', datetime('now'), ?2)",
            params![stage_name, config_json],
        )?;
        Ok(())
    }

    /// Mark a pipeline stage as complete
    pub fn complete_stage(&mut self, stage_name: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE pipeline_stages SET status = 'complete', completed_at = datetime('now')
             WHERE stage_name = ?1",
            params![stage_name],
        )?;
        Ok(())
    }

    /// Check if a stage is complete (for resume)
    pub fn is_stage_complete(&self, stage_name: &str) -> Result<bool> {
        let status: Option<String> = self
            .conn
            .query_row(
                "SELECT status FROM pipeline_stages WHERE stage_name = ?1",
                params![stage_name],
                |row| row.get(0),
            )
            .ok();
        Ok(status.is_some_and(|s| s == "complete"))
    }

    /// Iterate embeddings in batches (for SphereStage, RootsStage)
    ///
    /// Returns embedding records for each patch in the batch.
    pub fn iter_embeddings_batched(
        &self,
        batch_size: usize,
        offset: usize,
    ) -> Result<Vec<EmbeddingRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, embedding, prominence, entropy 
             FROM embeddings 
             ORDER BY doc_id, patch_idx
             LIMIT ?1 OFFSET ?2",
        )?;

        let mut results = Vec::with_capacity(batch_size);
        let mut rows = stmt.query(params![batch_size as i64, offset as i64])?;

        while let Some(row) = rows.next()? {
            results.push(EmbeddingRecord {
                id: row.get(0)?,
                embedding: row.get(1)?,
                prominence: row.get::<_, Option<f64>>(2)?.unwrap_or(1.0) as f32,
                entropy: row.get::<_, Option<f64>>(3)?.unwrap_or(0.0) as f32,
            });
        }

        Ok(results)
    }

    /// Update spherical coordinates for a batch of patches (SphereStage)
    pub fn update_sphere_coords_batch(
        &mut self,
        updates: &[(i64, f32, f32, f32)], // (id, theta, phi, r)
    ) -> Result<usize> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "UPDATE embeddings SET theta = ?2, phi = ?3, r = ?4 WHERE id = ?1",
            )?;

            for (id, theta, phi, r) in updates {
                stmt.execute(params![id, theta, phi, r])?;
            }
        }
        tx.commit()?;
        Ok(updates.len())
    }

    /// Update partition assignments for a batch of patches (RootsStage)
    pub fn update_partition_batch(
        &mut self,
        updates: &[(i64, i64)], // (patch_id, partition_id)
    ) -> Result<usize> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt =
                tx.prepare_cached("UPDATE embeddings SET partition_id = ?2 WHERE id = ?1")?;

            for (id, partition_id) in updates {
                stmt.execute(params![id, partition_id])?;
            }
        }
        tx.commit()?;
        Ok(updates.len())
    }

    /// Insert partition centroids (RootsStage)
    pub fn insert_partitions(&mut self, centroids: &[Vec<f32>]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt =
                tx.prepare_cached("INSERT INTO partitions (centroid, n_members) VALUES (?1, 0)")?;

            for centroid in centroids {
                let bytes: Vec<u8> = centroid.iter().flat_map(|f| f.to_le_bytes()).collect();
                stmt.execute(params![bytes])?;
            }
        }
        tx.commit()?;
        Ok(centroids.len())
    }

    /// Update partition member counts (RootsStage)
    pub fn update_partition_counts(&mut self) -> Result<()> {
        self.conn.execute(
            "UPDATE partitions SET n_members = (
                SELECT COUNT(*) FROM embeddings WHERE embeddings.partition_id = partitions.id
            )",
            [],
        )?;
        Ok(())
    }

    /// Clear and save ROOTS tree structure (RootsStage)
    ///
    /// Serializes a hierarchical RootsNode tree to the database.
    /// Called after ROOTS construction or recalculation.
    pub fn save_roots_tree(&mut self, tree_nodes: &[RootsTreeNode], version: u64) -> Result<usize> {
        let tx = self.conn.transaction()?;

        // Clear existing tree
        tx.execute("DELETE FROM roots_tree", [])?;

        // Insert nodes
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO roots_tree (node_id, parent_id, is_leaf, partition_id, 
                 left_child, right_child, centroid, point_count, 
                 radius_min, radius_max, prom_min, prom_max)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            )?;

            for node in tree_nodes {
                let centroid_bytes: Vec<u8> =
                    node.centroid.iter().flat_map(|f| f.to_le_bytes()).collect();

                stmt.execute(params![
                    node.node_id as i64,
                    node.parent_id.map(|p| p as i64),
                    node.is_leaf,
                    node.partition_id.map(|p| p as i64),
                    node.left_child.map(|c| c as i64),
                    node.right_child.map(|c| c as i64),
                    centroid_bytes,
                    node.point_count as i64,
                    node.radius_range.0,
                    node.radius_range.1,
                    node.prom_range.0,
                    node.prom_range.1,
                ])?;
            }
        }

        // Update version
        tx.execute(
            "INSERT OR REPLACE INTO roots_meta (key, value) VALUES ('tree_version', ?1)",
            params![version.to_string()],
        )?;
        tx.execute(
            "INSERT OR REPLACE INTO roots_meta (key, value) VALUES ('tree_built_at', datetime('now'))",
            [],
        )?;

        tx.commit()?;
        Ok(tree_nodes.len())
    }

    /// Load ROOTS tree structure from database
    pub fn load_roots_tree(&self) -> Result<Vec<RootsTreeNode>> {
        let mut stmt = self.conn.prepare(
            "SELECT node_id, parent_id, is_leaf, partition_id, left_child, right_child,
                    centroid, point_count, radius_min, radius_max, prom_min, prom_max
             FROM roots_tree ORDER BY node_id",
        )?;

        let mut nodes = Vec::new();
        let mut rows = stmt.query([])?;

        while let Some(row) = rows.next()? {
            let node_id: i64 = row.get(0)?;
            let parent_id: Option<i64> = row.get(1)?;
            let is_leaf: bool = row.get(2)?;
            let partition_id: Option<i64> = row.get(3)?;
            let left_child: Option<i64> = row.get(4)?;
            let right_child: Option<i64> = row.get(5)?;
            let centroid_bytes: Vec<u8> = row.get(6)?;
            let point_count: i64 = row.get(7)?;
            let radius_min: f32 = row.get(8)?;
            let radius_max: f32 = row.get(9)?;
            let prom_min: f32 = row.get(10)?;
            let prom_max: f32 = row.get(11)?;

            let centroid: Vec<f32> = centroid_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            nodes.push(RootsTreeNode {
                node_id: node_id as usize,
                parent_id: parent_id.map(|p| p as usize),
                is_leaf,
                partition_id: partition_id.map(|p| p as usize),
                left_child: left_child.map(|c| c as usize),
                right_child: right_child.map(|c| c as usize),
                centroid,
                point_count: point_count as usize,
                radius_range: (radius_min, radius_max),
                prom_range: (prom_min, prom_max),
            });
        }

        Ok(nodes)
    }

    /// Get ROOTS tree version (for staleness detection)
    pub fn get_roots_tree_version(&self) -> Result<Option<u64>> {
        let result: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM roots_meta WHERE key = 'tree_version'",
                [],
                |row| row.get(0),
            )
            .ok();

        Ok(result.and_then(|s| s.parse().ok()))
    }

    /// Check if ROOTS tree exists and is not empty
    pub fn has_roots_tree(&self) -> Result<bool> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM roots_tree", [], |row| row.get(0))?;
        Ok(count > 0)
    }

    /// Invalidate ROOTS tree (marks it as stale, e.g., after sphere optimization)
    pub fn invalidate_roots_tree(&mut self) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO roots_meta (key, value) VALUES ('tree_stale', 'true')",
            [],
        )?;
        Ok(())
    }

    /// Check if ROOTS tree is stale
    pub fn is_roots_tree_stale(&self) -> Result<bool> {
        let result: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM roots_meta WHERE key = 'tree_stale'",
                [],
                |row| row.get(0),
            )
            .ok();

        Ok(result.is_some_and(|s| s == "true"))
    }

    /// Clear stale flag (after tree rebuild)
    pub fn clear_roots_tree_stale(&mut self) -> Result<()> {
        self.conn
            .execute("DELETE FROM roots_meta WHERE key = 'tree_stale'", [])?;
        Ok(())
    }

    /// Insert harmonics basis (HarmonicsStage)
    pub fn insert_harmonics_basis(
        &mut self,
        band_limit: usize,
        n_theta: usize,
        n_phi: usize,
        basis_values: &[f64],
    ) -> Result<()> {
        let bytes: Vec<u8> = basis_values.iter().flat_map(|f| f.to_le_bytes()).collect();

        self.conn.execute(
            "INSERT OR REPLACE INTO harmonics_basis (id, band_limit, n_theta, n_phi, basis_values)
             VALUES (1, ?1, ?2, ?3, ?4)",
            params![band_limit as i64, n_theta as i64, n_phi as i64, bytes],
        )?;
        Ok(())
    }

    /// Get harmonics basis (HarmonicsStage)
    pub fn get_harmonics_basis(&self) -> Result<Option<HarmonicsBasisRecord>> {
        let result: Option<(i64, i64, i64, Vec<u8>)> = self
            .conn
            .query_row(
                "SELECT band_limit, n_theta, n_phi, basis_values FROM harmonics_basis WHERE id = 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .ok();

        match result {
            Some((bl, nt, np, bytes)) => {
                let basis_values: Vec<f64> = bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect();
                Ok(Some(HarmonicsBasisRecord {
                    band_limit: bl as usize,
                    n_theta: nt as usize,
                    n_phi: np as usize,
                    basis_values,
                }))
            }
            None => Ok(None),
        }
    }

    /// Insert hypergraph edges (HypergraphStage) - uses sphere_edges table format
    pub fn insert_sphere_edges_batch(
        &mut self,
        edges: &[(i64, i64, f32, &str)], // (src_patch, dst_patch, weight, label)
    ) -> Result<usize> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO sphere_edges (src_patch, dst_patch, weight, label)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;

            for (src, dst, weight, label) in edges {
                stmt.execute(params![src, dst, weight, label])?;
            }
        }
        tx.commit()?;
        Ok(edges.len())
    }

    /// Create sphere_edges table if needed (for direct edge insertion without hypergraph)
    pub fn ensure_sphere_edges_table(&mut self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sphere_edges (
                id         INTEGER PRIMARY KEY,
                src_patch  INTEGER NOT NULL,
                dst_patch  INTEGER NOT NULL,
                weight     REAL NOT NULL,
                label      TEXT NOT NULL DEFAULT 'next'
            );
            CREATE INDEX IF NOT EXISTS idx_sphere_edge_label ON sphere_edges(label);",
        )?;
        Ok(())
    }

    /// Get document patch counts for hypergraph edge construction
    pub fn get_doc_patch_counts(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT doc_id, patch_count FROM documents 
             WHERE status = 'complete' 
             ORDER BY doc_id",
        )?;

        let mut results = Vec::new();
        let mut rows = stmt.query([])?;

        while let Some(row) = rows.next()? {
            let doc_id: String = row.get(0)?;
            let patch_count: i64 = row.get(1)?;
            results.push((doc_id, patch_count));
        }

        Ok(results)
    }

    /// Get embedding dimension from database metadata
    pub fn embedding_dim(&self) -> usize {
        self.embed_dim
    }
}

/// Corpus statistics
#[derive(Debug, Clone)]
pub struct CorpusStats {
    pub total_docs: usize,
    pub complete_docs: usize,
    pub failed_docs: usize,
    pub total_patches: usize,
    pub total_bytes: usize,
    pub elapsed_secs: f64,
}

impl CorpusStats {
    pub fn print_summary(&self) {
        let rate = self.complete_docs as f64 / self.elapsed_secs.max(0.001);
        let bytes_per_sec = self.total_bytes as f64 / self.elapsed_secs.max(0.001);

        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║                    CORPUS INGESTION COMPLETE                  ║");
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║  Documents:  {:>10} complete, {:>6} failed              ║",
            self.complete_docs, self.failed_docs
        );
        println!(
            "║  Patches:    {:>10}                                      ║",
            self.total_patches
        );
        println!(
            "║  Data:       {:>10.2} MB                                   ║",
            self.total_bytes as f64 / 1_000_000.0
        );
        println!(
            "║  Time:       {:>10.1} seconds                              ║",
            self.elapsed_secs
        );
        println!(
            "║  Throughput: {:>10.1} docs/s ({:.2} MB/s)                 ║",
            rate,
            bytes_per_sec / 1_000_000.0
        );
        println!("╚══════════════════════════════════════════════════════════════╝\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    #[ignore] // Run manually: cargo test export_ai_corpus -- --ignored --nocapture
    fn test_export_ai_corpus() {
        use std::path::Path;

        let db_path = Path::new("/Volumes/ai/sphere_results/corpus.db");
        if !db_path.exists() {
            println!("Skipping: {} not found", db_path.display());
            return;
        }

        let db = CorpusDB::open(db_path, 2048).unwrap();

        // Export SafeTensors
        let st_path = Path::new("/Volumes/ai/sphere_results/corpus.safetensors");
        let n = db.export_safetensors(st_path).unwrap();
        println!("✅ Exported {} patches to {}", n, st_path.display());

        // Export hypergraph sidecar
        let hg_path = Path::new("/Volumes/ai/sphere_results/corpus.hypergraph.db");
        db.export_hypergraph_sidecar(hg_path).unwrap();
        println!("✅ Exported hypergraph sidecar to {}", hg_path.display());
    }

    #[test]
    fn test_corpus_db_create_and_resume() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("test_corpus.db");

        // Create and insert some data
        {
            let mut db = CorpusDB::open(&db_path, 768).unwrap();

            db.begin_document("doc1", 1000).unwrap();
            let embeddings: Vec<f32> = (0..768 * 3).map(|i| i as f32 * 0.001).collect();
            let prominence = vec![0.5, 0.8, 0.3];
            let entropies = vec![1.0, 1.5, 2.0];
            db.insert_embeddings("doc1", &embeddings, &prominence, &entropies)
                .unwrap();
            db.complete_document("doc1", 3).unwrap();

            assert!(db.is_doc_complete("doc1").unwrap());
            assert!(!db.is_doc_complete("doc2").unwrap());
        }

        // Reopen and verify resume works
        {
            let db = CorpusDB::open(&db_path, 768).unwrap();
            assert!(db.is_doc_complete("doc1").unwrap());
            assert_eq!(db.completed_count().unwrap(), 1);
            assert_eq!(db.total_patches().unwrap(), 3);
        }
    }
}
