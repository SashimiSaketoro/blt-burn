use crate::modalities::ByteSegment;
use bincode;
use hypergraph::{HyperedgeIndex, Hypergraph, VertexIndex};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SidecarError {
    #[error("SQLite error: {0}")]
    Sql(#[from] rusqlite::Error),

    #[error("serialization error: {0}")]
    Bincode(#[from] bincode::error::EncodeError),

    #[error("deserialization error: {0}")]
    BincodeDecode(#[from] bincode::error::DecodeError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// DTO for serializing the Hypergraph structure and data
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HypergraphSidecar {
    pub nodes: Vec<NodeData>,                 // The Flesh (Vertices)
    pub edges: Vec<EdgeData>,                 // The Flesh (Hyperedges)
    pub topology: TopologyDto,                // The Skeleton
    pub sharding: Option<ShardingInfo>,       // JAX sharding info (optional)
    pub replication: Option<ReplicationSpec>, // Replication info (optional)
}

/// For SQLite compatibility, we need a simpler edge structure
#[derive(Debug, Serialize, Deserialize)]
pub struct HypergraphEdge {
    pub vertices: Vec<u64>, // or whatever index type you use
    pub label: String,
    pub weight: f32,
}

/// JAX sharding information for distributed loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingInfo {
    pub global_shape: Vec<usize>,     // Full dataset shape
    pub shard_index: usize,           // Which shard this is
    pub num_shards: usize,            // Total number of shards
    pub process_index: Option<usize>, // Target process (optional)
    pub axis: usize,                  // Which dimension is sharded (usually 0 for batch)
}

/// Replication specification for model parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationSpec {
    pub replica_groups: Vec<Vec<usize>>, // Which devices share data
    pub replication_factor: usize,
}

/// Represents the topology of the hypergraph (adjacency list)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TopologyDto {
    // Adjacency list: EdgeIndex -> (EdgeWeightIndex, [VertexIndex, ...])
    // We map internal HyperedgeIndex back to our storage indices (usize)
    pub edges: Vec<(usize, Vec<usize>)>,
}

/// The data stored in each node (Vertex)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum NodeData {
    Trunk {
        source_hash: String,
        total_bytes: usize,
    },
    Branch {
        label: String,
        modality: String,
    },
    Leaf(ByteSegment), // Re-use existing ByteSegment
}

/// The data stored in each edge (Hyperedge)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EdgeData {
    pub label: String, // "contains", "next"
    pub weight: f32,   // logical weight
}

/// Builder to manage the mapping during ingest
pub struct HypergraphBuilder {
    // Internal graph structure: Vertices are indices into `nodes`, Edges are indices into `edges`
    graph: Hypergraph<usize, usize>,
    nodes: Vec<NodeData>,
    edges: Vec<EdgeData>,
    // Map back from internal graph indices to our DTO indices
    vertex_map: HashMap<VertexIndex, usize>,
    edge_map: HashMap<HyperedgeIndex, usize>,
    // Cache topology to avoid querying private graph API
    topology_cache: Vec<(usize, Vec<usize>)>,
}

impl HypergraphSidecar {
    pub fn save_to_sqlite(&self, path: &Path) -> Result<(), SidecarError> {
        // Make sure parent dirs exist
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)?;
            }
        }

        let mut conn = Connection::open(path)?;
        let tx = conn.transaction()?;

        // Simple meta table for versioning
        tx.execute(
            "CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
            [],
        )?;

        tx.execute(
            "CREATE TABLE IF NOT EXISTS nodes (
                id   INTEGER PRIMARY KEY,
                data BLOB NOT NULL
            )",
            [],
        )?;

        tx.execute(
            "CREATE TABLE IF NOT EXISTS hyperedges (
                id        INTEGER PRIMARY KEY,
                vertices  BLOB NOT NULL,
                data      BLOB NOT NULL
            )",
            [],
        )?;

        // Clear existing contents so repeated saves don't append forever.
        tx.execute("DELETE FROM nodes", [])?;
        tx.execute("DELETE FROM hyperedges", [])?;

        // Insert meta info
        tx.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', '2')",
            [],
        )?;

        // Insert sharding info if present
        if let Some(ref sharding) = self.sharding {
            let sharding_json = serde_json::to_string(sharding)?;
            tx.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('sharding_info', ?1)",
                params![sharding_json],
            )?;
        }

        // Insert replication info if present
        if let Some(ref replication) = self.replication {
            let replication_json = serde_json::to_string(replication)?;
            tx.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('replication_spec', ?1)",
                params![replication_json],
            )?;
        }

        // Insert nodes
        {
            let mut stmt = tx.prepare("INSERT INTO nodes (data) VALUES (?1)")?;
            for node in &self.nodes {
                let blob = bincode::serde::encode_to_vec(node, bincode::config::standard())?;
                stmt.execute(params![blob])?;
            }
        }

        // Insert edges - convert EdgeData + topology to HypergraphEdge format
        {
            let mut stmt = tx.prepare("INSERT INTO hyperedges (vertices, data) VALUES (?1, ?2)")?;
            for (_edge_idx, (edge_data_idx, vertex_indices)) in
                self.topology.edges.iter().enumerate()
            {
                if *edge_data_idx < self.edges.len() {
                    let edge_data = &self.edges[*edge_data_idx];
                    let edge = HypergraphEdge {
                        vertices: vertex_indices.iter().map(|&v| v as u64).collect(),
                        label: edge_data.label.clone(),
                        weight: edge_data.weight,
                    };
                    let vertices_blob =
                        bincode::serde::encode_to_vec(&edge.vertices, bincode::config::standard())?;
                    let data_blob =
                        bincode::serde::encode_to_vec(&edge, bincode::config::standard())?;
                    stmt.execute(params![vertices_blob, data_blob])?;
                }
            }
        }

        tx.commit()?;
        Ok(())
    }

    pub fn load_from_sqlite(path: &Path) -> Result<Self, SidecarError> {
        let conn = Connection::open(path)?;

        // Load metadata
        let mut sharding = None;
        let mut replication = None;

        if let Ok(sharding_json) = conn.query_row(
            "SELECT value FROM meta WHERE key='sharding_info'",
            [],
            |row| row.get::<_, String>(0),
        ) {
            sharding = Some(serde_json::from_str(&sharding_json)?);
        }

        if let Ok(replication_json) = conn.query_row(
            "SELECT value FROM meta WHERE key='replication_spec'",
            [],
            |row| row.get::<_, String>(0),
        ) {
            replication = Some(serde_json::from_str(&replication_json)?);
        }

        let mut nodes = Vec::new();
        {
            let mut stmt = conn.prepare("SELECT data FROM nodes ORDER BY id ASC")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let blob: Vec<u8> = row.get(0)?;
                let (node, _): (NodeData, _) =
                    bincode::serde::decode_from_slice(&blob, bincode::config::standard())?;
                nodes.push(node);
            }
        }

        let mut edges = Vec::new();
        let mut topology_edges = Vec::new();
        {
            let mut stmt = conn.prepare("SELECT data FROM hyperedges ORDER BY id ASC")?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let blob: Vec<u8> = row.get(0)?;
                let (edge, _): (HypergraphEdge, _) =
                    bincode::serde::decode_from_slice(&blob, bincode::config::standard())?;

                // Convert back to EdgeData and topology
                let edge_data = EdgeData {
                    label: edge.label,
                    weight: edge.weight,
                };
                let edge_idx = edges.len();
                edges.push(edge_data);

                let vertex_indices: Vec<usize> =
                    edge.vertices.iter().map(|&v| v as usize).collect();
                topology_edges.push((edge_idx, vertex_indices));
            }
        }

        Ok(HypergraphSidecar {
            nodes,
            edges,
            topology: TopologyDto {
                edges: topology_edges,
            },
            sharding,
            replication,
        })
    }

    /// Export to JSON for debugging/human inspection
    pub fn save_to_json(&self, path: &Path) -> Result<(), std::io::Error> {
        let file = fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }
}

impl HypergraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: Hypergraph::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            vertex_map: HashMap::new(),
            edge_map: HashMap::new(),
            topology_cache: Vec::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, data: NodeData) -> VertexIndex {
        let index = self.nodes.len();
        self.nodes.push(data);
        let v = self.graph.add_vertex(index).expect("Failed to add vertex");
        self.vertex_map.insert(v, index);
        v
    }

    /// Add a hyperedge connecting multiple nodes
    pub fn add_hyperedge(&mut self, nodes: Vec<VertexIndex>, data: EdgeData) -> HyperedgeIndex {
        let index = self.edges.len();
        self.edges.push(data);

        // Convert VertexIndex to internal usize index for caching
        let node_indices: Vec<usize> = nodes
            .iter()
            .map(|v| *self.vertex_map.get(v).expect("Vertex must exist"))
            .collect();

        // Store topology mapping: Edge Data Index -> Connected Node Data Indices
        self.topology_cache.push((index, node_indices));

        let e = self
            .graph
            .add_hyperedge(nodes, index)
            .expect("Failed to add hyperedge");
        self.edge_map.insert(e, index);
        e
    }

    /// Add cross-view hyperedges connecting nodes that share the same source_id.
    ///
    /// This is used for multi-view learning, where different representations
    /// of the same content (raw, text, image) should be linked in the graph
    /// and can be encouraged to stay close in the embedding space.
    ///
    /// Creates `same_source` hyperedges between all leaf nodes sharing a source_id.
    pub fn add_cross_view_edges(&mut self) {
        // Collect leaf nodes by their source_id
        let mut source_groups: HashMap<String, Vec<VertexIndex>> = HashMap::new();

        for (vertex_idx, node_idx) in &self.vertex_map {
            if let Some(NodeData::Leaf(segment)) = self.nodes.get(*node_idx) {
                if let Some(ref metadata) = segment.metadata {
                    if let Some(ref extra) = metadata.extra {
                        if let Some(source_id) = extra.get("source_id").and_then(|v| v.as_str()) {
                            source_groups
                                .entry(source_id.to_string())
                                .or_insert_with(Vec::new)
                                .push(*vertex_idx);
                        }
                    }
                }
            }
        }

        // Create hyperedges for groups with multiple views
        for (_source_id, vertices) in source_groups {
            if vertices.len() > 1 {
                // Create a single hyperedge connecting all views of this source
                self.add_hyperedge(
                    vertices,
                    EdgeData {
                        label: "same_source".to_string(),
                        weight: 1.0, // Full identity weight for cross-view association
                    },
                );
            }
        }
    }

    /// Get all nodes that share a source_id (for debugging/inspection).
    pub fn get_cross_view_groups(&self) -> HashMap<String, Vec<usize>> {
        let mut source_groups: HashMap<String, Vec<usize>> = HashMap::new();

        for (_, node_idx) in &self.vertex_map {
            if let Some(NodeData::Leaf(segment)) = self.nodes.get(*node_idx) {
                if let Some(ref metadata) = segment.metadata {
                    if let Some(ref extra) = metadata.extra {
                        if let Some(source_id) = extra.get("source_id").and_then(|v| v.as_str()) {
                            source_groups
                                .entry(source_id.to_string())
                                .or_insert_with(Vec::new)
                                .push(*node_idx);
                        }
                    }
                }
            }
        }

        source_groups
    }

    /// Build the final serializable sidecar DTO
    pub fn build(self) -> HypergraphSidecar {
        HypergraphSidecar {
            nodes: self.nodes,
            edges: self.edges,
            topology: TopologyDto {
                edges: self.topology_cache,
            },
            sharding: None,
            replication: None,
        }
    }

    /// Build with cross-view edges enabled (for multiview mode).
    ///
    /// This automatically adds `same_source` hyperedges connecting
    /// nodes that represent different views of the same content.
    pub fn build_with_cross_view_edges(mut self) -> HypergraphSidecar {
        self.add_cross_view_edges();
        self.build()
    }

    /// Build with sharding information
    pub fn build_with_sharding(self, sharding: ShardingInfo) -> HypergraphSidecar {
        let mut sidecar = self.build();
        sidecar.sharding = Some(sharding);
        sidecar
    }
}
