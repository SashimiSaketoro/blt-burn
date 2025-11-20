use serde::{Serialize, Deserialize};
use hypergraph::{Hypergraph, VertexIndex, HyperedgeIndex};
use crate::pretokenize::ByteSegment;
use std::collections::HashMap;

/// DTO for serializing the Hypergraph structure and data
#[derive(Serialize, Deserialize, Debug)]
pub struct HypergraphSidecar {
    pub nodes: Vec<NodeData>,           // The Flesh (Vertices)
    pub edges: Vec<EdgeData>,           // The Flesh (Hyperedges)
    pub topology: TopologyDto,          // The Skeleton
}

/// Represents the topology of the hypergraph (adjacency list)
#[derive(Serialize, Deserialize, Debug)]
pub struct TopologyDto {
    // Adjacency list: EdgeIndex -> (EdgeWeightIndex, [VertexIndex, ...])
    // We map internal HyperedgeIndex back to our storage indices (usize)
    pub edges: Vec<(usize, Vec<usize>)>,
}

/// The data stored in each node (Vertex)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum NodeData {
    Trunk { source_hash: String, total_bytes: usize },
    Branch { label: String, modality: String },
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
        let node_indices: Vec<usize> = nodes.iter()
            .map(|v| *self.vertex_map.get(v).expect("Vertex must exist"))
            .collect();
            
        // Store topology mapping: Edge Data Index -> Connected Node Data Indices
        self.topology_cache.push((index, node_indices));

        let e = self.graph.add_hyperedge(nodes, index).expect("Failed to add hyperedge");
        self.edge_map.insert(e, index);
        e
    }

    /// Build the final serializable sidecar DTO
    pub fn build(self) -> HypergraphSidecar {
        HypergraphSidecar {
            nodes: self.nodes,
            edges: self.edges,
            topology: TopologyDto {
                edges: self.topology_cache,
            },
        }
    }
}

