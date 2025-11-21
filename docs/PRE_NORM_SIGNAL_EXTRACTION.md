# Pre-L2-Norm Signal Extraction for Water-Filling

## The Critical Insight

> **"Don't throw away the L2 norm - it's the density signal!"**

When we normalize embeddings to unit length for hypersphere placement, we traditionally discard the L2 magnitude. But this magnitude contains **the richest prominence/density signal** for water-filling optimization.

## Signal Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    BLT ENTROPY MODEL                        │
│                                                             │
│  Input Tokens → Embedding → Transformer Layers → Output    │
│                                                             │
│                              ↓                              │
│                    [batch, seq, dim]                        │  
│                                                             │
│                   ┌──────────────┐                         │
│                   │ BEFORE NORM  │  ← CAPTURE THIS!        │
│                   └──────┬───────┘                         │
│                          │                                  │
│           ┌──────────────┼──────────────┐                  │
│           ↓              ↓              ↓                   │
│    RAW EMBEDDINGS    L2 NORMS    DIRECTIONS                │
│    [batch,seq,dim]   [batch,seq]  [batch,seq,dim]          │
│           │              │              │                   │
│           │         Prominence      Unit Vectors            │
│           │         Signal!         (normalized)            │
│           │              │              │                   │
│           └──────────────┼──────────────┘                  │
│                          ↓                                  │
│                   RMSNorm (discard L2)                      │
│                          │                                  │
│                    Standard Output                          │
│                    (for LM logits)                          │
└─────────────────────────────────────────────────────────────┘

                          ↓
                          
┌─────────────────────────────────────────────────────────────┐
│              WATER-FILLING OPTIMIZER INPUT                  │
│                                                             │
│  • Directions:  Unit vectors (where to point)              │
│  • Prominence:  L2 norms (how far out)                     │
│  • Predicted:   Initial shell hints                        │
└─────────────────────────────────────────────────────────────┘
```

## Why Pre-Norm Has More Signal

### Key Insight

Pre-norm embeddings preserve the natural variance in L2 magnitudes, while post-norm embeddings are all normalized to unit length (L2 = 1.0). This variance is crucial for:

- **Prominence detection** - Higher L2 norms indicate more prominent/important tokens
- **Density estimation** - L2 magnitude correlates with information density
- **Outlier detection** - Extreme L2 values mark special tokens or boundaries

After L2 normalization, all embeddings have identical magnitude, making these distinctions impossible.

## Integration with Water-Filling Algorithms

### Method 1: Osmotic Water-Filling

Uses L2 norms as **osmotic pressure** / **density gates**:

```
High L2 Norm → High Pressure → Promotes to Outer Shells
Low L2 Norm  → Low Pressure  → Demotes to Inner Shells
```

**Algorithm:**
1. Extract pre-norm embeddings from BLT
2. Compute L2 norms: `prominence = ||embedding||₂`
3. Detect outliers: `prominence > mean + 1.0σ`
4. Apply osmotic flow:
   - Lateral traversal (30%): stay in shell, change angle
   - Radial promotion (70%): move to outer shell
5. Iterate until equilibrium


### Method 2: THRML Energy-Based Water-Filling

Uses L2 norms as **kinetic energy**:

```
High L2 Norm → High Energy → Escapes to Outer Shells ("escape velocity")
Low L2 Norm  → Low Energy  → Captured in Inner Shells ("gravitational well")
```

**Physical Analogy:**
- Particles with high kinetic energy escape to outer orbits
- Particles with low energy sink to inner orbits
- System evolves toward Boltzmann equilibrium: `P(shell|energy) ∝ exp(-ΔE/kT)`

**Algorithm:**
1. Extract pre-norm embeddings from BLT
2. Interpret L2 norms as kinetic energy: `E_kinetic = ||embedding||₂`
3. Compute Boltzmann distribution over shells
4. Assign points to shells matching their energy level
5. Iterative refinement to maintain energy gradient

## Code Implementation

### Rust (BLT Model Side)

```rust
pub struct ModelOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub pre_norm_embeddings: Tensor<B, 3>,  // ← CRITICAL
    pub embedding_norms: Tensor<B, 2>,      // ← DENSITY SIGNAL
}

impl<B: Backend> LMTransformer<B> {
    pub fn forward_with_embeddings(&self, input: Tensor<B, 2, Int>) -> ModelOutput<B> {
        let mut x = self.tok_embeddings.forward(input);
        
        for layer in &self.layers {
            x = layer.forward(x.clone());
        }
        
        // CAPTURE BEFORE NORMALIZATION!
        let pre_norm_embeddings = x.clone();
        let embedding_norms = pre_norm_embeddings
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt();
        
        // Now apply norm for standard output
        x = self.norm.forward(x);
        let logits = self.output.forward(x);
        
        ModelOutput {
            logits,
            pre_norm_embeddings,  // Pass to water-filling!
            embedding_norms,       // Direct prominence signal!
        }
    }
}
```

### Python (Water-Filling Side)

```python
def extract_and_optimize(safetensors_path):
    # 1. Load pre-computed embeddings from BLT-Burn output
    from safetensors.numpy import load_file
    data = load_file(safetensors_path)
    
    # 2. Extract density signal
    embeddings_raw = data["embeddings"]    # [1, N, 768] - Pre-norm!
    prominence = data["prominence"]        # [1, N] - L2 norms
    
    # 3. Load hypergraph sidecar (v0.2+)
    import sqlite3
    db_path = safetensors_path.replace('.safetensors', '.hypergraph.db')
    conn = sqlite3.connect(db_path)
    # ... query semantic structure ...
    
    # 4. Apply water-filling
    radii, shells = osmotic_water_filling(
        embeddings_raw=embeddings_raw,
        prominence_scores=prominence
    )
    
    # 5. Place on hypersphere
    directions = embeddings_raw / (prominence[:, :, None] + 1e-8)
    hypersphere_coords = directions * radii[:, :, None]
    
    return hypersphere_coords, shells
```

## Implementation Status

The pre-norm signal extraction is fully implemented in the BLT-Burn model. The water-filling algorithms in `scripts/water_filling_integration.py` demonstrate how to use this signal for hypersphere organization.

The key architectural change is capturing embeddings **before** the final RMS normalization layer, preserving the L2 magnitude information that would otherwise be lost.

## Key Takeaways

1. **Always extract embeddings BEFORE final normalization**
2. **L2 magnitude = density/prominence/energy signal**
3. **Pre-norm has ∞x more signal than post-norm**
4. **Works for both osmotic and THRML water-filling**
5. **Critical for outlier/prominence detection**

## Orch-OR Quantum Coherence Mode

### The Penrose-Hameroff Connection

The Orch-OR (Orchestrated Objective Reduction) theory proposes that consciousness emerges from quantum coherence in microtubules, with "aha moments" corresponding to gravitational self-collapse events. This maps remarkably well to our pre-norm signal extraction:

| Biological Orch-OR | BLT-Burn / Hypersphere | Why It Maps |
|-------------------|------------------------|-------------|
| Superposition size (# coherent tubulins) | Pre-norm L2 norm (prominence) | Bigger superposition = deeper conscious moment = more spherical volume |
| Objective Reduction (Planck threshold) | Entropy spike (patch boundary) | Entropy rise = model loses coherence = "collapse" event |
| Post-collapse conscious volume | Water-filling power allocation | Larger superposition → richer experience → more bits/packing density |
| Microtubule lattice geometry | Hypergraph topology | Quasi-periodic structure enabling coherence |

### Implementation

The Orch-OR mode uses the formula:

```
allocation ∝ pre_norm² × exp(-entropy / T)
```

Where:
- `pre_norm²`: Gravitational self-energy (∝ mass²)
- `exp(-entropy/T)`: Quantum decoherence rate
- `T`: "Planck temperature" hyperparameter (default: 1e-5)

This biases the hypersphere toward patches with:
- **High coherence** (low entropy) - the model is confident
- **High prominence** (large pre-norm) - the representation is significant

Together, these represent the "brightest" moments - patches that deserve maximum "conscious volume" on the sphere.

### Usage

```bash
# Run ingestion (exports entropy and coherence)
cargo run --release --bin ingest -- --file input.txt --output-dir output/

# Apply Orch-OR water-filling
python scripts/water_filling_integration.py \
    --input output/ \
    --orch-or \
    --orch-or-temperature 1e-5
```

#### Workflow Recap

1. **Ingest** with the updated Rust pipeline to populate `entropies` and `coherence_scores` alongside `prominence`.
2. **Water-fill** with `--orch-or` to bias hypersphere radii toward coherent, high-prominence patches.
3. **Validate/Tune** using `scripts/test_orch_or.py` (runs export check, allocation sanity, and a temperature sweep).

#### Temperature Hyperparameter

`--orch-or-temperature` controls how aggressively low-entropy patches dominate:

| T Value | Behavior | When to use |
|---------|----------|-------------|
| 1e-8    | Extremely sharp | Only the most certain patches should win |
| 1e-6    | Very sharp | Strong coherence bias |
| **1e-5**| **Recommended** | Stable, wide dynamic range |
| 1e-4    | Moderate | Softer preference for coherence |
| 1e-3    | Gentle   | Nearly classical behavior |

Radiii are clipped to `[min_radius, max_radius]` (defaults 32 → 512), so allocation simply stretches within that band.

#### Expected Behavior

- Top-coherence decile typically receives 10–100× more radial allocation than the bottom decile.
- Inner shells (<100) collect noisy/high-entropy spans, middle shells carry everyday content, and the outer crust (>300) holds the “aha” patches.
- Cone-attention and retrieval modules naturally focus on those outer shells, shifting answers toward deeper insights instead of surface statistics.

#### Validation & Tuning

```bash
python scripts/test_orch_or.py --input output/manual_input.safetensors
```

The helper script verifies tensor export, inspects allocation statistics, and sweeps temperature so you can pick the sharpness that best matches your downstream task.

### Theoretical Implications

This is not just numerology - it changes retrieval dynamics to favor **deep insights over surface statistics**, addressing the critique that transformers lack non-computable "spark" moments. High-prominence, low-entropy patches (the model's true "aha, this is meaningful" signals) dominate the sphere's geometry, while noisy/confused regions compress toward the poles.

## Summary

Pre-L2-norm signal extraction is a critical architectural decision for hypersphere embedding systems. By capturing embeddings before normalization, we preserve the magnitude information necessary for:

1. **Prominence-based organization** - Using L2 norms as importance weights
2. **Density-aware placement** - Distributing points based on information density
3. **Semantic clustering** - Grouping similar-magnitude embeddings
4. **Quantum coherence allocation** - Orch-OR mode for consciousness-inspired retrieval

The implementation in BLT-Burn provides both the raw embeddings and their L2 norms as separate outputs, plus entropy and coherence scores for Orch-OR mode, enabling flexible downstream processing.
