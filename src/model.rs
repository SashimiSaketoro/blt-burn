use burn::{
    config::Config,
    module::{Module, Param},
    nn::{self, Linear, LinearConfig},
    tensor::{
        activation::{silu, softmax},
        backend::Backend,
        Tensor,
    },
};
// RoPE implementation follows Meta's BLT with θ=500000

// ============================================================================
// RoPE (Rotary Position Embeddings) Implementation
// Following Meta's BLT implementation with θ=500000
// ============================================================================

/// Precompute the frequency tensor for RoPE (Rotary Position Embeddings).
///
/// This follows the standard RoPE formulation from Su et al. (2021):
/// freqs[i] = 1 / (θ^(2i/dim)) for i in 0..dim/2
///
/// Returns tensor of shape [max_seqlen, head_dim/2, 2] containing [cos, sin] pairs
pub fn precompute_freqs_cis<B: Backend>(
    head_dim: usize,
    max_seqlen: usize,
    theta: f64,
    device: &B::Device,
) -> Tensor<B, 3> {
    let half_dim = head_dim / 2;

    // Compute inverse frequencies: 1 / (θ^(2i/dim))
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64) as f32)
        .collect();

    // Create position indices [0, 1, 2, ..., max_seqlen-1]
    let positions: Vec<f32> = (0..max_seqlen).map(|i| i as f32).collect();

    // Compute outer product: positions × inv_freq -> [max_seqlen, half_dim]
    let mut freqs_data = Vec::with_capacity(max_seqlen * half_dim * 2);
    for pos in &positions {
        for freq in &inv_freq {
            let angle = pos * freq;
            freqs_data.push(angle.cos()); // cos component
            freqs_data.push(angle.sin()); // sin component
        }
    }

    // Return as [max_seqlen, half_dim, 2]
    Tensor::<B, 1>::from_floats(&freqs_data[..], device).reshape([max_seqlen, half_dim, 2])
}

/// Apply rotary position embeddings to query and key tensors.
///
/// Input shapes:
/// - xq: [batch, seq_len, n_heads, head_dim]
/// - xk: [batch, seq_len, n_kv_heads, head_dim]  
/// - freqs_cis: [max_seqlen, head_dim/2, 2]
///
/// The rotation is applied as:
/// x_rot = x * cos(θ) + rotate_half(x) * sin(θ)
///
/// where rotate_half swaps and negates adjacent pairs.
pub fn apply_rotary_emb<B: Backend>(
    xq: Tensor<B, 4>,
    xk: Tensor<B, 4>,
    freqs_cis: Tensor<B, 3>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch_size, seq_len, n_heads, head_dim] = xq.dims();
    let [_, _, n_kv_heads, _] = xk.dims();
    let half_dim = head_dim / 2;

    // Slice freqs_cis to match sequence length: [seq_len, half_dim, 2]
    let freqs = freqs_cis.slice([0..seq_len, 0..half_dim, 0..2]);

    // Extract cos and sin components
    // freqs[:, :, 0] = cos, freqs[:, :, 1] = sin
    let cos_freqs = freqs
        .clone()
        .slice([0..seq_len, 0..half_dim, 0..1])
        .reshape([1, seq_len, 1, half_dim]); // [1, seq_len, 1, half_dim]
    let sin_freqs = freqs
        .slice([0..seq_len, 0..half_dim, 1..2])
        .reshape([1, seq_len, 1, half_dim]); // [1, seq_len, 1, half_dim]

    // Apply rotation to queries
    let xq_rot = apply_rope_single(
        xq,
        cos_freqs.clone(),
        sin_freqs.clone(),
        batch_size,
        seq_len,
        n_heads,
        head_dim,
    );

    // Apply rotation to keys
    let xk_rot = apply_rope_single(
        xk, cos_freqs, sin_freqs, batch_size, seq_len, n_kv_heads, head_dim,
    );

    (xq_rot, xk_rot)
}

/// Apply RoPE to a single tensor (either Q or K)
fn apply_rope_single<B: Backend>(
    x: Tensor<B, 4>,
    cos_freqs: Tensor<B, 4>,
    sin_freqs: Tensor<B, 4>,
    batch_size: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Tensor<B, 4> {
    let half_dim = head_dim / 2;

    // Reshape x to [batch, seq, heads, 2, half_dim] to separate pairs
    let x_reshaped = x.reshape([batch_size, seq_len, n_heads, 2, half_dim]);

    // Extract even and odd indices (x1 and x2 in the rotation formula)
    let x1 = x_reshaped
        .clone()
        .slice([0..batch_size, 0..seq_len, 0..n_heads, 0..1, 0..half_dim])
        .reshape([batch_size, seq_len, n_heads, half_dim]);
    let x2 = x_reshaped
        .slice([0..batch_size, 0..seq_len, 0..n_heads, 1..2, 0..half_dim])
        .reshape([batch_size, seq_len, n_heads, half_dim]);

    // Broadcast cos/sin to [batch, seq, heads, half_dim]
    let cos_broad = cos_freqs.repeat_dim(0, batch_size).repeat_dim(2, n_heads);
    let sin_broad = sin_freqs.repeat_dim(0, batch_size).repeat_dim(2, n_heads);

    // Apply rotation:
    // x1_rot = x1 * cos - x2 * sin
    // x2_rot = x1 * sin + x2 * cos
    let x1_rot = x1.clone() * cos_broad.clone() - x2.clone() * sin_broad.clone();
    let x2_rot = x1 * sin_broad + x2 * cos_broad;

    // Interleave x1_rot and x2_rot back to original shape
    // Stack along new dim then reshape
    let x1_expanded = x1_rot.reshape([batch_size, seq_len, n_heads, 1, half_dim]);
    let x2_expanded = x2_rot.reshape([batch_size, seq_len, n_heads, 1, half_dim]);

    // Concatenate and reshape back to [batch, seq, heads, head_dim]
    Tensor::cat(vec![x1_expanded, x2_expanded], 3).reshape([batch_size, seq_len, n_heads, head_dim])
}

/// Create a causal attention mask for autoregressive attention.
///
/// Returns a mask where position i can only attend to positions 0..=i
/// Mask values: 0.0 for allowed positions, -inf for masked positions
pub fn create_causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    // Create lower triangular mask
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                // Future position - mask it out
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::<B, 1>::from_floats(&mask_data[..], device).reshape([seq_len, seq_len])
}

#[derive(Config, Debug)]
pub struct LMTransformerConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: Option<usize>,
    pub n_heads: Option<usize>,
    pub n_kv_heads: Option<usize>,
    pub ffn_dim_multiplier: Option<f64>,
    pub multiple_of: usize,
    pub norm_eps: f64,
    pub rope_theta: f64,
    pub max_seqlen: usize,
    pub vocab_size: usize,
}

impl LMTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LMTransformer<B> {
        let head_dim = self
            .head_dim
            .unwrap_or(self.dim / self.n_heads.unwrap_or(1));
        let n_heads = self.n_heads.unwrap_or(self.dim / head_dim);
        let n_kv_heads = self.n_kv_heads.unwrap_or(n_heads);

        let tok_embeddings = nn::EmbeddingConfig::new(self.vocab_size, self.dim).init(device);

        let layers = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(
                    self.dim,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    self.norm_eps,
                    self.rope_theta,
                    self.multiple_of,
                )
                .with_ffn_dim_multiplier(self.ffn_dim_multiplier)
                .init(device)
            })
            .collect();

        let norm = RmsNormConfig::new(self.dim)
            .with_epsilon(self.norm_eps)
            .init(device);

        let output = LinearConfig::new(self.dim, self.vocab_size)
            .with_bias(false)
            .init(device);

        LMTransformer {
            tok_embeddings,
            layers,
            norm,
            output,
            max_seqlen: self.max_seqlen,
            head_dim,
            rope_theta: self.rope_theta,
        }
    }
}

/// Output from the model including pre-normalization embeddings for water-filling
#[derive(Debug, Clone)]
pub struct ModelOutput<B: Backend> {
    /// Standard logits output [batch, seq, vocab]
    pub logits: Tensor<B, 3>,
    /// Pre-L2-norm embeddings BEFORE final norm [batch, seq, dim]
    /// This preserves the natural L2 magnitude distribution - the "density signal"
    pub pre_norm_embeddings: Tensor<B, 3>,
    /// L2 norms of pre-norm embeddings [batch, seq]
    /// Direct prominence/density signal for water-filling
    pub embedding_norms: Tensor<B, 2>,
    /// Entropy computed from logits [batch, seq]
    /// Used for entropy-weighted prominence allocation
    pub entropies: Option<Tensor<B, 2>>,
    /// Coherence scores: pre_norm^2 / entropy [batch, seq]
    /// Signal combining prominence and confidence
    pub coherence_scores: Option<Tensor<B, 2>>,
}

#[derive(Module, Debug)]
pub struct LMTransformer<B: Backend> {
    pub tok_embeddings: nn::Embedding<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub norm: RmsNorm<B>,
    pub output: Linear<B>,
    pub max_seqlen: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
}

/// Main implementation for LMTransformer.
/// When `fused-entropy` feature is enabled, requires FusedEntropyBackend for entropy calculation.
#[cfg(feature = "fused-entropy")]
impl<B: Backend + crate::fused_ops::FusedEntropyBackend> LMTransformer<B> {
    /// Forward pass with standard logits output
    pub fn forward(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        self.forward_with_embeddings(input).logits
    }

    /// Forward pass that also returns pre-normalization embeddings for water-filling
    /// This is the primary method for extracting density signals
    pub fn forward_with_embeddings(
        &self,
        input: Tensor<B, 2, burn::tensor::Int>,
    ) -> ModelOutput<B> {
        let [batch_size, seq_len] = input.dims();
        let device = input.device();
        let mut x = self.tok_embeddings.forward(input);

        // Precompute RoPE frequencies for this sequence
        let freqs_cis =
            precompute_freqs_cis::<B>(self.head_dim, self.max_seqlen, self.rope_theta, &device);

        // Create causal mask for attention
        let causal_mask = create_causal_mask::<B>(seq_len, &device);

        // Pass through transformer layers with RoPE frequencies and causal mask
        for layer in &self.layers {
            x = layer.forward_with_rope(x.clone(), freqs_cis.clone(), causal_mask.clone());
        }

        // Important: Capture embeddings BEFORE normalization
        // This is where the L2 magnitude signal resides
        let pre_norm_embeddings = x.clone();

        // Compute L2 norms for each position (useful for water-filling)
        // Shape: [batch, seq, dim] -> [batch, seq]
        // Optimization: Use element-wise multiplication instead of powf_scalar(2.0)
        let squared = pre_norm_embeddings.clone() * pre_norm_embeddings.clone();
        let embedding_norms = squared
            .sum_dim(2) // Sum over dim dimension
            .sqrt() // Take square root
            .reshape([batch_size, seq_len]); // [batch, seq, 1] -> [batch, seq]

        // Now apply normalization for standard output
        x = self.norm.forward(x);
        let logits = self.output.forward(x);

        // Compute entropy from logits for entropy-weighted allocation
        let entropies = crate::patcher::entropy(logits.clone());

        // Compute coherence: pre_norm^2 / entropy (with stability constant)
        // This combines prominence signal with inverse entropy as confidence weighting
        const ENTROPY_FLOOR: f64 = 1e-6; // Prevent explosion at low entropy
        let coherence_scores =
            embedding_norms.clone().powf_scalar(2.0) / (entropies.clone() + ENTROPY_FLOOR);

        ModelOutput {
            logits,
            pre_norm_embeddings,
            embedding_norms,
            entropies: Some(entropies),
            coherence_scores: Some(coherence_scores),
        }
    }

    /// Extract just the embeddings (no logits) - efficient for water-filling pipeline
    pub fn extract_embeddings(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        let [_batch_size, seq_len] = input.dims();
        let device = input.device();
        let mut x = self.tok_embeddings.forward(input);

        // Precompute RoPE frequencies
        let freqs_cis =
            precompute_freqs_cis::<B>(self.head_dim, self.max_seqlen, self.rope_theta, &device);

        // Create causal mask
        let causal_mask = create_causal_mask::<B>(seq_len, &device);

        for layer in &self.layers {
            x = layer.forward_with_rope(x.clone(), freqs_cis.clone(), causal_mask.clone());
        }

        // Return PRE-normalization for maximum signal
        x
    }
}

/// Main implementation for LMTransformer when fused-entropy feature is disabled.
/// Uses standard Backend trait without fused entropy optimization.
#[cfg(not(feature = "fused-entropy"))]
impl<B: Backend> LMTransformer<B> {
    /// Forward pass with standard logits output
    pub fn forward(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        self.forward_with_embeddings(input).logits
    }

    /// Forward pass that also returns pre-normalization embeddings for water-filling
    /// This is the primary method for extracting density signals
    pub fn forward_with_embeddings(
        &self,
        input: Tensor<B, 2, burn::tensor::Int>,
    ) -> ModelOutput<B> {
        let [batch_size, seq_len] = input.dims();
        let device = input.device();
        let mut x = self.tok_embeddings.forward(input);

        // Precompute RoPE frequencies for this sequence
        let freqs_cis =
            precompute_freqs_cis::<B>(self.head_dim, self.max_seqlen, self.rope_theta, &device);

        // Create causal mask for attention
        let causal_mask = create_causal_mask::<B>(seq_len, &device);

        // Pass through transformer layers with RoPE frequencies and causal mask
        for layer in &self.layers {
            x = layer.forward_with_rope(x.clone(), freqs_cis.clone(), causal_mask.clone());
        }

        // Important: Capture embeddings BEFORE normalization
        // This is where the L2 magnitude signal resides
        let pre_norm_embeddings = x.clone();

        // Compute L2 norms for each position (useful for water-filling)
        // Shape: [batch, seq, dim] -> [batch, seq]
        // Optimization: Use element-wise multiplication instead of powf_scalar(2.0)
        let squared = pre_norm_embeddings.clone() * pre_norm_embeddings.clone();
        let embedding_norms = squared
            .sum_dim(2) // Sum over dim dimension
            .sqrt() // Take square root
            .reshape([batch_size, seq_len]); // [batch, seq, 1] -> [batch, seq]

        // Now apply normalization for standard output
        x = self.norm.forward(x);
        let logits = self.output.forward(x);

        // Compute entropy from logits for entropy-weighted allocation
        let entropies = crate::patcher::entropy(logits.clone());

        // Compute coherence: pre_norm^2 / entropy (with stability constant)
        // This combines prominence signal with inverse entropy as confidence weighting
        const ENTROPY_FLOOR: f64 = 1e-6; // Prevent explosion at low entropy
        let coherence_scores =
            embedding_norms.clone().powf_scalar(2.0) / (entropies.clone() + ENTROPY_FLOOR);

        ModelOutput {
            logits,
            pre_norm_embeddings,
            embedding_norms,
            entropies: Some(entropies),
            coherence_scores: Some(coherence_scores),
        }
    }

    /// Extract just the embeddings (no logits) - efficient for water-filling pipeline
    pub fn extract_embeddings(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        let [_batch_size, seq_len] = input.dims();
        let device = input.device();
        let mut x = self.tok_embeddings.forward(input);

        // Precompute RoPE frequencies
        let freqs_cis =
            precompute_freqs_cis::<B>(self.head_dim, self.max_seqlen, self.rope_theta, &device);

        // Create causal mask
        let causal_mask = create_causal_mask::<B>(seq_len, &device);

        for layer in &self.layers {
            x = layer.forward_with_rope(x.clone(), freqs_cis.clone(), causal_mask.clone());
        }

        // Return PRE-normalization for maximum signal
        x
    }
}

#[derive(Config, Debug)]
pub struct TransformerBlockConfig {
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    norm_eps: f64,
    rope_theta: f64,
    multiple_of: usize,
    #[config(default = "None")]
    ffn_dim_multiplier: Option<f64>,
}

impl TransformerBlockConfig {
    // Config derive already creates new(), so just add this helper

    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerBlock<B> {
        let attention_norm = RmsNormConfig::new(self.dim)
            .with_epsilon(self.norm_eps)
            .init(device);

        let attention = AttentionConfig::new(
            self.dim,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.rope_theta,
        )
        .init(device);

        let ffn_norm = RmsNormConfig::new(self.dim)
            .with_epsilon(self.norm_eps)
            .init(device);

        let ffn_dim = self
            .ffn_dim_multiplier
            .map_or(self.dim * 4, |m| (m * self.dim as f64) as usize);

        // Round up to multiple_of
        let ffn_dim_rounded = self.multiple_of * ffn_dim.div_ceil(self.multiple_of);

        let feed_forward = FeedForwardConfig {
            dim: self.dim,
            hidden_dim: ffn_dim_rounded,
            multiple_of: self.multiple_of,
        }
        .init(device);

        TransformerBlock {
            attention_norm,
            attention,
            ffn_norm,
            feed_forward,
        }
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub attention_norm: RmsNorm<B>,
    pub attention: Attention<B>,
    pub ffn_norm: RmsNorm<B>,
    pub feed_forward: FeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    /// Legacy forward without RoPE (for backwards compatibility)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm architecture (without RoPE - legacy mode)
        let h = x.clone() + self.attention.forward(self.attention_norm.forward(x));
        h.clone() + self.feed_forward.forward(self.ffn_norm.forward(h))
    }

    /// Forward pass with RoPE and causal mask
    pub fn forward_with_rope(
        &self,
        x: Tensor<B, 3>,
        freqs_cis: Tensor<B, 3>,
        causal_mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        // Pre-norm architecture with RoPE
        let h = x.clone()
            + self.attention.forward_with_rope(
                self.attention_norm.forward(x),
                freqs_cis,
                causal_mask,
            );
        h.clone() + self.feed_forward.forward(self.ffn_norm.forward(h))
    }
}

#[derive(Config, Debug)]
pub struct AttentionConfig {
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        let wq = LinearConfig::new(self.dim, self.n_heads * self.head_dim)
            .with_bias(false)
            .init(device);

        let wk = LinearConfig::new(self.dim, self.n_kv_heads * self.head_dim)
            .with_bias(false)
            .init(device);

        let wv = LinearConfig::new(self.dim, self.n_kv_heads * self.head_dim)
            .with_bias(false)
            .init(device);

        let wo = LinearConfig::new(self.n_heads * self.head_dim, self.dim)
            .with_bias(false)
            .init(device);

        Attention {
            wq,
            wk,
            wv,
            wo,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub wq: Linear<B>,
    pub wk: Linear<B>,
    pub wv: Linear<B>,
    pub wo: Linear<B>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl<B: Backend> Attention<B> {
    /// Legacy forward without RoPE (for backwards compatibility)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        // Reshape for multi-head attention: [batch, heads, seq, head_dim]
        let q = q
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Repeat k,v heads if using grouped-query attention
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = Self::repeat_kv(k, n_rep);
        let v = Self::repeat_kv(v, n_rep);

        // Compute attention scores
        let scores = q
            .matmul(k.clone().swap_dims(2, 3))
            .div_scalar((self.head_dim as f64).sqrt());

        let attn = softmax(scores, 3);
        let output = attn.matmul(v);

        // Reshape back to [batch, seq, n_heads * head_dim]
        let output =
            output
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);

        self.wo.forward(output)
    }

    /// Forward pass with RoPE (Rotary Position Embeddings) and causal attention mask
    ///
    /// This is the correct implementation following Meta's BLT architecture.
    pub fn forward_with_rope(
        &self,
        x: Tensor<B, 3>,
        freqs_cis: Tensor<B, 3>,
        causal_mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        // Reshape for multi-head attention: [batch, seq, heads, head_dim]
        // (keeping seq before heads for RoPE application)
        let q = q.reshape([batch_size, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2); // [batch, heads, seq, head_dim]

        // Apply RoPE to queries and keys
        let (q, k) = apply_rotary_emb(q, k, freqs_cis);

        // Transpose to [batch, heads, seq, head_dim] for attention computation
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);

        // Repeat k,v heads if using grouped-query attention
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = Self::repeat_kv(k, n_rep);
        let v = Self::repeat_kv(v, n_rep);

        // Compute attention scores: [batch, heads, seq, seq]
        let scores = q
            .matmul(k.clone().swap_dims(2, 3))
            .div_scalar((self.head_dim as f64).sqrt());

        // Apply causal mask
        // Expand mask from [seq, seq] to [batch, heads, seq, seq]
        let mask = causal_mask.reshape([1, 1, seq_len, seq_len]);
        let scores = scores + mask;

        let attn = softmax(scores, 3);
        let output = attn.matmul(v);

        // Reshape back to [batch, seq, n_heads * head_dim]
        let output =
            output
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);

        self.wo.forward(output)
    }

    /// Repeat KV heads to match number of Q heads (for Grouped Query Attention)
    fn repeat_kv(x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        if n_rep == 1 {
            return x;
        }
        // x: [batch, n_kv_heads, seq, head_dim]
        // output: [batch, n_kv_heads * n_rep, seq, head_dim]
        let [batch_size, n_kv_heads, seq_len, head_dim] = x.dims();

        // Expand and reshape to repeat each head n_rep times
        // [batch, kv_heads, seq, dim] -> [batch, kv_heads, 1, seq, dim]
        let x = x.reshape([batch_size, n_kv_heads, 1, seq_len, head_dim]);

        // Repeat along the new dimension
        let x = x.repeat_dim(2, n_rep);

        // Reshape to merge kv_heads and rep dimensions
        x.reshape([batch_size, n_kv_heads * n_rep, seq_len, head_dim])
    }
}

#[derive(Config, Debug)]
pub struct FeedForwardConfig {
    dim: usize,
    hidden_dim: usize,
    multiple_of: usize,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        let w1 = LinearConfig::new(self.dim, self.hidden_dim)
            .with_bias(false)
            .init(device);

        let w2 = LinearConfig::new(self.hidden_dim, self.dim)
            .with_bias(false)
            .init(device);

        let w3 = LinearConfig::new(self.dim, self.hidden_dim)
            .with_bias(false)
            .init(device);

        FeedForward { w1, w2, w3 }
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub w1: Linear<B>,
    pub w2: Linear<B>,
    pub w3: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = silu(self.w1.forward(x.clone())) * self.w3.forward(x);
        self.w2.forward(hidden)
    }
}

// Custom RmsNorm to match PyTorch weight naming
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    dim: usize,
    #[config(default = "1e-6")]
    epsilon: f64,
}

impl RmsNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        let weight = Param::from_tensor(Tensor::ones([self.dim], device));
        RmsNorm {
            weight,
            epsilon: self.epsilon,
        }
    }
}

#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub epsilon: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Optimization: Use element-wise multiplication instead of powf_scalar(2.0)
        let squared = x.clone() * x.clone();
        let norm = squared.mean_dim(2).sqrt().add_scalar(self.epsilon);

        // norm is [batch, seq, 1]
        // weight is [dim] -> reshape to [1, 1, dim] for broadcasting
        let [_, _, dim] = x.dims();
        let weight = self.weight.val().reshape([1, 1, dim]);

        x.div(norm) * weight
    }
}
