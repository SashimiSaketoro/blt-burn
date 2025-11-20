use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        self,
        Linear, LinearConfig,
    },
    tensor::{activation::{softmax, silu}, backend::Backend, Tensor},
};

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
        let head_dim = self.head_dim.unwrap_or(self.dim / self.n_heads.unwrap_or(1));
        let n_heads = self.n_heads.unwrap_or(self.dim / head_dim);
        let n_kv_heads = self.n_kv_heads.unwrap_or(n_heads);

        let tok_embeddings = nn::EmbeddingConfig::new(self.vocab_size, self.dim)
            .init(device);

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
}

#[derive(Module, Debug)]
pub struct LMTransformer<B: Backend> {
    pub tok_embeddings: nn::Embedding<B>,
    pub layers: Vec<TransformerBlock<B>>,
    pub norm: RmsNorm<B>,
    pub output: Linear<B>,
    pub max_seqlen: usize,
}

impl<B: Backend> LMTransformer<B> {
    /// Forward pass with standard logits output
    pub fn forward(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        self.forward_with_embeddings(input).logits
    }

    /// Forward pass that also returns pre-normalization embeddings for water-filling
    /// This is the KEY method for extracting density signals
    pub fn forward_with_embeddings(&self, input: Tensor<B, 2, burn::tensor::Int>) -> ModelOutput<B> {
        let [batch_size, seq_len] = input.dims();
        let mut x = self.tok_embeddings.forward(input);

        // Pass through transformer layers
        for layer in &self.layers {
            x = layer.forward(x.clone());
        }

        // CRITICAL: Capture embeddings BEFORE normalization
        // This is where the L2 magnitude signal lives!
        let pre_norm_embeddings = x.clone();
        
        // Compute L2 norms for each position (useful for water-filling)
        // Shape: [batch, seq, dim] -> [batch, seq]
        // Optimization: Use element-wise multiplication instead of powf_scalar(2.0)
        let squared = pre_norm_embeddings.clone() * pre_norm_embeddings.clone();
        let embedding_norms = squared
            .sum_dim(2)        // Sum over dim dimension
            .sqrt()           // Take square root
            .reshape([batch_size, seq_len]);      // [batch, seq, 1] -> [batch, seq]
        
        // Now apply normalization for standard output
        x = self.norm.forward(x);
        let logits = self.output.forward(x);

        ModelOutput {
            logits,
            pre_norm_embeddings,
            embedding_norms,
        }
    }

    /// Extract just the embeddings (no logits) - efficient for water-filling pipeline
    pub fn extract_embeddings(&self, input: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        let mut x = self.tok_embeddings.forward(input);

        for layer in &self.layers {
            x = layer.forward(x.clone());
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
            .map(|m| (m * self.dim as f64) as usize)
            .unwrap_or(self.dim * 4);
        
        // Round up to multiple_of
        let ffn_dim_rounded = self.multiple_of * ((ffn_dim + self.multiple_of - 1) / self.multiple_of);

        let feed_forward = FeedForwardConfig {
            dim: self.dim,
            hidden_dim: ffn_dim_rounded,
            multiple_of: self.multiple_of,
        }.init(device);

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
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm architecture
        let h = x.clone() + self.attention.forward(self.attention_norm.forward(x));
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
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        // Reshape for multi-head attention
        let q = q.reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k.reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v.reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // TODO: Apply RoPE here
        // let q = apply_rotary_emb(q, freqs_cis);
        // let k = apply_rotary_emb(k, freqs_cis);

        // Repeat k,v heads if using grouped-query attention
        let n_rep = self.n_heads / self.n_kv_heads;
        let k = Self::repeat_kv(k, n_rep);
        let v = Self::repeat_kv(v, n_rep);

        // Compute attention scores
        let scores = q.matmul(k.clone().swap_dims(2, 3))
            .div_scalar((self.head_dim as f64).sqrt());

        // TODO: Apply Causal Mask here
        // let scores = scores.mask_fill(causal_mask, f32::NEG_INFINITY);

        let attn = softmax(scores, 3);
        let output = attn.matmul(v);

        // Reshape back to [batch, seq, n_heads * head_dim]
        let output = output
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.n_heads * self.head_dim]);

        self.wo.forward(output)
    }

    fn repeat_kv<const D: usize>(x: Tensor<B, D>, n_rep: usize) -> Tensor<B, D> {
        if n_rep == 1 {
            return x;
        }
        // For GQA: repeat KV heads to match number of Q heads
        // This is a simplified implementation - may need adjustment for actual shape manipulation
        x
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
        let norm = squared
            .mean_dim(2)
            .sqrt()
            .add_scalar(self.epsilon);

        // norm is [batch, seq, 1]
        // weight is [dim] -> reshape to [1, 1, dim] for broadcasting
        let [_, _, dim] = x.dims();
        let weight = self.weight.val().reshape([1, 1, dim]);

        x.div(norm) * weight
    }
}
