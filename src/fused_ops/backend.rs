//! Backend implementations for fused operations.
//!
//! This module provides `FusedOpsBackend` implementations for:
//! - `CubeBackend<R, F, I, BT>` - Direct CubeCL kernel execution
//! - `Fusion<B>` (e.g., Wgpu) - Fallback to reference implementation

use super::{
    attention_kernel::{fused_attention_kernel, fused_cross_attention_kernel},
    coherence_kernel::fused_coherence_kernel,
    hash_embedding_kernel::rolling_hash_float_kernel,
    kernel::{fused_entropy_kernel, fused_entropy_kernel_optimized},
    l2_norm_kernel::{fused_l2_norm_kernel, fused_l2_norm_kernel_optimized},
    rms_norm_kernel::{fused_rms_norm_kernel, fused_rms_norm_kernel_optimized},
    silu_gate_kernel::fused_silu_gate_kernel,
    softmax_kernel::fused_softmax_kernel,
    FusedOpsBackend,
};
use burn::tensor::{
    activation::{log_softmax, silu, softmax},
    ops::FloatTensor,
    Shape, Tensor, TensorPrimitive,
};
use burn_cubecl::{
    element::BoolElement, kernel::into_contiguous, tensor::CubeTensor, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use burn_fusion::{Fusion, FusionBackend};
use cubecl::{prelude::ScalarArg, CubeCount, CubeDim};

// ============================================================================
// CubeBackend Implementation - Direct kernel execution
// ============================================================================

impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FusedOpsBackend
    for CubeBackend<R, F, I, BT>
{
    fn fused_entropy(logits: FloatTensor<Self>, vocab_size: usize) -> FloatTensor<Self> {
        let logits = into_contiguous(logits);
        let [batch_size, seq_len, _vocab] = logits.shape.dims();
        let total_rows = batch_size * seq_len;

        let shape_out = Shape::from([batch_size, seq_len]);
        let buffer = logits
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        let output = CubeTensor::new_contiguous(
            logits.client.clone(),
            logits.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        // Use optimized kernel with plane-based parallelism for larger vocab sizes
        // Each plane (warp) of 32 threads handles one row cooperatively
        if vocab_size >= 64 {
            // Optimized: Each plane (x=32 threads) handles one row
            // Multiple rows per cube (y dimension)
            let plane_size = 32u32; // Typical warp size
            let rows_per_cube = 8u32; // Process 8 rows per cube
            let cube_dim = CubeDim {
                x: plane_size,
                y: rows_per_cube,
                z: 1,
            };
            let cubes_needed = (total_rows as f32 / rows_per_cube as f32).ceil() as u32;
            let cube_count = CubeCount::Static(cubes_needed, 1, 1);

            fused_entropy_kernel_optimized::launch::<F, R>(
                &logits.client,
                cube_count,
                cube_dim,
                logits.as_tensor_arg::<F>(1),
                output.as_tensor_arg::<F>(1),
                ScalarArg::new(vocab_size as u32),
            );
        } else {
            // Simple kernel for small vocab sizes
            let cube_dim = CubeDim { x: 16, y: 16, z: 1 };
            let cubes_x = (batch_size as f32 / cube_dim.x as f32).ceil() as u32;
            let cubes_y = (seq_len as f32 / cube_dim.y as f32).ceil() as u32;
            let cube_count = CubeCount::Static(cubes_x, cubes_y, 1);

            fused_entropy_kernel::launch::<F, R>(
                &logits.client,
                cube_count,
                cube_dim,
                logits.as_tensor_arg::<F>(1),
                output.as_tensor_arg::<F>(1),
                ScalarArg::new(vocab_size as u32),
            );
        }

        output
    }

    fn fused_rms_norm(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        epsilon: f32,
        dim: usize,
    ) -> FloatTensor<Self> {
        let input = into_contiguous(input);
        let weight = into_contiguous(weight);
        let [batch_size, seq_len, _dim] = input.shape.dims();
        let total_rows = batch_size * seq_len;

        let shape_out = input.shape.clone();
        let buffer = input
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        let output = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        // Note: epsilon is hardcoded in the kernel (1e-6)
        let _ = epsilon;

        // Use optimized kernel with plane-based parallelism for larger dims
        if dim >= 64 {
            let plane_size = 32u32;
            let rows_per_cube = 8u32;
            let cube_dim = CubeDim {
                x: plane_size,
                y: rows_per_cube,
                z: 1,
            };
            let cubes_needed = (total_rows as f32 / rows_per_cube as f32).ceil() as u32;
            let cube_count = CubeCount::Static(cubes_needed, 1, 1);

            fused_rms_norm_kernel_optimized::launch::<F, R>(
                &input.client,
                cube_count,
                cube_dim,
                input.as_tensor_arg::<F>(1),
                weight.as_tensor_arg::<F>(1),
                output.as_tensor_arg::<F>(1),
                ScalarArg::new(dim as u32),
            );
        } else {
            let cube_dim = CubeDim { x: 16, y: 16, z: 1 };
            let cubes_x = (batch_size as f32 / cube_dim.x as f32).ceil() as u32;
            let cubes_y = (seq_len as f32 / cube_dim.y as f32).ceil() as u32;
            let cube_count = CubeCount::Static(cubes_x, cubes_y, 1);

            fused_rms_norm_kernel::launch::<F, R>(
                &input.client,
                cube_count,
                cube_dim,
                input.as_tensor_arg::<F>(1),
                weight.as_tensor_arg::<F>(1),
                output.as_tensor_arg::<F>(1),
                ScalarArg::new(dim as u32),
            );
        }

        output
    }

    fn fused_softmax(input: FloatTensor<Self>, dim_size: usize) -> FloatTensor<Self> {
        let input = into_contiguous(input);
        let total_elements = input.shape.num_elements();
        let num_rows = total_elements / dim_size;

        let shape_out = input.shape.clone();
        let buffer = input
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        let output = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        let cube_dim = CubeDim::default();
        let cubes_needed = (num_rows as f32 / cube_dim.x as f32).ceil() as u32;
        let cube_count = CubeCount::Static(cubes_needed, 1, 1);

        fused_softmax_kernel::launch::<F, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
            ScalarArg::new(dim_size as u32),
        );

        output
    }

    fn fused_l2_norm(input: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let input = into_contiguous(input);
        let [batch_size, seq_len, _dim] = input.shape.dims();
        let total_rows = batch_size * seq_len;

        let shape_out = Shape::from([batch_size, seq_len]);
        let buffer = input
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        let output = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        // Use optimized kernel with plane-based parallelism for larger dims
        if dim >= 64 {
            let plane_size = 32u32;
            let rows_per_cube = 8u32;
            let cube_dim = CubeDim {
                x: plane_size,
                y: rows_per_cube,
                z: 1,
            };
            let cubes_needed = (total_rows as f32 / rows_per_cube as f32).ceil() as u32;
            let cube_count = CubeCount::Static(cubes_needed, 1, 1);

            fused_l2_norm_kernel_optimized::launch::<F, R>(
                &input.client,
                cube_count,
                cube_dim,
                input.as_tensor_arg::<F>(1),
                output.as_tensor_arg::<F>(1),
                ScalarArg::new(dim as u32),
            );
        } else {
            let cube_dim = CubeDim { x: 16, y: 16, z: 1 };
            let cubes_x = (batch_size as f32 / cube_dim.x as f32).ceil() as u32;
            let cubes_y = (seq_len as f32 / cube_dim.y as f32).ceil() as u32;
            let cube_count = CubeCount::Static(cubes_x, cubes_y, 1);

            fused_l2_norm_kernel::launch::<F, R>(
                &input.client,
                cube_count,
                cube_dim,
                input.as_tensor_arg::<F>(1),
                output.as_tensor_arg::<F>(1),
                ScalarArg::new(dim as u32),
            );
        }

        output
    }

    fn fused_silu_gate(
        gate_input: FloatTensor<Self>,
        up_input: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let gate_input = into_contiguous(gate_input);
        let up_input = into_contiguous(up_input);
        let total_elements = gate_input.shape.num_elements();

        let shape_out = gate_input.shape.clone();
        let buffer = gate_input
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        let output = CubeTensor::new_contiguous(
            gate_input.client.clone(),
            gate_input.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        let cube_dim = CubeDim::default();
        let cubes_needed = (total_elements as f32 / cube_dim.x as f32).ceil() as u32;
        let cube_count = CubeCount::Static(cubes_needed, 1, 1);

        fused_silu_gate_kernel::launch::<F, R>(
            &gate_input.client,
            cube_count,
            cube_dim,
            gate_input.as_tensor_arg::<F>(1),
            up_input.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
        );

        output
    }

    fn fused_coherence(
        norms: FloatTensor<Self>,
        entropies: FloatTensor<Self>,
        epsilon: f32,
    ) -> FloatTensor<Self> {
        let norms = into_contiguous(norms);
        let entropies = into_contiguous(entropies);
        let total_elements = norms.shape.num_elements();

        let shape_out = norms.shape.clone();
        let buffer = norms
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        let output = CubeTensor::new_contiguous(
            norms.client.clone(),
            norms.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        let cube_dim = CubeDim::default();
        let cubes_needed = (total_elements as f32 / cube_dim.x as f32).ceil() as u32;
        let cube_count = CubeCount::Static(cubes_needed, 1, 1);

        // Note: epsilon is hardcoded in the kernel (1e-6)
        let _ = epsilon;
        fused_coherence_kernel::launch::<F, R>(
            &norms.client,
            cube_count,
            cube_dim,
            norms.as_tensor_arg::<F>(1),
            entropies.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
        );

        output
    }

    fn fused_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        head_dim: usize,
        use_causal_mask: bool,
    ) -> FloatTensor<Self> {
        let q = into_contiguous(q);
        let k = into_contiguous(k);
        let v = into_contiguous(v);

        let [batch_size, n_heads, seq_q, _head_dim] = q.shape.dims();
        let [_, _, seq_kv, _] = k.shape.dims();

        let shape_out = q.shape.clone();
        let buffer = q
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        let output = CubeTensor::new_contiguous(
            q.client.clone(),
            q.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        let scale = F::from_elem(1.0 / (head_dim as f32).sqrt());
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };
        let cubes_x = (seq_q as f32 / cube_dim.x as f32).ceil() as u32;
        let cubes_y = (n_heads as f32 / cube_dim.y as f32).ceil() as u32;
        let cubes_z = batch_size as u32;
        let cube_count = CubeCount::Static(cubes_x, cubes_y, cubes_z);

        fused_attention_kernel::launch::<F, R>(
            &q.client,
            cube_count,
            cube_dim,
            q.as_tensor_arg::<F>(1),
            k.as_tensor_arg::<F>(1),
            v.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
            ScalarArg::new(head_dim as u32),
            ScalarArg::new(seq_kv as u32),
            ScalarArg::new(scale),
            ScalarArg::new(if use_causal_mask { 1u32 } else { 0u32 }),
        );

        output
    }

    fn fused_cross_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        head_dim: usize,
    ) -> FloatTensor<Self> {
        let q = into_contiguous(q);
        let k = into_contiguous(k);
        let v = into_contiguous(v);

        let [batch_size, n_patches, n_heads, _head_dim] = q.shape.dims();
        let [_, n_bytes, _, _] = k.shape.dims();

        let shape_out = q.shape.clone();
        let buffer = q
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        let output = CubeTensor::new_contiguous(
            q.client.clone(),
            q.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        let scale = F::from_elem(1.0 / (head_dim as f32).sqrt());
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };
        let cubes_x = (n_patches as f32 / cube_dim.x as f32).ceil() as u32;
        let cubes_y = (n_heads as f32 / cube_dim.y as f32).ceil() as u32;
        let cubes_z = batch_size as u32;
        let cube_count = CubeCount::Static(cubes_x, cubes_y, cubes_z);

        fused_cross_attention_kernel::launch::<F, R>(
            &q.client,
            cube_count,
            cube_dim,
            q.as_tensor_arg::<F>(1),
            k.as_tensor_arg::<F>(1),
            v.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
            ScalarArg::new(head_dim as u32),
            ScalarArg::new(n_bytes as u32),
            ScalarArg::new(scale),
        );

        output
    }

    fn fused_hash_indices(
        bytes: FloatTensor<Self>,
        n_bytes: usize,
        hash_vocab: f32,
    ) -> FloatTensor<Self> {
        let bytes = into_contiguous(bytes);
        let [batch_size, _n_bytes] = bytes.shape.dims();
        let total_positions = batch_size * n_bytes;

        let shape_out = Shape::from([batch_size, n_bytes, 6]);
        let buffer = bytes
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        let output = CubeTensor::new_contiguous(
            bytes.client.clone(),
            bytes.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        let cube_dim = CubeDim::default();
        let cubes_needed = (total_positions as f32 / cube_dim.x as f32).ceil() as u32;
        let cube_count = CubeCount::Static(cubes_needed, 1, 1);

        rolling_hash_float_kernel::launch::<F, R>(
            &bytes.client,
            cube_count,
            cube_dim,
            bytes.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
            ScalarArg::new(n_bytes as u32),
            ScalarArg::new(F::from_elem(hash_vocab)),
        );

        output
    }
}

// ============================================================================
// Fusion<B> Implementation - Fallback to reference implementations
// ============================================================================

impl<B: FusionBackend> FusedOpsBackend for Fusion<B> {
    fn fused_entropy(logits: FloatTensor<Self>, _vocab_size: usize) -> FloatTensor<Self> {
        let logits_tensor: Tensor<Fusion<B>, 3> =
            Tensor::from_primitive(TensorPrimitive::Float(logits));
        let [bs, seq_len, _vocab] = logits_tensor.dims();

        let log_probs = log_softmax(logits_tensor, 2);
        let probs = log_probs.clone().exp();
        let p_log_p = log_probs * probs;
        let entropy_values = p_log_p.sum_dim(2).neg().reshape([bs, seq_len]);

        entropy_values.into_primitive().tensor()
    }

    fn fused_rms_norm(
        input: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        epsilon: f32,
        _dim: usize,
    ) -> FloatTensor<Self> {
        let x: Tensor<Fusion<B>, 3> = Tensor::from_primitive(TensorPrimitive::Float(input));
        let weight: Tensor<Fusion<B>, 1> = Tensor::from_primitive(TensorPrimitive::Float(weight));
        let [_, _, dim] = x.dims();

        // RMS norm: x / sqrt(mean(x^2) + eps) * weight
        let squared = x.clone() * x.clone();
        let norm = squared.mean_dim(2).sqrt().add_scalar(epsilon as f64);
        let weight = weight.reshape([1, 1, dim]);
        let result = x.div(norm) * weight;

        result.into_primitive().tensor()
    }

    fn fused_softmax(input: FloatTensor<Self>, _dim_size: usize) -> FloatTensor<Self> {
        // Determine the rank from shape
        let rank = input.shape.len();

        // Use Burn's softmax over the last dimension
        // Note: This is a simplified version that works for common cases
        match rank {
            2 => {
                let tensor: Tensor<Fusion<B>, 2> =
                    Tensor::from_primitive(TensorPrimitive::Float(input));
                softmax(tensor, 1).into_primitive().tensor()
            }
            3 => {
                let tensor: Tensor<Fusion<B>, 3> =
                    Tensor::from_primitive(TensorPrimitive::Float(input));
                softmax(tensor, 2).into_primitive().tensor()
            }
            4 => {
                let tensor: Tensor<Fusion<B>, 4> =
                    Tensor::from_primitive(TensorPrimitive::Float(input));
                softmax(tensor, 3).into_primitive().tensor()
            }
            _ => {
                // Fallback: assume 3D
                let tensor: Tensor<Fusion<B>, 3> =
                    Tensor::from_primitive(TensorPrimitive::Float(input));
                softmax(tensor, 2).into_primitive().tensor()
            }
        }
    }

    fn fused_l2_norm(input: FloatTensor<Self>, _dim: usize) -> FloatTensor<Self> {
        let x: Tensor<Fusion<B>, 3> = Tensor::from_primitive(TensorPrimitive::Float(input));
        let [bs, seq_len, _dim] = x.dims();

        let squared = x.clone() * x;
        let norm = squared.sum_dim(2).sqrt().reshape([bs, seq_len]);

        norm.into_primitive().tensor()
    }

    fn fused_silu_gate(
        gate_input: FloatTensor<Self>,
        up_input: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let gate: Tensor<Fusion<B>, 3> = Tensor::from_primitive(TensorPrimitive::Float(gate_input));
        let up: Tensor<Fusion<B>, 3> = Tensor::from_primitive(TensorPrimitive::Float(up_input));

        let result = silu(gate) * up;
        result.into_primitive().tensor()
    }

    fn fused_coherence(
        norms: FloatTensor<Self>,
        entropies: FloatTensor<Self>,
        epsilon: f32,
    ) -> FloatTensor<Self> {
        let norms: Tensor<Fusion<B>, 2> = Tensor::from_primitive(TensorPrimitive::Float(norms));
        let entropies: Tensor<Fusion<B>, 2> =
            Tensor::from_primitive(TensorPrimitive::Float(entropies));

        let result = norms.clone().powf_scalar(2.0) / (entropies + epsilon as f64);
        result.into_primitive().tensor()
    }

    fn fused_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        head_dim: usize,
        use_causal_mask: bool,
    ) -> FloatTensor<Self> {
        // Reference implementation using standard ops
        let q: Tensor<Fusion<B>, 4> = Tensor::from_primitive(TensorPrimitive::Float(q));
        let k: Tensor<Fusion<B>, 4> = Tensor::from_primitive(TensorPrimitive::Float(k));
        let v: Tensor<Fusion<B>, 4> = Tensor::from_primitive(TensorPrimitive::Float(v));

        let [_batch, _n_heads, seq_q, _] = q.dims();
        let [_, _, seq_kv, _] = k.dims();

        // Q·K^T / sqrt(d)
        let scale = (head_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;

        // Apply causal mask if needed
        let scores = if use_causal_mask {
            let device = scores.device();
            let mut mask_data = vec![0.0f32; seq_q * seq_kv];
            for i in 0..seq_q {
                for j in 0..seq_kv {
                    if j > i {
                        mask_data[i * seq_kv + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let mask: Tensor<Fusion<B>, 2> = Tensor::from_data(
                burn::tensor::TensorData::new(mask_data, [seq_q, seq_kv]),
                &device,
            );
            let mask_4d: Tensor<Fusion<B>, 4> = mask.unsqueeze::<3>().unsqueeze();
            scores + mask_4d
        } else {
            scores
        };

        // Softmax
        let attn = softmax(scores, 3);

        // Attention · V
        let result = attn.matmul(v);
        result.into_primitive().tensor()
    }

    fn fused_cross_attention(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        head_dim: usize,
    ) -> FloatTensor<Self> {
        // Reference implementation - no causal mask for cross-attention
        let q: Tensor<Fusion<B>, 4> = Tensor::from_primitive(TensorPrimitive::Float(q));
        let k: Tensor<Fusion<B>, 4> = Tensor::from_primitive(TensorPrimitive::Float(k));
        let v: Tensor<Fusion<B>, 4> = Tensor::from_primitive(TensorPrimitive::Float(v));

        // Q·K^T / sqrt(d)
        let scale = (head_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;

        // Softmax
        let attn = softmax(scores, 3);

        // Attention · V
        let result = attn.matmul(v);
        result.into_primitive().tensor()
    }

    fn fused_hash_indices(
        bytes: FloatTensor<Self>,
        _n_bytes: usize,
        hash_vocab: f32,
    ) -> FloatTensor<Self> {
        // CPU fallback: compute hash indices using float arithmetic
        let bytes_tensor: Tensor<Fusion<B>, 2> =
            Tensor::from_primitive(TensorPrimitive::Float(bytes));
        let [batch_size, seq_len] = bytes_tensor.dims();
        let device = bytes_tensor.device();

        // Prime powers (small primes keep values in f32 exact range)
        const P0_POWERS: [f32; 3] = [1.0, 31.0, 961.0]; // 31^0, 31^1, 31^2
        const P1_POWERS: [f32; 3] = [1.0, 37.0, 1369.0]; // 37^0, 37^1, 37^2
        const P2_POWERS: [f32; 3] = [1.0, 41.0, 1681.0]; // 41^0, 41^1, 41^2

        let bytes_data = bytes_tensor.into_data();
        let bytes_vec: Vec<f32> = bytes_data.iter::<f32>().collect();

        let mut hash_indices = vec![0.0f32; batch_size * seq_len * 6];
        let vocab = hash_vocab as f64;

        for b in 0..batch_size {
            for pos in 0..seq_len {
                let b0 = bytes_vec[b * seq_len + pos] as f64;
                let b1 = if pos >= 1 {
                    bytes_vec[b * seq_len + pos - 1] as f64
                } else {
                    0.0
                };
                let b2 = if pos >= 2 {
                    bytes_vec[b * seq_len + pos - 2] as f64
                } else {
                    0.0
                };

                let base = (b * seq_len + pos) * 6;

                // Table 0: prime=31, ngram=2
                let h0 = b0 * P0_POWERS[0] as f64 + b1 * P0_POWERS[1] as f64;
                hash_indices[base] = (h0 - (h0 / vocab).floor() * vocab).abs() as f32;

                // Table 1: prime=31, ngram=3
                let h1 = h0 + b2 * P0_POWERS[2] as f64;
                hash_indices[base + 1] = (h1 - (h1 / vocab).floor() * vocab).abs() as f32;

                // Table 2: prime=37, ngram=2
                let h2 = b0 * P1_POWERS[0] as f64 + b1 * P1_POWERS[1] as f64;
                hash_indices[base + 2] = (h2 - (h2 / vocab).floor() * vocab).abs() as f32;

                // Table 3: prime=37, ngram=3
                let h3 = h2 + b2 * P1_POWERS[2] as f64;
                hash_indices[base + 3] = (h3 - (h3 / vocab).floor() * vocab).abs() as f32;

                // Table 4: prime=41, ngram=2
                let h4 = b0 * P2_POWERS[0] as f64 + b1 * P2_POWERS[1] as f64;
                hash_indices[base + 4] = (h4 - (h4 / vocab).floor() * vocab).abs() as f32;

                // Table 5: prime=41, ngram=3
                let h5 = h4 + b2 * P2_POWERS[2] as f64;
                hash_indices[base + 5] = (h5 - (h5 / vocab).floor() * vocab).abs() as f32;
            }
        }

        let result: Tensor<Fusion<B>, 3> = Tensor::from_data(
            burn::tensor::TensorData::new(hash_indices, [batch_size, seq_len, 6]),
            &device,
        );
        result.into_primitive().tensor()
    }
}
