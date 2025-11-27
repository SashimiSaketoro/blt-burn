//! Backend implementations for fused operations.
//!
//! This module provides `FusedOpsBackend` implementations for:
//! - `CubeBackend<R, F, I, BT>` - Direct CubeCL kernel execution
//! - `Fusion<B>` (e.g., Wgpu) - Fallback to reference implementation

use super::{
    coherence_kernel::fused_coherence_kernel,
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
        let gate: Tensor<Fusion<B>, 3> =
            Tensor::from_primitive(TensorPrimitive::Float(gate_input));
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
}

