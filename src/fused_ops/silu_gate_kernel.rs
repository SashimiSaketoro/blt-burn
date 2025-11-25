//! Fused SiLU (Swish) + Gate kernel for FFN.
//!
//! SiLU Gate: silu(w1_out) * w3_out
//! where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//!
//! This is used in the SwiGLU FFN architecture from LLaMA/BLT.
//!
//! This kernel fuses:
//! 1. Compute sigmoid of w1_out
//! 2. Multiply w1_out by sigmoid (SiLU)
//! 3. Multiply by w3_out (gating)
//!
//! Current: 3+ kernel launches
//! Fused: 1 kernel launch

use cubecl::{cube, prelude::*};

/// Fused SiLU activation and gating kernel.
///
/// Computes: silu(gate_input) * up_input
/// where silu(x) = x * sigmoid(x)
///
/// Both inputs must have the same shape.
///
/// # Type Parameters
/// * `F` - Float type (f16, f32, etc.)
///
/// # Arguments
/// * `gate_input` - Input to SiLU (w1 output), shape [batch, seq, hidden]
/// * `up_input` - Gating input (w3 output), shape [batch, seq, hidden]
/// * `output` - Output tensor, same shape as inputs
#[cube(launch)]
pub fn fused_silu_gate_kernel<F: Float>(
    gate_input: &Tensor<F>,
    up_input: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let idx = ABSOLUTE_POS_X;

    if idx >= output.len() {
        terminate!();
    }

    let gate = gate_input[idx];
    let up = up_input[idx];

    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // For numerical stability with large negative values:
    // sigmoid(x) = 1 / (1 + exp(-x)) for x >= 0
    // sigmoid(x) = exp(x) / (1 + exp(x)) for x < 0
    
    // Simple version (works well for typical ranges):
    let sigmoid = F::new(1.0) / (F::new(1.0) + F::exp(F::new(0.0) - gate));
    let silu = gate * sigmoid;

    output[idx] = silu * up;
}

/// Fused SiLU activation only (without gating).
///
/// Computes: silu(x) = x * sigmoid(x)
#[cube(launch)]
pub fn fused_silu_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    let idx = ABSOLUTE_POS_X;

    if idx >= output.len() {
        terminate!();
    }

    let x = input[idx];
    let sigmoid = F::new(1.0) / (F::new(1.0) + F::exp(F::new(0.0) - x));
    output[idx] = x * sigmoid;
}

