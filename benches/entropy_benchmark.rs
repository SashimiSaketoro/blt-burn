//! Benchmark comparing fused vs reference entropy implementations.
//!
//! Run with: `cargo bench --features fused-entropy`
//!
//! This benchmark compares:
//! 1. Reference implementation (standard tensor ops)
//! 2. Fused CubeCL kernel (single kernel launch)

use burn::tensor::{activation::log_softmax, backend::Backend, Distribution, Tensor};
use burn_cubecl::CubeBackend;
use burn_wgpu::WgpuRuntime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// Use CubeBackend directly to see the real fused kernel performance
type DirectBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

/// Reference entropy implementation (non-fused, uses multiple kernels).
fn entropy_reference<B: Backend>(scores: Tensor<B, 3>) -> Tensor<B, 2> {
    let [bs, seq_len, _vocab] = scores.dims();
    let log_probs = log_softmax(scores, 2);
    let probs = log_probs.clone().exp();
    let p_log_p = log_probs * probs;
    p_log_p.sum_dim(2).neg().reshape([bs, seq_len])
}

fn benchmark_entropy_direct(c: &mut Criterion) {
    use burn_wgpu::WgpuDevice;

    let device = WgpuDevice::default();

    // BLT typical dimensions
    let vocab_size = 260; // BLT vocab size

    let mut group = c.benchmark_group("entropy_direct");

    // Test different sizes
    for (batch_size, seq_len) in [(2, 64), (4, 128), (8, 256)] {
        let param_str = format!("b{batch_size}_s{seq_len}_v{vocab_size}");

        // Pre-create a tensor to warm up the device
        let warmup = Tensor::<DirectBackend, 3>::random(
            [batch_size, seq_len, vocab_size],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let _ = warmup.into_data();

        // Benchmark reference implementation (multiple kernel launches)
        group.bench_with_input(
            BenchmarkId::new("reference", &param_str),
            &(batch_size, seq_len),
            |b, &(bs, sl)| {
                b.iter(|| {
                    let logits = Tensor::<DirectBackend, 3>::random(
                        [bs, sl, vocab_size],
                        Distribution::Normal(0.0, 1.0),
                        &device,
                    );
                    let result = entropy_reference(black_box(logits));
                    // Force sync to measure actual GPU time
                    let _ = result.into_data();
                });
            },
        );

        // Benchmark fused implementation (single kernel launch)
        #[cfg(feature = "fused-entropy")]
        group.bench_with_input(
            BenchmarkId::new("fused", &param_str),
            &(batch_size, seq_len),
            |b, &(bs, sl)| {
                b.iter(|| {
                    let logits = Tensor::<DirectBackend, 3>::random(
                        [bs, sl, vocab_size],
                        Distribution::Normal(0.0, 1.0),
                        &device,
                    );
                    let result = blt_burn::fused_ops::fused_entropy(black_box(logits), vocab_size);
                    // Force sync to measure actual GPU time
                    let _ = result.into_data();
                });
            },
        );
    }

    group.finish();
}

fn benchmark_all_fused_ops(c: &mut Criterion) {
    use burn_wgpu::WgpuDevice;

    let device = WgpuDevice::default();

    let mut group = c.benchmark_group("fused_ops");
    group.sample_size(50);

    // Test dimensions
    let batch_size = 4;
    let seq_len = 128;
    let dim = 256;
    let vocab_size = 260;

    // Warmup
    let warmup = Tensor::<DirectBackend, 3>::random(
        [batch_size, seq_len, dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let _ = warmup.into_data();

    // ========== RMS Norm ==========
    group.bench_function("rms_norm/reference", |b| {
        b.iter(|| {
            let x = Tensor::<DirectBackend, 3>::random(
                [batch_size, seq_len, dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let weight = Tensor::<DirectBackend, 1>::ones([dim], &device);
            // Reference: x / sqrt(mean(x^2) + eps) * weight
            let squared = x.clone() * x.clone();
            let norm = squared.mean_dim(2).sqrt().add_scalar(1e-6);
            let weight = weight.reshape([1, 1, dim]);
            let result = x.div(norm) * weight;
            let _ = result.into_data();
        });
    });

    #[cfg(feature = "fused-entropy")]
    group.bench_function("rms_norm/fused", |b| {
        b.iter(|| {
            let x = Tensor::<DirectBackend, 3>::random(
                [batch_size, seq_len, dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let weight = Tensor::<DirectBackend, 1>::ones([dim], &device);
            let result = blt_burn::fused_ops::fused_rms_norm(x, weight, 1e-6);
            let _ = result.into_data();
        });
    });

    // ========== L2 Norm ==========
    group.bench_function("l2_norm/reference", |b| {
        b.iter(|| {
            let x = Tensor::<DirectBackend, 3>::random(
                [batch_size, seq_len, dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let squared = x.clone() * x;
            let result = squared.sum_dim(2).sqrt().reshape([batch_size, seq_len]);
            let _ = result.into_data();
        });
    });

    #[cfg(feature = "fused-entropy")]
    group.bench_function("l2_norm/fused", |b| {
        b.iter(|| {
            let x = Tensor::<DirectBackend, 3>::random(
                [batch_size, seq_len, dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let result = blt_burn::fused_ops::fused_l2_norm(x);
            let _ = result.into_data();
        });
    });

    // ========== SiLU Gate ==========
    group.bench_function("silu_gate/reference", |b| {
        b.iter(|| {
            let gate = Tensor::<DirectBackend, 3>::random(
                [batch_size, seq_len, dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let up = Tensor::<DirectBackend, 3>::random(
                [batch_size, seq_len, dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let result = burn::tensor::activation::silu(gate) * up;
            let _ = result.into_data();
        });
    });

    #[cfg(feature = "fused-entropy")]
    group.bench_function("silu_gate/fused", |b| {
        b.iter(|| {
            let gate = Tensor::<DirectBackend, 3>::random(
                [batch_size, seq_len, dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let up = Tensor::<DirectBackend, 3>::random(
                [batch_size, seq_len, dim],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let result = blt_burn::fused_ops::fused_silu_gate(gate, up);
            let _ = result.into_data();
        });
    });

    // ========== Entropy ==========
    group.bench_function("entropy/reference", |b| {
        b.iter(|| {
            let logits = Tensor::<DirectBackend, 3>::random(
                [batch_size, seq_len, vocab_size],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let result = entropy_reference(logits);
            let _ = result.into_data();
        });
    });

    #[cfg(feature = "fused-entropy")]
    group.bench_function("entropy/fused", |b| {
        b.iter(|| {
            let logits = Tensor::<DirectBackend, 3>::random(
                [batch_size, seq_len, vocab_size],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let result = blt_burn::fused_ops::fused_entropy(logits, vocab_size);
            let _ = result.into_data();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_entropy_direct, benchmark_all_fused_ops);
criterion_main!(benches);
