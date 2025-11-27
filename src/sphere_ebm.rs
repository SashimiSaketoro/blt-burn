//! Sphere Energy-Based Model for Water-Filling Optimization
//!
//! This module provides a Rust-native implementation of sphere optimization
//! using Langevin dynamics and optional thrml-rs EBM integration.
//!
//! ## Energy Function
//!
//! The Hamiltonian combines:
//! - **Radial Gravity**: Forces points to their ideal radii based on prominence ranking
//!   `E_gravity = (r - r_ideal)^2`
//! - **Lateral Surface Tension**: Attracts semantically similar points
//!   `E_lateral = -sum(Similarity_ij * Gaussian(dist_ij))`
//!
//! ## Sampling
//!
//! Overdamped Langevin dynamics: `dx = -∇H(x)dt + sqrt(2Tdt)ξ`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

/// Spherical coordinates: (radius, theta, phi)
#[derive(Clone, Debug)]
pub struct SphericalCoords {
    pub r: Array1<f32>,
    pub theta: Array1<f32>,
    pub phi: Array1<f32>,
}

impl SphericalCoords {
    /// Convert spherical to Cartesian coordinates
    pub fn to_cartesian(&self) -> Array2<f32> {
        let n = self.r.len();
        let cart_vec: Vec<f32> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let r = self.r[i];
                let st = self.theta[i].sin();
                let ct = self.theta[i].cos();
                let sp = self.phi[i].sin();
                let cp = self.phi[i].cos();
                vec![r * st * cp, r * st * sp, r * ct]
            })
            .collect();

        Array2::from_shape_vec((n, 3), cart_vec).unwrap()
    }

    /// Convert Cartesian to spherical coordinates
    pub fn from_cartesian(cart: ArrayView2<f32>) -> Self {
        let n = cart.nrows();
        let r_vec: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let x = cart[[i, 0]];
                let y = cart[[i, 1]];
                let z = cart[[i, 2]];
                (x * x + y * y + z * z).sqrt()
            })
            .collect();

        let theta_vec: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let z = cart[[i, 2]];
                let r = r_vec[i];
                (z / (r + 1e-8)).clamp(-1.0, 1.0).acos()
            })
            .collect();

        let phi_vec: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let x = cart[[i, 0]];
                let y = cart[[i, 1]];
                y.atan2(x)
            })
            .collect();

        Self {
            r: Array1::from_vec(r_vec),
            theta: Array1::from_vec(theta_vec),
            phi: Array1::from_vec(phi_vec),
        }
    }
}

/// Scale profile for different corpus sizes
#[derive(Clone, Copy, Debug, Default)]
pub enum ScaleProfile {
    #[default]
    Dev,
    Medium,
    Large,
    Planetary,
}

/// Settings derived from scale profile
#[derive(Clone, Copy, Debug)]
pub struct ScaleSettings {
    pub min_radius: f32,
    pub max_radius: f32,
    pub interaction_radius: f32,
    pub step_size: f32,
    pub temperature: f32,
    pub n_steps: usize,
}

impl ScaleProfile {
    pub fn settings(self) -> ScaleSettings {
        match self {
            ScaleProfile::Dev => ScaleSettings {
                min_radius: 32.0,
                max_radius: 512.0,
                interaction_radius: 1.0,
                step_size: 0.5,
                temperature: 0.1,
                n_steps: 120,
            },
            ScaleProfile::Medium => ScaleSettings {
                min_radius: 64.0,
                max_radius: 2_048.0,
                interaction_radius: 1.25,
                step_size: 0.4,
                temperature: 0.08,
                n_steps: 240,
            },
            ScaleProfile::Large => ScaleSettings {
                min_radius: 96.0,
                max_radius: 8_192.0,
                interaction_radius: 1.5,
                step_size: 0.35,
                temperature: 0.06,
                n_steps: 480,
            },
            ScaleProfile::Planetary => ScaleSettings {
                min_radius: 128.0,
                max_radius: 32_768.0,
                interaction_radius: 2.0,
                step_size: 0.3,
                temperature: 0.05,
                n_steps: 960,
            },
        }
    }
}

/// Box-Muller transform: generate two independent standard normal samples
#[inline]
fn box_muller_pair() -> (f32, f32) {
    let u1 = fastrand::f32().max(1e-10);
    let u2 = fastrand::f32();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Generate a single standard normal sample
#[inline]
fn randn() -> f32 {
    box_muller_pair().0
}

/// Compute cosine similarity matrix in parallel
pub fn cosine_similarity_matrix(embeddings: ArrayView2<f32>) -> Array2<f32> {
    let n = embeddings.nrows();

    // Normalize embeddings (L2 norm per row)
    let norms: Array1<f32> = embeddings
        .axis_iter(Axis(0))
        .map(|row| row.iter().map(|&x| x * x).sum::<f32>().sqrt())
        .collect();

    let normalized_flat: Vec<f32> = embeddings
        .axis_iter(Axis(0))
        .zip(norms.iter())
        .flat_map(|(row, &norm)| {
            let v: Vec<f32> = row.iter().map(|&x| x / (norm + 1e-8)).collect();
            v.into_iter()
        })
        .collect();
    let normalized = Array2::from_shape_vec((n, embeddings.ncols()), normalized_flat).unwrap();

    // Parallel pairwise similarity computation
    let normalized_vec: Vec<Vec<f32>> = normalized
        .axis_iter(Axis(0))
        .map(|row| row.iter().copied().collect())
        .collect();

    let similarity_vec: Vec<f32> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let vec_i = &normalized_vec[i];
            let row: Vec<f32> = (0..n)
                .map(|j| {
                    if i != j {
                        let vec_j = &normalized_vec[j];
                        vec_i
                            .iter()
                            .zip(vec_j.iter())
                            .map(|(&a, &b)| a * b)
                            .sum::<f32>()
                    } else {
                        0.0 // Zero out self-similarity
                    }
                })
                .collect();
            row.into_iter()
        })
        .collect();

    Array2::from_shape_vec((n, n), similarity_vec).unwrap()
}

/// Compute pairwise squared distances in parallel
pub fn pairwise_distances_sq(positions: ArrayView2<f32>) -> Array2<f32> {
    let n = positions.nrows();

    let pos_vec: Vec<Vec<f32>> = positions
        .axis_iter(Axis(0))
        .map(|row| row.iter().copied().collect())
        .collect();

    let dist_sq_vec: Vec<f32> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let pos_i = &pos_vec[i];
            let row: Vec<f32> = (0..n)
                .map(|j| {
                    if i != j {
                        let pos_j = &pos_vec[j];
                        pos_i
                            .iter()
                            .zip(pos_j.iter())
                            .map(|(&a, &b)| (a - b) * (a - b))
                            .sum()
                    } else {
                        0.0
                    }
                })
                .collect();
            row.into_iter()
        })
        .collect();

    Array2::from_shape_vec((n, n), dist_sq_vec).unwrap()
}

/// Gravity force: F_g = -2 * (r - r_ideal)
pub fn compute_gravity_force(r: ArrayView1<f32>, ideal_r: ArrayView1<f32>) -> Array1<f32> {
    r.iter()
        .zip(ideal_r.iter())
        .map(|(&r_val, &ideal)| -2.0 * (r_val - ideal))
        .collect()
}

/// Lateral force from similarity-weighted Gaussian kernel
pub fn compute_lateral_force(
    positions: ArrayView2<f32>,
    similarity: ArrayView2<f32>,
    interaction_radius: f32,
) -> Array2<f32> {
    let n = positions.nrows();
    let sigma2 = interaction_radius * interaction_radius;
    let dist_sq = pairwise_distances_sq(positions);

    // Gaussian weights: similarity * exp(-dist^2 / sigma^2)
    let weights: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let sim = similarity[[i, j]];
                    let d = dist_sq[[i, j]];
                    sim * (-d / sigma2).exp()
                })
                .collect()
        })
        .collect();

    let pos_vec: Vec<Vec<f32>> = positions
        .axis_iter(Axis(0))
        .map(|row| row.iter().copied().collect())
        .collect();

    // Force accumulation: sum_j weights_ij * (pos_j - pos_i) * (2 / sigma^2)
    // The 2/sigma^2 factor comes from the gradient of exp(-d^2 / sigma^2)
    // This matches thrml-sphere/hamiltonian.rs lateral_force_cartesian
    let scale = 2.0 / sigma2;
    let forces_vec: Vec<f32> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let pos_i = &pos_vec[i];
            let mut force = [0.0f32; 3];
            for j in 0..n {
                if i != j {
                    let pos_j = &pos_vec[j];
                    let weight = weights[i][j];
                    for k in 0..3 {
                        force[k] += weight * (pos_j[k] - pos_i[k]);
                    }
                }
            }
            // Apply scaling factor
            [force[0] * scale, force[1] * scale, force[2] * scale]
        })
        .collect();

    Array2::from_shape_vec((n, 3), forces_vec).unwrap()
}

/// Compute ideal radii using r^1.5 capacity law
///
/// When `entropies` is provided, weights are entropy-biased:
/// Higher entropy → outer shells (more "spread out" concepts)
pub fn compute_ideal_radii(
    prominence: ArrayView1<f32>,
    entropies: Option<ArrayView1<f32>>,
    min_radius: f32,
    max_radius: f32,
) -> Array1<f32> {
    let n = prominence.len();

    // Compute weighted prominence if entropies provided
    let weights: Vec<f32> = if let Some(ent) = entropies {
        // Entropy-weighted: high entropy → higher rank → outer shell
        prominence
            .iter()
            .zip(ent.iter())
            .map(|(&p, &e)| p * (1.0 + e.max(0.0)))
            .collect()
    } else {
        prominence.iter().copied().collect()
    };

    // Sort and rank - DESCENDING: higher weight = smaller rank = smaller radius (closer to core)
    // This matches thrml-sphere/sphere_ebm.rs behavior
    let mut indexed: Vec<(usize, f32)> = weights.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0; n];
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        ranks[idx] = rank;
    }

    let radius_span = (max_radius - min_radius).max(1.0);

    // r^1.5 law: r = min_radius + span * normalized^(1/1.5)
    ranks
        .iter()
        .map(|&rank| {
            let normalized_rank = (rank as f32 + 0.5) / n as f32;
            min_radius + radius_span * normalized_rank.powf(1.0 / 1.5)
        })
        .collect()
}

/// Langevin dynamics step with parallel force computation
pub fn langevin_step(
    coords: &SphericalCoords,
    ideal_radii: ArrayView1<f32>,
    similarity: ArrayView2<f32>,
    interaction_radius: f32,
    step_size: f32,
    temperature: f32,
) -> SphericalCoords {
    let positions = coords.to_cartesian();
    let gravity_force = compute_gravity_force(coords.r.view(), ideal_radii);

    let n = coords.r.len();
    let gravity_cart: Array2<f32> = Array2::from_shape_vec(
        (n, 3),
        (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let r_hat = [
                    coords.theta[i].sin() * coords.phi[i].cos(),
                    coords.theta[i].sin() * coords.phi[i].sin(),
                    coords.theta[i].cos(),
                ];
                let f_g = gravity_force[i];
                [r_hat[0] * f_g, r_hat[1] * f_g, r_hat[2] * f_g]
            })
            .collect(),
    )
    .unwrap();

    let lateral_force = compute_lateral_force(positions.view(), similarity, interaction_radius);
    let total_force = &gravity_cart + &lateral_force;
    let drift = &total_force * step_size;

    let noise_scale = (2.0 * temperature * step_size).sqrt();
    let noise: Array2<f32> = Array2::from_shape_vec(
        (n, 3),
        (0..n)
            .into_par_iter()
            .flat_map_iter(|_| {
                let (z0, z1) = box_muller_pair();
                let z2 = randn();
                [z0, z1, z2]
            })
            .collect(),
    )
    .unwrap();
    let diffusion = &noise * noise_scale;

    let new_positions = &positions + &drift + &diffusion;
    SphericalCoords::from_cartesian(new_positions.view())
}

/// Main Sphere Optimizer using Langevin dynamics
#[derive(Clone, Debug)]
pub struct SphereOptimizer {
    pub step_size: f32,
    pub temperature: f32,
    pub interaction_radius: f32,
    pub min_radius: f32,
    pub max_radius: f32,
    pub n_steps: usize,
    pub entropy_weighted: bool,
}

impl Default for SphereOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SphereOptimizer {
    pub fn new() -> Self {
        Self {
            step_size: 0.5,
            temperature: 0.1,
            interaction_radius: 1.0,
            min_radius: 32.0,
            max_radius: 512.0,
            n_steps: 100,
            entropy_weighted: false,
        }
    }

    pub fn from_profile(profile: ScaleProfile) -> Self {
        let settings = profile.settings();
        Self::new().with_scale_settings(settings)
    }

    pub fn with_step_size(mut self, step_size: f32) -> Self {
        self.step_size = step_size;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_interaction_radius(mut self, interaction_radius: f32) -> Self {
        self.interaction_radius = interaction_radius;
        self
    }

    pub fn with_min_radius(mut self, min_radius: f32) -> Self {
        self.min_radius = min_radius;
        self
    }

    pub fn with_max_radius(mut self, max_radius: f32) -> Self {
        self.max_radius = max_radius;
        self
    }

    pub fn with_steps(mut self, n_steps: usize) -> Self {
        self.n_steps = n_steps;
        self
    }

    pub fn with_entropy_weighted(mut self, enabled: bool) -> Self {
        self.entropy_weighted = enabled;
        self
    }

    pub fn with_scale_settings(mut self, settings: ScaleSettings) -> Self {
        self.min_radius = settings.min_radius;
        self.max_radius = settings.max_radius;
        self.interaction_radius = settings.interaction_radius;
        self.step_size = settings.step_size;
        self.temperature = settings.temperature;
        self.n_steps = settings.n_steps;
        self
    }

    /// Run sphere optimization on embeddings
    ///
    /// # Arguments
    /// * `embeddings` - [N, D] embedding matrix
    /// * `prominence` - [N] prominence scores (embedding norms)
    /// * `entropies` - Optional [N] entropy values for entropy-weighted mode
    ///
    /// # Returns
    /// Optimized spherical coordinates
    pub fn optimize(
        &self,
        embeddings: ArrayView2<f32>,
        prominence: ArrayView1<f32>,
        entropies: Option<ArrayView1<f32>>,
    ) -> SphericalCoords {
        let n = embeddings.nrows();

        // Compute similarity matrix
        let similarity = cosine_similarity_matrix(embeddings);

        // Compute ideal radii (entropy-weighted if enabled)
        let ideal_radii = if self.entropy_weighted {
            compute_ideal_radii(prominence, entropies, self.min_radius, self.max_radius)
        } else {
            compute_ideal_radii(prominence, None, self.min_radius, self.max_radius)
        };

        // Initialize positions with random angles
        let norms: Array1<f32> = embeddings
            .axis_iter(Axis(0))
            .map(|row| row.iter().map(|&x| x * x).sum::<f32>().sqrt())
            .collect();

        let mut rng = fastrand::Rng::new();
        rng.seed(42);

        let mut coords = SphericalCoords {
            r: norms,
            theta: (0..n).map(|_| rng.f32() * std::f32::consts::PI).collect(),
            phi: (0..n)
                .map(|_| rng.f32() * 2.0 * std::f32::consts::PI)
                .collect(),
        };

        // Run Langevin dynamics
        for _step in 0..self.n_steps {
            coords = langevin_step(
                &coords,
                ideal_radii.view(),
                similarity.view(),
                self.interaction_radius,
                self.step_size,
                self.temperature,
            );
        }

        coords
    }
}

/// Result of sphere optimization with metadata
#[derive(Clone, Debug)]
pub struct SphereResult {
    pub coords: SphericalCoords,
    pub cartesian: Array2<f32>,
    pub ideal_radii: Array1<f32>,
}

impl SphereResult {
    pub fn new(coords: SphericalCoords, ideal_radii: Array1<f32>) -> Self {
        let cartesian = coords.to_cartesian();
        Self {
            coords,
            cartesian,
            ideal_radii,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical_conversion_roundtrip() {
        let coords = SphericalCoords {
            r: Array1::from_vec(vec![1.0, 2.0, 3.0]),
            theta: Array1::from_vec(vec![
                std::f32::consts::PI / 2.0,
                std::f32::consts::PI / 4.0,
                std::f32::consts::PI / 3.0,
            ]),
            phi: Array1::from_vec(vec![0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI]),
        };

        let cart = coords.to_cartesian();
        let recovered = SphericalCoords::from_cartesian(cart.view());

        for i in 0..3 {
            assert!((coords.r[i] - recovered.r[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let embeddings =
            Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();

        let sim = cosine_similarity_matrix(embeddings.view());
        assert!((sim[[0, 1]] - 1.0).abs() < 1e-5); // Identical vectors
        assert!(sim[[0, 0]].abs() < 1e-5); // Self-similarity zeroed
    }

    #[test]
    fn test_entropy_weighted_radii() {
        // Use values where entropy changes the ranking order
        // Plain ranking: [3.0, 2.5, 2.0] → idx 0 highest
        // With entropy: 3.0 * 1.1 = 3.3, 2.5 * 2.0 = 5.0, 2.0 * 1.0 = 2.0
        // Weighted ranking: [5.0, 3.3, 2.0] → idx 1 now highest
        let prominence = Array1::from_vec(vec![3.0, 2.5, 2.0]);
        let entropies = Array1::from_vec(vec![0.1, 1.0, 0.0]);  // High entropy on idx 1

        let radii_plain = compute_ideal_radii(prominence.view(), None, 32.0, 512.0);
        let radii_weighted =
            compute_ideal_radii(prominence.view(), Some(entropies.view()), 32.0, 512.0);

        // With entropy weighting, ordering should change
        // idx 0 had smallest radius (highest prominence), but with entropy
        // idx 1 should now have smallest radius (highest weighted score)
        assert!(radii_plain != radii_weighted, "Entropy weighting should change radii");
        assert!(radii_weighted[1] < radii_weighted[0], 
            "High entropy idx 1 should get smaller radius than idx 0");
    }

    #[test]
    fn test_optimizer_basic() {
        let embeddings = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let prominence = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.732]);

        let optimizer = SphereOptimizer::new().with_steps(10);
        let result = optimizer.optimize(embeddings.view(), prominence.view(), None);

        assert_eq!(result.r.len(), 4);
        assert!(result.r.iter().all(|&r| r > 0.0));
    }
}


