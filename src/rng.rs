//! High-performance deterministic pseudorandom number generator (Xoroshiro128++)
//! and Gaussian Box-Muller distribution sampling for stochastic thermostats.

use crate::types::{Topology, Vector3D};

pub struct FastRng {
    s: [u64; 2],
}

impl FastRng {
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut rng = Self {
            s: [seed.wrapping_add(0x9E3779B97F4A7C15), seed.wrapping_mul(0xBF58476D1CE4E5B9)],
        };
        // Warm up state
        for _ in 0..16 {
            rng.next_u64();
        }
        rng
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s0 = self.s[0];
        let mut s1 = self.s[1];
        let result = s0.wrapping_add(s1).rotate_left(17).wrapping_add(s0);

        s1 ^= s0;
        self.s[0] = s0.rotate_left(49) ^ s1 ^ (s1 << 21);
        self.s[1] = s1.rotate_left(28);

        result
    }

    /// Uniform float in [0.0, 1.0).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal Gaussian random variable N(0, 1) via Box-Muller transform.
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// 3D Gaussian random vector with specified standard deviation sigma.
    pub fn next_gaussian_vector3d(&mut self, sigma: f64) -> Vector3D {
        Vector3D::new(
            self.next_gaussian() * sigma,
            self.next_gaussian() * sigma,
            self.next_gaussian() * sigma,
        )
    }

    /// Assign velocities according to the Maxwell-Boltzmann distribution at target temperature T,
    /// removing net center-of-mass linear momentum.
    pub fn initialize_maxwell_boltzmann(&mut self, topology: &mut Topology, target_temp_k: f64) {
        const KB: f64 = 8.314462618e-3; // kJ / (mol * K)
        let mut total_p = Vector3D::ZERO;
        let mut total_m = 0.0;

        for p in &mut topology.particles {
            if p.mass > 0.0 {
                let sigma = (KB * target_temp_k / p.mass).sqrt();
                let v = self.next_gaussian_vector3d(sigma);
                p.velocity = v;
                total_p += v * p.mass;
                total_m += p.mass;
            }
        }

        // Subtract center-of-mass velocity drift
        if total_m > 0.0 {
            let v_com = total_p / total_m;
            for p in &mut topology.particles {
                if p.mass > 0.0 {
                    p.velocity -= v_com;
                }
            }
        }

        // Rescale to exact target temperature
        let current_temp = topology.instantaneous_temperature();
        if current_temp > 1e-6 {
            let scale = (target_temp_k / current_temp).sqrt();
            for p in &mut topology.particles {
                p.velocity *= scale;
            }
        }
    }
}
