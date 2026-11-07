use crate::rng::FastRng;
use crate::types::Topology;

/// Symplectic BAOAB Langevin dynamics integrator for canonical NVT sampling.
/// References: Leimkuhler & Matthews (2013), "Rational Construction of Stochastic Numerical Methods".
pub struct LangevinIntegrator {
    pub friction_gamma: f64, // Collision friction coefficient in 1/ps
    pub target_temp: f64,    // Target temperature in Kelvin
}

impl LangevinIntegrator {
    pub fn new(friction_gamma: f64, target_temp: f64) -> Self {
        Self {
            friction_gamma,
            target_temp,
        }
    }

    /// Step 1: Execute B and A components (Half-velocity kick + Half-position drift)
    /// B: v <- v + 0.5 * dt * (F / m)
    /// A: r <- r + 0.5 * dt * v
    pub fn step_ba(&self, topology: &mut Topology, dt: f64) {
        let dt_half = 0.5 * dt;

        for p in &mut topology.particles {
            if p.mass > 0.0 {
                let accel = p.force * (1.0 / p.mass);
                p.velocity += accel * dt_half;
                p.position += p.velocity * dt_half;
            }
        }
    }

    /// Step 2: Execute O component (Ornstein-Uhlenbeck stochastic thermostat friction + noise)
    /// v <- c1 * v + c2 * sqrt(k_B * T / m) * GaussianNoise
    /// where c1 = exp(-gamma * dt), c2 = sqrt(1 - c1^2)
    pub fn step_o(&self, topology: &mut Topology, dt: f64, rng: &mut FastRng) {
        const KB: f64 = 8.314462618e-3; // kJ / (mol * K)
        let c1 = (-self.friction_gamma * dt).exp();
        let c2 = (1.0 - c1 * c1).max(0.0).sqrt();

        for p in &mut topology.particles {
            if p.mass > 0.0 {
                let thermal_velocity = (KB * self.target_temp / p.mass).sqrt();
                let noise = rng.next_gaussian_vector3d(thermal_velocity);
                p.velocity = p.velocity * c1 + noise * c2;
            }
        }
    }

    /// Step 3: Execute second A component (Second half-position drift)
    /// A: r <- r + 0.5 * dt * v
    pub fn step_a(&self, topology: &mut Topology, dt: f64) {
        let dt_half = 0.5 * dt;

        for p in &mut topology.particles {
            if p.mass > 0.0 {
                p.position += p.velocity * dt_half;
            }
        }
    }

    /// Step 4: Execute final B component after force calculation
    /// B: v <- v + 0.5 * dt * (F / m)
    pub fn step_b(&self, topology: &mut Topology, dt: f64) {
        let dt_half = 0.5 * dt;

        for p in &mut topology.particles {
            if p.mass > 0.0 {
                let accel = p.force * (1.0 / p.mass);
                p.velocity += accel * dt_half;
            }
        }
    }
}
