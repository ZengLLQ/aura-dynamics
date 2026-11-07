use crate::types::Topology;

/// Symplectic Velocity-Verlet numerical time integration algorithm.
pub struct VelocityVerlet;

impl VelocityVerlet {
    /// Perform the first half-step: update positions and half-step velocities.
    /// r(t + dt) = r(t) + v(t)*dt + 0.5 * (F(t)/m) * dt^2
    /// v(t + 0.5*dt) = v(t) + 0.5 * (F(t)/m) * dt
    pub fn step_half_position(topology: &mut Topology, dt: f64) {
        let dt_half = 0.5 * dt;

        for p in &mut topology.particles {
            if p.mass > 0.0 {
                let inv_m = 1.0 / p.mass;
                let accel = p.force * inv_m;

                p.position += p.velocity * dt + accel * (0.5 * dt * dt);
                p.velocity += accel * dt_half;
            }
        }
    }

    /// Perform the second half-step: update final velocities after force calculation.
    /// v(t + dt) = v(t + 0.5*dt) + 0.5 * (F(t + dt)/m) * dt
    pub fn step_half_velocity(topology: &mut Topology, dt: f64) {
        let dt_half = 0.5 * dt;

        for p in &mut topology.particles {
            if p.mass > 0.0 {
                let inv_m = 1.0 / p.mass;
                let accel = p.force * inv_m;
                p.velocity += accel * dt_half;
            }
        }
    }
}
