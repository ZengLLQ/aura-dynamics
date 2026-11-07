use crate::types::Vector3D;

/// Velocity Autocorrelation Function (VACF) and Vibrational Density of States (VDOS).
#[derive(Debug, Clone)]
pub struct VelocityAutocorrelation {
    pub velocity_history: Vec<Vec<Vector3D>>,
    pub dt: f64,
}

impl VelocityAutocorrelation {
    pub fn new(dt: f64) -> Self {
        Self {
            velocity_history: Vec::new(),
            dt,
        }
    }

    /// Record snapshot of particle velocities at the current timestep.
    pub fn record_frame(&mut self, velocities: &[Vector3D]) {
        self.velocity_history.push(velocities.to_vec());
    }

    /// Compute normalized VACF C_v(tau) for lag times up to max_lag_steps.
    pub fn compute_vacf(&self, max_lag_steps: usize) -> Vec<(f64, f64)> {
        let total_frames = self.velocity_history.len();
        if total_frames < 2 || max_lag_steps == 0 {
            return Vec::new();
        }

        let lag_limit = max_lag_steps.min(total_frames - 1);
        let num_particles = self.velocity_history[0].len();
        let mut vacf = Vec::with_capacity(lag_limit + 1);

        // Compute C_v(0) normalization factor
        let mut norm_sum = 0.0;
        for frame in &self.velocity_history {
            for v in frame {
                norm_sum += v.norm_sq();
            }
        }
        let c0 = norm_sum / (total_frames * num_particles) as f64;
        if c0 < 1e-14 {
            return Vec::new();
        }

        for lag in 0..=lag_limit {
            let mut dot_sum = 0.0;
            let count = total_frames - lag;

            for t in 0..count {
                let v_t = &self.velocity_history[t];
                let v_t_lag = &self.velocity_history[t + lag];

                for i in 0..num_particles {
                    dot_sum += v_t[i].dot(v_t_lag[i]);
                }
            }

            let c_tau = dot_sum / (count * num_particles) as f64;
            let normalized_c = c_tau / c0;
            let time_lag = lag as f64 * self.dt;
            vacf.push((time_lag, normalized_c));
        }

        vacf
    }

    /// Compute Vibrational Density of States VDOS(omega) via Cosine Fourier Transform of VACF.
    pub fn compute_vdos(&self, vacf: &[(f64, f64)], omega_max: f64, num_points: usize) -> Vec<(f64, f64)> {
        let mut vdos = Vec::with_capacity(num_points);
        let d_omega = omega_max / num_points as f64;

        for i in 0..num_points {
            let omega = i as f64 * d_omega;
            let mut integral = 0.0;

            for window in vacf.windows(2) {
                let (t0, c0) = window[0];
                let (t1, c1) = window[1];
                let dt = t1 - t0;

                // Trapezoidal rule integration: C(t) * cos(omega * t)
                let f0 = c0 * (omega * t0).cos();
                let f1 = c1 * (omega * t1).cos();
                integral += 0.5 * (f0 + f1) * dt;
            }

            vdos.push((omega, integral.max(0.0)));
        }

        vdos
    }
}
