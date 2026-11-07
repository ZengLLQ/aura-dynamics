use super::colvars::CollectiveVariable;
use crate::types::Topology;

/// Gaussian hill deposited in the collective variable space.
#[derive(Debug, Clone, Copy)]
pub struct GaussianHill {
    pub center: f64, // Position s_t
    pub height: f64, // Well-tempered scaled height W_t
    pub sigma: f64,  // Width parameter
}

/// Well-Tempered Metadynamics engine for free energy landscape exploration.
/// Reference: Barducci, Bussi, Parrinello (2008), "Well-Tempered Metadynamics: A Smoothly Converging and Tunable Free-Energy Method".
pub struct Metadynamics {
    pub colvar: Box<dyn CollectiveVariable>,
    pub initial_height: f64, // Initial hill height W_0 (kJ/mol)
    pub sigma: f64,          // Gaussian width sigma
    pub pace: usize,         // Deposition interval (steps)
    pub bias_factor: f64,    // gamma = (T + DeltaT) / DeltaT (e.g. 10.0)
    pub temperature: f64,    // System temperature T (Kelvin)
    pub hills: Vec<GaussianHill>,
    pub step_counter: usize,
}

impl Metadynamics {
    pub fn new(
        colvar: Box<dyn CollectiveVariable>,
        initial_height: f64,
        sigma: f64,
        pace: usize,
        bias_factor: f64,
        temperature: f64,
    ) -> Self {
        assert!(bias_factor > 1.0, "Bias factor gamma must be > 1.0");
        Self {
            colvar,
            initial_height,
            sigma,
            pace,
            bias_factor,
            temperature,
            hills: Vec::new(),
            step_counter: 0,
        }
    }

    /// Calculate the accumulated bias potential V_bias(s) at collective variable value s.
    pub fn evaluate_bias_at(&self, s: f64) -> f64 {
        let mut bias = 0.0;
        let two_sigma_sq = 2.0 * self.sigma * self.sigma;

        for hill in &self.hills {
            let diff = s - hill.center;
            bias += hill.height * (-diff * diff / two_sigma_sq).exp();
        }

        bias
    }

    /// Calculate the derivative of the bias potential -dV_bias / ds at collective variable value s.
    pub fn evaluate_bias_force_at(&self, s: f64) -> f64 {
        let mut force_s = 0.0;
        let sigma_sq = self.sigma * self.sigma;
        let two_sigma_sq = 2.0 * sigma_sq;

        for hill in &self.hills {
            let diff = s - hill.center;
            let gaussian = hill.height * (-diff * diff / two_sigma_sq).exp();
            // -dV/ds = sum W * (s - s_0)/sigma^2 * exp(...)
            force_s += (diff / sigma_sq) * gaussian;
        }

        force_s
    }

    /// Apply metadynamics bias force to atomic coordinates and deposit hills when step % pace == 0.
    /// Returns (current_cv_val, current_bias_energy).
    pub fn apply(&mut self, topology: &mut Topology) -> (f64, f64) {
        self.step_counter += 1;
        let s = self.colvar.calculate(topology);
        let bias_energy = self.evaluate_bias_at(s);

        // Deposit hill at scheduled intervals
        if self.step_counter % self.pace == 0 {
            const KB: f64 = 8.314462618e-3; // kJ / (mol * K)
            // DeltaT = T * (gamma - 1)
            let delta_t = self.temperature * (self.bias_factor - 1.0);
            let k_delta_t = KB * delta_t;

            // Well-tempered height: W(t) = W_0 * exp(-V_bias(s, t) / (k_B * DeltaT))
            let current_v = self.evaluate_bias_at(s);
            let w_t = self.initial_height * (-current_v / k_delta_t).exp();

            self.hills.push(GaussianHill {
                center: s,
                height: w_t,
                sigma: self.sigma,
            });
        }

        // Compute force on CV: F_s = -dV_bias / ds
        let f_s = self.evaluate_bias_force_at(s);

        // Project force back to atomic Cartesian coordinates: F_i += F_s * grad_i(s)
        let grads = self.colvar.gradients(topology);
        for (i, grad) in grads.iter().enumerate() {
            topology.particles[i].force += *grad * f_s;
        }

        (s, bias_energy)
    }

    /// Reconstruct the 1D Free Energy Surface (FES): F(s) = -(gamma / (gamma - 1)) * V_bias(s)
    pub fn reconstruct_free_energy_surface(
        &self,
        s_min: f64,
        s_max: f64,
        num_grid_points: usize,
    ) -> Vec<(f64, f64)> {
        let mut fes = Vec::with_capacity(num_grid_points);
        let ds = (s_max - s_min) / (num_grid_points - 1) as f64;
        let gamma_factor = self.bias_factor / (self.bias_factor - 1.0);

        for i in 0..num_grid_points {
            let s = s_min + i as f64 * ds;
            let v_bias = self.evaluate_bias_at(s);
            let free_energy = -gamma_factor * v_bias;
            fes.push((s, free_energy));
        }

        // Normalize minimum free energy to 0.0 kJ/mol
        if let Some(min_val) = fes.iter().map(|&(_, f)| f).min_by(|a, b| a.partial_cmp(b).unwrap()) {
            for (_, f) in &mut fes {
                *f -= min_val;
            }
        }

        fes
    }
}
