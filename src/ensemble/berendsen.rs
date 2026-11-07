use crate::types::Topology;

/// Berendsen weak-coupling thermostat for rapid thermal equilibration.
pub struct BerendsenThermostat {
    pub target_temp: f64,
    pub tau_t: f64, // Coupling time constant in ps (e.g. 0.1 ps)
}

impl BerendsenThermostat {
    pub fn new(target_temp: f64, tau_t: f64) -> Self {
        Self { target_temp, tau_t }
    }

    /// Rescale velocities according to the weak coupling equation:
    /// lambda = sqrt(1 + (dt / tau_T) * (T_0 / T - 1))
    pub fn apply(&self, topology: &mut Topology, dt: f64) -> f64 {
        let current_temp = topology.instantaneous_temperature();
        if current_temp < 1e-6 {
            return 1.0;
        }

        let ratio = self.target_temp / current_temp;
        let lambda_sq = 1.0 + (dt / self.tau_t) * (ratio - 1.0);
        let lambda = lambda_sq.max(0.01).sqrt();

        for p in &mut topology.particles {
            p.velocity *= lambda;
        }

        lambda
    }
}

/// Berendsen isotropic barostat for NPT simulations.
pub struct BerendsenBarostat {
    pub target_pressure_bar: f64,
    pub tau_p: f64,          // Pressure coupling time constant (ps)
    pub compressibility: f64, // Isothermal compressibility (bar^-1, e.g. 4.5e-5 for water)
}

impl BerendsenBarostat {
    pub fn new(target_pressure_bar: f64, tau_p: f64, compressibility: f64) -> Self {
        Self {
            target_pressure_bar,
            tau_p,
            compressibility,
        }
    }

    /// Apply isotropic coordinate and box length rescaling based on current virial pressure.
    pub fn apply(&self, topology: &mut Topology, current_pressure_bar: f64, dt: f64) -> f64 {
        let delta_p = self.target_pressure_bar - current_pressure_bar;
        let mu_cubed = 1.0 - (dt * self.compressibility / self.tau_p) * delta_p;
        let mu = mu_cubed.clamp(0.9, 1.1).cbrt();

        // Scale box
        topology.pbc.lx *= mu;
        topology.pbc.ly *= mu;
        topology.pbc.lz *= mu;

        // Scale particle coordinates
        for p in &mut topology.particles {
            p.position *= mu;
        }

        mu
    }
}
