use crate::types::Topology;

/// Nosé-Hoover thermostat for deterministic symplectic canonical NVT sampling.
/// References: Martyna, Klein, Tuckerman (1992), "Nosé-Hoover chains: The canonical ensemble via continuous dynamics".
#[derive(Debug, Clone)]
pub struct NoseHooverThermostat {
    pub target_temp: f64,
    pub tau: f64,          // Relaxation time scale in ps (e.g. 0.1 to 0.5 ps)
    pub xi: f64,           // Thermostat friction variable
    pub v_xi: f64,         // Thermostat velocity
    pub q_mass: f64,       // Inertial mass parameter Q
}

impl NoseHooverThermostat {
    pub fn new(target_temp: f64, tau: f64, dof: f64) -> Self {
        const KB: f64 = 8.314462618e-3; // kJ / (mol * K)
        let q_mass = dof * KB * target_temp * tau * tau;
        Self {
            target_temp,
            tau,
            xi: 0.0,
            v_xi: 0.0,
            q_mass,
        }
    }

    /// Propagate thermostat velocity: v_xi <- v_xi + dt_half * G / Q
    pub fn step_thermostat_velocity(&mut self, current_ke: f64, dof: f64, dt_half: f64) {
        const KB: f64 = 8.314462618e-3;
        let g = 2.0 * current_ke - dof * KB * self.target_temp;
        let accel_xi = g / self.q_mass;
        self.v_xi += accel_xi * dt_half;
    }

    /// Propagate thermostat coordinate: xi <- xi + dt_half * v_xi
    pub fn step_thermostat_coord(&mut self, dt_half: f64) {
        self.xi += self.v_xi * dt_half;
    }

    /// Scale particle velocities by exp(-v_xi * dt_half)
    pub fn scale_velocities(&mut self, topology: &mut Topology, dt_half: f64) {
        let scale = (-self.v_xi * dt_half).exp();
        for p in &mut topology.particles {
            p.velocity *= scale;
        }
    }

    /// Conserved quantity (Hamiltonian / Pseudo-energy) in the extended phase space:
    /// H_extended = E_kin + E_pot + 0.5 * Q * v_xi^2 + N_dof * k_B * T * xi
    pub fn extended_conserved_energy(&self, total_energy: f64, dof: f64) -> f64 {
        const KB: f64 = 8.314462618e-3;
        let thermostat_ke = 0.5 * self.q_mass * self.v_xi * self.v_xi;
        let thermostat_pe = dof * KB * self.target_temp * self.xi;
        total_energy + thermostat_ke + thermostat_pe
    }
}
