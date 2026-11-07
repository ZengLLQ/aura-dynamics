use crate::ensemble::{BerendsenThermostat, NoseHooverThermostat};
use crate::enhanced::Metadynamics;
use crate::forcefield::{EnergyBreakdown, ForceField};
use crate::integrators::{LangevinIntegrator, VelocityVerlet};
use crate::rng::FastRng;
use crate::types::Topology;

/// Simulation ensemble selection.
pub enum Ensemble {
    NVE,
    NVTLangevin(LangevinIntegrator),
    NVTNoseHoover(NoseHooverThermostat),
    NVTBerendsen(BerendsenThermostat),
}

/// Simulation snapshot state report.
#[derive(Debug, Clone, Copy)]
pub struct SimulationState {
    pub step: usize,
    pub time_ps: f64,
    pub temperature_k: f64,
    pub energy: EnergyBreakdown,
    pub conserved_quantity: f64, // E_total in NVE, extended H in Nose-Hoover
    pub cv_value: Option<f64>,
    pub bias_energy: Option<f64>,
}

/// High-level Molecular Dynamics Simulation Engine.
pub struct SimulationEngine {
    pub topology: Topology,
    pub forcefield: ForceField,
    pub ensemble: Ensemble,
    pub metadynamics: Option<Metadynamics>,
    pub rng: FastRng,
    pub step: usize,
    pub time_ps: f64,
}

impl SimulationEngine {
    pub fn new(
        topology: Topology,
        forcefield: ForceField,
        ensemble: Ensemble,
        rng_seed: u64,
    ) -> Self {
        Self {
            topology,
            forcefield,
            ensemble,
            metadynamics: None,
            rng: FastRng::seed_from_u64(rng_seed),
            step: 0,
            time_ps: 0.0,
        }
    }

    pub fn with_metadynamics(mut self, metadynamics: Metadynamics) -> Self {
        self.metadynamics = Some(metadynamics);
        self
    }

    /// Initialize forces before starting simulation steps.
    pub fn initialize(&mut self) -> EnergyBreakdown {
        self.forcefield.compute_forces(&mut self.topology)
    }

    /// Advance the simulation by a single timestep dt (ps).
    pub fn step(&mut self, dt: f64) -> SimulationState {
        self.step += 1;
        self.time_ps += dt;

        let mut cv_val = None;
        let mut bias_e = None;

        match &mut self.ensemble {
            Ensemble::NVE => {
                // Velocity-Verlet Step 1: Half-position and half-velocity
                VelocityVerlet::step_half_position(&mut self.topology, dt);

                // Compute new forces at updated positions
                let mut energy = self.forcefield.compute_forces(&mut self.topology);

                // Apply metadynamics bias force if active
                if let Some(meta) = &mut self.metadynamics {
                    let (s, b_e) = meta.apply(&mut self.topology);
                    cv_val = Some(s);
                    bias_e = Some(b_e);
                }

                // Velocity-Verlet Step 2: Final half-velocity
                VelocityVerlet::step_half_velocity(&mut self.topology, dt);

                energy.kinetic = self.topology.particles.iter().map(|p| p.kinetic_energy()).sum();
                let temp = self.topology.instantaneous_temperature();
                let conserved = energy.total() + bias_e.unwrap_or(0.0);

                SimulationState {
                    step: self.step,
                    time_ps: self.time_ps,
                    temperature_k: temp,
                    energy,
                    conserved_quantity: conserved,
                    cv_value: cv_val,
                    bias_energy: bias_e,
                }
            }
            Ensemble::NVTLangevin(langevin) => {
                // BAOAB Splitting: B then A
                langevin.step_ba(&mut self.topology, dt);

                // O: Stochastic Ornstein-Uhlenbeck thermal kick
                langevin.step_o(&mut self.topology, dt, &mut self.rng);

                // A: Second half-position drift
                langevin.step_a(&mut self.topology, dt);

                // Compute forces at new coordinates
                let mut energy = self.forcefield.compute_forces(&mut self.topology);

                // Apply metadynamics bias
                if let Some(meta) = &mut self.metadynamics {
                    let (s, b_e) = meta.apply(&mut self.topology);
                    cv_val = Some(s);
                    bias_e = Some(b_e);
                }

                // Final B: Velocity kick
                langevin.step_b(&mut self.topology, dt);

                energy.kinetic = self.topology.particles.iter().map(|p| p.kinetic_energy()).sum();
                let temp = self.topology.instantaneous_temperature();

                SimulationState {
                    step: self.step,
                    time_ps: self.time_ps,
                    temperature_k: temp,
                    energy,
                    conserved_quantity: energy.total() + bias_e.unwrap_or(0.0),
                    cv_value: cv_val,
                    bias_energy: bias_e,
                }
            }
            Ensemble::NVTNoseHoover(nh) => {
                let dt_half = 0.5 * dt;
                let dof = (3 * self.topology.num_particles()).saturating_sub(3).max(1) as f64;

                // 1. Advance thermostat coordinate and velocity by dt / 2
                let ke0: f64 = self.topology.particles.iter().map(|p| p.kinetic_energy()).sum();
                nh.step_thermostat_velocity(ke0, dof, dt_half);
                nh.step_thermostat_coord(dt_half);

                // 2. Scale particle velocities by exp(-v_xi * dt / 2)
                nh.scale_velocities(&mut self.topology, dt_half);

                // 3. Velocity-Verlet position drift
                VelocityVerlet::step_half_position(&mut self.topology, dt);

                // 4. Compute molecular forces
                let mut energy = self.forcefield.compute_forces(&mut self.topology);

                // Metadynamics bias
                if let Some(meta) = &mut self.metadynamics {
                    let (s, b_e) = meta.apply(&mut self.topology);
                    cv_val = Some(s);
                    bias_e = Some(b_e);
                }

                // 5. Half-velocity update from forces
                VelocityVerlet::step_half_velocity(&mut self.topology, dt);

                // 6. Scale particle velocities by exp(-v_xi * dt / 2)
                nh.scale_velocities(&mut self.topology, dt_half);

                // 7. Advance thermostat coordinate and velocity by dt / 2
                let ke1: f64 = self.topology.particles.iter().map(|p| p.kinetic_energy()).sum();
                nh.step_thermostat_coord(dt_half);
                nh.step_thermostat_velocity(ke1, dof, dt_half);

                energy.kinetic = ke1;
                let temp = self.topology.instantaneous_temperature();
                let conserved = nh.extended_conserved_energy(energy.total(), dof) + bias_e.unwrap_or(0.0);

                SimulationState {
                    step: self.step,
                    time_ps: self.time_ps,
                    temperature_k: temp,
                    energy,
                    conserved_quantity: conserved,
                    cv_value: cv_val,
                    bias_energy: bias_e,
                }
            }
            Ensemble::NVTBerendsen(berendsen) => {
                VelocityVerlet::step_half_position(&mut self.topology, dt);
                let mut energy = self.forcefield.compute_forces(&mut self.topology);

                if let Some(meta) = &mut self.metadynamics {
                    let (s, b_e) = meta.apply(&mut self.topology);
                    cv_val = Some(s);
                    bias_e = Some(b_e);
                }

                VelocityVerlet::step_half_velocity(&mut self.topology, dt);
                berendsen.apply(&mut self.topology, dt);

                energy.kinetic = self.topology.particles.iter().map(|p| p.kinetic_energy()).sum();
                let temp = self.topology.instantaneous_temperature();

                SimulationState {
                    step: self.step,
                    time_ps: self.time_ps,
                    temperature_k: temp,
                    energy,
                    conserved_quantity: energy.total() + bias_e.unwrap_or(0.0),
                    cv_value: cv_val,
                    bias_energy: bias_e,
                }
            }
        }
    }
}
