use aura_dynamics::{
    ensemble::NoseHooverThermostat,
    forcefield::ForceField,
    integrators::LangevinIntegrator,
    rng::FastRng,
    types::{AtomType, PeriodicBox, Topology, Vector3D},
    Ensemble, SimulationEngine,
};

#[test]
fn test_langevin_temperature_control() {
    let pbc = PeriodicBox::cubic(20.0);
    let mut topology = Topology::new(pbc);
    let ar_type = topology.add_atom_type(AtomType::argon());

    // 27 Argon particles
    for x in 0..3 {
        for y in 0..3 {
            for z in 0..3 {
                let pos = Vector3D::new(5.0 + x as f64 * 3.6, 5.0 + y as f64 * 3.6, 5.0 + z as f64 * 3.6);
                topology.add_particle(ar_type, pos);
            }
        }
    }

    let target_temp = 120.0;
    let mut rng = FastRng::seed_from_u64(999);
    // Start colder at 50 K
    rng.initialize_maxwell_boltzmann(&mut topology, 50.0);

    let ff = ForceField::standard(8.0);
    let langevin = LangevinIntegrator::new(5.0, target_temp); // Strong coupling for rapid equilibration
    let mut sim = SimulationEngine::new(topology, ff, Ensemble::NVTLangevin(langevin), 999);
    sim.initialize();

    let dt = 0.002;
    let mut temp_samples = Vec::new();

    for step in 1..=500 {
        let state = sim.step(dt);
        // Sample temperature in the second half of the trajectory
        if step > 250 {
            temp_samples.push(state.temperature_k);
        }
    }

    let avg_temp: f64 = temp_samples.iter().sum::<f64>() / temp_samples.len() as f64;
    println!("Langevin target: {:.1} K, measured average: {:.1} K", target_temp, avg_temp);

    // Should equilibrate within 20% of target temperature
    assert!(
        (avg_temp - target_temp).abs() < 25.0,
        "Langevin average temperature out of bounds: {}",
        avg_temp
    );
}

#[test]
fn test_nose_hoover_thermostat() {
    let pbc = PeriodicBox::cubic(25.0);
    let mut topology = Topology::new(pbc);
    let ar_type = topology.add_atom_type(AtomType::argon());

    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                let pos = Vector3D::new(10.0 + x as f64 * 4.0, 10.0 + y as f64 * 4.0, 10.0 + z as f64 * 4.0);
                topology.add_particle(ar_type, pos);
            }
        }
    }

    let target_temp = 100.0;
    let mut rng = FastRng::seed_from_u64(777);
    rng.initialize_maxwell_boltzmann(&mut topology, target_temp);

    let dof = (3 * topology.num_particles()).saturating_sub(3) as f64;
    let nh = NoseHooverThermostat::new(target_temp, 0.1, dof);
    let ff = ForceField::standard(8.0);

    let mut sim = SimulationEngine::new(topology, ff, Ensemble::NVTNoseHoover(nh), 777);
    sim.initialize();

    let dt = 0.001;
    let mut state = sim.step(dt);
    let initial_conserved = state.conserved_quantity;

    for _ in 0..200 {
        state = sim.step(dt);
    }

    let final_conserved = state.conserved_quantity;
    let drift = (final_conserved - initial_conserved).abs() / initial_conserved.abs().max(1.0);
    println!("Nose-Hoover extended energy drift: {:.6e}", drift);

    // Extended Hamiltonian should be conserved within numerical tolerance for 8 particles
    assert!(
        drift < 0.05,
        "Nose-Hoover extended energy drift too large: {}",
        drift
    );
}
