use aura_dynamics::{
    forcefield::ForceField,
    types::{AtomType, PeriodicBox, Topology, Vector3D},
    Ensemble, SimulationEngine,
};

#[test]
fn test_nve_energy_conservation() {
    let pbc = PeriodicBox::cubic(20.0);
    let mut topology = Topology::new(pbc);
    let ar_type = topology.add_atom_type(AtomType::argon());

    // 8 particles arranged on a small grid
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                let pos = Vector3D::new(8.0 + x as f64 * 3.8, 8.0 + y as f64 * 3.8, 8.0 + z as f64 * 3.8);
                topology.add_particle(ar_type, pos);
            }
        }
    }

    // Give small initial velocities with zero net momentum
    topology.particles[0].velocity = Vector3D::new(0.05, -0.02, 0.03);
    topology.particles[1].velocity = Vector3D::new(-0.05, 0.02, -0.03);
    topology.particles[2].velocity = Vector3D::new(0.02, 0.04, -0.01);
    topology.particles[3].velocity = Vector3D::new(-0.02, -0.04, 0.01);

    let ff = ForceField::standard(8.0);
    let mut sim = SimulationEngine::new(topology, ff, Ensemble::NVE, 42);
    let initial_energy = sim.initialize();
    let e0 = initial_energy.total();

    let dt = 0.001; // 1 fs
    let mut max_energy_drift = 0.0;

    for _ in 0..300 {
        let state = sim.step(dt);
        let current_e = state.conserved_quantity;
        let drift = (current_e - e0).abs() / e0.abs().max(1.0);
        if drift > max_energy_drift {
            max_energy_drift = drift;
        }
    }

    println!("NVE max relative energy drift: {:.6e}", max_energy_drift);
    // Symplectic Velocity-Verlet must conserve energy with relative drift < 1e-3
    assert!(
        max_energy_drift < 1e-3,
        "NVE energy drift too high: {}",
        max_energy_drift
    );
}

#[test]
fn test_bonded_harmonic_forces() {
    let pbc = PeriodicBox::cubic(30.0);
    let mut topology = Topology::new(pbc);
    let c_type = topology.add_atom_type(AtomType::new("C", 12.011, 0.0, 3.4, 0.36));

    let idx1 = topology.add_particle(c_type, Vector3D::new(10.0, 10.0, 10.0));
    let idx2 = topology.add_particle(c_type, Vector3D::new(11.5, 10.0, 10.0)); // stretched from 1.2 to 1.5

    // Equilibrium length 1.2 A, k_b = 1000.0 kJ/(mol * A^2)
    topology.add_bond(idx1, idx2, 1000.0, 1.2);

    let mut ff = ForceField::standard(10.0);
    let energy = ff.compute_forces(&mut topology);

    // Stretched by dx = 0.3 A
    // Expected energy: 0.5 * 1000 * 0.3^2 = 45.0 kJ/mol
    let expected_energy = 0.5 * 1000.0 * 0.3 * 0.3;
    assert!((energy.bond - expected_energy).abs() < 1e-6);

    // Force magnitude: -k_b * (r - r_0) = -1000 * 0.3 = -300.0
    // F2x should be -300.0 (restoring towards x=10), F1x should be +300.0
    assert!((topology.particles[idx2].force.x - (-300.0)).abs() < 1e-6);
    assert!((topology.particles[idx1].force.x - 300.0).abs() < 1e-6);
}
