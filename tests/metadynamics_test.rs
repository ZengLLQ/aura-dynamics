use aura_dynamics::{
    enhanced::{DistanceCV, Metadynamics, RadiusOfGyrationCV, CollectiveVariable},
    forcefield::ForceField,
    integrators::LangevinIntegrator,
    types::{AtomType, PeriodicBox, Topology, Vector3D},
    Ensemble, SimulationEngine,
};

#[test]
fn test_distance_cv_gradients() {
    let pbc = PeriodicBox::cubic(20.0);
    let mut topology = Topology::new(pbc);
    let c_type = topology.add_atom_type(AtomType::new("C", 12.011, 0.0, 3.4, 0.36));

    let idx1 = topology.add_particle(c_type, Vector3D::new(5.0, 5.0, 5.0));
    let idx2 = topology.add_particle(c_type, Vector3D::new(8.0, 5.0, 5.0));

    let cv = DistanceCV::new("d12", idx1, idx2);
    let dist = cv.calculate(&topology);
    assert!((dist - 3.0).abs() < 1e-10);

    let grads = cv.gradients(&topology);
    // Gradient on atom 2 should be unit vector +X: (1, 0, 0)
    assert!((grads[idx2].x - 1.0).abs() < 1e-10);
    assert!(grads[idx2].y.abs() < 1e-10);
    assert!(grads[idx2].z.abs() < 1e-10);

    // Gradient on atom 1 should be -X: (-1, 0, 0)
    assert!((grads[idx1].x - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_radius_of_gyration_cv() {
    let pbc = PeriodicBox::cubic(30.0);
    let mut topology = Topology::new(pbc);
    let c_type = topology.add_atom_type(AtomType::new("C", 12.0, 0.0, 3.4, 0.36));

    // Two identical atoms at distance 4 A apart: center of mass is midpoint (dist = 2 A each)
    let idx1 = topology.add_particle(c_type, Vector3D::new(10.0, 15.0, 15.0));
    let idx2 = topology.add_particle(c_type, Vector3D::new(14.0, 15.0, 15.0));

    let cv = RadiusOfGyrationCV::new("rg", vec![idx1, idx2]);
    let rg = cv.calculate(&topology);
    // Rg = sqrt((m * 2^2 + m * 2^2) / (2m)) = sqrt(8m / 2m) = sqrt(4) = 2.0 A
    assert!((rg - 2.0).abs() < 1e-10);
}

#[test]
fn test_metadynamics_hill_deposition_and_fes() {
    let pbc = PeriodicBox::cubic(20.0);
    let mut topology = Topology::new(pbc);
    let c_type = topology.add_atom_type(AtomType::new("C", 12.0, 0.0, 3.4, 0.36));

    let idx1 = topology.add_particle(c_type, Vector3D::new(8.0, 10.0, 10.0));
    let idx2 = topology.add_particle(c_type, Vector3D::new(11.0, 10.0, 10.0));

    let cv = Box::new(DistanceCV::new("d12", idx1, idx2));
    let meta = Metadynamics::new(cv, 2.0, 0.25, 5, 5.0, 300.0);

    let ff = ForceField::standard(8.0);
    let langevin = LangevinIntegrator::new(2.0, 300.0);
    let mut sim = SimulationEngine::new(topology, ff, Ensemble::NVTLangevin(langevin), 42)
        .with_metadynamics(meta);

    sim.initialize();

    for _ in 0..50 {
        sim.step(0.001);
    }

    let meta_ref = sim.metadynamics.as_ref().unwrap();
    // In 50 steps with pace 5, exactly 10 hills should be deposited
    assert_eq!(meta_ref.hills.len(), 10);

    let fes = meta_ref.reconstruct_free_energy_surface(2.0, 5.0, 30);
    assert_eq!(fes.len(), 30);

    // Free energy at minimum must be normalized to 0.0
    let min_fes = fes.iter().map(|&(_, f)| f).fold(f64::INFINITY, f64::min);
    assert!(min_fes.abs() < 1e-10);
}
