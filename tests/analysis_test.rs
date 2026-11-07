use aura_dynamics::{
    analysis::{MeanSquaredDisplacement, RadialDistributionFunction, VelocityAutocorrelation},
    types::{PeriodicBox, Vector3D},
};

#[test]
fn test_radial_distribution_function() {
    let pbc = PeriodicBox::cubic(20.0);
    let mut rdf = RadialDistributionFunction::new(10.0, 50);

    // Place 4 atoms in a square with side 4.0 A
    let positions = vec![
        Vector3D::new(5.0, 5.0, 5.0),
        Vector3D::new(9.0, 5.0, 5.0),
        Vector3D::new(5.0, 9.0, 5.0),
        Vector3D::new(9.0, 9.0, 5.0),
    ];

    rdf.accumulate(&positions, &pbc);
    let gr = rdf.compute_gr();
    assert_eq!(gr.len(), 50);

    // Target pair distance is exactly 4.0 A.
    // In binning with dr = 0.2: bin = (4.0 / 0.2).floor() = 20 (r_mid = 4.1 A).
    let target_bin = (4.0 / rdf.dr).floor() as usize;
    assert!(target_bin < gr.len());
    assert!(gr[target_bin].1 > 0.0, "Expected non-zero g(r) at 4.0 A distance");
}

#[test]
fn test_msd_and_diffusion_coefficient() {
    let initial = vec![Vector3D::ZERO];
    let mut msd = MeanSquaredDisplacement::new(&initial);
    let box_lengths = Vector3D::new(100.0, 100.0, 100.0);

    // Simulate ballistic or linear motion where pos(t) = (sqrt(6 * D * t), 0, 0)
    // with D = 2.0 A^2/ps
    let target_d = 2.0;
    for step in 1..=100 {
        let t = step as f64 * 0.1;
        let x = (6.0 * target_d * t).sqrt();
        let cur = vec![Vector3D::new(x, 0.0, 0.0)];
        msd.record_step(&cur, box_lengths, t);
    }

    let estimated_d = msd.calculate_diffusion_coefficient();
    println!("Target D = {}, estimated D = {}", target_d, estimated_d);
    assert!((estimated_d - target_d).abs() < 0.2);
}

#[test]
fn test_vacf_and_vdos() {
    let dt = 0.01;
    let mut vacf_calc = VelocityAutocorrelation::new(dt);

    // Velocity harmonic oscillation v(t) = cos(omega0 * t) with omega0 = 10.0 rad/ps
    let omega0 = 10.0;
    for step in 0..100 {
        let t = step as f64 * dt;
        let v = vec![Vector3D::new((omega0 * t).cos(), 0.0, 0.0)];
        vacf_calc.record_frame(&v);
    }

    let vacf = vacf_calc.compute_vacf(50);
    assert!(!vacf.is_empty());
    // VACF at lag 0 should be normalized to 1.0
    assert!((vacf[0].1 - 1.0).abs() < 1e-4);

    let vdos = vacf_calc.compute_vdos(&vacf, 20.0, 40);
    assert_eq!(vdos.len(), 40);
}
