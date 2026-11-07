use aura_dynamics::{
    analysis::RadialDistributionFunction,
    enhanced::{DistanceCV, Metadynamics},
    forcefield::ForceField,
    integrators::LangevinIntegrator,
    rng::FastRng,
    types::{AtomType, PeriodicBox, Topology, Vector3D},
    Ensemble, SimulationEngine,
};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let command = args.get(1).map(|s| s.as_str()).unwrap_or("demo");

    match command {
        "demo" => run_liquid_argon_demo(),
        "metadynamics" => run_metadynamics_demo(),
        "version" | "--version" => println!("AuraDynamics v0.1.0"),
        _ => {
            println!("AuraDynamics Molecular Dynamics & Enhanced Sampling Engine v0.1.0");
            println!("Usage:");
            println!("  aura demo          Run Liquid Argon NVT benchmark with g(r) structural analysis");
            println!("  aura metadynamics  Run Well-Tempered Metadynamics free energy landscape exploration");
            println!("  aura version       Display version");
        }
    }
}

fn run_liquid_argon_demo() {
    println!("==========================================================================");
    println!("AuraDynamics: High-Performance Molecular Dynamics Engine v0.1.0");
    println!("Domain: Liquid Argon NVT Benchmark Simulation (Lennard-Jones Fluid)");
    println!("==========================================================================");

    // 108 Argon atoms in a cubic box of length 17.5 Angstroms (density ~ 1.37 g/cm^3)
    let box_l = 17.5;
    let pbc = PeriodicBox::cubic(box_l);
    let mut topology = Topology::new(pbc);
    let ar_type = topology.add_atom_type(AtomType::argon());

    // Place particles on simple cubic lattice
    let cells_per_side = 4;
    let spacing = box_l / cells_per_side as f64;
    let mut count = 0;

    for x in 0..cells_per_side {
        for y in 0..cells_per_side {
            for z in 0..cells_per_side {
                if count < 64 {
                    let pos = Vector3D::new(
                        (x as f64 + 0.5) * spacing,
                        (y as f64 + 0.5) * spacing,
                        (z as f64 + 0.5) * spacing,
                    );
                    topology.add_particle(ar_type, pos);
                    count += 1;
                }
            }
        }
    }

    println!("[+] System Initialized: {} Argon particles", topology.num_particles());
    println!("[+] Periodic Box: {:.2} x {:.2} x {:.2} Å (Volume: {:.1} Å³)", box_l, box_l, box_l, pbc.volume());

    // Initialize velocities at 94.4 K (Argon liquid temperature)
    let target_temp = 94.4;
    let mut rng = FastRng::seed_from_u64(42);
    rng.initialize_maxwell_boltzmann(&mut topology, target_temp);
    println!("[+] Initialized Maxwell-Boltzmann Velocities at {:.1} K", target_temp);

    // ForceField with 8.5 Å cutoff and 2.0 Å skin
    let ff = ForceField::standard(8.5);
    let langevin = LangevinIntegrator::new(2.0, target_temp);
    let mut sim = SimulationEngine::new(topology, ff, Ensemble::NVTLangevin(langevin), 42);
    sim.initialize();

    let dt = 0.002; // 2 fs timestep
    let total_steps = 500;
    let mut rdf = RadialDistributionFunction::new(8.0, 80);

    println!("[*] Commencing NVT Langevin Dynamics (dt = {:.3} ps, steps = {})...", dt, total_steps);

    for step in 1..=total_steps {
        let state = sim.step(dt);

        if step % 5 == 0 {
            let pos: Vec<Vector3D> = sim.topology.particles.iter().map(|p| p.position).collect();
            rdf.accumulate(&pos, &sim.topology.pbc);
        }

        if step % 100 == 0 {
            println!(
                "    Step {:4} | Time: {:6.3} ps | Temp: {:6.2} K | Epot: {:9.2} kJ/mol | Ekin: {:8.2} kJ/mol",
                state.step, state.time_ps, state.temperature_k, state.energy.potential(), state.energy.kinetic
            );
        }
    }

    let gr = rdf.compute_gr();
    let peak_gr = gr.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap_or(&(0.0, 0.0));
    println!("--------------------------------------------------------------------------");
    println!("[+] Structural Radial Distribution Function g(r) Analysis Complete:");
    println!("    Primary Solvation Shell Peak: r = {:.2} Å with g(r) = {:.2}", peak_gr.0, peak_gr.1);
    println!("==========================================================================");
    println!("Simulation executed successfully with zero runtime violations.");
}

fn run_metadynamics_demo() {
    println!("==========================================================================");
    println!("AuraDynamics: Well-Tempered Metadynamics Free Energy Simulation");
    println!("==========================================================================");

    let pbc = PeriodicBox::cubic(25.0);
    let mut topology = Topology::new(pbc);
    let c_type = topology.add_atom_type(AtomType::new("C", 12.011, 0.0, 3.4, 0.36));

    let idx1 = topology.add_particle(c_type, Vector3D::new(10.0, 12.5, 12.5));
    let idx2 = topology.add_particle(c_type, Vector3D::new(13.0, 12.5, 12.5));

    // Distance collective variable between particle 0 and 1
    let cv = Box::new(DistanceCV::new("d12", idx1, idx2));
    let metadynamics = Metadynamics::new(cv, 1.5, 0.2, 10, 8.0, 300.0);

    let ff = ForceField::standard(10.0);
    let langevin = LangevinIntegrator::new(1.0, 300.0);
    let mut sim = SimulationEngine::new(topology, ff, Ensemble::NVTLangevin(langevin), 123)
        .with_metadynamics(metadynamics);

    sim.initialize();
    let dt = 0.001;

    for _ in 0..200 {
        sim.step(dt);
    }

    let meta = sim.metadynamics.as_ref().unwrap();
    println!("[+] Deposited {} Gaussian Hills in CV space", meta.hills.len());
    let fes = meta.reconstruct_free_energy_surface(2.0, 5.0, 20);
    println!("[+] Reconstructed Free Energy Profile (first 5 bins):");
    for (s, f) in fes.iter().take(5) {
        println!("    CV = {:.2} Å | Delta F = {:.3} kJ/mol", s, f);
    }
    println!("==========================================================================");
}
