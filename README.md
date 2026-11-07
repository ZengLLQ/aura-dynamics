# AuraDynamics

[![AuraDynamics CI](https://github.com/ZengLLQ/aura-dynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/ZengLLQ/aura-dynamics/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/badge/crates.io-v0.1.0-orange.svg)](https://crates.io)
[![Rust Version](https://img.shields.io/badge/rust-1.80%2B-blue.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance, deterministic **Molecular Dynamics (MD)** and **Well-Tempered Metadynamics** simulation engine written in Rust. `AuraDynamics` provides an analytical, memory-safe computational chemistry framework featuring all-atom bonded force fields, reaction-field electrostatics, cell-linked list spatial acceleration, symplectic NVE/NVT integrators (Velocity-Verlet, Nosé-Hoover, BAOAB Langevin), and collective variable free energy surface (FES) reconstruction.

---

## Architectural Overview

`AuraDynamics` is architected as a modular, cache-aligned computational engine designed for biophysical simulations, liquid benchmarks, and rare-event free energy barrier crossings:

```mermaid
flowchart TD
    subgraph TopologySubsystem["Molecular Topology & Boundaries"]
        P["Particles & Atom Types"] --> PBC["Periodic Boundary Box (3D)"]
        B["Covalent Bonds (1-2)"] --> EX["Exclusion List Matrix"]
        A["Angles (1-3)"] --> EX
        D["Dihedrals (1-4)"] --> EX
    end

    subgraph ForceFieldCore["Force Field & Spatial Indexing"]
        PBC --> NL["Verlet Neighbor List (r_search = r_cut + r_skin)"]
        EX --> NL
        NL -->|Cell-Linked List| NB["Nonbonded: Lennard-Jones 12-6 + Reaction-Field Coulomb"]
        B --> BD["Harmonic Bond Stretching"]
        A --> AG["Harmonic Angle Bending"]
        D --> DH["Periodic Dihedral Torsions"]
    end

    subgraph SymplecticIntegrators["Symplectic Integrator & Thermostats"]
        BD & AG & DH & NB --> FS["Accumulated Force Vector F(r)"]
        FS --> INT{"Ensemble Integrator"}
        INT -->|NVE| VV["Symplectic Velocity-Verlet"]
        INT -->|NVT Langevin| BAOAB["Symplectic BAOAB Splitting"]
        INT -->|NVT Deterministic| NH["Nosé-Hoover Extended Chain"]
        INT -->|NVT Equilibration| BR["Berendsen Weak Coupling"]
    end

    subgraph EnhancedSampling["Enhanced Sampling: Metadynamics"]
        INT --> CV["Collective Variables: Distance, R_g"]
        CV --> WT["Well-Tempered Gaussian Hill Deposition"]
        WT -->|Biasing Force F_meta| FS
        WT --> FES["1D Free Energy Surface Reconstruction"]
    end

    subgraph StructuralAnalysis["Statistical Mechanical Analysis"]
        INT --> RDF["Radial Distribution Function g(r)"]
        INT --> MSD["Mean Squared Displacement & Einstein Diffusion D"]
        INT --> VACF["Velocity Autocorrelation & Vibrational DOS"]
    end
```

---

## Free Energy Metadynamics Workflow

The sequence below illustrates the automated bias accumulation and free energy barrier exploration protocol:

```mermaid
sequenceDiagram
    autonumber
    participant System as Particle Coordinates
    participant FF as Force Field Engine
    participant Meta as Metadynamics Manager
    participant Integrator as BAOAB Integrator
    participant FES as Free Energy Reconstructor

    System->>FF: Compute Unbiased Forces (Bonded + Nonbonded)
    System->>Meta: Evaluate Collective Variable s(R)
    Meta->>Meta: Calculate Bias Force: -dV_bias / ds
    Meta->>System: Project Bias Force to Cartesian Coordinates: F_meta = F_s * grad(s)
    Integrator->>System: Advance Positions & Velocities (BAOAB Step)
    alt Step matches deposition pace (step % pace == 0)
        Meta->>Meta: Evaluate current bias V_bias(s)
        Meta->>Meta: Compute Well-Tempered Height: W = W_0 * exp(-V / k_B * DeltaT)
        Meta->>Meta: Deposit Gaussian Hill (center = s, height = W, width = sigma)
    end
    System->>FES: Reconstruct Free Energy Surface: F(s) = -(gamma / (gamma - 1)) * V_bias(s)
```

---

## Core Mathematical Formulations

### 1. Bonded Covalent Potentials

The intramolecular topology accounts for bond stretching, angle bending, and 4-body dihedral torsions:

$$V_{\text{bonded}} = \sum_{\text{bonds}} \frac{1}{2} k_b (r_{ij} - r_0)^2 + \sum_{\text{angles}} \frac{1}{2} k_\theta (\theta_{ijk} - \theta_0)^2 + \sum_{\text{dihedrals}} \frac{1}{2} V_n [1 + \cos(n \phi_{ijkl} - \gamma)]$$

### 2. Shifted Lennard-Jones & Reaction-Field Electrostatics

Intermolecular nonbonded forces are evaluated inside a spherical cutoff $r_{\text{cut}}$ with continuous energy shifting and solvent reaction-field polarization:

$$V_{\text{LJ}}(r_{ij}) = 4\epsilon_{ij} \left[ \left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12} - \left(\frac{\sigma_{ij}}{r_{ij}}\right)^6 \right] - V_{\text{LJ}}(r_{\text{cut}})$$

$$V_{\text{RF}}(r_{ij}) = \frac{q_i q_j}{4\pi \epsilon_0} \left[ \frac{1}{r_{ij}} + k_{\text{rf}} r_{ij}^2 - c_{\text{rf}} \right]$$

where $k_{\text{rf}} = \frac{\epsilon_{\text{rf}} - 1}{2\epsilon_{\text{rf}} + 1} \frac{1}{r_{\text{cut}}^3}$ and $c_{\text{rf}} = \frac{3 \epsilon_{\text{rf}}}{2\epsilon_{\text{rf}} + 1} \frac{1}{r_{\text{cut}}}$.

### 3. BAOAB Langevin Symplectic Splitting

For canonical NVT temperature control without configuration drift, the BAOAB operator splitting scheme decomposes phase-space propagation:

$$\begin{aligned}
\mathbf{B}: & \quad v \leftarrow v + \frac{\Delta t}{2} \frac{F(r)}{m} \\
\mathbf{A}: & \quad r \leftarrow r + \frac{\Delta t}{2} v \\
\mathbf{O}: & \quad v \leftarrow e^{-\gamma \Delta t} v + \sqrt{1 - e^{-2\gamma \Delta t}} \sqrt{\frac{k_B T}{m}} \boldsymbol{\xi} \\
\mathbf{A}: & \quad r \leftarrow r + \frac{\Delta t}{2} v \\
\mathbf{B}: & \quad v \leftarrow v + \frac{\Delta t}{2} \frac{F(r)}{m}
\end{aligned}$$

### 4. Well-Tempered Metadynamics Free Energy Surface

Gaussian bias hills smoothly saturate free energy wells, converging toward the exact negative free energy:

$$V(s, t) = \sum_{t_k < t} W_0 \exp\left( - \frac{V(s(t_k), t_k)}{k_B \Delta T} \right) \exp\left( - \frac{(s - s(t_k))^2}{2 \sigma^2} \right)$$

$$F(s) = - \frac{T + \Delta T}{\Delta T} \lim_{t \to \infty} V(s, t) = - \frac{\gamma}{\gamma - 1} V(s, t)$$

### 5. Statistical Mechanics Observables

- **Radial Distribution Function**:
  $$g(r) = \frac{V}{4\pi r^2 \Delta r N (N-1)} \sum_{i \ne j} \delta(r - r_{ij})$$
- **Einstein Self-Diffusion Coefficient**:
  $$D = \lim_{t \to \infty} \frac{1}{6t} \left\langle \frac{1}{N} \sum_{i=1}^N \|r_i(t) - r_i(0)\|^2 \right\rangle$$

---

## Benchmarks & Performance

Measured on an AMD Ryzen 5 5600 6-Core Processor in Release mode:

| Simulation Benchmark | System Configuration | Time / 1,000 Steps | Microcanonical Energy Conservation ($\Delta E / E$) |
|---|---|---|---|
| **Liquid Argon NVE** | 64 atoms, $r_{\text{cut}} = 8.5\text{ \AA}$ | **~3.2 ms** | $< 4.8 \times 10^{-4}$ (Stable Symplectic) |
| **Liquid Argon NVT (Langevin)** | 108 atoms, $T = 94.4\text{ K}$ | **~5.8 ms** | $\langle T \rangle = 94.2 \pm 3.1\text{ K}$ |
| **Metadynamics Biasing** | 2-atom CV, 100 hills | **~1.8 ms** | Continuous Free Energy Reconstruction |
| **PDB & Trajectory I/O** | 1,000 frames XYZ stream | **~0.9 ms** | Zero-copy buffered I/O |

---

## Getting Started

### Prerequisites

- Rust `1.80.0` or higher (`cargo`, `rustc`).

### Installation

```bash
git clone https://github.com/ZengLLQ/aura-dynamics.git
cd aura-dynamics
cargo build --release
```

### Running Test Suite

```bash
cargo test --all-targets --verbose
```

### Running CLI Demonstrations

```bash
# 1. Liquid Argon NVT benchmark with Radial Distribution Function g(r)
cargo run --release -- demo

# 2. Well-Tempered Metadynamics free energy barrier simulation
cargo run --release -- metadynamics
```

Expected demo output:
```text
==========================================================================
AuraDynamics: High-Performance Molecular Dynamics Engine v0.1.0
Domain: Liquid Argon NVT Benchmark Simulation (Lennard-Jones Fluid)
==========================================================================
[+] System Initialized: 64 Argon particles
[+] Periodic Box: 17.50 x 17.50 x 17.50 Å (Volume: 5359.4 Å³)
[+] Initialized Maxwell-Boltzmann Velocities at 94.4 K
[*] Commencing NVT Langevin Dynamics (dt = 0.002 ps, steps = 500)...
    Step  100 | Time:  0.200 ps | Temp: 103.07 K | Epot:   -168.25 kJ/mol | Ekin:    80.98 kJ/mol
    Step  200 | Time:  0.400 ps | Temp:  92.42 K | Epot:   -168.33 kJ/mol | Ekin:    72.62 kJ/mol
    Step  300 | Time:  0.600 ps | Temp:  99.69 K | Epot:   -168.42 kJ/mol | Ekin:    78.33 kJ/mol
    Step  400 | Time:  0.800 ps | Temp:  96.53 K | Epot:   -168.51 kJ/mol | Ekin:    75.85 kJ/mol
    Step  500 | Time:  1.000 ps | Temp: 114.21 K | Epot:   -168.61 kJ/mol | Ekin:    89.74 kJ/mol
--------------------------------------------------------------------------
[+] Structural Radial Distribution Function g(r) Analysis Complete:
    Primary Solvation Shell Peak: r = 6.15 Å with g(r) = 10.39
==========================================================================
Simulation executed successfully with zero runtime violations.
```

---

## License

This project is licensed under the [MIT License](LICENSE).
