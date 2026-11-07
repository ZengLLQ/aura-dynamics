pub mod harmonic;
pub mod neighbor;
pub mod nonbonded;
pub mod torsion;

use crate::types::{Topology, Vector3D};
use neighbor::NeighborList;
use nonbonded::NonbondedParams;

/// Breakdown of molecular potential energy components in kJ/mol.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EnergyBreakdown {
    pub bond: f64,
    pub angle: f64,
    pub dihedral: f64,
    pub vdw: f64,
    pub coulomb: f64,
    pub kinetic: f64,
}

impl EnergyBreakdown {
    pub fn potential(&self) -> f64 {
        self.bond + self.angle + self.dihedral + self.vdw + self.coulomb
    }

    pub fn total(&self) -> f64 {
        self.potential() + self.kinetic
    }
}

/// Force field manager managing neighbor lists and calculating molecular forces.
pub struct ForceField {
    pub neighbor_list: NeighborList,
    pub nonbonded_params: NonbondedParams,
}

impl ForceField {
    pub fn new(r_cut: f64, r_skin: f64, epsilon_rf: f64) -> Self {
        Self {
            neighbor_list: NeighborList::new(r_cut, r_skin),
            nonbonded_params: NonbondedParams::new(r_cut, epsilon_rf),
        }
    }

    pub fn standard(r_cut: f64) -> Self {
        Self::new(r_cut, 2.0, 78.3)
    }

    /// Evaluates all forces and returns the energy breakdown.
    pub fn compute_forces(&mut self, topology: &mut Topology) -> EnergyBreakdown {
        let n = topology.particles.len();

        // Zero all forces
        for p in &mut topology.particles {
            p.force = Vector3D::ZERO;
        }

        let positions: Vec<Vector3D> = topology.particles.iter().map(|p| p.position).collect();
        let type_ids: Vec<usize> = topology.particles.iter().map(|p| p.type_id).collect();
        let charges: Vec<f64> = topology.particles.iter().map(|p| p.charge).collect();
        let mut forces = vec![Vector3D::ZERO; n];

        // 1. Bond stretching
        let bond_energy = harmonic::evaluate_bonds(
            &topology.bonds,
            &positions,
            &mut forces,
            &topology.pbc,
        );

        // 2. Angle bending
        let angle_energy = harmonic::evaluate_angles(
            &topology.angles,
            &positions,
            &mut forces,
            &topology.pbc,
        );

        // 3. Dihedral torsions
        let dihedral_energy = torsion::evaluate_dihedrals(
            &topology.dihedrals,
            &positions,
            &mut forces,
            &topology.pbc,
        );

        // 4. Neighbor list check & rebuild
        if self.neighbor_list.needs_rebuild(&positions, &topology.pbc) {
            self.neighbor_list.rebuild(&positions, &topology.pbc, &topology.exclusions);
        }

        // 5. Nonbonded van der Waals & Coulomb
        let (vdw_energy, coul_energy) = nonbonded::evaluate_nonbonded(
            &self.neighbor_list.pairs,
            &positions,
            &type_ids,
            &topology.atom_types,
            &charges,
            &mut forces,
            &topology.pbc,
            &self.nonbonded_params,
        );

        // Write accumulated forces back to particles
        for (i, p) in topology.particles.iter_mut().enumerate() {
            p.force = forces[i];
        }

        let kinetic_energy: f64 = topology.particles.iter().map(|p| p.kinetic_energy()).sum();

        EnergyBreakdown {
            bond: bond_energy,
            angle: angle_energy,
            dihedral: dihedral_energy,
            vdw: vdw_energy,
            coulomb: coul_energy,
            kinetic: kinetic_energy,
        }
    }
}
