use crate::types::{AtomType, PeriodicBox, Vector3D};

/// Coulomb conversion factor: 1 / (4 * pi * epsilon_0) in (kJ * A) / (mol * e^2)
pub const COULOMB_FACTOR: f64 = 138.935458;

/// Nonbonded evaluation parameters.
#[derive(Debug, Clone, Copy)]
pub struct NonbondedParams {
    pub r_cut: f64,
    pub epsilon_rf: f64, // Solvent dielectric for Reaction-Field (e.g. 78.3 for water)
    pub k_rf: f64,
    pub c_rf: f64,
}

impl NonbondedParams {
    pub fn new(r_cut: f64, epsilon_rf: f64) -> Self {
        let r_cut_cubed = r_cut * r_cut * r_cut;
        let k_rf = ((epsilon_rf - 1.0) / (2.0 * epsilon_rf + 1.0)) / r_cut_cubed;
        let c_rf = (3.0 * epsilon_rf / (2.0 * epsilon_rf + 1.0)) / r_cut;

        Self {
            r_cut,
            epsilon_rf,
            k_rf,
            c_rf,
        }
    }

    /// Standard reaction-field with infinite dielectric (conducting boundary)
    pub fn conducting(r_cut: f64) -> Self {
        Self::new(r_cut, 1e6)
    }
}

/// Evaluates nonbonded Lennard-Jones and Reaction-Field Coulomb interactions over neighbor pairs.
/// Returns (vdw_energy, coulomb_energy).
pub fn evaluate_nonbonded(
    pairs: &[(usize, usize)],
    positions: &[Vector3D],
    type_ids: &[usize],
    atom_types: &[AtomType],
    charges: &[f64],
    forces: &mut [Vector3D],
    pbc: &PeriodicBox,
    params: &NonbondedParams,
) -> (f64, f64) {
    let mut vdw_total = 0.0;
    let mut coul_total = 0.0;
    let r_cut_sq = params.r_cut * params.r_cut;

    for &(i, j) in pairs {
        let delta = positions[j] - positions[i];
        let r_ij = pbc.minimum_image(delta);
        let dist_sq = r_ij.norm_sq();

        if dist_sq < r_cut_sq && dist_sq > 1e-12 {
            let dist = dist_sq.sqrt();
            let inv_r = 1.0 / dist;
            let inv_r2 = inv_r * inv_r;

            // Lorentz-Berthelot mixing rules
            let type_i = &atom_types[type_ids[i]];
            let type_j = &atom_types[type_ids[j]];

            let sigma = 0.5 * (type_i.sigma + type_j.sigma);
            let epsilon = (type_i.epsilon * type_j.epsilon).sqrt();

            let mut f_pair = 0.0;

            // Lennard-Jones 12-6 interaction
            if epsilon > 1e-12 && sigma > 1e-12 {
                let s_r = sigma * inv_r;
                let s_r2 = s_r * s_r;
                let s_r6 = s_r2 * s_r2 * s_r2;
                let s_r12 = s_r6 * s_r6;

                // Shifted potential
                let s_rc = sigma / params.r_cut;
                let s_rc2 = s_rc * s_rc;
                let s_rc6 = s_rc2 * s_rc2 * s_rc2;
                let s_rc12 = s_rc6 * s_rc6;
                let v_shift = 4.0 * epsilon * (s_rc12 - s_rc6);

                let v_lj = 4.0 * epsilon * (s_r12 - s_r6) - v_shift;
                vdw_total += v_lj;

                // Force: -dV_lj/dr = 24 * eps / r * (2 * (sig/r)^12 - (sig/r)^6)
                f_pair += 24.0 * epsilon * inv_r * (2.0 * s_r12 - s_r6);
            }

            // Reaction-Field Coulomb interaction
            let q_product = charges[i] * charges[j];
            if q_product.abs() > 1e-12 {
                let v_coul = COULOMB_FACTOR * q_product * (inv_r + params.k_rf * dist_sq - params.c_rf);
                coul_total += v_coul;

                // Force: -dV_coul/dr = C * q1 * q2 * (1 / r^2 - 2 * k_rf * r)
                f_pair += COULOMB_FACTOR * q_product * (inv_r2 - 2.0 * params.k_rf * dist);
            }

            // Apply force vector: F_j += f_pair * (r_ij / dist), F_i -= f_pair * (r_ij / dist)
            let f_vec = (r_ij * inv_r) * f_pair;
            forces[j] += f_vec;
            forces[i] -= f_vec;
        }
    }

    (vdw_total, coul_total)
}
