use crate::types::{Dihedral, PeriodicBox, Vector3D};

/// Evaluate periodic dihedral torsion energies and analytical forces.
/// Potential: V_dihedral = 0.5 * V_n * [1 + cos(n * phi - gamma)]
pub fn evaluate_dihedrals(
    dihedrals: &[Dihedral],
    positions: &[Vector3D],
    forces: &mut [Vector3D],
    pbc: &PeriodicBox,
) -> f64 {
    let mut energy = 0.0;

    for dih in dihedrals {
        let i = dih.i;
        let j = dih.j;
        let k = dih.k;
        let l = dih.l;

        // Vectors along the 3 bonds: b1 = j - i, b2 = k - j, b3 = l - k
        let b1 = pbc.minimum_image(positions[j] - positions[i]);
        let b2 = pbc.minimum_image(positions[k] - positions[j]);
        let b3 = pbc.minimum_image(positions[l] - positions[k]);

        let b2_norm = b2.norm();
        if b2_norm < 1e-12 {
            continue;
        }

        // Normal vectors to the two intersecting planes
        let n1 = b1.cross(b2);
        let n2 = b2.cross(b3);

        let n1_norm_sq = n1.norm_sq();
        let n2_norm_sq = n2.norm_sq();

        if n1_norm_sq < 1e-14 || n2_norm_sq < 1e-14 {
            continue;
        }

        let n1_norm = n1_norm_sq.sqrt();
        let n2_norm = n2_norm_sq.sqrt();

        // Calculate dihedral angle phi in [-pi, pi]
        let m1 = n1.cross(b2 / b2_norm);
        let cos_phi = (n1.dot(n2) / (n1_norm * n2_norm)).clamp(-1.0, 1.0);
        let sin_phi = (m1.dot(n2) / (n1_norm * n2_norm)).clamp(-1.0, 1.0);
        let phi = sin_phi.atan2(cos_phi);

        // Potential energy: 0.5 * V_n * [1 + cos(n * phi - gamma)]
        let arg = dih.n as f64 * phi - dih.gamma;
        energy += 0.5 * dih.v_n * (1.0 + arg.cos());

        // Derivative dV / dphi = -0.5 * V_n * n * sin(n * phi - gamma)
        let d_v_d_phi = -0.5 * dih.v_n * (dih.n as f64) * arg.sin();

        // Chain-rule gradients on each of the 4 atoms
        let f_i = (n1 / (n1_norm_sq)) * (-d_v_d_phi * b2_norm);
        let f_l = (n2 / (n2_norm_sq)) * (d_v_d_phi * b2_norm);

        // Moments on central atoms j and k
        let b1_dot_b2 = b1.dot(b2);
        let b3_dot_b2 = b3.dot(b2);
        let b2_norm_sq = b2_norm * b2_norm;

        let f_j = f_i * (b1_dot_b2 / b2_norm_sq - 1.0) + f_l * (b3_dot_b2 / b2_norm_sq);
        let f_k = -(f_i + f_j + f_l);

        forces[i] += f_i;
        forces[j] += f_j;
        forces[k] += f_k;
        forces[l] += f_l;
    }

    energy
}
