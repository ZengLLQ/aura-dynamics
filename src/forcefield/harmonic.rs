use crate::types::{Angle, Bond, PeriodicBox, Vector3D};

/// Evaluate harmonic bond stretching energies and forces.
/// Potential: V_bond = 0.5 * k_b * (r_ij - r_0)^2
pub fn evaluate_bonds(
    bonds: &[Bond],
    positions: &[Vector3D],
    forces: &mut [Vector3D],
    pbc: &PeriodicBox,
) -> f64 {
    let mut energy = 0.0;

    for bond in bonds {
        let i = bond.i;
        let j = bond.j;

        let delta = positions[j] - positions[i];
        let r_ij = pbc.minimum_image(delta);
        let dist = r_ij.norm();

        if dist > 1e-12 {
            let diff = dist - bond.r_0;
            energy += 0.5 * bond.k_b * diff * diff;

            // Force magnitude: -dV/dr = -k_b * (dist - r_0)
            let f_mag = -bond.k_b * diff;
            let f_vec = (r_ij / dist) * f_mag;

            // Action-reaction: F_j += f_vec, F_i -= f_vec
            forces[j] += f_vec;
            forces[i] -= f_vec;
        }
    }

    energy
}

/// Evaluate harmonic angle bending energies and forces.
/// Potential: V_angle = 0.5 * k_theta * (theta - theta_0)^2
pub fn evaluate_angles(
    angles: &[Angle],
    positions: &[Vector3D],
    forces: &mut [Vector3D],
    pbc: &PeriodicBox,
) -> f64 {
    let mut energy = 0.0;

    for angle in angles {
        let i = angle.i;
        let j = angle.j; // Central atom
        let k = angle.k;

        let r_ji = pbc.minimum_image(positions[i] - positions[j]);
        let r_jk = pbc.minimum_image(positions[k] - positions[j]);

        let d_ji = r_ji.norm();
        let d_jk = r_jk.norm();

        if d_ji > 1e-12 && d_jk > 1e-12 {
            let cos_theta = (r_ji.dot(r_jk) / (d_ji * d_jk)).clamp(-1.0, 1.0);
            let theta = cos_theta.acos();

            let diff = theta - angle.theta_0;
            energy += 0.5 * angle.k_theta * diff * diff;

            let sin_theta = (1.0 - cos_theta * cos_theta).max(1e-14).sqrt();
            let d_v_d_theta = angle.k_theta * diff;
            let factor = -d_v_d_theta / sin_theta;

            // Analytical gradient vectors with respect to positions r_i, r_k, and central r_j
            let f_i = (r_jk / (d_ji * d_jk) - r_ji * (cos_theta / (d_ji * d_ji))) * factor;
            let f_k = (r_ji / (d_ji * d_jk) - r_jk * (cos_theta / (d_jk * d_jk))) * factor;
            let f_j = -(f_i + f_k);

            forces[i] += f_i;
            forces[j] += f_j;
            forces[k] += f_k;
        }
    }

    energy
}
