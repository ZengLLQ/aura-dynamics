use crate::types::{Topology, Vector3D};

/// Trait defining a differentiable collective variable (reaction coordinate).
pub trait CollectiveVariable: Send + Sync {
    /// Name of the collective variable
    fn name(&self) -> &str;

    /// Calculate the current value of the CV s(R)
    fn calculate(&self, topology: &Topology) -> f64;

    /// Calculate the spatial gradient of the CV with respect to each particle: d(s) / d(r_i)
    fn gradients(&self, topology: &Topology) -> Vec<Vector3D>;
}

/// Collective variable: Distance between two atoms ||r_i - r_j||.
#[derive(Debug, Clone)]
pub struct DistanceCV {
    pub name: String,
    pub atom_i: usize,
    pub atom_j: usize,
}

impl DistanceCV {
    pub fn new(name: impl Into<String>, atom_i: usize, atom_j: usize) -> Self {
        Self {
            name: name.into(),
            atom_i,
            atom_j,
        }
    }
}

impl CollectiveVariable for DistanceCV {
    fn name(&self) -> &str {
        &self.name
    }

    fn calculate(&self, topology: &Topology) -> f64 {
        let delta = topology.particles[self.atom_j].position - topology.particles[self.atom_i].position;
        let r_ij = topology.pbc.minimum_image(delta);
        r_ij.norm()
    }

    fn gradients(&self, topology: &Topology) -> Vec<Vector3D> {
        let mut grads = vec![Vector3D::ZERO; topology.particles.len()];
        let delta = topology.particles[self.atom_j].position - topology.particles[self.atom_i].position;
        let r_ij = topology.pbc.minimum_image(delta);
        let dist = r_ij.norm();

        if dist > 1e-12 {
            let unit = r_ij / dist;
            grads[self.atom_j] = unit;
            grads[self.atom_i] = -unit;
        }

        grads
    }
}

/// Collective variable: Radius of Gyration R_g.
#[derive(Debug, Clone)]
pub struct RadiusOfGyrationCV {
    pub name: String,
    pub atom_indices: Vec<usize>,
}

impl RadiusOfGyrationCV {
    pub fn new(name: impl Into<String>, atom_indices: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            atom_indices,
        }
    }
}

impl CollectiveVariable for RadiusOfGyrationCV {
    fn name(&self) -> &str {
        &self.name
    }

    fn calculate(&self, topology: &Topology) -> f64 {
        let total_mass: f64 = self.atom_indices.iter().map(|&idx| topology.particles[idx].mass).sum();
        if total_mass <= 0.0 {
            return 0.0;
        }

        let mut com = Vector3D::ZERO;
        for &idx in &self.atom_indices {
            com += topology.particles[idx].position * topology.particles[idx].mass;
        }
        com /= total_mass;

        let mut sum_sq = 0.0;
        for &idx in &self.atom_indices {
            let d = topology.pbc.minimum_image(topology.particles[idx].position - com);
            sum_sq += topology.particles[idx].mass * d.norm_sq();
        }

        (sum_sq / total_mass).sqrt()
    }

    fn gradients(&self, topology: &Topology) -> Vec<Vector3D> {
        let mut grads = vec![Vector3D::ZERO; topology.particles.len()];
        let rg = self.calculate(topology);
        if rg < 1e-12 {
            return grads;
        }

        let total_mass: f64 = self.atom_indices.iter().map(|&idx| topology.particles[idx].mass).sum();
        if total_mass <= 0.0 {
            return grads;
        }

        let mut com = Vector3D::ZERO;
        for &idx in &self.atom_indices {
            com += topology.particles[idx].position * topology.particles[idx].mass;
        }
        com /= total_mass;

        for &idx in &self.atom_indices {
            let d = topology.pbc.minimum_image(topology.particles[idx].position - com);
            // d(Rg)/d(r_i) = (m_i / (M * Rg)) * (r_i - com)
            grads[idx] = d * (topology.particles[idx].mass / (total_mass * rg));
        }

        grads
    }
}
