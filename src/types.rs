use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Three-dimensional Euclidean vector with high-performance operator implementations.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3D {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn norm_sq(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n > 1e-14 {
            self / n
        } else {
            Self::ZERO
        }
    }
}

impl Add for Vector3D {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vector3D {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vector3D {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vector3D {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Mul<f64> for Vector3D {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f64) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

impl Mul<Vector3D> for f64 {
    type Output = Vector3D;
    #[inline]
    fn mul(self, vec: Vector3D) -> Vector3D {
        vec * self
    }
}

impl MulAssign<f64> for Vector3D {
    #[inline]
    fn mul_assign(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl Div<f64> for Vector3D {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f64) -> Self {
        let inv = 1.0 / scalar;
        Self::new(self.x * inv, self.y * inv, self.z * inv)
    }
}

impl DivAssign<f64> for Vector3D {
    #[inline]
    fn div_assign(&mut self, scalar: f64) {
        let inv = 1.0 / scalar;
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

impl Neg for Vector3D {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Orthorhombic periodic boundary simulation box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicBox {
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
}

impl PeriodicBox {
    pub fn new(lx: f64, ly: f64, lz: f64) -> Self {
        assert!(lx > 0.0 && ly > 0.0 && lz > 0.0, "Box dimensions must be positive");
        Self { lx, ly, lz }
    }

    pub fn cubic(l: f64) -> Self {
        Self::new(l, l, l)
    }

    #[inline]
    pub fn volume(&self) -> f64 {
        self.lx * self.ly * self.lz
    }

    /// Minimum image convention for displacement vector r_ij = r_j - r_i.
    #[inline]
    pub fn minimum_image(&self, delta: Vector3D) -> Vector3D {
        Vector3D::new(
            delta.x - self.lx * (delta.x / self.lx).round(),
            delta.y - self.ly * (delta.y / self.ly).round(),
            delta.z - self.lz * (delta.z / self.lz).round(),
        )
    }

    /// Wrap particle position into the primary cell [0, L).
    #[inline]
    pub fn wrap(&self, pos: Vector3D) -> Vector3D {
        Vector3D::new(
            pos.x.rem_euclid(self.lx),
            pos.y.rem_euclid(self.ly),
            pos.z.rem_euclid(self.lz),
        )
    }
}

/// Chemical atom type parameters for force field evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomType {
    pub name: String,
    pub mass: f64,       // In atomic mass units (amu or g/mol)
    pub charge: f64,     // In units of elementary charge e
    pub sigma: f64,      // Lennard-Jones distance parameter (Angstroms)
    pub epsilon: f64,    // Lennard-Jones well depth (kJ/mol)
}

impl AtomType {
    pub fn new(name: impl Into<String>, mass: f64, charge: f64, sigma: f64, epsilon: f64) -> Self {
        Self {
            name: name.into(),
            mass,
            charge,
            sigma,
            epsilon,
        }
    }

    /// Standard Lennard-Jones argon parameter set.
    pub fn argon() -> Self {
        Self::new("Ar", 39.948, 0.0, 3.405, 0.996)
    }

    /// Standard TIP3P water oxygen.
    pub fn tip3p_oxygen() -> Self {
        Self::new("OW", 15.9994, -0.834, 3.1506, 0.6364)
    }

    /// Standard TIP3P water hydrogen.
    pub fn tip3p_hydrogen() -> Self {
        Self::new("HW", 1.008, 0.417, 0.0, 0.0)
    }
}

/// Simulation particle representing an atom or coarse-grained bead.
#[derive(Debug, Clone, PartialEq)]
pub struct Particle {
    pub id: usize,
    pub type_id: usize,
    pub position: Vector3D,
    pub velocity: Vector3D,
    pub force: Vector3D,
    pub mass: f64,
    pub charge: f64,
}

impl Particle {
    pub fn new(id: usize, type_id: usize, position: Vector3D, mass: f64, charge: f64) -> Self {
        Self {
            id,
            type_id,
            position,
            velocity: Vector3D::ZERO,
            force: Vector3D::ZERO,
            mass,
            charge,
        }
    }

    #[inline]
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self.mass * self.velocity.norm_sq()
    }
}

/// Covalent 2-body bond definition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bond {
    pub i: usize,
    pub j: usize,
    pub k_b: f64, // Spring constant: kJ/(mol * A^2)
    pub r_0: f64, // Equilibrium bond distance (A)
}

impl Bond {
    pub fn new(i: usize, j: usize, k_b: f64, r_0: f64) -> Self {
        Self { i, j, k_b, r_0 }
    }
}

/// Covalent 3-body angle definition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Angle {
    pub i: usize,
    pub j: usize, // Central vertex atom
    pub k: usize,
    pub k_theta: f64, // Angle force constant: kJ/(mol * rad^2)
    pub theta_0: f64, // Equilibrium angle in radians
}

impl Angle {
    pub fn new(i: usize, j: usize, k: usize, k_theta: f64, theta_0: f64) -> Self {
        Self { i, j, k, k_theta, theta_0 }
    }
}

/// 4-body dihedral torsion angle definition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dihedral {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub l: usize,
    pub v_n: f64,      // Barrier height: kJ/mol
    pub n: usize,      // Periodicity (multiplicity)
    pub gamma: f64,    // Phase angle in radians
}

impl Dihedral {
    pub fn new(i: usize, j: usize, k: usize, l: usize, v_n: f64, n: usize, gamma: f64) -> Self {
        Self { i, j, k, l, v_n, n, gamma }
    }
}

/// Molecular topology containing particle data, covalent bonds, angles, and boundary conditions.
#[derive(Debug, Clone)]
pub struct Topology {
    pub atom_types: Vec<AtomType>,
    pub particles: Vec<Particle>,
    pub bonds: Vec<Bond>,
    pub angles: Vec<Angle>,
    pub dihedrals: Vec<Dihedral>,
    pub pbc: PeriodicBox,
    pub exclusions: Vec<std::collections::HashSet<usize>>,
}

impl Topology {
    pub fn new(pbc: PeriodicBox) -> Self {
        Self {
            atom_types: Vec::new(),
            particles: Vec::new(),
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            pbc,
            exclusions: Vec::new(),
        }
    }

    pub fn add_atom_type(&mut self, atom_type: AtomType) -> usize {
        let id = self.atom_types.len();
        self.atom_types.push(atom_type);
        id
    }

    pub fn add_particle(&mut self, type_id: usize, position: Vector3D) -> usize {
        let id = self.particles.len();
        let atom_type = &self.atom_types[type_id];
        let p = Particle::new(id, type_id, position, atom_type.mass, atom_type.charge);
        self.particles.push(p);
        self.exclusions.push(std::collections::HashSet::new());
        id
    }

    pub fn add_bond(&mut self, i: usize, j: usize, k_b: f64, r_0: f64) {
        self.bonds.push(Bond::new(i, j, k_b, r_0));
        // Add 1-2 exclusions for nonbonded interaction
        self.exclusions[i].insert(j);
        self.exclusions[j].insert(i);
    }

    pub fn add_angle(&mut self, i: usize, j: usize, k: usize, k_theta: f64, theta_0: f64) {
        self.angles.push(Angle::new(i, j, k, k_theta, theta_0));
        // Add 1-3 exclusions
        self.exclusions[i].insert(k);
        self.exclusions[k].insert(i);
    }

    pub fn add_dihedral(&mut self, i: usize, j: usize, k: usize, l: usize, v_n: f64, n: usize, gamma: f64) {
        self.dihedrals.push(Dihedral::new(i, j, k, l, v_n, n, gamma));
    }

    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    pub fn total_mass(&self) -> f64 {
        self.particles.iter().map(|p| p.mass).sum()
    }

    pub fn center_of_mass(&self) -> Vector3D {
        let total_m = self.total_mass();
        if total_m <= 0.0 {
            return Vector3D::ZERO;
        }
        let weighted_sum = self.particles.iter().fold(Vector3D::ZERO, |acc, p| {
            acc + p.position * p.mass
        });
        weighted_sum / total_m
    }

    pub fn instantaneous_temperature(&self) -> f64 {
        let n = self.num_particles();
        if n == 0 {
            return 0.0;
        }
        let total_ke: f64 = self.particles.iter().map(|p| p.kinetic_energy()).sum();
        // Degrees of freedom: 3N - 3 (removing center-of-mass motion)
        let dof = (3 * n).saturating_sub(3).max(1) as f64;
        // Boltzmann constant k_B in (kJ / (mol * K)) = 8.314462618e-3 kJ/(mol*K)
        const KB: f64 = 8.314462618e-3;
        (2.0 * total_ke) / (dof * KB)
    }
}
