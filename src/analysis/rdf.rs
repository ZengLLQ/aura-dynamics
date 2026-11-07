use crate::types::{PeriodicBox, Vector3D};

/// Radial Distribution Function g(r) accumulator.
#[derive(Debug, Clone)]
pub struct RadialDistributionFunction {
    pub r_max: f64,
    pub dr: f64,
    pub num_bins: usize,
    pub histogram: Vec<u64>,
    pub num_samples: usize,
    pub num_particles: usize,
    pub box_volume: f64,
}

impl RadialDistributionFunction {
    pub fn new(r_max: f64, num_bins: usize) -> Self {
        let dr = r_max / num_bins as f64;
        Self {
            r_max,
            dr,
            num_bins,
            histogram: vec![0; num_bins],
            num_samples: 0,
            num_particles: 0,
            box_volume: 0.0,
        }
    }

    /// Accumulate distance pairs from a simulation frame.
    pub fn accumulate(&mut self, positions: &[Vector3D], pbc: &PeriodicBox) {
        let n = positions.len();
        self.num_particles = n;
        self.box_volume = pbc.volume();
        self.num_samples += 1;

        for i in 0..n {
            for j in (i + 1)..n {
                let delta = positions[j] - positions[i];
                let dist = pbc.minimum_image(delta).norm();

                if dist < self.r_max {
                    let bin = (dist / self.dr).floor() as usize;
                    if bin < self.num_bins {
                        self.histogram[bin] += 2; // Count both (i, j) and (j, i)
                    }
                }
            }
        }
    }

    /// Calculate normalized g(r) profile: Vec<(r_mid, g_r)>
    pub fn compute_gr(&self) -> Vec<(f64, f64)> {
        if self.num_samples == 0 || self.num_particles < 2 {
            return Vec::new();
        }

        let mut gr = Vec::with_capacity(self.num_bins);
        let n = self.num_particles as f64;
        let bulk_density = n / self.box_volume;

        for bin in 0..self.num_bins {
            let r_lower = bin as f64 * self.dr;
            let r_upper = r_lower + self.dr;
            let r_mid = r_lower + 0.5 * self.dr;

            // Volume of spherical shell: 4/3 * pi * (r_upper^3 - r_lower^3)
            let shell_volume = (4.0 / 3.0) * std::f64::consts::PI * (r_upper.powi(3) - r_lower.powi(3));
            let ideal_count = self.num_samples as f64 * n * bulk_density * shell_volume;

            let g_val = if ideal_count > 0.0 {
                self.histogram[bin] as f64 / ideal_count
            } else {
                0.0
            };

            gr.push((r_mid, g_val));
        }

        gr
    }
}
