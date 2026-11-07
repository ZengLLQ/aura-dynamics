use crate::types::Vector3D;

/// Mean Squared Displacement (MSD) and Einstein self-diffusion coefficient estimator.
#[derive(Debug, Clone)]
pub struct MeanSquaredDisplacement {
    pub initial_unwrapped_positions: Vec<Vector3D>,
    pub current_unwrapped_positions: Vec<Vector3D>,
    pub previous_positions: Vec<Vector3D>,
    pub time_points: Vec<f64>,
    pub msd_values: Vec<f64>,
}

impl MeanSquaredDisplacement {
    pub fn new(initial_positions: &[Vector3D]) -> Self {
        Self {
            initial_unwrapped_positions: initial_positions.to_vec(),
            current_unwrapped_positions: initial_positions.to_vec(),
            previous_positions: initial_positions.to_vec(),
            time_points: Vec::new(),
            msd_values: Vec::new(),
        }
    }

    /// Record displacement at time t (ps), updating continuous unwrapped trajectories across PBC jumps.
    pub fn record_step(&mut self, current_positions: &[Vector3D], box_lengths: Vector3D, time_ps: f64) {
        let n = current_positions.len();

        for i in 0..n {
            // Detect periodic boundary wrap-around jumps
            let mut dx = current_positions[i].x - self.previous_positions[i].x;
            let mut dy = current_positions[i].y - self.previous_positions[i].y;
            let mut dz = current_positions[i].z - self.previous_positions[i].z;

            dx -= box_lengths.x * (dx / box_lengths.x).round();
            dy -= box_lengths.y * (dy / box_lengths.y).round();
            dz -= box_lengths.z * (dz / box_lengths.z).round();

            self.current_unwrapped_positions[i].x += dx;
            self.current_unwrapped_positions[i].y += dy;
            self.current_unwrapped_positions[i].z += dz;
            self.previous_positions[i] = current_positions[i];
        }

        let mut sum_sq_disp = 0.0;
        for i in 0..n {
            let disp = self.current_unwrapped_positions[i] - self.initial_unwrapped_positions[i];
            sum_sq_disp += disp.norm_sq();
        }

        let msd = sum_sq_disp / n as f64;
        self.time_points.push(time_ps);
        self.msd_values.push(msd);
    }

    /// Calculate self-diffusion coefficient D in (A^2 / ps) using linear regression on MSD vs time.
    /// In 3D: MSD(t) = 6 * D * t  =>  D = slope / 6
    /// Note: 1 A^2/ps = 1e-8 m^2 / 1e-12 s = 1e-4 m^2/s = 1e-5 cm^2/s
    pub fn calculate_diffusion_coefficient(&self) -> f64 {
        let n = self.time_points.len();
        if n < 2 {
            return 0.0;
        }

        // Linear regression: slope = (N * sum(t*MSD) - sum(t)*sum(MSD)) / (N * sum(t^2) - (sum(t))^2)
        // Discard first 20% ballistic regime to measure the diffusive slope
        let start_idx = n / 5;
        let points = &self.time_points[start_idx..];
        let values = &self.msd_values[start_idx..];
        let m = points.len() as f64;

        let sum_t: f64 = points.iter().sum();
        let sum_y: f64 = values.iter().sum();
        let sum_tt: f64 = points.iter().map(|&t| t * t).sum();
        let sum_ty: f64 = points.iter().zip(values.iter()).map(|(&t, &y)| t * y).sum();

        let denom = m * sum_tt - sum_t * sum_t;
        if denom.abs() < 1e-12 {
            return 0.0;
        }

        let slope = (m * sum_ty - sum_t * sum_y) / denom;
        (slope / 6.0).max(0.0)
    }
}
