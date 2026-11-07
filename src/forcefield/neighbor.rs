use crate::types::{PeriodicBox, Vector3D};
use std::collections::HashSet;

/// Verlet neighbor list with automatic skin displacement checking.
#[derive(Debug, Clone)]
pub struct NeighborList {
    pub r_cut: f64,
    pub r_skin: f64,
    pub r_search: f64,
    pub pairs: Vec<(usize, usize)>,
    pub reference_positions: Vec<Vector3D>,
}

impl NeighborList {
    pub fn new(r_cut: f64, r_skin: f64) -> Self {
        let r_search = r_cut + r_skin;
        Self {
            r_cut,
            r_skin,
            r_search,
            pairs: Vec::new(),
            reference_positions: Vec::new(),
        }
    }

    /// Check whether particle displacements require rebuilding the neighbor list.
    #[inline]
    pub fn needs_rebuild(&self, positions: &[Vector3D], pbc: &PeriodicBox) -> bool {
        if self.reference_positions.len() != positions.len() {
            return true;
        }

        let mut max_disp_sq: f64 = 0.0;
        for (i, pos) in positions.iter().enumerate() {
            let delta = pbc.minimum_image(*pos - self.reference_positions[i]);
            let d_sq = delta.norm_sq();
            if d_sq > max_disp_sq {
                max_disp_sq = d_sq;
            }
        }

        // Trigger rebuild when 2 * max_disp >= r_skin (i.e. max_disp >= r_skin / 2)
        let threshold = 0.5 * self.r_skin;
        max_disp_sq >= threshold * threshold
    }

    /// Rebuild neighbor list using Spatial Cell Lists for O(N) scaling.
    pub fn rebuild(
        &mut self,
        positions: &[Vector3D],
        pbc: &PeriodicBox,
        exclusions: &[HashSet<usize>],
    ) {
        let n = positions.len();
        self.pairs.clear();
        self.reference_positions = positions.to_vec();

        let r_search_sq = self.r_search * self.r_search;

        // Cell dimensions
        let num_cells_x = ((pbc.lx / self.r_search).floor() as isize).max(1);
        let num_cells_y = ((pbc.ly / self.r_search).floor() as isize).max(1);
        let num_cells_z = ((pbc.lz / self.r_search).floor() as isize).max(1);

        let cell_size_x = pbc.lx / num_cells_x as f64;
        let cell_size_y = pbc.ly / num_cells_y as f64;
        let cell_size_z = pbc.lz / num_cells_z as f64;

        let total_cells = (num_cells_x * num_cells_y * num_cells_z) as usize;
        let mut head: Vec<isize> = vec![-1; total_cells];
        let mut next: Vec<isize> = vec![-1; n];

        // Map particle to primary cell [0, L) and populate linked list
        for (i, pos) in positions.iter().enumerate() {
            let wrapped = pbc.wrap(*pos);
            let cx = ((wrapped.x / cell_size_x).floor() as isize).clamp(0, num_cells_x - 1);
            let cy = ((wrapped.y / cell_size_y).floor() as isize).clamp(0, num_cells_y - 1);
            let cz = ((wrapped.z / cell_size_z).floor() as isize).clamp(0, num_cells_z - 1);

            let cell_idx = ((cx * num_cells_y + cy) * num_cells_z + cz) as usize;
            next[i] = head[cell_idx];
            head[cell_idx] = i as isize;
        }

        // Loop over neighboring cells
        for cx in 0..num_cells_x {
            for cy in 0..num_cells_y {
                for cz in 0..num_cells_z {
                    let cell_idx = ((cx * num_cells_y + cy) * num_cells_z + cz) as usize;

                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                let ncx = (cx + dx).rem_euclid(num_cells_x);
                                let ncy = (cy + dy).rem_euclid(num_cells_y);
                                let ncz = (cz + dz).rem_euclid(num_cells_z);
                                let ncell_idx = ((ncx * num_cells_y + ncy) * num_cells_z + ncz) as usize;

                                let mut i_node = head[cell_idx];
                                while i_node != -1 {
                                    let i = i_node as usize;
                                    let mut j_node = head[ncell_idx];

                                    while j_node != -1 {
                                        let j = j_node as usize;

                                        // Ensure unique pairs with i < j
                                        if i < j {
                                            // Check covalent exclusion
                                            let is_excluded = exclusions
                                                .get(i)
                                                .map_or(false, |ex| ex.contains(&j));

                                            if !is_excluded {
                                                let delta = pbc.minimum_image(positions[j] - positions[i]);
                                                if delta.norm_sq() <= r_search_sq {
                                                    self.pairs.push((i, j));
                                                }
                                            }
                                        }

                                        j_node = next[j];
                                    }

                                    i_node = next[i];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate pairs in case small boxes mapped multiple neighbors to identical cells
        self.pairs.sort_unstable();
        self.pairs.dedup();
    }
}
