use crate::types::Topology;
use std::io::Write;

/// Multi-frame XYZ trajectory writer.
pub struct XyzTrajectoryWriter<W: Write> {
    writer: W,
}

impl<W: Write> XyzTrajectoryWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Write single trajectory snapshot frame.
    pub fn write_frame(
        &mut self,
        topology: &Topology,
        step: usize,
        time_ps: f64,
        total_energy_kj: f64,
    ) -> Result<(), std::io::Error> {
        let n = topology.particles.len();
        writeln!(self.writer, "{}", n)?;
        writeln!(
            self.writer,
            "Step: {} | Time: {:.4} ps | Total Energy: {:.4} kJ/mol | Box: {:.3}x{:.3}x{:.3} A",
            step, time_ps, total_energy_kj, topology.pbc.lx, topology.pbc.ly, topology.pbc.lz
        )?;

        for p in &topology.particles {
            let atom_name = &topology.atom_types[p.type_id].name;
            writeln!(
                self.writer,
                "{:<4} {:12.6} {:12.6} {:12.6}",
                atom_name, p.position.x, p.position.y, p.position.z
            )?;
        }

        self.writer.flush()?;
        Ok(())
    }
}
