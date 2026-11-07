use crate::types::{PeriodicBox, Topology, Vector3D};
use std::io::{BufRead, BufReader, Read, Write};

/// PDB atom entry.
#[derive(Debug, Clone)]
pub struct PdbAtom {
    pub serial: usize,
    pub name: String,
    pub res_name: String,
    pub res_seq: usize,
    pub position: Vector3D,
    pub element: String,
}

/// Reads molecular coordinates from a PDB file string or reader.
pub fn read_pdb<R: Read>(reader: R) -> Result<(Vec<PdbAtom>, Option<PeriodicBox>), String> {
    let buf = BufReader::new(reader);
    let mut atoms = Vec::new();
    let mut pbc = None;

    for line_res in buf.lines() {
        let line = line_res.map_err(|e| e.to_string())?;

        if line.starts_with("CRYST1") && line.len() >= 54 {
            let lx: f64 = line[6..15].trim().parse().unwrap_or(0.0);
            let ly: f64 = line[15..24].trim().parse().unwrap_or(0.0);
            let lz: f64 = line[24..33].trim().parse().unwrap_or(0.0);
            if lx > 0.0 && ly > 0.0 && lz > 0.0 {
                pbc = Some(PeriodicBox::new(lx, ly, lz));
            }
        } else if line.starts_with("ATOM") || line.starts_with("HETATM") {
            if line.len() < 54 {
                continue;
            }
            let serial: usize = line[6..11].trim().parse().unwrap_or(atoms.len() + 1);
            let name = line[12..16].trim().to_string();
            let res_name = line[17..20].trim().to_string();
            let res_seq: usize = line[22..26].trim().parse().unwrap_or(1);

            let x: f64 = line[30..38].trim().parse().map_err(|e: std::num::ParseFloatError| e.to_string())?;
            let y: f64 = line[38..46].trim().parse().map_err(|e: std::num::ParseFloatError| e.to_string())?;
            let z: f64 = line[46..54].trim().parse().map_err(|e: std::num::ParseFloatError| e.to_string())?;

            let element = if line.len() >= 78 {
                line[76..78].trim().to_string()
            } else {
                name.chars().next().unwrap_or('C').to_string()
            };

            atoms.push(PdbAtom {
                serial,
                name,
                res_name,
                res_seq,
                position: Vector3D::new(x, y, z),
                element,
            });
        }
    }

    Ok((atoms, pbc))
}

/// Write current topology snapshot to standard PDB format.
pub fn write_pdb<W: Write>(writer: &mut W, topology: &Topology, title: &str) -> Result<(), std::io::Error> {
    writeln!(writer, "TITLE     {}", title)?;
    writeln!(
        writer,
        "CRYST1{:9.3}{:9.3}{:9.3}  90.00  90.00  90.00 P 1           1",
        topology.pbc.lx, topology.pbc.ly, topology.pbc.lz
    )?;

    for (i, p) in topology.particles.iter().enumerate() {
        let type_name = &topology.atom_types[p.type_id].name;
        let serial = (i + 1) % 100000;
        let element = if type_name.len() > 2 { &type_name[0..1] } else { type_name };

        writeln!(
            writer,
            "ATOM  {:5} {:^4} MOL A{:4}    {:8.3}{:8.3}{:8.3}  1.00 20.00          {:>2}",
            serial,
            type_name,
            1,
            p.position.x,
            p.position.y,
            p.position.z,
            element
        )?;
    }

    writeln!(writer, "END")?;
    Ok(())
}
