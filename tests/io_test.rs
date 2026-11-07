use aura_dynamics::{
    io::{read_pdb, write_pdb, XyzTrajectoryWriter},
    types::{AtomType, PeriodicBox, Topology, Vector3D},
};

#[test]
fn test_pdb_reader_and_writer() {
    let pdb_content = "\
CRYST1   30.000   30.000   30.000  90.00  90.00  90.00 P 1           1
ATOM      1  CA  ALA A   1      10.500  12.300  14.200  1.00 20.00           C
ATOM      2  CB  ALA A   1      11.800  13.100  14.500  1.00 20.00           C
END
";
    let (atoms, pbc) = read_pdb(pdb_content.as_bytes()).expect("Failed to parse PDB");
    assert_eq!(atoms.len(), 2);
    assert_eq!(atoms[0].name, "CA");
    assert_eq!(atoms[1].name, "CB");
    assert!((atoms[0].position.x - 10.5).abs() < 1e-4);

    let pbc_box = pbc.expect("CRYST1 should be parsed");
    assert!((pbc_box.lx - 30.0).abs() < 1e-4);

    // Build topology and write back to buffer
    let mut topology = Topology::new(pbc_box);
    let c_type = topology.add_atom_type(AtomType::new("C", 12.011, 0.0, 3.4, 0.36));
    topology.add_particle(c_type, atoms[0].position);
    topology.add_particle(c_type, atoms[1].position);

    let mut out_buffer = Vec::new();
    write_pdb(&mut out_buffer, &topology, "Test Snapshot").expect("Write PDB failed");
    let written_str = String::from_utf8(out_buffer).expect("Valid UTF8");
    assert!(written_str.contains("CRYST1   30.000   30.000   30.000"));
    assert!(written_str.contains("ATOM      1"));
    assert!(written_str.contains("MOL A   1"));
}

#[test]
fn test_xyz_trajectory_writer() {
    let pbc = PeriodicBox::cubic(20.0);
    let mut topology = Topology::new(pbc);
    let ar_type = topology.add_atom_type(AtomType::argon());
    topology.add_particle(ar_type, Vector3D::new(1.0, 2.0, 3.0));

    let mut buffer = Vec::new();
    {
        let mut traj = XyzTrajectoryWriter::new(&mut buffer);
        traj.write_frame(&topology, 1, 0.002, -150.25).expect("Write frame failed");
    }

    let out_str = String::from_utf8(buffer).expect("Valid UTF8");
    assert!(out_str.starts_with("1\n"));
    assert!(out_str.contains("Step: 1 | Time: 0.0020 ps"));
    assert!(out_str.contains("Ar"));
}
