import numpy as np
from rdkit import Chem

atom_int_dict = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "Fe": 26,
    "FE": 26,
    "I": 53,
}


int_atom_dict = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    26: "Fe",
    53: "I",
}


def pdb_to_xyz(file):
    with open(file, "r") as f:
        lines = f.readlines()
    xyz = []
    charge = []
    atom = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            # xyz.append([float(i) for i in line.split()[6:9]])
            charge.append(float(line.split()[-2]))
            atom.append(atom_int_dict[line.split()[-1]])
    return xyz, charge, atom


def get_proto_mol(atoms):
    """ """
    mol = Chem.MolFromSmarts("[#" + str(atoms[0]) + "]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atoms)):
        a = Chem.Atom(atoms[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def xyz2AC_vdW(atoms, xyz):
    # Get mol template
    mol = get_proto_mol(atoms)

    # Set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    AC = get_AC(mol)

    return AC, mol


def get_AC(atoms, xyz, covalent_factor=1.4):
    pt = Chem.GetPeriodicTable()
    # xyz to distance matrix
    xyz = np.array(xyz)
    dist_mat = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    num_atoms = xyz.shape[0]

    AC = np.zeros((num_atoms, num_atoms), dtype=int)
    for i in range(num_atoms):
        a_i = atoms[i]
        Rcov_i = pt.GetRcovalent(a_i) * covalent_factor
        for j in range(i + 1, num_atoms):
            a_j = atoms[j]
            Rcov_j = pt.GetRcovalent(a_j) * covalent_factor
            if dist_mat[i, j] <= Rcov_i + Rcov_j:
                AC[i, j] = 1
    return AC
