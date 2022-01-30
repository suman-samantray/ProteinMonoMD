#!/usr/bin/env python
#title           :LegendCodeMonomer.py
#description     :Methods used for analyzing MD trajectories
#format          :All output is formatted consistently
#python_version  :3.7
#creater         :Suman Samantray
#email           :suman.rishis@gmail.com
#web             :suman-samantray.github.io
#==============================================================================

import os
import errno
import mdtraj as mdt
import numpy as np
import itertools
import shutil


def calc_rmsd(trajectory, reference_pdb):
    """Compute the rmsd by frame of trajectory compared to ref structure

    returns: returns list of tuples: (frame_index, rmsd)
    """
    pdb = mdt.load_pdb(reference_pdb)
    rmsd_list = list(mdt.rmsd(trajectory, pdb, 0))
    frame_index = list(range(1, len(rmsd_list)+1))
    return list(zip(frame_index, rmsd_list))


def calc_center_of_mass(trajectory):
    """Compute the center of mass by frame of trajectory compared to ref structure

    returns: returns list of tuples: (frame_index, com)
    """
    com_list = mdt.compute_center_of_mass(trajectory)
    frame_index = list(range(1, len(com_list) + 1))
    return list(zip(frame_index, com_list))


def calc_atom_SASA(trajectory):
    """Calculates solvent accessible surface area of each atom in a simulation frame using shrake-rupley method.
    This code uses the golden section spiral algorithm to approximate a unit sphere

        References
    ----------
    ..[1] Shrake, A; Rupley, JA. (1973) J Mol Biol 79 (2): 351–71

    returns list of tuples: (frame_index, atom_1_SASA, ..., atom_n_SASA)
    """
    atom_designation = ['']

    atomSASA = mdt.shrake_rupley(trajectory,
                      probe_radius=0.14,
                      n_sphere_points=960,
                      mode='atom',
                      change_radii=None,
                      get_mapping=True)
    templist = []
    frame_index = 1
    for i in range(len(atomSASA[0])):
        templist.append(tuple([frame_index]) + tuple(atomSASA[0][i]))
        frame_index += 1
    atomSASA_return = list(templist)

    for i in range(len(templist[0])-1):
        atom_designation.append(str(trajectory.topology.atom(i)))

    result_matrix = []
    result_matrix.append(tuple(atom_designation))
    result_matrix.extend(atomSASA_return)

    return result_matrix


def calc_residue_SASA(trajectory):
    """Calculates solvent accessible surface area of each residue in a simulation frame using shrake-rupley method.
    This code uses the golden section spiral algorithm to approximate a unit sphere

        References
    ----------
    ..[1] Shrake, A; Rupley, JA. (1973) J Mol Biol 79 (2): 351–71

    returns list of tuples: (frame_index, residue_1_SASA, ..., residue_n_SASA)
    """

    residue_designation = ['']

    residueSASA = mdt.shrake_rupley(trajectory,
                                 probe_radius=0.14,
                                 n_sphere_points=960,
                                 mode='residue',
                                 change_radii=None,
                                 get_mapping=False)
    templist = []
    frame_index = 1
    for i in range(len(residueSASA)):
        templist.append(tuple([frame_index]) + tuple(residueSASA[i]))
        frame_index += 1
    residueSASA_return = list(templist)

    for i in range(len(templist[0])-1):
        residue_designation.append(str(trajectory.topology.residue(i)))

    result_matrix = []
    result_matrix.append(tuple(residue_designation))
    result_matrix.extend(residueSASA_return)

    return result_matrix


def calc_rg(trajectory):
    """Calculates the radius of gyration of each frame in a trajectory. Assumes equal masses.

    returns list of tuples: (frame_index, rg)
    """
    rg_list = mdt.compute_rg(trajectory, masses=None)
    frame_index = list(range(1, len(rg_list) + 1))
    return list(zip(frame_index, rg_list))


def calc_phi(trajectory):
    """Calculates all phi angles in each frame of trajectory.

    returns two list(s) of tuples.
        The first list: tuples of the atom_index groups used to calculate the angles. Example:
            (Angle1_atom_1, Angle1_atom_2, Angle1_atom_3, Angle1_atom_4), ... , (Anglen_atom_1, Anglen_atom_2, Anglen_atom_3, Anglen_atom_4)
        The second list:
            (frame_index, Angle_1, ... , Angle_n)
    """

    atom_group_list = ['']
    angles_list = []
    ca_index = 2

    atom_group, angles = mdt.compute_phi(trajectory, periodic=True, opt=True)

    for group in atom_group:
        atom_group_list.append(str(trajectory.topology.atom(group[ca_index])).split('-')[0])

    frame_index = 1
    for i in range(len(angles)):
        angles_list.append(tuple([frame_index]) + tuple(angles[i]))
        frame_index += 1

    result_matrix = []
    result_matrix.append(tuple(atom_group_list))
    result_matrix.extend(angles_list)

    return result_matrix


def calc_psi(trajectory):
    """Calculates all psi angles in each frame of trajectory.

    returns two list(s) of tuples.
    The first list: tuples of the atom_index groups used to calculate the angles. Example:
        (Angle1_atom_1, Angle1_atom_2, Angle1_atom_3, Angle1_atom_4), ... , (Anglen_atom_1, Anglen_atom_2, Anglen_atom_3, Anglen_atom_4)
    The second list:
        (frame_index, Angle_1, ... , Angle_n)
    """

    atom_group_list = ['']
    angles_list = []
    ca_index = 1

    atom_group, angles = mdt.compute_psi(trajectory, periodic=True, opt=True)

    for group in atom_group:
        atom_group_list.append(str(trajectory.topology.atom(group[ca_index])).split('-')[0])

    frame_index = 1
    for i in range(len(angles)):
        angles_list.append(tuple([frame_index]) + tuple(angles[i]))
        frame_index += 1

    result_matrix = []
    result_matrix.append(tuple(atom_group_list))
    result_matrix.extend(angles_list)

    return result_matrix


def calc_chi1(trajectory):
    """Calculates all chi1 angles in each frame of trajectory.

    returns two list(s) of tuples.
    The first list: tuples of the atom_index groups used to calculate the angles. Example:
        (Angle1_atom_1, Angle1_atom_2, Angle1_atom_3, Angle1_atom_4), ... , (Anglen_atom_1, Anglen_atom_2, Anglen_atom_3, Anglen_atom_4)
    The second list:
        (frame_index, Angle_1, ... , Angle_n)
    """

    atom_group_list = ['']
    angles_list = []
    ca_index = 1

    atom_group, angles = mdt.compute_chi1(trajectory, periodic=True, opt=True)

    for group in atom_group:
        atom_group_list.append(str(trajectory.topology.atom(group[ca_index])).split('-')[0])

    frame_index = 1
    for i in range(len(angles)):
        angles_list.append(tuple([frame_index]) + tuple(angles[i]))
        frame_index += 1

    result_matrix = []
    result_matrix.append(tuple(atom_group_list))
    result_matrix.extend(angles_list)

    return result_matrix


def calc_chi2(trajectory):
    """Calculates all chi2 angles in each frame of trajectory.

    returns two list(s) of tuples.
    The first list: tuples of the atom_index groups used to calculate the angles. Example:
        (Angle1_atom_1, Angle1_atom_2, Angle1_atom_3, Angle1_atom_4), ... , (Anglen_atom_1, Anglen_atom_2, Anglen_atom_3, Anglen_atom_4)
    The second list:
        (frame_index, Angle_1, ... , Angle_n)
    """

    atom_group_list = ['']
    angles_list = []
    ca_index = 0

    atom_group, angles = mdt.compute_chi2(trajectory, periodic=True, opt=True)

    for group in atom_group:
        atom_group_list.append(str(trajectory.topology.atom(group[ca_index])).split('-')[0])

    frame_index = 1
    for i in range(len(angles)):
        angles_list.append(tuple([frame_index]) + tuple(angles[i]))
        frame_index += 1

    result_matrix = []
    result_matrix.append(tuple(atom_group_list))
    result_matrix.extend(angles_list)

    return result_matrix


def calc_chi3(trajectory):
    """Calculates all chi3 angles in each frame of trajectory.

    returns two list(s) of tuples.
    The first list: tuples of the atom_index groups used to calculate the angles. Example:
        (Angle1_atom_1, Angle1_atom_2, Angle1_atom_3, Angle1_atom_4), ... , (Anglen_atom_1, Anglen_atom_2, Anglen_atom_3, Anglen_atom_4)
    The second list:
        (frame_index, Angle_1, ... , Angle_n)
    """

    atom_group_list = ['']
    angles_list = []
    cb_index = 0

    atom_group, angles = mdt.compute_chi3(trajectory, periodic=True, opt=True)

    for group in atom_group:
        atom_group_list.append(str(trajectory.topology.atom(group[cb_index])).split('-')[0])

    frame_index = 1
    for i in range(len(angles)):
        angles_list.append(tuple([frame_index]) + tuple(angles[i]))
        frame_index += 1

    result_matrix = []
    result_matrix.append(tuple(atom_group_list))
    result_matrix.extend(angles_list)

    return result_matrix


def calc_chi4(trajectory):
    """Calculates all chi4 angles in each frame of trajectory.

    returns two list(s) of tuples.
    The first list: tuples of the atom_index groups used to calculate the angles. Example:
        (Angle1_atom_1, Angle1_atom_2, Angle1_atom_3, Angle1_atom_4), ... , (Anglen_atom_1, Anglen_atom_2, Anglen_atom_3, Anglen_atom_4)
    The second list:
        (frame_index, Angle_1, ... , Angle_n)
    """

    atom_group_list = ['']
    angles_list = []
    cg_index = 0

    atom_group, angles = mdt.compute_chi4(trajectory, periodic=True, opt=True)

    for group in atom_group:
        atom_group_list.append(str(trajectory.topology.atom(group[cg_index])).split('-')[0])

    frame_index = 1
    for i in range(len(angles)):
        angles_list.append(tuple([frame_index]) + tuple(angles[i]))
        frame_index += 1

    result_matrix = []
    result_matrix.append(tuple(atom_group_list))
    result_matrix.extend(angles_list)

    return result_matrix


def calc_omega(trajectory):
    """Calculates all omega angles in each frame of trajectory.

    returns two list(s) of tuples.
    The first list: tuples of the atom_index groups used to calculate the angles. Example:
        (Angle1_atom_1, Angle1_atom_2, Angle1_atom_3, Angle1_atom_4), ... , (Anglen_atom_1, Anglen_atom_2, Anglen_atom_3, Anglen_atom_4)
    The second list:
        (frame_index, Angle_1, ... , Angle_n)
    """

    atom_group_list = ['']
    angles_list = []
    ca_index_1 = 0
    ca_index_2 = 3

    atom_group, angles = mdt.compute_omega(trajectory, periodic=True, opt=True)

    for group in atom_group:
        atom_group_list.append(str(trajectory.topology.atom(group[ca_index_1])).split('-')[0] +
                               "_" + str(trajectory.topology.atom(group[ca_index_2])).split('-')[0])

    frame_index = 1
    for i in range(len(angles)):
        angles_list.append(tuple([frame_index]) + tuple(angles[i]))
        frame_index += 1

    result_matrix = []
    result_matrix.append(tuple(atom_group_list))
    result_matrix.extend(angles_list)

    return result_matrix


def calc_density(trajectory):
    """Calculates the mass density of each frame in a trajectory. Assumes equal masses.

    returns list of tuples: (frame_index, mass_density)
    """
    mass_density_list = mdt.density(trajectory, masses=None)
    frame_index = list(range(1, len(mass_density_list) + 1))
    return list(zip(frame_index, mass_density_list))


def calc_secondary_structure(trajectory):
    """Calculates the secondary structure at each AA in a frame

    The DSSP assignment codes are:
        ‘H’ : Alpha helix
        ‘B’ : Residue in isolated beta-bridge
        ‘E’ : Extended strand, participates in beta ladder
        ‘G’ : 3-helix (3/10 helix)
        ‘I’ : 5 helix (pi helix)
        ‘T’ : hydrogen bonded turn
        ‘S’ : bend
        ‘ ‘ : Loops and irregular elements

    returns list of tuples: (frame_index, dssp_residue_1, ... , dssp_residue_n)
    """
    dssp_list = mdt.compute_dssp(trajectory, simplified=False)
    dssp_list = [tuple(x) for x in dssp_list]
    frame_index = list(range(1, len(dssp_list) + 1))
    templist = []
    residue_designation = ['']

    for i in range(len(dssp_list)):
        templist.append(tuple([frame_index[i]]) + dssp_list[i])
    dssp_list = templist

    for i in range(len(templist[0])-1):
        residue_designation.append(str(trajectory.topology.residue(i)))

    result_matrix = []
    result_matrix.append(tuple(residue_designation))
    result_matrix.extend(dssp_list)

    return result_matrix


def calc_fraction_native_contacts(trajectory, reference_pdb_file):
    """Compute the fraction of native contacts according the definition from
    Best, Hummer and Eaton [1]

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used

    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)

    returns: returns list of tuples: (frame_index, Fraction_native_contacts)
    """

    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers

    reference_pdb = mdt.load(reference_pdb_file)
    # get the indices of all of the heavy atoms
    heavy = reference_pdb.topology.select_atom_indices('heavy')
    # get the pairs of heavy atoms which are farther than 3
    # residues apart
    heavy_pairs = np.array(
        [(i,j) for (i,j) in itertools.combinations(heavy, 2)
            if abs(reference_pdb.topology.atom(i).residue.index - \
                   reference_pdb.topology.atom(j).residue.index) > 3])

    # compute the distances between these pairs in the native state
    heavy_pairs_distances = mdt.compute_distances(reference_pdb[0], heavy_pairs)[0]
    # and get the pairs s.t. the distance is less than NATIVE_CUTOFF
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]

    # now compute these distances for the whole trajectory
    r = mdt.compute_distances(trajectory, native_contacts)
    # and recompute them for just the native state
    r0 = mdt.compute_distances(reference_pdb[0], native_contacts)

    fraction_native_contacts = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)

    frame_index = list(range(1, len(fraction_native_contacts)+1))

    return list(zip(frame_index, list(fraction_native_contacts)))


def calc_all_pairwise_contacts(trajectory, scheme):
    """Compute the residue to residue distances between all non-adjacent residues, using defined scheme.

    scheme to determine the distance between two residues:

        ‘ca’ :distance between two residues is given by the distance between their alpha carbons

        ‘closest’ :distance is the closest distance between any two atoms in the residues

        ‘closest-heavy’ :distance is the closest distance between any two non-hydrogen atoms in the residues

        ‘sidechain’ :distance is the closest distance between any two atoms in residue sidechains

        ‘sidechain-heavy’ :distance is the closest distance between any two non-hydrogen atoms in residue sidechains
    """
    residue_designation = ['']
    distances_list = []

    distances, residue_pairs = mdt.compute_contacts(trajectory, contacts='all', scheme=scheme, ignore_nonprotein=True)

    for sublist in residue_pairs:
        residue_designation.append(str(trajectory.topology.residue(sublist[0])) +
                                   "_" + str(trajectory.topology.residue(sublist[1])))
    for i in range(len(distances)):
        distances_list.append(tuple([i+1]) + tuple(distances[i]))

    result_matrix = []
    result_matrix.append(tuple(residue_designation))
    result_matrix.extend(distances_list)

    return result_matrix


def calc_pairwise_contacts(trajectory, res_group_one, res_group_two, scheme):
    """Compute the residue to residue distances between residues in provided groups, using defined scheme.
    All combinations of residue pairs across groups are calculated. Residue indexes begin at 0

    scheme to determine the distance between two residues:

        ‘ca’ :distance between two residues is given by the distance between their alpha carbons

        ‘closest’ :distance is the closest distance between any two atoms in the residues

        ‘closest-heavy’ :distance is the closest distance between any two non-hydrogen atoms in the residues

        ‘sidechain’ :distance is the closest distance between any two atoms in residue sidechains

        ‘sidechain-heavy’ :distance is the closest distance between any two non-hydrogen atoms in residue sidechains
    """
    residue_designation = ['']
    distances_list = []

    for i in range(len(res_group_one)):
        res_group_one[i] = int(res_group_one[i])
    for i in range(len(res_group_two)):
        res_group_two[i] = int(res_group_two[i])

    pairs = list(itertools.product(res_group_one, res_group_two))
    distances, residue_pairs = mdt.compute_contacts(trajectory, pairs, scheme=scheme)

    for sublist in residue_pairs:
        residue_designation.append(str(trajectory.topology.residue(sublist[0]-1)) +
                                   "_" + str(trajectory.topology.residue(sublist[1]-1)))
    for i in range(len(distances)):
        distances_list.append(tuple([i+1]) + tuple(distances[i]))

    result_matrix = []
    result_matrix.append(tuple(residue_designation))
    result_matrix.extend(distances_list)

    return result_matrix


def calc_atom_distance(trajectory, atom_group_one, atom_group_two):
    """Compute the atom to atom distances between all atoms in provided groups.
    All combinations of atom pairs across groups are calculated. Atom indexes begin at 0
    """
    atom_designation = ['']
    distances_list = []

    for i in range(len(atom_group_one)):
        atom_group_one[i] = int(atom_group_one[i])
    for i in range(len(atom_group_two)):
        atom_group_two[i] = int(atom_group_two[i])

    pairs = list(itertools.product(atom_group_one, atom_group_two))
    distances = mdt.compute_distances(trajectory, pairs)

    for sublist in pairs:
        atom_designation.append(str(trajectory.topology.atom(sublist[0]-1)) +
                                   "_" + str(trajectory.topology.atom(sublist[1]-1)))
    for i in range(len(distances)):
        distances_list.append(tuple([i+1]) + tuple(distances[i]))

    result_matrix = []
    result_matrix.append(tuple(atom_designation))
    result_matrix.extend(distances_list)

    return result_matrix


def find_residues_within_radius(trajectory, res_group, radius_distance_nm=1):
    """Return List of residues and distances within defined radius."""
    all_ca_atoms = trajectory.topology.select('name CA')
    res_group_ca_atoms = []
    list_out = []
    distances_list = []
    frame_list = []
    residue_pairs = ['']

    if not isinstance(res_group, list):
        res_group = [res_group]
    for i in range(len(res_group)):
        res_group_ca_atoms.append(int(trajectory.topology.select('name CA and resid ' + str(res_group[i]-1))[0]))

    ca_pairs = list(itertools.product(res_group_ca_atoms, all_ca_atoms))
    all_distances = mdt.compute_distances(trajectory, ca_pairs)

    for i in range(len(ca_pairs)):
        list_out.append([0] * len(all_distances))
        for j in range(len(all_distances)):
            list_out[i][j] = all_distances[j][i]

    count = 1
    for i in range(len(list_out)):
        if min(list_out[i]) <= radius_distance_nm and sum(list_out[i]) > 0:
            res1 = str(trajectory.topology.atom(ca_pairs[i][0])).split('-')[0]
            res2 = str(trajectory.topology.atom(ca_pairs[i][1])).split('-')[0]
            residue_pairs.append(res1 + '_' + res2)
            distances_list.append(list_out[i])
            count += 1

    result_matrix = []
    result_matrix.append(tuple(residue_pairs))
    for i in range(0, trajectory.n_frames):
        distance_temp = []
        for j in range(len(distances_list)):
            distance_temp.append(distances_list[j][i])
        result_matrix.append(tuple([i+1]) + tuple(distance_temp))

    return result_matrix


def calc_hydrogen_bonds(trajectory):
    """Determines hydrogen bonds and calculates bond energy using
    the kabsch sander method [1].  Returned matrix is a list of atom pairs and energies (kcal/mol):
    [
    ["GLU1-N , GLU1-OE2", "1.2"]
    ["...","..."]
    ]

    References
    ----------
    ..[1] Kabsch W, Sander C (1983). “Dictionary of protein secondary structure: pattern
          recognition of hydrogen-bonded and geometrical features”. Biopolymers 22 (12): 2577-637
    """
    energies = mdt.kabsch_sander(trajectory)
    toreturn = []
    pairs = ['']
    pair_list = [[]]

    for i in range(len(energies)):
        if ',' not in str(energies[i]):
            continue
        temp_entries = list(str(energies[i]).strip().split('\n'))
        for item in temp_entries:
            temp_list = (str(item).strip().split('\t')[0])[1:-1].split(',')
            if temp_list == ['']:
                continue
            res_str = str(trajectory.topology.residue(int(temp_list[0]))) + "_" + str(trajectory.topology.residue(int(temp_list[1])))
            if res_str not in pairs:
                pairs.append(res_str)
                pair_list.append([temp_list[0], temp_list[1]])
    toreturn.append(tuple(pairs))
    for i in range(len(energies)):
        if ',' not in str(energies[i]):
            templine = [0] * len(pairs)
            templine[0] = i + 1
            toreturn.append(tuple(templine))
            continue
        temp_entries = list(str(energies[i]).strip().split('\n'))
        templine = [0] * len(pairs)
        templine[0] = i + 1
        for item in temp_entries:
            temp_val = str(item).strip().split('\t')
            temp_list = (temp_val[0])[1:-1].split(',')
            temp_energy = temp_val[1]
            if temp_list in pair_list:
                templine[pair_list.index(temp_list)] = temp_energy
        toreturn.append(tuple(templine))

    return toreturn


def list_of_tuple_to_txt(matrix, filename):
    with safe_open_w(filename) as output:
        for line in matrix:
            output.write('\t'.join(map(str, line)) + '\n')
    return True

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.'''
    dname = os.path.dirname(path)
    try:
        os.makedirs(dname)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(dname):
            pass
        else:
            raise
    return open(path, 'w')