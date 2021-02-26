import logging as log
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# controls loggin level
log.basicConfig(level=log.DEBUG)

def _get_conf_gen_params(num_threads):
    # this includes addtional small ring torsion potentials and
    # macrocycle ring torsion potentials and macrocycle-specific handles
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    params.numThreads = num_threads
    return params


def _determine_conf_num(mol):
    # please cite: dx.doi.org/10.1021/ci2004658
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if rot_bonds <= 7:
        return 50
    elif rot_bonds <= 12:
        return 200
    else:
        return 300

def _filter_and_sort_conformers(mol, energies, remove_unconverged):
    confs = mol.GetConformers()
    energies_idx = np.argsort([x[1] for x in energies])  # sort the indices by energy values
    filtered_energies = []
    mol.RemoveAllConformers()
    for eidx in energies_idx:
        unconverged, e = energies[eidx]
        if remove_unconverged:
            if not unconverged:
                mol.AddConformer(confs[eidx])  # add the conformer to the molecule, but this time ordered by energy
                filtered_energies.append(e)
        else:
            mol.AddConformer(confs[eidx])  # add the conformer to the molecule, but this time ordered by energy
            filtered_energies.append(e)

    if remove_unconverged:
        log.debug(f"Started with {len(confs)} conformers, ended with {len(filtered_energies)}")

    if len(filtered_energies) != len(mol.GetConformers()):
        raise RuntimeError("conformers and energies length mismatch")

    return (mol, filtered_energies)

def _energy_minimize(mol, max_tries = 10, remove_unconverged = True, lowest_e_only = False):
    conf_num = len(mol.GetConformers())
    not_converged =  conf_num # at the start all confs. are not converged
    retries = 0
    while not_converged > 0 and retries <= max_tries: # retry until all converged or
        energies = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0) # this returns a tuple (not_converged_flag, energy)
        not_converged = sum([e[0] for e in energies])
        log.debug(f"{not_converged} conformers not converged (try #{retries})")
        retries += 1

    mol_ordered_e, e = _filter_and_sort_conformers(mol, energies, remove_unconverged)

    if lowest_e_only: # keep only one conformer, the first and lowest energy
        confs = mol_ordered_e.GetConformers()
        for c_idx in range(len(confs) - 1, 0, -1):
            mol_ordered_e.RemoveConformer(confs[c_idx].GetId())
            del e[c_idx]

    return mol_ordered_e, e

def generate_conformers(smiles, lowest_e_only=True, num_threads=0):

    # first step standardize the molecule
    std_mol = Chem.MolFromSmiles(rdMolStandardize.StandardizeSmiles(smiles))

    # how many confs to generate?
    conf_num = _determine_conf_num(std_mol)

    # add hydrogens, critical for good conformer generation
    std_mol_h = Chem.AddHs(std_mol)

    # run ETKDG (by default)

    cids = AllChem.EmbedMultipleConfs(std_mol_h, conf_num, params = _get_conf_gen_params(num_threads))

    # energy minimize, we need this
    mol_e_min, e = _energy_minimize(std_mol_h, lowest_e_only = True)

    return Chem.RemoveHs(mol_e_min), e

