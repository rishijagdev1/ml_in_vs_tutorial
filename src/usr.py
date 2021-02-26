import logging as log

from rdkit.Chem import rdMolDescriptors

# controls loggin level
log.basicConfig(level=log.DEBUG)

def get_usr_descriptor(mol):
    return rdMolDescriptors.GetUSR(mol)

