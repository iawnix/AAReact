
from ast import Tuple
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdmolops, DataStructs, rdFingerprintGenerator, rdDepictor, Descriptors
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors

import numpy as np
from numpy.typing import NDArray
from typing import List, Union, Tuple

#+++++++++++++++++++++++++++++++++++++++# READ ME #++++++++++++++++++++++++++++++++++++++++++++++++#
# add_dative
# unbond_metal
# cal_mogan_fp
# gen_3D
# check_atom_charge
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def add_dative(mol: Mol, meta_idx: int, coord_idx: List) -> Mol:
    rw_mol = Chem.RWMol(mol)
    for i in coord_idx:
        if rw_mol.GetBondBetweenAtoms(i, meta_idx) is None:
            rw_mol.AddBond(i, meta_idx, order=Chem.BondType.DATIVE)
    new_mol = rw_mol.GetMol()
    return new_mol

def MolFromGVMol2(fp: str) ->Mol:
    mol_block = None
    with open(fp, "r+") as F:
        mol_block = "".join(F.readlines())
        mol_block = mol_block.replace("Ar", "1")
    mol = Chem.MolFromMol2Block(mol_block, removeHs=False, sanitize=False)
    return mol

def bond_all_to_single(mol:Mol) -> Mol:
    new_mol = Chem.Mol(mol)
    #for atom in new_mol.GetAtoms():
    #    atom.SetIsAromatic(False)
    for bond in new_mol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return new_mol

def unbond_metal(mol:Mol, meta_idx: List[int]) -> Mol:
    rw_mol = Chem.RWMol(mol)
    for idx in meta_idx:
        atom = rw_mol.GetAtomWithIdx(idx)
        neighbor_indices = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        for nbr_idx in neighbor_indices:
            rw_mol.RemoveBond(idx, nbr_idx)
    new_mol = rw_mol.GetMol()
    return new_mol

def cal_mogan_fp(smiles: str, radius: int = 2, n_bits: int = 1024, use_features: bool = False) -> ExplicitBitVect:
    mol = Chem.MolFromSmiles(smiles)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius = radius, fpSize = n_bits)
    fp = morgan_gen.GetFingerprint(mol)
    return fp

def cal_rdkit_descrip(smiles: str) -> Tuple[List, NDArray]: 
    mol = Chem.MolFromSmiles(smiles)
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    return descs, np.array(desc_calc.CalcDescriptors(mol))

def gen_3D(smi: str, mol: Union[bool, Mol] = False) -> Mol:
    if mol == False:
        mol = Chem.MolFromSmiles(smi)

    rdDepictor.SetPreferCoordGen(True)
    AllChem.Compute2DCoords(mol)
    mol_3d = Chem.AddHs(mol) 
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.maxIterations = 5000
    AllChem.EmbedMolecule(mol_3d, params)
    try:
        AllChem.MMFFOptimizeMolecule(mol_3d)
    except:
        AllChem.UFFOptimizeMolecule(mol_3d, ignoreInterfragInteractions=False)
    return mol_3d

def check_atom_charge(mol: Mol, atom_idx: List[int]) -> None:
    for idx in atom_idx:
        atom = mol.GetAtomWithIdx(idx)
        print("Info[iaw]>: ATOM[{}], charge: {}, Valence: {}".format(atom.GetSymbol()
                                                                     , atom.GetFormalCharge()
                                                                     , atom.GetTotalValence()))





