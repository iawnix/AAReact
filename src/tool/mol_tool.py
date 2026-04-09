import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter, Draw, rdmolops, DataStructs, rdFingerprintGenerator, rdDepictor, Descriptors
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors

import numpy as np
from numpy.typing import NDArray
from typing import List, Union, Tuple

import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config.constants import METAL_TYPE, COORD_TYPE

from copy import deepcopy

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
        else:
            print("Warning[iaw]:> Please remove the bond between atom[{}] and atom[{}]!".format(i, meta_idx))
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
            nbr_atm = rw_mol.GetAtomWithIdx(nbr_idx)
            new_nbr_chrg = nbr_atm.GetFormalCharge() - 1
            nbr_atm.SetFormalCharge(new_nbr_chrg)
    new_mol = rw_mol.GetMol()
    Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    return new_mol

def unbond_metal_sym(mol:Mol, metal_Sym: List[str] = METAL_TYPE) -> Tuple[Mol, List[int], List[int]]:
    rw_mol = Chem.RWMol(mol)
    
    metal_idx = []
    coord_idx = []
    for atm in rw_mol.GetAtoms():
        if atm.GetSymbol() in metal_Sym:
            metal_idx.append(atm.GetIdx())
    for idx in metal_idx:
        atom = rw_mol.GetAtomWithIdx(idx)
        neighbor_indices = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        coord_idx.append(neighbor_indices)
        for nbr_idx in neighbor_indices:
            rw_mol.RemoveBond(idx, nbr_idx)
            nbr_atm = rw_mol.GetAtomWithIdx(nbr_idx)
            
            new_nbr_chrg = nbr_atm.GetFormalCharge() - 1
            nbr_atm.SetFormalCharge(new_nbr_chrg)
            
            #new_nbr_valence = nbr_atm.GetExplicitValence() -1
            #nbr_atm.SetExplicitValence(new_nbr_valence)

    new_mol = rw_mol.GetMol()
    Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    return new_mol, metal_idx, coord_idx

def find_coordinating_atoms(mol:Mol) -> List[int]:
    
    coord_atom_idx = []
    for atm in mol.GetAtoms():
        if atm.GetSymbol() not in COORD_TYPE:
            continue
        if atm.GetSymbol() in ["N", "P"]:
            if atm.GetDegree < 4:
                # 重N没有排除
                coord_atom_idx.append(atm.GetIdx())
        else:
            pass
        if atm.GetFormalCharge() > 0:
            continue
        hyb = atm.GetHybridization()
        if hyb in [Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]:
            coord_atom_idx.append(atm.GetIdx())
    
    return coord_atom_idx

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
        print("Info[iaw]>: ATOM[{}-{}], Charge: {}, Valence: {}, RadicalElectron: {}".format(atom.GetSymbol()
                                                                     , idx
                                                                     , atom.GetFormalCharge()
                                                                     , atom.GetTotalValence()
                                                                     , atom.GetNumRadicalElectrons()))

def find_atom_radicalE(mol: Mol) -> List[int]:
    out = []
    for idx in list(range(len(mol.GetAtoms()))):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetNumRadicalElectrons() != 0:
            print("Info[iaw]>: ATOM[{}-{}], Charge: {}, Valence: {}, RadicalElectron: {}".format(atom.GetSymbol()
                                                                         , idx
                                                                         , atom.GetFormalCharge()
                                                                         , atom.GetTotalValence()
                                                                         , atom.GetNumRadicalElectrons()))
            out.append(idx)
    return out

def correct_valence(mol: Mol, infos: Tuple[int, int, int]) -> Mol:
    mol = Chem.RWMol(mol)
    idx, radical, charge = infos
    atom = mol.GetAtomWithIdx(idx)
    atom.SetFormalCharge(charge)
    atom.SetNumRadicalElectrons(radical)
    mol = mol.GetMol()
    return mol


def split_cat_smi() -> Tuple[str, str]:
    pass


def repair_mol2(mol: Mol, ion_coor_s: List[Tuple[int, List[int]]]
                    , valence_info_s: List[Tuple[int, int, int]], print_out: bool = False) -> Union[Tuple[str, Mol], Tuple[None, None]]:
    """
    - `ion_coor_s` format: `[(int_1, (int2, ...)), ....]`
        - `(int_1, (int2, ...))`: group1
        - `int_1`: ion atom in group1
        - `(int2, ...)`: coordinating atoms in group1
    - `valence_info_s` format: `[(int_1, int_2, int_3), ...]`
        - `(int_1, int_2, int_3)`: atom1
        - `int_1`: index (0-based) of atom1
        - `int_2`: NumRadicalElectrons of atom1
        - `int_3`: FormalCharge of atom1
    """
    mol_fixed = deepcopy(mol)
    try:
        # repair 1
        for i_ion_coor in ion_coor_s:
            mol_fixed = add_dative(mol_fixed, i_ion_coor[0], i_ion_coor[1])
            mol_fixed.UpdatePropertyCache(strict=False)
    
        # repair 2
        for i_info in valence_info_s:
            mol_fixed = correct_valence(mol_fixed, i_info)
            mol_fixed.UpdatePropertyCache(strict=False)

        smi_fixed = Chem.MolToSmiles(mol_fixed, isomericSmiles=True)
    except:
        if print_out:
            print("Error[iaw]:> can not repair mol2")
        smi_fixed = None
        mol_fixed = None
    
    return (smi_fixed, mol_fixed)
   

def repair_mol2_and_save_as_sdf(mol_name: str, mol2_fp: str, sdf_saved_fp: str
                        , ion_coor_s: List[Tuple[int, List[int]]]
                        , valence_info_s: List[Tuple[int, int, int]]) -> Union[Tuple[str, Mol], Tuple[None, None]]:
    """
    - `ion_coor_s` format: `[(int_1, (int2, ...)), ....]`
        - `(int_1, (int2, ...))`: group1
        - `int_1`: ion atom in group1
        - `(int2, ...)`: coordinating atoms in group1
    - `valence_info_s` format: `[(int_1, int_2, int_3), ...]`
        - `(int_1, int_2, int_3)`: atom1
        - `int_1`: index (0-based) of atom1
        - `int_2`: NumRadicalElectrons of atom1
        - `int_3`: FormalCharge of atom1
    """
    try:
        # read mol2
        mol = Chem.MolFromMol2File(mol2_fp, removeHs = False)
    except:
        print("Error[iaw]:> can read mol from `{}`".format(mol2_fp))
    

    smi_fixed, mol_fixed = repair_mol2(mol, ion_coor_s, valence_info_s, print_out = False)
    if smi_fixed == None or mol_fixed == None:
        print("Error[iaw]:> can not repair mol2 `{}`".format(mol_name))
    else:
        with SDWriter(sdf_saved_fp) as writer:
            mol_fixed.SetProp("_Name", mol_name)
            mol_fixed.SetProp("SMILES", smi_fixed)
            writer.write(mol_fixed)

    return smi_fixed, mol_fixed





