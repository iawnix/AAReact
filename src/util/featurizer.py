from typing import Any, Dict
from unimol_tools import UniMolRepr
import numpy as np
from numpy.typing import NDArray

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdmolops, DataStructs, rdFingerprintGenerator, rdDepictor, Descriptors
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors


from dscribe.descriptors import ACSF,SOAP,LMBTR,MBTR

import sys
from typing import List, Union, Tuple

def condition_featurizer():
    pass

def mol_featurizer(model_size: str) -> NDArray:
    clf = UniMolRepr(data_type='molecule', 
        remove_hs=False,
        model_name="unimolv2",
        model_size=model_size,
        use_cuda=True
        )
    return clf

def calc_mogan_fp(smiles: str, radius: int = 2, n_bits: int = 1024, use_features: bool = False) -> ExplicitBitVect:
    mol = Chem.MolFromSmiles(smiles)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius = radius, fpSize = n_bits)
    fp = morgan_gen.GetFingerprint(mol)
    return fp

def calc_rdkit_descrip(smiles: str) -> Tuple[List, NDArray]: 
    mol = Chem.MolFromSmiles(smiles)
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    return descs, np.array(desc_calc.CalcDescriptors(mol))


def __calc_mol_center_of_mass__(mol: Mol) -> int:
    """
    该函数用于计算距离分子质心最近的原子
    """
    pos = mol.GetConformer().GetPositions()
    atom_mass = np.array([tmp_atom.GetMass() for tmp_atom in mol.GetAtoms()]).reshape(-1,1)
    atom_mass = np.concatenate([atom_mass,atom_mass,atom_mass],axis=1)
    mass_center = np.sum(pos*atom_mass,axis=0)/atom_mass.sum()
    mass_center_idx = np.argmin(np.sum((pos - mass_center)**2,axis=1))
    return mass_center_idx

def calc_dscribe_descrip(sdf_fp: str, config, sanitize: bool = True) -> Any:

    atom_syms: List[str] = []
    dscriber: Any = None

    # rdkit mol -> ASE mol
    mol = Chem.SDMolSupplier(sdf_fp, removeHs=False, sanitize = sanitize)[0]
    if mol is None:
        print("Error[iaw]>: cannot read mol from sdf file, {}!".format(sdf_fp))
        sys.exit(1)
    
    for atom in mol.GetAtoms():
        atom_syms.append(atom.GetSymbol())
    
    match config.mol_type:
        case "reactant":
            pass
        case "product":
            pass
        case "solvent":
            pass
        case "catalyst":
            pass
        case _:
            print("Error[iaw]>: only support 'reactant', 'product', 'solvent' and 'catalyst' mol_type!")
            sys.exit(1)


    match config.type:
        case 'soap':
            dscriber = SOAP()
        case 'acsf':
            dscriber = ACSF()
        case 'lmbtr':
            dscriber = LMBTR()
        case 'mbtr':
            dscriber = MBTR()
        case _:
            print("Error[iaw]>: only support 'soap', 'acsf', 'lmbtr' and 'mbtr' !")
            sys.exit(1)



    