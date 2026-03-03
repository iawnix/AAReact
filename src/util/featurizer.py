from tkinter import N
from typing import Any, Dict
import venv
import numpy as np
from numpy.typing import NDArray

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdmolops, DataStructs, rdFingerprintGenerator, rdDepictor, Descriptors
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors


from dscribe.descriptors import ACSF,SOAP,LMBTR,MBTR

from ase import Atoms as ASE_ATOMS
import ase


import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util.constants import METAL_TYPE, ELEMENT_LIST

from typing import List, Union, Tuple

def condition_featurizer():
    pass

def mol_featurizer(model_size: str) -> NDArray:
    from unimol_tools import UniMolRepr
    clf = UniMolRepr(data_type='molecule', 
        remove_hs=False,
        model_name="unimolv2",
        model_size=model_size,
        use_cuda=True
        )
    return clf

class rdkit_featurizer():
    def __init__(self, smi: str) -> None:
        self.mol = Chem.MolFromSmiles(smi)

        self.morgan_generator: Any = None
        self.rdkit_desc_name = [desc_name[0] for desc_name in Descriptors._descList]
        self.rdkit_desc_generator: Any = MoleculeDescriptors.MolecularDescriptorCalculator(self.rdkit_desc_name)

    def calc_morgan_fp(self, radius: int = 2, n_bits: int = 1024, use_features: bool = False) -> ExplicitBitVect:
        if self.morgan_generator == None:
            self.morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius = radius, fpSize = n_bits)
        fp = self.morgan_generator.GetFingerprint(self.mol)
        return fp
    
    def calc_rdkit_descrip(self) -> NDArray: 
        return np.array(self.rdkit_desc_generator.CalcDescriptors(self.mol))

class dscribe_featurizer():
    def __init__(self, sdf_fp: str, sanitize: bool = True) -> None:
        self.mol = Chem.SDMolSupplier(sdf_fp, removeHs=False, sanitize = sanitize)[0]
        self.fp = sdf_fp
        if self.mol is None:
            print("Error[iaw]>: cannot read mol from sdf file, {}!".format(sdf_fp))
            sys.exit(1)
        self.metal_type = METAL_TYPE
        self.atom_syms = ELEMENT_LIST
        self.atoms = self.convert_ASE_mol()

        self.center_atom_idx = None

    def __calc_mol_center_of_mass__(self) -> int:
        """
        该函数用于计算距离分子质心最近的原子的索引
        """
        pos = self.mol.GetConformer().GetPositions()
        atom_mass = np.array([tmp_atom.GetMass() for tmp_atom in self.mol.GetAtoms()]).reshape(-1,1)
        atom_mass = np.concatenate([atom_mass,atom_mass,atom_mass],axis=1)
        mass_center = np.sum(pos*atom_mass,axis=0)/atom_mass.sum()
        mass_center_idx = np.argmin(np.sum((pos - mass_center)**2,axis=1))

        return int(mass_center_idx)

    def convert_ASE_mol(self) -> ase.atoms.Atoms:
        positions = self.mol.GetConformer().GetPositions()
        atom_syms = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        atoms = ASE_ATOMS(symbols = atom_syms, positions = positions, velocities = None, momenta = np.zeros_like(positions))
        #if hasattr(atoms, 'initial_momenta'):
        #    del atoms.initial_momenta
        #if hasattr(atoms, 'momenta'):
        #    del atoms.momenta

        return atoms

    def init_featurizer(self, config) -> Any:
        match config.type:
            case 'soap':
                dscriber = SOAP(species = self.atom_syms
                                , r_cut = config.rcut
                                , n_max = config.nmax
                                , l_max = config.lmax)
            case 'acsf':
                dscriber = ACSF(species=self.atom_syms
                                , r_cut=config.rcut
                                , g2_params = config.g2_params
                                , g4_params = config.g4_params)
            case 'lmbtr':
                dscriber = LMBTR(species = self.atom_syms
                                 , geometry = config.geometry
                                 , grid = config.grid
                                 , weighting = config.weighting
                                 , periodic=False)
            case 'mbtr':
                dscriber = MBTR(species = self.atom_syms
                                , geometry = config.geometry
                                , grid = config.grid
                                , weighting = config.weighting
                                , periodic = False)
            case _:
                print("Error[iaw]>: only support 'soap', 'acsf', 'lmbtr' and 'mbtr' !")
                sys.exit(1)
        
        match config.mol_type:
            case "reactant":
                self.center_atom_idx = [self.__calc_mol_center_of_mass__()]
            case "product":
                self.center_atom_idx = [self.__calc_mol_center_of_mass__()]
            case "solvent":
                self.center_atom_idx = [self.__calc_mol_center_of_mass__()]
            case "catalyst":
                self.center_atom_idx = []
                for atom in self.mol.GetAtoms():
                    if atom.GetSymbol() in self.metal_type:
                        self.center_atom_idx.append(atom.GetIdx())
                
                if len(self.center_atom_idx) == 0:
                    self.center_atom_idx = None
                    print("Warning[iaw]>: no metal atom found in catalyst mol! fp, {}".format(self.fp))
                    sys.exit(1)
                elif len(self.center_atom_idx) > 1:
                    print("Warning[iaw]>: more than one metal atom found in catalyst mol, use first metal atom as center atoms!")
                    # 中心原子需要确定一下, 暂时取第一个
                    self.center_atom_idx = [self.center_atom_idx[0]]
                else:
                    pass
            case _:
                print("Error[iaw]>: only support 'reactant', 'product', 'solvent' and 'catalyst' mol_type!")
                sys.exit(1)

        return dscriber

    def calc_soap(self, config) -> NDArray:
        dscriber = self.init_featurizer(config)
        #print("Debug[iaw]>: n_atoms in mol: {}, n_sym in mol: {}".format(len(self.atoms), len(self.atom_syms)))
        soap_feat = dscriber.create(self.atoms, centers = self.center_atom_idx)
        return soap_feat

    def calc_acsf(self, config) -> NDArray:
        dscriber = self.init_featurizer(config)
        #print("Debug[iaw]>: n_atoms in mol: {}, n_sym in mol: {}".format(len(self.atoms), len(self.atom_syms)))
        acsf_feat = dscriber.create(self.atoms, centers = self.center_atom_idx)
        return acsf_feat
    
    def calc_lmbtr(self, config) -> NDArray:
        dscriber = self.init_featurizer(config)
        #print("Debug[iaw]>: n_atoms in mol: {}, n_sym in mol: {}".format(len(self.atoms), len(self.atom_syms)))
        lmbtr_feat = dscriber.create(self.atoms, centers = self.center_atom_idx)
        return lmbtr_feat
    
    def calc_mbtr(self, config) -> NDArray:
        dscriber = self.init_featurizer(config)
        #print("Debug[iaw]>: n_atoms in mol: {}, n_sym in mol: {}".format(len(self.atoms), len(self.atom_syms)))
        mbtr_feat = dscriber.create(self.atoms)
        return mbtr_feat
    



    