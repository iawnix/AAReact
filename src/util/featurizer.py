from typing import Any, Dict, List
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

import os
import re
import sys

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config.constants import METAL_TYPE, ELEMENT_LIST, HOMO_LUMO_GAP_NUM_2, HOMO_LUMO_GAP_NUM_4, SOAP_FIX_PARAMETER
from tool.cli import CMD_RUN
from tool.molden_xtb import molden_mol

from typing import List, Union, Tuple

import shutil

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

    def reset_rdkit_desc_generator(self, desc_name_list: List[str]) -> None:
        self.rdkit_desc_name = desc_name_list
        self.rdkit_desc_generator = MoleculeDescriptors.MolecularDescriptorCalculator(self.rdkit_desc_name)

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
                temp_center_atom_idx = []
                # 催化剂的时候哦, 这个地方先存放的是一个元组, 包含元素类型以及idx
                for atom in self.mol.GetAtoms():
                    if atom.GetSymbol() in self.metal_type:
                        temp_center_atom_idx.append((atom.GetSymbol(), atom.GetIdx()))
                
                if len(temp_center_atom_idx) == 0:
                    self.center_atom_idx = None
                    print("Warning[iaw]>: no metal atom found in catalyst mol! fp, {}".format(self.fp))
                    sys.exit(1)
                elif len(temp_center_atom_idx) > 1:
                    #print("Warning[iaw]>: more than one metal atom found in catalyst mol, use first metal atom as center atoms!")
                    # 中心原子需要确定一下, 暂时取第一个
                    # 20260320更新, 如果存在Fe则先排除Fe, 然后再随机选择一个
                    self.center_atom_idx = [] 
                    for (i_sym, i_idx) in temp_center_atom_idx:
                        if i_sym != "Fe":
                            self.center_atom_idx.append(i_idx)
                    self.center_atom_idx = [self.center_atom_idx[0]]
                else:
                    # 不出意外, 这里铁定是1, 老铁!
                    self.center_atom_idx = [temp_center_atom_idx[0][-1]]
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



def xtb_log_to_data(log_path: str, sign: str, addition: Any = None) -> Union[Tuple[Tuple[Union[NDArray, None], bool], Tuple[Union[NDArray, None], bool]], Tuple[Union[NDArray, None], bool]]:
    """
    这个函数会依赖于grep命令, addition 主要为了给HomoLumoGap传入HOMO_LUMO_GAP_NUM
    """

    if sign == "Vipea":
        # Vertical Ionization Potentials and Electron Affinities
        out1 = CMD_RUN("grep 'delta SCC IP (eV):' {}".format(log_path))
        ip_error_if = False
        if type(out1) == bool:
            ip_error_if = True
            ip = None
        else:
            if out1 == "":
                ip_error_if = True
                ip = None
            else:
                ip = np.array([float(
                    out1.rstrip("\n").replace("delta SCC IP (eV):", "").replace(" ", "")
                )])

        out2 = CMD_RUN("grep 'delta SCC EA (eV):' {}".format(log_path))
        ea_error_if = False
        if type(out2) == bool:
            ea_error_if = True
            ea = None
        else:
            if out2 == "":
                ea_error_if = True
                ea = None
            else:
                ea = np.array([float(
                    out2.rstrip("\n").replace("delta SCC EA (eV):", "").replace(" ", "")
                )])
        return (ip, ip_error_if), (ea, ea_error_if)
    elif sign == "Vfukui":
        # Fukui Index

        with open(log_path, "r+") as F:
            lines = F.readlines()
            start = None
            end = -1

            for i, line in enumerate(lines):
                if "Fukui functions:" in line:
                    start = i

                if (
                    type(start) == int
                    and "-------------------------------------------------" in line
                ):
                    end = i - 1
                    break

            # assert type(start) == int and end > start+2, "Error: Can not find Fukui functions in {}".format(log_path)
            if type(start) != int and end == -1:
                fukui_data = np.array([])
                fukui_error_if = False
                # print("Here")
            else:
                # print( type(start) != int , type(end) != int , end, start)
                fukui_error_if = False
                fukui_data = []
                if not fukui_error_if:
                    for line in lines[start + 2 : end + 1]:
                        line1 = line.rstrip("\n")
                        line1 = line1.replace("\t", " ").replace("  ", " ")
                        var = line1.split(" ")
                        var = [i for i in var if i != ""]

                        atm_idx = eval(
                            re.findall(r"\d+|[A-Za-z]+", var[0])[0]
                        )  # 1C 这里需要提取数字
                        fukui_plus = eval(var[1])
                        fukui_minus = eval(var[2])
                        fukui_radical = eval(var[3])

                        fukui_data.append([fukui_plus, fukui_minus, fukui_radical])
                    fukui_data = np.array(fukui_data)

            return (fukui_data, fukui_error_if)

    elif sign == "Vomega":
        # Global Electrophilicity Index
        out1 = CMD_RUN("grep 'Global electrophilicity index (eV):' {}".format(log_path))
        gei_error_if = False
        if isinstance(out1, bool):
            gei_error_if = True
            gei = None
        else:
            if out1 == "":
                gei_error_if = True
                gei = None
            else:
                gei = np.array([float(
                    out1.rstrip("\n")
                    .replace("Global electrophilicity index (eV):", "")
                    .replace(" ", "")
                )])
        return (gei, gei_error_if)

    elif sign == "HomoLumoGap":
        hlg_error_if = False
        
        # 这个时候必须需要这个额外的参数
        if addition == None and isinstance(addition, int):
            return (None, True)
        
        n_hlg = addition
        try:
            molden = molden_mol(log_path)

            # 计算homo_lumo的gap
            lumo_ene, homo_ene = molden.FO()
            if (n_l := lumo_ene.shape[0]) < n_hlg and ( n_h := homo_ene.shape[0]) < n_hlg:
                print("Error[iaw]:> The number of molecular orbitals of this molecule is not sufficient to calculate a sufficient gap: HOMO: {}, LUMO: {}".format(
                    n_h, n_l))
            
            lumo_ene_select = lumo_ene[:n_hlg]
            homo_ene_select = homo_ene[:n_hlg]

            lumo_homo_diff = np.subtract.outer(lumo_ene_select, homo_ene_select)
        except:
            lumo_homo_diff =  None
            hlg_error_if = True

        return (lumo_homo_diff, hlg_error_if)
    else:
        raise RuntimeError("Error[iaw]:> Support Keys: Vipea, Vfukui, Vomega, HomoLumoGap")

class xtb_featurizer():

    def __init__(self, sdf_fp: str, sanitize: bool = True) -> None:
        self.mol = Chem.SDMolSupplier(sdf_fp, removeHs=False, sanitize = sanitize)[0]
        self.fp = sdf_fp
        if self.mol is None:
            print("Error[iaw]>: cannot read mol from sdf file, {}!".format(sdf_fp))
            sys.exit(1)

    def __calc_chrg_uhf__(self) -> Tuple[float, float]:
        chrg = Chem.GetFormalCharge(self.mol)
        re = Descriptors.NumRadicalElectrons(self.mol)
        ve = Descriptors.NumValenceElectrons(self.mol)

        if (ve + chrg) % 2 == 0 and re == 0:
            uhf = 0
        else:
            uhf = re
        return (chrg, uhf)

    def __init_featurizer__(self, config) -> Any:
        """
        初始化一个路径
        config:
            bachend:
            workpath: 
        """
        root_path = os.getcwd()
        if os.path.exists(config.workpath):
            shutil.rmtree(config.workpath)
        os.makedirs(config.workpath, exist_ok=True)
        os.chdir(config.workpath)
        return root_path
    
    def calc_xtb(self, config) -> Union[NDArray, None]:
        self.root_path = self.__init_featurizer__(config)
        
        try:
            chrg, uhf = self.__calc_chrg_uhf__()
            # 这SB 不支持V3000 的SDF
            _tmp_xyz = os.path.basename(self.fp)[:-len(".sdf")]
            CMD_RUN("{} -isdf {} -oxyz -O {}.xyz".format(config.obabel_bachend, self.fp, _tmp_xyz))
            CMD_RUN("{} {}.xyz --opt normal --ohess --gfn 1 --chrg {} --uhf {}  --molden > opt.log".format(
                        config.xtb_bachend, _tmp_xyz, chrg, uhf))
            CMD_RUN("mv wbo wbo.opt")
            CMD_RUN("mv charges charges.opt")

            CMD_RUN("{} xtbopt.xyz --gfn 1 --chrg {} --uhf {} --vipea > Vipea.log".format(
                    config.xtb_bachend, chrg, uhf))  
        
            # 这里加入--sp会造成错误
            CMD_RUN("mv wbo wbo.Vipea")
            CMD_RUN("mv charges charges.Vipea")
            #CMD_RUN("{} xtbopt.xyz --gfn 2 --chrg {} --uhf {} --vfukui > Vfukui.log".format(
            #        config.xtb_bachend, chrg, uhf))
        
            #CMD_RUN("mv wbo wbo.Vfukui")
            #CMD_RUN("mv charges charges.Vfukui")
            CMD_RUN("{} xtbopt.xyz --gfn 1 --chrg {} --uhf {} --vomega > Vomega.log".format(
                    config.xtb_bachend, chrg, uhf))
            CMD_RUN("mv wbo wbo.Vomega")
            CMD_RUN("mv charges charges.Vomega")

            # 读取数据
            (ip, ip_error_if), (ea, ea_error_if) = xtb_log_to_data(
                log_path="./Vipea.log", sign="Vipea"
            )
            #fukui_data, fukui_error_if = xtb_log_to_data(
            #    log_path="./Vfukui.log", sign="Vfukui"
            #)
            gei, gei_error_if = xtb_log_to_data(log_path="./Vomega.log", sign="Vomega")
            match config.mol_type:
                case "reactant":
                    n_homo_lumo_gap = HOMO_LUMO_GAP_NUM_4
                case "product":
                    n_homo_lumo_gap = HOMO_LUMO_GAP_NUM_4
                case "solvent":
                    n_homo_lumo_gap = HOMO_LUMO_GAP_NUM_4
                case "catalyst":
                    n_homo_lumo_gap = HOMO_LUMO_GAP_NUM_4
                case _:
                    print("Error[iaw]>: only support 'reactant', 'product', 'solvent' and 'catalyst' mol_type!")
                    sys.exit(-1)
            hlg, hlg_error_if = xtb_log_to_data(log_path="./molden.input", sign="HomoLumoGap", addition=n_homo_lumo_gap)

            all_feat = None
            if not ip_error_if and not ea_error_if and not gei_error_if and not hlg_error_if:
                #print("Debug[iaw]:> ip.shape: {}, ea.shape: {}, gei.shape: {}, hlg.shape: {}".format(ip.shape, 
                #                                     ea.shape, gei.shape, hlg.shape))
                # -> Debug[iaw]:> ip.shape: (1,), ea.shape: (1,), gei.shape: (1,), hlg.shape: (4, 4)
                all_feat = np.concatenate([ip, ea, gei, hlg.flatten()])
            #else:
            #    print(ip_error_if, ea_error_if, gei_error_if, hlg_error_if)
        except Exception as e:
            print("Error[iaw]:> {}".format(e))
            all_feat = None
        finally:
            os.chdir(self.root_path)
        return all_feat

def soap_label_search(label_idx: int) -> Union[str, None]:
    
    # init label
    label = []
    for i in ELEMENT_LIST:
        for j in ELEMENT_LIST:
            for h in range(SOAP_FIX_PARAMETER["nmax"]):
                for k in range(SOAP_FIX_PARAMETER["lmax"]):
                    label.append("{}-{}-n_{}-l_{}".format(i, j, h, k))
    
    if label_idx >= (n_label:=len(label)):
        print("Error[iaw]:> Check index[{}], out of label range (the num of label is {})!".format(label_idx, n_label))
        out = None
    else:
        out = label[label_idx]
    return out

def xtb_label_search(label_idx: int) -> Union[str, None]:
    
    # init label
    label = ["IP", "EA", "GEI"]
    for i in range(HOMO_LUMO_GAP_NUM_4):
        for j in range(HOMO_LUMO_GAP_NUM_4):
            label.append("LH-{}_{}".format(i, j))
            
    if label_idx >= (n_label:=len(label)):
        print("Error[iaw]:> Check index[{}], out of label range (the num of label is {})!".format(label_idx, n_label))
        out = None
    else:
        out = label[label_idx]
    return out

    