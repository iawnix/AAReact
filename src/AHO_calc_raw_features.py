import pandas as pd
import numpy as np
from numpy.typing import NDArray
from rich.progress import track
from types import SimpleNamespace
import os
import sys

from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util.featurizer import dscribe_featurizer, rdkit_featurizer, xtb_featurizer
from config.constants import SOAP_FIX_PARAMETER, ACSF_FIX_PARAMETER, XTB_BACHEND, OBABEL_BACHEND, XTB_WORK_SCRATCH, RAW_CSV_COLUMNS, SUPPORTED_DESC_TYPES
from copy import deepcopy

from typing import List, Tuple, Dict, Any, Union

import argparse
from argparse import Namespace

"""
# 重新设计
1. 统一csv的格式, 删除冗余的FILE_PATH的信息
2. 加入缓存机制, 避免重复计算同一分子的特征
3. CLASS按照REA的类型进行划分
"""

def check_raw_csv(data_fp: str, data: Union[pd.DataFrame, False] =  False) -> bool:
    if data is False:
        dat = pd.read_csv(data_fp)
    else:
        dat = data

    dat_columns = dat.columns.to_list()
    if dat_columns == RAW_CSV_COLUMNS:
        return True
    else:
        return False

def calc_features(dat: pd.DataFrame, desc_type: str, out_fp: str) -> None:

    cache = {}
    features_name = None
    features_data = []
    features_class = []

    smi_col_n = ["REA_SMI", "SOL_SMI", "CAT_SMI"]
    name_col_n = ["REA_NAME", "SOL_NAME", "CAT_NAME"]

    for i in track(range(dat.shape[0])):
        # smi_s: rea_smi, sol_smi, cat_smi
        i_smi_s = dat.loc[i, smi_col_n].to_list()

        # sdf_fp_s: rea_sdf_fp, sol_sdf_fp, cat_sdf_fp
        i_sdf_s = [os.path.join(SDF_HOME, name + ".sdf") for name in dat.loc[i, name_col_n].to_list()]

        # name_s: rea_name, sol_name, cat_name
        i_name_s = dat.loc[i, name_col_n].to_list()

        # get class based on rea_idx
        i_rea_idx = dat.loc[i, "REA_NAME"].split("-")[1]
        features_class.append(int(i_rea_idx))

        if i == 0:
            first = True
        else:
            first = False

        # calc feat
        if desc_type == "xtb":
            i_feat, i_feat_name, cache = calc_xtb_features(i_sdf_s, i_name_s, cache, first)
        elif desc_type == "rdkit":
            i_feat, i_feat_name, cache = calc_rdkit_desc_features(i_smi_s, i_name_s, cache, first)
        elif desc_type == "morgan":
            i_feat, i_feat_name, cache = calc_rdkit_morgan_features(i_smi_s, i_name_s, cache, first)
        elif desc_type == "soap":
            i_feat, i_feat_name, cache = calc_soap_features(i_sdf_s, i_name_s, cache, first)
        elif desc_type == "acsf":
            i_feat, i_feat_name, cache = calc_acsf_features(i_sdf_s, i_name_s, cache, first)
        else:
            print("Error[iaw]:> Unsupported type, only supported types: {}.".format(", ".join(SUPPORTED_DESC_TYPES)))

        if features_name is None and first == True:
            features_name = i_feat_name
        features_data.append(i_feat)
    
    if features_name is None:
        print("Error[iaw]:> Can not gen feature names!")
        
    # merge all data's feat
    dat = pd.concat([dat, pd.DataFrame(features_data, columns=features_name), pd.DataFrame({"CLASS": features_class})], axis=1)
    dat.to_csv(out_fp, index = False)

def calc_xtb_features(sdf_s: Tuple[str, str, str]
                      , name_s: Tuple[str, str, str]
                      , cache: Dict[str, Any], first: bool = False) -> Tuple[NDArray, List[str], Dict[str, Any]]:
    
    out_feat = []
    features_name = []

    for idx, i_sdf in enumerate(sdf_s):
        i_name = name_s[idx]
        if i_name in cache.keys():
            out_feat.append(cache[i_name])
        else:
            match i_name[:3]:
                case "CAT":
                    mol_type = "catalyst"
                case "REA":
                    mol_type = "reactant"
                case "PRO":
                    mol_type = "product"
                case "SOL":
                    mol_type = "solvent"             
                case _:
                    print("Error[iaw]>: unsupport!")
                    mol_type = "NAN"
            xtb_config = SimpleNamespace(**{
                  "mol_type": mol_type
                , "xtb_bachend": XTB_BACHEND
                , "obabel_bachend": OBABEL_BACHEND
                , "workpath": XTB_WORK_SCRATCH})
            
            i_featurizer = xtb_featurizer(sdf_fp = i_sdf)
            _i_feat = i_featurizer.calc_xtb(xtb_config)
            out_feat.append(_i_feat)
            cache[i_name] = deepcopy(_i_feat)

        if first:
            n_xtb = out_feat[0].shape[-1]
            for j in range(n_xtb):
                features_name.append("{}_XTB{}".format(i_name[:-len("_NAME")], j))

    out_feat = np.array(out_feat).flatten()
    return out_feat, features_name, cache

def calc_rdkit_desc_features(smi_s: Tuple[str, str, str]
                      , name_s: Tuple[str, str, str]
                      , cache: Dict[str, Any], first: bool = False) -> Tuple[NDArray, List[str], Dict[str, Any]]:
    out_feat = []
    features_name = []

    for idx, i_smi in enumerate(smi_s):
        i_name = name_s[idx]
        if i_name in cache.keys():
            out_feat.append(cache[i_name])
        else:
            i_featurizer = rdkit_featurizer(smi = i_smi)
            _i_feat = i_featurizer.calc_rdkit_descrip()
            out_feat.append(_i_feat)
            cache[i_name] = deepcopy(_i_feat)

        # init feat_name when i == 0
        if first == 0:
            for j in i_featurizer.rdkit_desc_name:
                features_name.append("{}_{}".format(i_name[:-len("_NAME")], j))

    out_feat = np.array(out_feat).flatten()
    return out_feat, features_name, cache

def calc_rdkit_morgan_features(smi_s: Tuple[str, str, str]
                      , name_s: Tuple[str, str, str]
                      , cache: Dict[str, Any], first: bool = False) -> Tuple[NDArray, List[str], Dict[str, Any]]:
    out_feat = []
    features_name = []

    for idx, i_smi in enumerate(smi_s):
        i_name = name_s[idx]
        if i_name in cache.keys():
            out_feat.append(cache[i_name])
        else:

            i_featurizer = rdkit_featurizer(smi = i_smi)
            _i_feat = i_featurizer.calc_morgan_fp().ToList()
            out_feat.append(_i_feat)
            cache[i_name] = deepcopy(_i_feat)

        # init feat_name when i == 0
        if first == 0:
            for j in range(1024):
                features_name.append("{}_MORGAN{}".format(i_name[:-len("_NAME")], j))

    out_feat = np.array(out_feat).flatten()
    return out_feat, features_name, cache

def calc_soap_features(sdf_s: Tuple[str, str, str]
                      , name_s: Tuple[str, str, str]
                      , cache: Dict[str, Any], first: bool = False) -> Tuple[NDArray, List[str], Dict[str, Any]]:
    out_feat = []
    features_name = []

    for idx, i_sdf in enumerate(sdf_s):
        i_name = name_s[idx]

        if i_name in cache.keys():
            out_feat.append(cache[i_name])
        else:
            match i_name[:3]:
                case "CAT":
                    mol_type = "catalyst"
                case "REA":
                    mol_type = "reactant"
                case "PRO":
                    mol_type = "product"
                case "SOL":
                    mol_type = "solvent"             
                case _:
                    print("Error[iaw]>: unsupport!")
                    mol_type = "NAN"

            soap_config = SimpleNamespace(**{
                  "type": "soap"
                , "mol_type": mol_type
                , "rcut": SOAP_FIX_PARAMETER["rcut"]
                , "nmax": SOAP_FIX_PARAMETER["nmax"]
                , "lmax": SOAP_FIX_PARAMETER["lmax"]})
            
            i_featurizer = dscribe_featurizer(sdf_fp = i_sdf)
            _i_feat = i_featurizer.calc_soap(soap_config)
            out_feat.append(_i_feat)
            cache[i_name] = deepcopy(_i_feat)
            #print(mol_type, len(i_feat), i_feat[-1].shape)

        if first == 0:
            n_soap = out_feat[0].shape[-1]
            for j in range(n_soap):
                features_name.append("{}_SOAP{}".format(i_name[:-len("_NAME")], j))
    out_feat = np.array(out_feat).flatten()
    return out_feat, features_name, cache

def calc_acsf_features(sdf_s: Tuple[str, str, str]
                      , name_s: Tuple[str, str, str]
                      , cache: Dict[str, Any], first: bool = False) -> Tuple[NDArray, List[str], Dict[str, Any]]:
    out_feat = []
    features_name = []

    for idx, i_sdf in enumerate(sdf_s):
        i_name = name_s[idx]

        if i_name in cache.keys():
            out_feat.append(cache[i_name])
        else:
            match i_name[:3]:
                case "CAT":
                    mol_type = "catalyst"
                case "REA":
                    mol_type = "reactant"
                case "PRO":
                    mol_type = "product"
                case "SOL":
                    mol_type = "solvent"             
                case _:
                    print("Error[iaw]>: unsupport!")
                    mol_type = "NAN"

            acsf_config = SimpleNamespace(**{
                  "type": "acsf"
                , "mol_type": mol_type
                , "rcut": ACSF_FIX_PARAMETER["rcut"]
                , "g2_params": ACSF_FIX_PARAMETER["g2_params"]
                , "g4_params": ACSF_FIX_PARAMETER["g4_params"]})

            i_featurizer = dscribe_featurizer(sdf_fp = i_sdf)
            _i_feat = i_featurizer.calc_acsf(acsf_config)
            out_feat.append(_i_feat)
            cache[i_name] = deepcopy(_i_feat)
            #print(mol_type, len(i_feat), i_feat[-1].shape)

        if first == 0:
            n_acsf = out_feat[0].shape[-1]
            for j in range(n_acsf):
                features_name.append("{}_ACSF{}".format(i_name[:-len("_NAME")], j))
    out_feat = np.array(out_feat).flatten()

    return out_feat, features_name, cache


def Parm() -> Namespace:
    """
    cli 参数
    """
    parser = argparse.ArgumentParser(description="AAReact: For generating the initial feature file.")
    parser.add_argument("--in_csv", type=str, help="Input CSV file with fields: {}".format(", ".join(RAW_CSV_COLUMNS)))
    parser.add_argument("--desc_type", type=str, help="Descriptor type. Choices: {}".format(", ".join(SUPPORTED_DESC_TYPES)), 
                        choices=["all"] + SUPPORTED_DESC_TYPES)
    parser.add_argument("--sdf_path", type=str, help="SDF file path (required if desc_type=xtb/soap/acsf/all)")
    parser.add_argument("--save_path", type=str, help="Output Directory.")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    global N_SAMPLE, SDF_HOME
    myp = Parm()

    dat_fp = myp.in_csv
    dat = pd.read_csv(dat_fp)
    
    if not myp.sdf_path:
        if myp.desc_type in ["all", "xtb", "soap", "acsf"]:
            print("Error[iaw]:> SDF file path (required if desc_type=xtb/soap/acsf/all)")
            sys.exit(-1)
    else:
        SDF_HOME = myp.sdf_path


    if not check_raw_csv(dat_fp, dat):
        print("Error[iaw]:> Input CSV file with fields: {}".format(", ".join(RAW_CSV_COLUMNS)))
        sys.exit(-1)

    
    N_SAMPLE = dat.shape[0]
    out_path = myp.save_path
    if myp.desc_type == "all":
        for desc_type in SUPPORTED_DESC_TYPES:
            calc_features(dat, desc_type, out_fp=os.path.join(out_path, "{}_features.csv".format(desc_type)))
    else:
        calc_features(dat, myp.desc_type, out_fp=os.path.join(out_path, "{}_features.csv".format(myp.desc_type)))


