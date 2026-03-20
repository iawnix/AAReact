import argparse
from argparse import Namespace
import os
from re import T
import sys
from pathlib import Path
from joblib import dump, load
import pickle

from typing import Any, Callable, List, Tuple, Union, Dict

import numpy as np
from numpy.typing import NDArray

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from util.train_tools import norm_col_parms
from util.featurizer import rdkit_featurizer, dscribe_featurizer
from util.constants import XTB_BACHEND, OBABEL_BACHEND, XTB_WORK_SCRATCH
from rich import print as rp
from rich.status import Status

from types import SimpleNamespace

from rdkit import Chem

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++# READ ME #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# _split_feat: 内部函数, 用于辅助init_feat_label函数
# init_feat_label: 初始化待计算特征的标签类型
# Parm: 用于初始化终端中的输入
# _check_in_mol_type_is_sdf: 检查输入分子的格式是否为sdf, True: sdf, False: smi
# _calc_soap_feat: 内部函数, 计算输入分子的soap特征
# _calc_xtb_feat: 内部函数, 用于计算输入分子的xtb特征
# init_features: 初始化data_x, 计算输入数据的特征
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#




def _split_feat(feat_label_value: List[str], warning: bool = True) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    if warning:
        print("Warning[iaw]:> This function is only help function`init_feat_label` to split feat!")

    REA, SOL, CAT, TEMP, PRESSURE = [], [], [], [], []
    REA_idx, SOL_idx, CAT_idx, TEMP_idx, PRESSURE_idx = [], [], [], [], []
    for i, feat in enumerate(feat_label_value):
        if feat == "TEMP":
            TEMP.append(feat)
            TEMP_idx.append(i)
        elif feat == "PRESSURE":
            PRESSURE.append(feat)
            PRESSURE_idx.append(i)
        else:
            if feat.startswith("REA_"):
                REA.append(feat[len("REA_"):])
                REA_idx.append(i)
            elif feat.startswith("SOL_"):
                SOL.append(feat[len("SOL_"):])
                SOL_idx.append(i)
            elif feat.startswith("CAT_"):
                CAT.append(feat[len("CAT_"):])
                CAT_idx.append(i)
            else:
                print("Error[iaw]>: Unrecognized feature label: {}".format(feat))
    # Check feat的顺序必须是REA, SOL, CAT, TEMP, PRESSURE
    if REA_idx + SOL_idx + CAT_idx + TEMP_idx + PRESSURE_idx != list(range(len(feat_label_value))):
        print("Error[iaw]>: Feature label order is not REA, SOL, CAT, TEMP, PRESSURE")
    return REA, SOL, CAT, TEMP, PRESSURE


def init_feat_label(feat_label: Dict[str, List[str]]) -> List[Tuple[str, Tuple[List[str], List[str], List[str], List[str]], List[str]]]:

    """
    Dict[str, List[str]]: 嵌套的, 可以根据key的前半部分判断是何种描述符
        - 例如: `label1_rdkit`

    """
    out = []
    key_s = list(feat_label.keys())
    feat_key_withsign = sorted(key_s, key=lambda x: int(x.split('_')[0].replace('label', '')))
    feat_key = [item.split('_')[1] for item in feat_key_withsign]
    for i, i_feat_name in enumerate(feat_key):
        i_feat_splited = _split_feat(feat_label_value = feat_label[feat_key_withsign[i]], warning = False)
        out.append((i_feat_name, i_feat_splited))
    return out

def _check_in_mol_type_is_sdf(mol_s: Tuple[str, str, str], warning: bool = True) -> bool:
    if warning:
        print("Warning[iaw]:> This function is only used to check the input in main function!")
    
    sign_s = []
    
    for mol in mol_s:
        if os.path.exists(mol):
            sign_s.append(True)
        else:
            try:
                _mol = Chem.MolFromSmiles(mol, sanitize=False)
                if _mol is None:
                    raise RuntimeError("Error[iaw]:> the mol must be smi or sdf!")
            except:
                raise RuntimeError("Error[iaw]:> the mol must be smi or sdf!")
            sign_s.append(False)
    
    # 这里是取巧
    if len(sign_s) != 3:
        raise RuntimeError("Error[iaw]:> the mol must be 3!")

    match sum(sign_s): 
        case 3:
            return True
        case -3:
            return False
        case _:
            raise RuntimeError("Error[iaw]:> the mol must only be smi or sdf!")


def _calc_soap_feat(mol_type: str, sdf_fp: str, feat_label: List[str], warning: bool = True) -> NDArray:
    if warning:
        print("Warning[iaw]:> This function is only used to calc soap in main function!")

    soap_config = SimpleNamespace(**{
          "type": "soap"
        , "mol_type": mol_type
        , "rcut": 6.0
        , "nmax": 4
        , "lmax": 3})
    soap_featurizer = dscribe_featurizer(sdf_fp = sdf_fp)
    tmp_feat = soap_featurizer.calc_soap(soap_config)
    # tmp_feat: 1, n_soap
    n_tmp_feat = tmp_feat.shape[-1]
    tmp_feat_name = {"SOAP{}".format(j): j for j in range(n_tmp_feat)}
    select_idx = [tmp_feat_name[i] for i in feat_label]
    #print("Debug[iaw]:> the tmp_feat shape, {}, type, {}".format(tmp_feat.shape, type(tmp_feat)))
    out = tmp_feat[:, select_idx]
    # 1, n_select_soap -> n_select_soap,
    out = out.flatten()
    return out

def _calc_xtb_feat(mol_type: str, sdf_fp: str, feat_label: List[str], warning: bool = True) -> NDArray:
    if warning:
        print("Warning[iaw]:> This function is only used to calc soap in main function!")

    xtb_config = SimpleNamespace(**{
          "mol_type": mol_type
        ,  "xtb_bachend": XTB_BACHEND
        , "obabel_bachend": OBABEL_BACHEND
        , "workpath": XTB_WORK_SCRATCH
    })

    xtb_featurizer = xtb_featurizer(sdf_fp = sdf_fp)
    tmp_feat = xtb_featurizer.calc_xtb(xtb_config)
    # tmp_feat: n_xtb, 
    n_tmp_feat = tmp_feat.shape[0]
    tmp_feat_name = {"XTB{}".format(j): j for j in range(n_tmp_feat)}
    select_idx = [tmp_feat_name[i] for i in feat_label]
    out = tmp_feat[select_idx]
    return out


def init_features(  mol_s: Tuple[str, str, str]
                   , feat_type: str
                   , feat: Tuple[str, Tuple[List[str], List[str], List[str], List[str]], List[str]]
                   , temp: Union[None, float], pressure: Union[None, float]
                   , first: bool = False):

    all_feat = []
    REA_feat_label, SOL_feat_label, CAT_feat_label, TEMP_feat_label, PRESSURE_feat_label = feat
    if first:
        # 如果first为True的时候, 必须要检查一下temp跟pressure其中的一个存在
        if len(TEMP_feat_label) == 0 and len(PRESSURE_feat_label) == 0:
            print("Error[iaw]:> please put the TEMP and PRESSURE in first feature_class!")
    
    _extract_smi_from_sdf: Callable = lambda sdf_fp: Chem.SDMolSupplier(sdf_fp, removeHs=False, sanitize=False)[0].GetProp("SMILES")

    """
    这个下面有大量的重复逻辑, 需要进一步优化
    """
    match feat_type:
        case "rdkit":
            # check mol_s is (smi, smi, smi)
            smi_s = []
            if _check_in_mol_type_is_sdf(mol_s, warning=False):
                for i_sdf in mol_s:
                    smi_s.append(_extract_smi_from_sdf(i_sdf))
            else:
                smi_s = mol_s
            rea_smi, sol_smi, cat_smi = smi_s

            if len(REA_feat_label) != 0:
                rea_featurizer = rdkit_featurizer(rea_smi)
                rea_featurizer.reset_rdkit_desc_generator(REA_feat_label)
                rea_desc = rea_featurizer.calc_rdkit_descrip()
                all_feat.append(rea_desc)
            if len(SOL_feat_label) != 0:
                sol_featurizer = rdkit_featurizer(sol_smi)
                sol_featurizer.reset_rdkit_desc_generator(SOL_feat_label)
                sol_desc = sol_featurizer.calc_rdkit_descrip()
                all_feat.append(sol_desc)
            if len(CAT_feat_label) != 0:
                cat_featurizer = rdkit_featurizer(cat_smi)
                cat_featurizer.reset_rdkit_desc_generator(CAT_feat_label)
                cat_desc = cat_featurizer.calc_rdkit_descrip()
                all_feat.append(cat_desc)
            if len(TEMP_feat_label) == 1:
                all_feat.append(np.array([temp]))
            if len(PRESSURE_feat_label) == 1:
                all_feat.append(np.array([pressure]))
        case "soap":
            if not _check_in_mol_type_is_sdf(mol_s, warning=False):
                raise RuntimeError("Error[iaw]:> the mol must be sdf, when the feat type is soap!")
            if len(REA_feat_label) != 0:
                rea_feat = _calc_soap_feat(mol_type = "reactant", sdf_fp = mol_s[0], feat_label = REA_feat_label, warning = False)
                all_feat.append(rea_feat)
            if len(SOL_feat_label) != 0:
                sol_feat = _calc_soap_feat(mol_type = "solvent", sdf_fp = mol_s[1], feat_label = SOL_feat_label, warning = False)
                all_feat.append(sol_feat)
            if len(CAT_feat_label) != 0:
                cat_feat = _calc_soap_feat(mol_type = "catalyst", sdf_fp = mol_s[2], feat_label = CAT_feat_label, warning = False)
                all_feat.append(cat_feat)
            if len(TEMP_feat_label) == 1:
                all_feat.append(np.array([temp]))
            if len(PRESSURE_feat_label) == 1:
                all_feat.append(np.array([pressure]))
        case "xtb":
            if not _check_in_mol_type_is_sdf(mol_s, warning=False):
                raise RuntimeError("Error[iaw]:> the mol must be sdf, when the feat type is xtb!")
            if len(REA_feat_label) != 0:
                rea_feat = _calc_xtb_feat(mol_type = "reactant", sdf_fp = mol_s[0], feat_label = REA_feat_label, warning = False)
                all_feat.append(rea_feat)
            if len(SOL_feat_label) != 0:
                sol_feat = _calc_xtb_feat(mol_type = "solvent", sdf_fp = mol_s[1], feat_label = SOL_feat_label, warning = False)
                all_feat.append(sol_feat)
            if len(CAT_feat_label) != 0:
                cat_feat = _calc_xtb_feat(mol_type = "catalyst", sdf_fp = mol_s[2], feat_label = CAT_feat_label, warning = False)
                all_feat.append(cat_feat)
            if len(TEMP_feat_label) == 1:
                all_feat.append(np.array([temp]))
            if len(PRESSURE_feat_label) == 1:
                all_feat.append(np.array([pressure]))

        case _:
            raise RuntimeError("Error[iaw]:> please input feat, rdkit, soap!")
    return all_feat

def Parm() -> Namespace:
    """
    2026-03-19修订
    1. 主要修订`rea_smi`, `sol_smi`以及`cat_smi`的格式, 可以传入两种:
        1. 纯smiles: 模型所需要的描述符不依赖于3d信息
        2. sdf: 模型需要3d信息, 且sdf中必须存有smiles, 默认的属性名为`SMILES`
    """
    parser = argparse.ArgumentParser(description="AAReact: Atropic Acid Enantioselectivity Prediction")
    parser.add_argument("--task", type=str, help="The task to perform: ee or conv.")
    parser.add_argument("--rea", type=str, help="The smiles or sdf of Reatant.")
    parser.add_argument("--sol", type=str, help="The smiles or sdf of Solvent.")
    parser.add_argument("--cat", type=str, help="The smiles or sdf of Catalyst.")
    parser.add_argument("--temp", type=float, help="The temperature of the reaction.")
    parser.add_argument("--pressure", type=float, help="The pressure of the reaction.")
    parser.add_argument("--model", type=str, help="The path of the trained model (pkl).")
    parser.add_argument("--feat_label", type=str, help="The path of the feature labels (pkl).")
    parser.add_argument("--verbose", type=int, default=1, help="The print out level: 1, detail; 0, simple.")
    parser.add_argument("--save_feat", type=str, default=None, 
                    help="Path to save the features of data generated during the process (e.g., ./features/xx.npy). "
                    "If not provided, features will not be saved.")
    # parser.add_argument("--feat_norm", type=str, help="The path of the feature Normalization parameters (pkl).")
    args = parser.parse_args()
    return args


def main():
    args = Parm()
    if args.task == "ee":
        with Status("running...", spinner="dots") as status:
            # int model
            model = load(args.model)
            with open(args.feat_label, "rb") as f:
                feat_label = pickle.load(f)

            # min_max_normer
            #data_x_max = np.load("data_x_max.npy")
            #data_x_min = np.load("data_x_min.npy")
            #feature_normer = norm_col_parms(min_col= data_x_min, max_col=data_x_max)

            # featurizer
            inited_feat = init_feat_label(feat_label)
            all_feat = []
            first_sign = False
            for i, (feat_type, feat) in enumerate(inited_feat):
                
                if i == 0:
                    first_sign = True
                else:
                    first_sign = False
                
                i_feat = init_features( (args.rea, args.sol, args.cat) , feat_type , feat , args.temp, args.pressure , first_sign)
                all_feat.extend(i_feat)
            
            #for i in all_feat:
            #    if isinstance(i, list):
            #        print(type(i), len(i))
            #    elif isinstance(i, np.ndarray):
            #        print(type(i), i.shape)

            data_x = np.concatenate(all_feat)
            #print("Debug[iaw]:> the data_x.shape = {}".format(data_x.shape))
            # predict
            ee = model.predict(data_x.reshape(1, -1))[0]
        
        if args.save_feat is not None:
            # 保存过程中产生的特征
            np.save("{}".format(args.save_feat), data_x)

        if args.verbose == 1:
            rp("Info\\[iaw]>:\n\tThe Rea.: {}\n\tThe Sol.: {}\n\tTheCat.: {}\n\tThe Temp: {}\n\tThe Pressure: {}\n\tThe ee: {:.6f}".format(
                args.rea_smi
                , args.sol_smi
                , args.cat_smi
                , args.temp
                , args.pressure
                , ee
            ))
        else:
            rp("Info\\[iaw]>: The ee: {:.6f}".format(ee))


    elif args.task == "conv":
        pass
    else:
        rp("Error\\[iaw]>: Please specify the task to perform: ee or conv.")
        sys.exit(1)

    # init model


if __name__ == "__main__":
    main() 