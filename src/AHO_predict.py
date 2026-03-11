import argparse
from argparse import Namespace
import os
import sys
from pathlib import Path
from joblib import dump, load
import pickle

from typing import Any, List, Tuple

import numpy as np
from numpy.typing import NDArray

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from util.train_tools import norm_col_parms
from util.featurizer import rdkit_featurizer

from rich import print as rp
from rich.status import Status

def init_feat_label(feat_label: list[str]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    REA, SOL, CAT, TEMP, PRESSURE = [], [], [], [], []
    REA_idx, SOL_idx, CAT_idx, TEMP_idx, PRESSURE_idx = [], [], [], [], []
    for i, feat in enumerate(feat_label):
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
    if REA_idx + SOL_idx + CAT_idx + TEMP_idx + PRESSURE_idx != list(range(len(feat_label))):
        print("Error[iaw]>: Feature label order is not REA, SOL, CAT, TEMP, PRESSURE")
    return REA, SOL, CAT, TEMP, PRESSURE



def Parm() -> Namespace:
    parser = argparse.ArgumentParser(description="AAReact: Atropic Acid Enantioselectivity Prediction")
    parser.add_argument("--task", type=str, help="The task to perform: ee or conv.")
    parser.add_argument("--rea_smi", type=str, help="The smiles of Reatant.")
    parser.add_argument("--sol_smi", type=str, help="The smiles of Solvent.")
    parser.add_argument("--cat_smi", type=str, help="The smiles of Catalyst.")
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
            all_feat = []
            REA_feat_label, SOL_feat_label, CAT_feat_label, TEMP_feat_label, PRESSURE_feat_label = init_feat_label(feat_label)
            if len(REA_feat_label) != 0:
                rea_featurizer = rdkit_featurizer(args.rea_smi)
                rea_featurizer.reset_rdkit_desc_generator(REA_feat_label)
                rea_desc = rea_featurizer.calc_rdkit_descrip()
                all_feat.append(rea_desc)
            if len(SOL_feat_label) != 0:
                sol_featurizer = rdkit_featurizer(args.sol_smi)
                sol_featurizer.reset_rdkit_desc_generator(SOL_feat_label)
                sol_desc = sol_featurizer.calc_rdkit_descrip()
                all_feat.append(sol_desc)
            if len(CAT_feat_label) != 0:
                cat_featurizer = rdkit_featurizer(args.cat_smi)
                cat_featurizer.reset_rdkit_desc_generator(CAT_feat_label)
                cat_desc = cat_featurizer.calc_rdkit_descrip()
                all_feat.append(cat_desc)
            if len(TEMP_feat_label) == 1:
                all_feat.append(np.array([args.temp]))
            if len(PRESSURE_feat_label) == 1:
                all_feat.append(np.array([args.pressure]))
            data_x = np.concatenate(all_feat)
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