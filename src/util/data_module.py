import torch
import pandas as pd

import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from util.constants import DF_COLUMNS
from typing import Any, Dict, Callable, Union, List, Tuple
from types import NoneType
import numpy as np

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

class AARDataSet(torch.utils.data.Dataset):
    def __init__(self, data_fp: str, cache_fp: str, mol_featurizer: Any = False) -> None:
        
        self.data_s: Union[List, NoneType] = None

        if mol_featurizer == False:
            self.data_s = self.__load_data_s__(cache_fp)
        else:
            self.mol_featurizer = mol_featurizer
            self.data_s = self.__process_data_s__(data_fp)
            self.__save_data_s__(self.data_s, cache_fp)
        
    def __process_data_s__(self, data_fp: str) -> List[Dict]:
        # 判断字段
        df = pd.read_csv(data_fp)
        assert df.columns.to_list() == DF_COLUMNS, print("Error[iaw]>: please check {}.".format(data_fp))
        self.data_id = df["DATA_ID"].to_numpy()
        self.reactant_smi = df["REACTANT_SMI"].to_numpy()
        self.product_r_smi = df["PRODUCT_R_SMI"].to_numpy()
        self.product_s_smi = df["PRODUCT_S_SMI"].to_numpy()
        self.solvent_smi = df["SOLVENT_SMI"].to_numpy()
        self.ion_smi = df["ION_SMI"].to_numpy()
        self.ligand_smi = df["LIGAND_SMI"].to_numpy()
        self.temp = df["TEMP"].to_numpy()
        self.pressure = df["PRESSURE"].to_numpy()
        self.y = df["EE"].to_numpy()

        all_data: List[Dict] = []

        for idx in range(df.shape[0]):
            try:
                data = {
                 "data_id": self.data_id[idx]
                , "reactant_smi": self.reactant_smi[idx]
                , "product_r_smi": self.product_r_smi[idx]
                , "product_s_smi": self.product_s_smi[idx]
                , "solvent_smi": self.solvent_smi[idx]
                , "ion_smi": self.ion_smi[idx]
                , "ligand_smi": self.ligand_smi[idx]
                , "temp": self.temp[idx]
                , "pressure": self.pressure[idx]
                , "unimol_reactant_embed": np.array(self.mol_featurizer.get_repr(self.reactant_smi[idx], return_atomic_reprs=False)).flatten()
                , "unimol_product_r_embed": np.array(self.mol_featurizer.get_repr(self.product_r_smi[idx], return_atomic_reprs=False)).flatten()
                , "unimol_product_s_embed": np.array(self.mol_featurizer.get_repr(self.product_s_smi[idx], return_atomic_reprs=False)).flatten()
                , "unimol_solvent_embed": np.array(self.mol_featurizer.get_repr(self.solvent_smi[idx], return_atomic_reprs=False)).flatten()
                , "unimol_ion_embed": np.array(self.mol_featurizer.get_repr(self.ion_smi[idx], return_atomic_reprs=False)).flatten()
                , "unimol_ligand_embed": np.array(self.mol_featurizer.get_repr(self.ligand_smi[idx], return_atomic_reprs=False)).flatten()
                , "y": self.y[idx]
                }
            except:
                print("Error[iaw]>: unimol error for {}.".format(self.data_id[idx]))
                sys.exit(1)

            # -> torch
            data["temp"] = torch.tensor(data["temp"]).to(dtype=torch.float32)
            data["pressure"] = torch.tensor(data["pressure"]).to(dtype=torch.float32)
            data["unimol_reactant_embed"] = torch.from_numpy(data["unimol_reactant_embed"]).to(dtype=torch.float32)
            data["unimol_product_r_embed"] = torch.from_numpy(data["unimol_product_r_embed"]).to(dtype=torch.float32)
            data["unimol_product_s_embed"] = torch.from_numpy(data["unimol_product_s_embed"]).to(dtype=torch.float32)
            data["unimol_solvent_embed"] = torch.from_numpy(data["unimol_solvent_embed"]).to(dtype=torch.float32)
            data["unimol_ion_embed"] = torch.from_numpy(data["unimol_ion_embed"]).to(dtype=torch.float32)
            data["unimol_ligand_embed"] = torch.from_numpy(data["unimol_ligand_embed"]).to(dtype=torch.float32)
            data["y"] = torch.tensor(data["y"]).to(dtype=torch.float32)

            all_data.append(data)
        return all_data

    def __save_data_s__(self, data_s: List[Dict], cache_fp: str) -> None:
        torch.save(data_s, cache_fp)

    def __load_data_s__(self, cache_fp: str) -> List[Dict]:
        return torch.load(cache_fp)

    def __len__(self) -> int:
        return len(self.data_s)

    def __getitem__(self, idx: int) -> Dict:
        return self.data_s[idx]

class AARDataModule(LightningDataModule):
    def __init__(self, dataset: AARDataSet, train_valid_test: Tuple[float, float], batch_size: int, seed: int):
        super().__init__()
        self.full_dataset = dataset
        self.train_valid_ratio = train_valid_test[0]
        self.valid_test_ratio = train_valid_test[1]
        self.batch_size = batch_size
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        dataset_size = len(self.full_dataset)
        all_idx = np.arange(dataset_size)
        train_indices, val_test_indices = train_test_split(
            all_idx,
            test_size=self.train_valid_ratio,
            random_state=self.seed,
            shuffle=True
        )
        val_indices, test_indices = train_test_split(
            val_test_indices,
            test_size=self.valid_test_ratio,
            random_state=self.seed,
            shuffle=True
        )
        self.train_dataset = torch.utils.data.Subset(self.full_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(self.full_dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(self.full_dataset, test_indices)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
              dataset=self.train_dataset
            , batch_size=self.batch_size
            , shuffle=True
            , num_workers = 0)
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
              dataset=self.val_dataset
            , batch_size=self.batch_size
            , shuffle=False
            , num_workers = 0)
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
              dataset=self.test_dataset
            , batch_size=self.batch_size
            , shuffle=False
            , num_workers = 0)

if __name__ == "__main__":
    import logging
    #from util.featurizer import mol_featurizer
    #unimol_logger = logging.getLogger("unimol_tools")
    #unimol_logger.disabled = True
    

    print("Info[iaw]>: Test")
    test_df = "/home/iaw/DATA2/AAReact/DataSet/debug/debug_df.csv"
    cache_fp = "/home/iaw/DATA2/AAReact/DataSet/debug/debug_df_310m.pt"
    test_model_size = "310m"
    #clf = mol_featurizer(test_model_size)
    #dataset = AARDataSet(test_df, cache_fp, clf)
    dataset = AARDataSet(test_df, cache_fp, False)
    train_set, valid_set = train_test_split(dataset, test_size=0.2, random_state=42)
    print(train_set)
    dataloader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    for batch_idx, batch_data in enumerate(dataloader):
        print("Info[iaw]>: [{}], num: {}".format(batch_idx, len(batch_data['data_id'])))
        print("Info[iaw]>: [{}], temp shape: {}".format(batch_idx, batch_data['temp'].shape))
        print("Info[iaw]>: [{}], product_r shape: {}".format(batch_idx, batch_data['unimol_product_r_embed'].shape))
        print("Info[iaw]>: [{}], y shape: {}".format(batch_idx, batch_data['y'].shape))
        break  
