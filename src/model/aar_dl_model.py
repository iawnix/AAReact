from typing import Tuple, Union
import torch
from torch.nn import functional as F
import pytorch_lightning
from sklearn.metrics import r2_score
import sys

class AARmodel(pytorch_lightning.LightningModule):
    def __init__(self, config):
        super(AARmodel, self).__init__()
        self.config = config
        self.diff_r_embed = torch.nn.Linear(config.model.mol_in_dim, config.model.mol_hidden_dim)
        self.diff_s_embed = torch.nn.Linear(config.model.mol_in_dim, config.model.mol_hidden_dim)

        # 将ion, ligand, 以及溶剂拼接
        self.environ_embed = torch.nn.Linear(config.model.mol_in_dim*3, config.model.mol_hidden_dim)

        # 与环境进行交互
        self.diff_r_environ = torch.nn.MultiheadAttention(embed_dim = config.model.mol_hidden_dim
                                        , batch_first=True
                                        , num_heads=config.model.cross_n_heads
                                        , dropout = config.model.cross_dropout)
        self.diff_s_environ = torch.nn.MultiheadAttention(embed_dim = config.model.mol_hidden_dim
                                        , batch_first=True
                                        , num_heads=config.model.cross_n_heads
                                        , dropout = config.model.cross_dropout)

        # temp, pressure
        self.other_embed = torch.nn.Linear(2, 4)

        # 最后的特征融合
        self.fuse_mode = None
        if config.model.fuse_embed_mode == "concat":
            self.fuse_embed = torch.nn.Sequential(
                  torch.nn.Linear(2 * config.model.mol_hidden_dim+4, config.model.mol_hidden_dim)
                , torch.nn.ReLU()
            )
            self.fuse_mode = "concat"
        elif config.model.fuse_embed_mode == "weight":
            self.embed_weights = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
            self.fuse_embed = torch.nn.Linear(config.model.mol_hidden_dim, config.model.mol_hidden_dim)
            self.fuse_mode = "weight"
        else:
            raise RuntimeError("Error[iaw]>: please must provide one fuse mode!")
        # 输出
        self.output = torch.nn.Sequential(
              torch.nn.Linear(config.model.mol_hidden_dim, config.model.mol_hidden_dim)
            , torch.nn.ReLU()
            , torch.nn.Linear(config.model.mol_hidden_dim, config.model.mol_hidden_dim)
            , torch.nn.ReLU()
            , torch.nn.Linear(config.model.mol_hidden_dim, 1)
        )

    def forward(self, batch_data):
        # batch_size, unimol_embed_dim -> batch_size, mol_hidden_dim
        batch_data_diff_r = batch_data['unimol_product_r_embed'] - batch_data['unimol_reactant_embed']
        batch_data_diff_s = batch_data['unimol_product_s_embed'] - batch_data['unimol_reactant_embed']
        batch_data_diff_r = self.diff_r_embed(batch_data_diff_r)
        batch_data_diff_s = self.diff_s_embed(batch_data_diff_s)

        # batch_size, unimol_embed_dim*3 -> batch_size, mol_hidden_dim
        batch_data_environ = torch.cat([batch_data['unimol_ion_embed']
                                         , batch_data['unimol_ligand_embed']
                                         , batch_data['unimol_solvent_embed']], dim=-1)
        batch_data_environ = self.environ_embed(batch_data_environ)

        # batch_size, mol_hidden_dim -> batch_size, 1, mol_hidden_dim
        batch_data_diff_r = batch_data_diff_r.unsqueeze(1)
        batch_data_diff_s = batch_data_diff_s.unsqueeze(1)
        batch_data_environ = batch_data_environ.unsqueeze(1)

        # cross_atten
        r_attn_output, r_attn_weights = self.diff_r_environ(query = batch_data_diff_r
                            , key = batch_data_environ
                            , value = batch_data_environ
                            , need_weights = True)
        
        s_attn_output, s_attn_weights = self.diff_s_environ(query = batch_data_diff_s
                            , key = batch_data_environ
                            , value = batch_data_environ
                            , need_weights = True)
        
        # batch_size, 1, mol_hidden_dim -> batch_size, mol_hidden_dim
        r_attn_output = r_attn_output.squeeze(1)   
        s_attn_output = s_attn_output.squeeze(1)

        # other
        temp_pressure = self.other_embed(torch.cat([batch_data['temp'].unsqueeze(-1), batch_data['pressure'].unsqueeze(-1)], dim=-1))
        # fuse
        if self.config.model.fuse_embed_mode == "concat":
            concat_embed = torch.cat([r_attn_output, s_attn_output, temp_pressure], dim=-1)
            fuse_embed = self.fuse_embed(concat_embed)
        elif self.config.model.fuse_embed_mode == "weight":
            normalized_weights = torch.nn.functional.softmax(self.embed_weights, dim=0)
            fuse_embed = normalized_weights[0] * r_attn_output + normalized_weights[1] * s_attn_output
            fuse_embed = self.fuse_embed(fuse_embed)

        # output
        output = self.output(fuse_embed)

        # -> y_pred, y_true
        return output, batch_data["y"]

    def get_loss(self, y_pred, y_true, stage) -> torch.tensor:
        """
        y_pred: batch_size, 1
        y_true: batch_size
        """
        pred = y_pred.squeeze(-1)
        loss_fn = torch.nn.MSELoss(reduction='mean')
        #print("pred:", pred, "true:",y_true)
        loss = loss_fn(pred, y_true)
        if stage != None:
            self.log("MSE[{}]".format(stage), loss
                    , prog_bar=False, logger=True, sync_dist=True
                    , batch_size = y_true.size(0))
        return loss
    
    def evaluate(self, batch_data, stage: Union[str, bool] = False):
        y_pred, y_true = self(batch_data)
        self.get_loss(y_pred, y_true, stage)
        self.calculate_metrics(y_pred, y_true, stage)

    def calculate_metrics(self, y_pred, y_true, stage) -> Tuple:
        pred = y_pred.squeeze(-1)
        mae = F.l1_loss(pred, y_true, reduction='mean')
        mse = F.mse_loss(pred, y_true, reduction='mean')
        rmse = torch.sqrt(mse + 1e-8) 

        pred_np = pred.detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        r2 = r2_score(y_true_np, pred_np)

        metrics = {
            "MAE[{}]".format(stage): mae,
            "RMSE[{}]".format(stage): rmse,
            "R2[{}]".format(stage): r2
        }

        self.log_dict(
            metrics,
            prog_bar=False, 
            logger=True, 
            sync_dist=True, 
            batch_size = y_true.size(0)
        )
        return mae, rmse, r2

    def training_step(self, batch, batch_idx):
        y_pred, y_true = self(batch)
        loss = self.get_loss(y_pred, y_true, "train")
        return loss
    
    def validation_step(self, batch, batch_idx) -> None:
        return self.evaluate(batch, 'valid')
    
    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, 'test')

    def configure_optimizers(self):
        if self.config.optimizer.name == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)
        elif self.config.optimizer.name == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.config.optimizer.lr)
        elif self.config.optimizer.name == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr=self.config.optimizer.lr)
        else:
            print("Error[iaw]>: can not support {}".format(self.config.optimizer.name))
            sys.exit(1)

