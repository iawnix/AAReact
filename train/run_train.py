import wandb
import sys
import os
SRC_PATH="/home/iaw/DATA2/AAReact/src"
sys.path.append(SRC_PATH)
from tool.train_tool import init_config_toml, set_seed, namespace_to_double_dict
from util.data_module import AARDataSet
from model.aar_dl_model import AARmodel
from util.data_module import AARDataSet, AARDataModule

import pytorch_lightning as pl

from pathlib import Path
import shutil
if __name__ == "__main__":

    config_fp = "/home/iaw/DATA2/AAReact/config/debug.toml"
    config = init_config_toml(config_fp)
    Path(config.job.out_path).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_fp, config.job.out_path)


    wandb_logger = pl.loggers.WandbLogger(
        project="AAReact",
        name=config.job.name,
        config=namespace_to_double_dict(config),
        save_dir=str(config.job.out_path),
        offline=False,
        anonymous="allow"
    )

    if config.data.init_unimol_embedding == False:
        mol_featurizer = False
    else:
        pass

    full_dataset = AARDataSet(data_fp = config.data.data_fp, cache_fp = config.data.cache_fp
                             , mol_featurizer  = mol_featurizer)
    
    dm = AARDataModule(full_dataset, config.data.train_valid_test, config.data.batch_size, config.data.seed)
    model = AARmodel(config)
    # 模型保存回调
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="MSE[valid]",
        mode="min",
        save_top_k=3,
        save_last = True,
        dirpath=config.job.out_path +  "/" + "checkpoints",
        filename=config.job.name
    )
    
    # 初始化 Trainer
    trainer = pl.Trainer(
        max_epochs=config.train.epoch,
        accelerator=config.train.cudu_device,
        devices=config.train.gpu_idx,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        deterministic=True,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        log_every_n_steps=1.0,
    )
    
    # 启动训练
    trainer.fit(model, datamodule=dm)
    
    # 启动测试
    best_ckpt_path = checkpoint_callback.best_model_path
    print("Infor[iaw]>: best_ckpt_path: {}".format(best_ckpt_path))
    model = AARmodel.load_from_checkpoint(best_ckpt_path, config=config)
    trainer.test(model, datamodule=dm)
    
    # 关闭 WandB
    wandb.finish()
