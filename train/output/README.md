# 训练结果索引

第一层按训练所用的数据集命名。模型、预测、指标属于哪个实验，以该实验的配置和 manifest 为准。

```text
train/output/
├── Data_All/
│   └── pt/ddg/hyper_seed_1_test_0-2/search_optuna/
│       └── seed_<种子>_test_<比例>/<模型>_<描述符>.pkl
├── Data_All3/
│   └── rea1_train_rea234_test/
│       ├── model_xgb_rdkit_xtb_acsf.pkl
│       ├── predictions.csv
│       ├── metrics.json
│       ├── manifest.json
│       └── optuna/
└── Data_All4/
    └── xgb_rdkit_xtb_acsf/
        ├── model_xgb_rdkit_xtb_acsf.pkl
        ├── train_predictions.csv
        ├── test_predictions.csv
        ├── extra_predictions.csv
        ├── pool_predictions.csv
        ├── metrics.json
        └── run_manifest.json
```

## 三组结果分别是什么

| 数据集 | 实验和数据规模 | 运行入口（从项目根目录执行） |
|---|---|---|
| Data_All | BATCH=0 的 408 条，按 3 个 seed × 3 个测试比例 × 3 个模型 × 15 个描述符生成 405 个模型；主划分为 326/82 | `bash train/1_train_ml_model.sh` |
| Data_All3 | REA-1 的 102 条训练，REA-2/3/4 的 306 条外部测试 | `bash run_rea1_external_test.sh` |
| Data_All4 | 固定 350 条训练、82 条内部测试、64 条 BATCH=3 extra，另预测 830 条未标注候选；复用 Data_All 主流程超参数 | `bash train/8_train_data_all4.sh` |

Data_All4 的轮次记录在 `run_manifest.json` 的 `round` 字段中，当前为 1。
`pool_predictions.csv` 按预测 DDG 的绝对值排序，这不是模型不确定性排名。
Data_All2 是预测候选的来源，目前没有在其上独立训练的模型，因此这里不单设 Data_All2 目录。

## 去哪里看结果

- Data_All4 本轮指标：[metrics.json](Data_All4/xgb_rdkit_xtb_acsf/metrics.json)。
- Data_All4 新增外部测试预测：[extra_predictions.csv](Data_All4/xgb_rdkit_xtb_acsf/extra_predictions.csv)。
- Data_All4 未标注候选预测：[pool_predictions.csv](Data_All4/xgb_rdkit_xtb_acsf/pool_predictions.csv)。
- Data_All3 实验说明：[manifest.json](Data_All3/rea1_train_rea234_test/manifest.json)。
- Data_All 旧主流程汇总仍在 `train/train.csv`；分析结果在 `train/analysis/Data_All/`，预测结果在 `train/predictions/`，训练日志在 `train/logs/`。

## 2026-11-02 路径整理

| 迁移前（相对于 train/output） | 迁移后 |
|---|---|
| `pt/` | `Data_All/pt/` |
| `rea1_train_rea234_test/` | `Data_All3/rea1_train_rea234_test/` |
| `data_all4/round_1/` | `Data_All4/` |

本次整理迁移已有文件，不重新训练。配置生成器、现有训练 TOML、预测与分析入口同步采用新路径。
已有 JSON 记录中的路径更新后，通过 `path_migration` 保留原路径和迁移说明；历史日志保留当时记录的路径。
`test/iter_pred_new_*.sh` 和旧 SHAP notebook 已进一步接到现有 Data_All DDG 模型及其配套特征；旧分子编号也已按结构核对更新。使用方法见 [旧预测脚本说明](../../test/README.md)。

默认入口再次运行会写入对应实验目录；需要保留不同运行结果时，为支持该选项的入口指定独立 `OUTPUT_DIR`。Data_All4 已支持该环境变量。
Git 跟踪本索引，生成的模型和结果继续忽略。

## 训练目录索引

| 目录 | 用途 |
|---|---|
| `train/analysis/Data_All/` | Data_All 的模型选择、训练指标和 SHAP/历史分析 |
| `train/predictions/Data_All/` | 使用 Data_All 模型产生的额外预测 |
| `train/predictions/Data_All2/` | Data_All2 候选数据的历史预测和 notebook |
| `train/logs/Data_All/` | Data_All 主流程的训练日志 |
| `train/analysis/` | 分数据集保存分析 notebook、模型解释和指标图表 |
| `train/predictions/` | 分数据集保存预测 CSV 和预测 notebook |
| `train/logs/` | 分数据集保存训练日志 |

Data_All3 和 Data_All4 的实验结果直接保存在 `train/output/Data_All3/` 和 `train/output/Data_All4/`，避免与历史分析文件混在一起。
