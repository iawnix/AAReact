# 旧分子预测脚本的当前用法

这两份脚本现已使用 Data_All 当前保存的 DDG 模型，保留旧实验的反应条件。
它们用于逐个分子计算描述符并预测，不训练模型，也不是 Data_All4 主动学习入口。

## 运行

从项目根目录或任意工作目录均可调用脚本。默认使用项目 Python 3.12 环境，也可通过 `PYTHON` 指定。

```bash
# 只核对 8 个模型、对应特征标签和 SDF 路径，不进行分子计算。
bash test/iter_pred_new_cat.sh --check-only
bash test/iter_pred_new_rea.sh --check-only

# 默认运行 RF / XGBoost × RDKit、RDKit+SOAP、RDKit+SOAP+xTB、SOAP+xTB。
bash test/iter_pred_new_cat.sh
bash test/iter_pred_new_rea.sh

# 仅运行一个模型，并指定输出位置。
MODELS="xgb_rdkit" OUTPUT_DIR=/tmp/new_rea_ddg bash test/iter_pred_new_rea.sh
```

默认固定 `SPLIT=seed_1_test_0-2`、`HYPER_SPLIT=seed_1_test_0-2`、`SEARCH_METHOD=optuna`。
脚本从 `train/output/Data_All/pt/ddg/` 读取对应模型，从
`DataSet/Data_All/3_data_for_train/ddg/` 读取同一划分的特征标签。
启动时核对模型目标、列数和标签顺序；输出为 **DDG，单位 kcal/mol**，不能直接作为 EE 解读。

每个样本保存 `.npy` 特征和 `.log` 预测输出。默认位置分别是：

- `test/iter_1/new_cat_ddg/seed_1_test_0-2/`：16 个条件 × 8 个模型。
- `test/iter_1/new_rea_ddg/seed_1_test_0-2/`：12 个条件 × 8 个模型。

再次运行同一输出目录会更新对应文件。xTB 沿用项目共用的临时目录，这两份脚本应顺序运行。

## 结构编号核对

旧目录已经不存在，因此使用 `DataSet/Data_All/1_sdf/` 中的结构。通过规范化 SMILES、保留立体化学并去除原子映射编号核对得到：

| 旧脚本编号 | 当前结构编号 |
|---|---|
| CAT-71 | CAT-22 |
| CAT-72 | CAT-39 |
| CAT-73 | CAT-38 |
| CAT-41 | CAT-35 |
| CAT-42 | CAT-36 |
| REA-1 到 REA-10 | 编号不变 |
| SOL-1、SOL-2 | 编号不变 |

新催化剂脚本中 CAT-39/38 仍使用旧脚本指定的 **SOL-2**，与 Data_All 现有 extra 中这两个催化剂使用 SOL-1 的记录不同。
温度保持 60°C；CAT-22 使用 20/100 的压力条件，其余组合使用 20。

## 旧 SHAP notebook

`train/analysis/Data_All/shap/1_ana_shap.ipynb` 已切换到当前 XGBoost + RDKit/xTB 的 DDG 模型。
模型和特征从同一训练 TOML 读取，并按数据 manifest 复现 326/82 划分。
旧输出已清空，避免与当前 DDG 结果混用。重新执行后，图表保存到
`train/analysis/Data_All/shap/data_all_ddg_rdkit_xtb/`，也可通过 `SHAP_OUTPUT_DIR` 指定位置。
