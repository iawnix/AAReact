# Data_All2：新增底物候选数据

## 数据规模

| BATCH | 行数 | 含义 |
|---:|---:|---|
| 2 | 918 | 新增底物候选数据，当前原始文件中目标值为空 |
| 合计 | 918 | Data_All2 数据集 |

数据文件是 `full_data_918-20260723.csv`，包含 REA-11 到 REA-19 共 9 种新底物，
每种底物 102 条记录。当前 `EE`、`DDG` 和 `CONV` 在该文件中均为空，不能直接作为有监督训练数据。

## 2026-10-28 标注数据的关系

`Data_Raw/2026-10-28/data.csv` 中的 88 条记录全部已经存在于 Data_All2：

- BATCH=2：24 条，已经获得 `EE/DDG`，后续加入 Data_All4 的训练集；
- BATCH=3：64 条，已经获得 `EE/DDG`，后续作为 Data_All4 的额外 extra 测试集。

因此这些记录不能再次按行追加到 Data_All2，而应按 `DATA_ID` 回填标签并更新 batch 角色。
Data_All2 剩余尚未标注的记录只作为未来预测候选，不进入当前监督训练或测试。

## 现有特征

`2_raw_features/` 中已有 918 条记录的 RDKit、xTB、SOAP、ACSF 特征；
`3_data_for_predict/` 中已有此前生成的新底物预测矩阵。

Data_All2 是扩展候选数据集，不改变 Data_All 的原有内部测试划分。
