# Data_All3：固定底物外部测试实验

## 数据规模

| BATCH | REA 范围 | 行数 | 角色 |
|---:|---|---:|---|
| 0 | REA-1 | 102 | 训练 |
| 0 | REA-2、REA-3、REA-4 | 306 | 固定外部测试 |
| 合计 | REA-1 到 REA-4 | 408 | 独立实验数据 |

Data_All3 只包含 Data_All 原始 BATCH=0 中的 REA-1 到 REA-4，不是 Data_All4 的组成部分。

## 实验规则

- 训练只使用 REA-1 的 102 条记录；
- REA-2、REA-3、REA-4 各 102 条，只作为外部测试；
- 特征筛选和超参数搜索只在 REA-1 内完成；
- 目标为 DDG，模型为 XGBoost，描述符为 RDKit+xTB+ACSF。

该目录用于检验跨底物泛化，结果应与 Data_All 和 Data_All4 分开保存。
