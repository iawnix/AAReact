# Data_All4：扩展训练与额外测试数据集

## 目的

Data_All4 用于在不改变 Data_All 原有内部测试集的前提下，加入 2026-10-28 新获得的标签：

- `data.csv` 中 BATCH=2 的 24 条记录加入训练集；
- `data.csv` 中 BATCH=3 的 64 条记录单独作为新的 extra 测试集；
- Data_All 原有测试集成员保持不变。

## 数据规模

| 来源/角色 | 行数 | Data_All4 中的用途 |
|---|---:|---|
| Data_All 原有训练集 | 326 | 保持原有训练成员不变 |
| 新数据 BATCH=2 | 24 | 新增到训练集 |
| Data_All 原有内部测试集 | 82 | 保持原有测试成员和顺序不变 |
| 新数据 BATCH=3 | 64 | 新的 extra 测试集 |
| Data_All 原有 BATCH=1 | 28 | 历史 extra，单独保留，不混入本次主评估 |

按当前默认 `seed=1`、`test_size=0.2` 计算，Data_All4 的主训练部分应为：

```text
326 条原有训练数据 + 24 条新 BATCH=2 = 350 条
```

原有内部测试仍然是 82 条，新的 extra 测试是 64 条。

Data_All4 合并源文件 `full_data_merged_20261102.csv` 共 1354 条唯一记录，按最终
`BATCH` 统计为：BATCH=0 共 408 条、BATCH=1 共 28 条、BATCH=2 共 854 条（其中只有
新标注的 24 条进入本次训练，其余 830 条仍未标注）、BATCH=3 共 64 条。

## 已生成的训练文件

`3_data_for_train/ddg/seed_1_test_0-2/` 下已生成 RDKit、xTB、SOAP、ACSF 以及常用组合：

- `train_*`：350 条固定训练数据；
- `test_*`：82 条固定内部测试数据；
- `extra_*`：64 条 BATCH=3 独立测试数据；
- `train_test_*`：350 条训练数据加 82 条内部测试数据的兼容副本，供现有训练脚本读取。

每个描述符目录还包含 `manifest.json`、`selector.json` 和 `selector.pkl`，记录特征筛选和
固定划分信息。所有筛选只使用训练侧数据拟合，BATCH=3 不参与特征筛选或模型选择。

如果训练流程需要严格保持 350/82 的固定划分，应读取 `train_*` 和 `test_*`；现有只接受
`train_test_*` 的旧入口可能会再次随机切分这 432 条记录，不能用来复现这里的固定测试集。

## 与 Data_All2 的关系

Data_All2 共 918 条，全部标记为 BATCH=2，但目标值为空。新 CSV 中的 88 条记录
全部已经存在于 Data_All2：

- 24 条被标注为 BATCH=2，并获得 EE/DDG；
- 64 条被重新标注为 BATCH=3，并获得 EE/DDG。

因此 Data_All4 不应把这 88 条再次追加成重复 DATA_ID，而应按 `DATA_ID` 更新标签和 batch。
Data_All2 中剩余未标注的记录保留为未来预测候选，不进入本次训练或测试。

## 划分原则

Data_All4 必须先读取 Data_All 当前划分产生的 `train_ids` 和 `test_ids`，冻结原有 326/82
成员关系，再把 24 条 BATCH=2 记录追加到训练侧。不能把 BATCH=0 和 BATCH=2 合并后重新
随机切分，否则会改变原有测试集。

BATCH=3 只在训练完成后作为 extra 输入，用于独立评估；它不能参与特征筛选、超参数搜索或模型选择。
