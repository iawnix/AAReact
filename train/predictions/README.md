# 预测结果

预测文件按数据来源归档：

- `Data_All/`：Data_All 主模型对额外样本的预测；
- `Data_All2/`：Data_All2 候选数据的历史预测；

这里的目录名表示被预测数据的来源，并不表示在该数据集上训练过模型。
Data_All2 的历史候选预测不能直接作为 Data_All4 新模型的预测结果。
Data_All3、Data_All4 的预测随模型、指标一起保存在
`train/output/Data_All3/`、`train/output/Data_All4/`，以对应 manifest 为准，
不在这里重复复制。

`train/5_predict_new_data.sh` 默认预测 Data_All 的 extra，写入 `Data_All/`。
手动切换 `DATA_DIR` 到其他数据集时，请同时指定 `OUTPUT`；
目录名不会自动改变实际使用的模型或输入数据。
