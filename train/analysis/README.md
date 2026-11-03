# 训练分析结果

分析结果按训练数据集归档。当前主流程的结果在 `Data_All/`，其中：

- `selected_model/`：主模型的回归、符号准确率和 SHAP 结果；
- `train_metrics/`：全部模型/描述符的指标汇总和图表；
- `shap/`：SHAP notebook 及其输出；
- `legacy/`：早期分析 notebook 和历史图表，保留用于追溯。

## 历史文件的使用边界

`Data_All/legacy/` 按原目录来源归档，不代表其中每份旧图表都对应当前
326/82 划分。`2_ana_data_distribution.ipynb` 仍引用早期
`DataSet/AtropicAcid_416/` 数据，是历史记录，未迁移为当前数据分析。
旧的回归图和 SHAP 图也不应当作更新后的 SHAP notebook 的输出。

`selected_model/1.ipynb` 是临时绘图草稿，引用项目外的
`/home/iaw/correlation_statistics.csv`，不是正式分析入口。
当前主模型分析使用 `bash train/7_analyze_selected_model.sh`；
RDKit/xTB 模型的 notebook 使用 `Data_All/shap/1_ana_shap.ipynb`。
