#!/bin/bash

train_log="train.log"
train_metric="train.csv"

# title
echo "model,desc,R2_train,R2_test,R_train,R_test,Spearmanr_train,Spearmanr_test,MSE_train,MSE_test,MAE_train,MAE_test,RMSE_train,RMSE_test" >> $train_metric

# 提取
awk '
/^Train: model/ {
    split($2, a, /\[|\]/); model = a[2];
    split($3, b, /\[|\]/); desc = b[2];
}
/^Info\[iaw\]:> Result/ {
    sub(/^Info\[iaw\]:> Result, /, "");
    gsub(/, /, ",");  # 把 ", " 变成 "," 保证纯逗号分隔
    print model "," desc "," $0;
}
' "$train_log" >> $train_metric
