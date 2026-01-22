#!/bin/bash

# ======================== 配置区 ========================
# 定义要测试的命令
C_CMD="./xtb_mol2 -imol2 ./CAT-9.mol2 -itraj ./xtbtrj.xyz -omol2 ./CAT-9_new.mol2"
PY_CMD="python xtb_mol2.py"
# 测试次数（多次运行取平均，减少误差）
RUN_TIMES=5
# 临时文件（存储每次运行的耗时）
C_TIME_FILE="./c_times.tmp"
PY_TIME_FILE="./py_times.tmp"
# =======================================================

# 检查必要文件是否存在
check_files() {
    if [ ! -f "./xtb_mol2" ]; then
        echo "错误：未找到C程序可执行文件 ./xtb_mol2，请先编译！"
        exit 1
    fi
    if [ ! -f "./xtb_mol2.py" ]; then
        echo "错误：未找到Python脚本 xtb_mol2.py！"
        exit 1
    fi
    if [ ! -f "./CAT-9.mol2" ]; then
        echo "错误：未找到文件 ./CAT-9.mol2！"
        exit 1
    fi
    if [ ! -f "./xtbtrj.xyz" ]; then
        echo "错误：未找到文件 ./xtbtrj.xyz！"
        exit 1
    fi
    # 清空临时文件
    > $C_TIME_FILE
    > $PY_TIME_FILE
}

# 单次运行并记录耗时（单位：秒，保留3位小数）
run_and_record() {
    local cmd=$1
    local time_file=$2
    # 使用/usr/bin/time精准计时（输出格式：real 0.001 user 0.000 sys 0.000）
    # 重定向程序输出到/dev/null，只保留计时信息
    local time_output=$(/usr/bin/time -f "real %e user %U sys %S" bash -c "$cmd" 2>&1 >/dev/null)
    # 提取real耗时（实际运行时间）
    local real_time=$(echo $time_output | grep -oP 'real \K\d+\.\d+' | head -1)
    # 记录到临时文件
    echo $real_time >> $time_file
    echo "  耗时：$real_time 秒"
}

# 计算平均值
calc_average() {
    local time_file=$1
    # 使用awk计算平均值
    local avg=$(awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}' $time_file)
    # 保留3位小数
    printf "%.8f" $avg
}

# 主测试流程
main() {
    echo "==================================== 速度测试开始 ===================================="
    echo "测试命令："
    echo "  C版本：$C_CMD"
    echo "  Python版本：$PY_CMD"
    echo "测试次数：$RUN_TIMES 次（取平均值）"
    echo "--------------------------------------------------------------------------------------"

    # 1. 检查文件
    check_files

    # 2. 测试C版本
    echo -e "\n【C语言版本】开始运行（$RUN_TIMES 次）："
    for ((i=1; i<=RUN_TIMES; i++)); do
        echo -n "  第 $i 次运行："
        run_and_record "$C_CMD" "$C_TIME_FILE"
    done
    C_AVG=$(calc_average $C_TIME_FILE)
    echo -e "  C版本平均耗时：$C_AVG 秒\n"

    # 3. 测试Python版本
    echo -e "\n【Python版本】开始运行（$RUN_TIMES 次）："
    for ((i=1; i<=RUN_TIMES; i++)); do
        echo -n "  第 $i 次运行："
        run_and_record "$PY_CMD" "$PY_TIME_FILE"
    done
    PY_AVG=$(calc_average $PY_TIME_FILE)
    echo -e "  Python版本平均耗时：$PY_AVG 秒\n"

    # 4. 对比结果
    echo "==================================== 测试结果对比 ===================================="
    echo "  C版本平均耗时：   $C_AVG 秒"
    echo "  Python版本平均耗时：$PY_AVG 秒"
    # 计算倍数
    if (( $(echo "$C_AVG > 0" | bc -l) )); then
        MULTIPLE=$(echo "scale=2; $PY_AVG / $C_AVG" | bc -l)
        echo "  Python版本耗时是C版本的 $MULTIPLE 倍"
    fi
    echo "======================================================================================"

    # 5. 清理临时文件
    rm -f $C_TIME_FILE $PY_TIME_FILE
}

# 启动主流程
main

