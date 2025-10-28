#!/bin/bash

# ------------------- 主要配置 -------------------
PROJECT_DIR="/Users/rui/Documents/fed_project"
CONDA_ENV_NAME="uav"
TOTAL_NODES=5
MALICIOUS_NODES=2

# --- 定义所有要测试的聚合算法 ---
ALGORITHMS_TO_TEST="A B C D E F G H"

# --- 定义所有要进行的消融实验配置 ---
# 格式: "人类可读的实验名称:R因子组合"
declare -a ABLATION_CONFIGS=(
    "FullModel:R1,R2,R3,R4,R5"
    "Without_R1:R2,R3,R4,R5"
    "Without_R2:R1,R3,R4,R5"
    "Without_R3:R1,R2,R4,R5"
    "Without_R4:R1,R2,R3,R5"
    "Without_R5:R1,R2,R3,R4"
)

# ------------------- 脚本主体 -------------------

# 设置PyTorch环境变量
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# 激活conda环境
source ~/.zshrc
conda activate "$CONDA_ENV_NAME"
cd "$PROJECT_DIR"

echo "======================================================"
echo "  即将开始全自动消融实验..."
echo "======================================================"
sleep 2

# 外层循环：遍历所有消融实验配置
for config in "${ABLATION_CONFIGS[@]}"
do
    # 解析实验名称和R因子组合
    ABLATION_NAME=${config%%:*}
    R_FACTORS=${config##*:}

    # --- 1. 为本次消融实验创建专属结果目录 ---
    TIMESTAMP=$(date +'%Y-%m-%d-%H-%M')
    RUN_DIR="logs/${ABLATION_NAME}/${TIMESTAMP}-N-${TOTAL_NODES}-M-${MALICIOUS_NODES}"
    mkdir -p "$RUN_DIR"
    SUMMARY_CSV_PATH="$RUN_DIR/final_experiment_summary.csv"

    echo "######################################################"
    echo "  消融实验: $ABLATION_NAME (R因子: $R_FACTORS)"
    echo "  结果将保存在: $RUN_DIR"
    echo "######################################################"

    # 内层循环：遍历所有聚合算法
    for ALGO in $ALGORITHMS_TO_TEST
    do
        echo "--------------------------------------------------"
        echo "    开始测试聚合算法: $ALGO"
        echo "--------------------------------------------------"

        LOG_DIR="$RUN_DIR/$ALGO"
        mkdir -p "$LOG_DIR"
        echo "    本轮日志将保存在: $LOG_DIR"

        # --- 2. 启动服务器，传入算法、结果路径和消融配置 ---
        python -u Central_Server.py "$ALGO" "$SUMMARY_CSV_PATH" "$R_FACTORS" > "$LOG_DIR/central_server.log" 2>&1 &
        
        echo "    等待3秒让服务器初始化..."
        sleep 3

        # --- 3. 启动所有节点 ---
        python -u 1Drone_Node.py 5001 1 > "$LOG_DIR/drone_1.log" 2>&1 &
        python -u 1Drone_Node.py 5002 2 > "$LOG_DIR/drone_2.log" 2>&1 &
        python -u Malicious_Drone_Node.py 5003 3 > "$LOG_DIR/drone_3.log" 2>&1 &
        python -u 1Drone_Node.py 5004 4 > "$LOG_DIR/drone_4.log" 2>&1 &
        python -u Malicious_Drone_Node.py 5005 5 > "$LOG_DIR/drone_5.log" 2>&1 &

        echo "    算法 '$ALGO' 的所有进程已启动，等待本轮完成..."
        wait

        echo "    算法 '$ALGO' 测试完成。"
        sleep 5
    done

    echo "######################################################"
    echo "  消融实验 '$ABLATION_NAME' 已全部完成。"
    echo "######################################################"
    echo
done

echo "======================================================"
echo "  所有消融实验均已完成。"
_e_cho "======================================================"