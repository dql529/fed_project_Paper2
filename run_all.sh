#!/bin/bash

# 配置项目路径和conda环境名称（请修改为你的实际环境名）
PROJECT_DIR="/Users/rui/Documents/fed_project"
CONDA_ENV_NAME="uav" 

# 设置PyTorch环境变量
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# 启动中心服务器（激活conda环境后运行）
osascript -e 'tell application "Terminal" to do script "
    source ~/.zshrc;       # 如果用zsh请取消这行注释，注释上面的bash_profile
    conda activate '"$CONDA_ENV_NAME"';
    cd '"$PROJECT_DIR"';
    python -u Central_Server.py H;
    exit
"'

# 启动节点（3正常+2恶意）
osascript -e 'tell application "Terminal" to do script "
    source ~/.zshrc;
    conda activate '"$CONDA_ENV_NAME"';
    cd '"$PROJECT_DIR"';
    python -u 1Drone_node.py 5001 1;
    exit
"'

osascript -e 'tell application "Terminal" to do script "
    source ~/.zshrc;
    conda activate '"$CONDA_ENV_NAME"';
    cd '"$PROJECT_DIR"';
    python -u 1Drone_node.py 5002 2;
    exit
"'

osascript -e 'tell application "Terminal" to do script "
    source ~/.zshrc;
    conda activate '"$CONDA_ENV_NAME"';
    cd '"$PROJECT_DIR"';
    python -u Malicious_Drone_Node.py 5003 3;
    exit
"'

osascript -e 'tell application "Terminal" to do script "
    source ~/.zshrc;
    conda activate '"$CONDA_ENV_NAME"';
    cd '"$PROJECT_DIR"';
    python -u 1Drone_node.py 5004 4;
    exit
"'

osascript -e 'tell application "Terminal" to do script "
    source ~/.zshrc;
    conda activate '"$CONDA_ENV_NAME"';
    cd '"$PROJECT_DIR"';
    python -u Malicious_Drone_Node.py 5005 5;
    exit
"'

# 等待训练完成（根据实际情况调整时间）
sleep 25

# 关闭相关终端窗口
osascript -e 'tell application "Terminal"
    set closeWindows to every window whose (transcript contains "'"$PROJECT_DIR"'") and (transcript contains "python")
    repeat with win in closeWindows
        close win
    end repeat
end tell'