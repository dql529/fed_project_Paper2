@echo off
REM -------------------------------------------------------
REM 全局关闭 PyTorch weights_only 安全加载（仅当前会话）
REM -------------------------------------------------------
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

REM -------------------------------------------------------
REM 启动 Central Server（第一个窗口）
REM -------------------------------------------------------
start "CENTRAL_SERVER" cmd /c "python -u Central_Server.py H"

REM -------------------------------------------------------
REM 启动 Drone 节点（每行起一个新窗口）
REM 注意：要增加节点就复制一行并改端口/ID；要停掉节点就把对应行前面加 REM 注释
REM -------------------------------------------------------
                        @REM "A": "Reputation-Based Aggregation",
                        @REM "B": "Simple Average Aggregation",
                        @REM "C": "Fed-MAE Aggregation",
                        @REM "D": "median_aggregation",
                        @REM "E": "trimmed_mean_aggregation",
                        @REM "F": "krum_aggregation",
                        @REM "G": "multi_krum_aggregation",
                        @REM "H": "bulyan_aggregation",

@REM REM    3
@REM start "DRONE_1" cmd /c "python -u 1Drone_node.py 5001 1"
@REM start "DRONE_2" cmd /c "python -u 1Drone_node.py 5002 2"
@REM start "DRONE_3" cmd /c "python -u 1Drone_node.py 5003 3"

@REM @REM    4 + 1
@REM start "DRONE_1" cmd /c "python -u 1Drone_node.py 5001 1"
@REM start "DRONE_2" cmd /c "python -u 1Drone_node.py 5002 2"
@REM start "DRONE_3" cmd /c "python -u 1Drone_node.py 5003 3"
@REM start "DRONE_4" cmd /c "python -u 1Drone_node.py 5004 4"
@REM start "DRONE_5" cmd /c "python -u Malicious_Drone_Node.py 5005 5"

@REM  3 + 2
start "DRONE_1" cmd /c "python -u 1Drone_node.py 5001 1"
start "DRONE_2" cmd /c "python -u 1Drone_node.py 5002 2"
start "DRONE_3" cmd /c "python -u Malicious_Drone_Node.py 5003 3"
start "DRONE_4" cmd /c "python -u 1Drone_node.py 5004 4"
start "DRONE_5" cmd /c "python -u Malicious_Drone_Node.py 5005 5"

@REM @REM 6 + 1
@REM start "DRONE_1" cmd /c "python -u 1Drone_node.py 5001 1"
@REM start "DRONE_2" cmd /c "python -u 1Drone_node.py 5002 2"
@REM start "DRONE_3" cmd /c "python -u 1Drone_node.py 5003 3"
@REM start "DRONE_4" cmd /c "python -u 1Drone_node.py 5004 4"
@REM start "DRONE_4" cmd /c "python -u 1Drone_node.py 5005 5"
@REM start "DRONE_4" cmd /c "python -u 1Drone_node.py 5006 6"
@REM start "DRONE_7" cmd /c "python -u Malicious_Drone_Node.py 50077"

@REM @REM 4+3
@REM start "DRONE_1" cmd /c "python -u 1Drone_node.py 5001 1"
@REM start "DRONE_2" cmd /c "python -u 1Drone_node.py 5002 2"
@REM start "DRONE_3" cmd /c "python -u 1Drone_node.py 5003 3"
@REM start "DRONE_4" cmd /c "python -u 1Drone_node.py 5004 4"
@REM start "DRONE_5" cmd /c "python -u Malicious_Drone_Node.py 5005 5"
@REM start "DRONE_6" cmd /c "python -u Malicious_Drone_Node.py 5006 6"
@REM start "DRONE_7" cmd /c "python -u Malicious_Drone_Node.py 5007 7"

@REM @REM 5+2
@REM start "DRONE_1" cmd /c "python -u 1Drone_node.py 5001 1"
@REM start "DRONE_2" cmd /c "python -u 1Drone_node.py 5002 2"
@REM start "DRONE_3" cmd /c "python -u 1Drone_node.py 5003 3"
@REM start "DRONE_4" cmd /c "python -u 1Drone_node.py 5004 4"
@REM start "DRONE_5" cmd /c "python -u 1Drone_node.py 5005 5"
@REM start "DRONE_6" cmd /c "python -u Malicious_Drone_Node.py 5006 6"
@REM start "DRONE_7" cmd /c "python -u Malicious_Drone_Node.py 5007 7"



REM === 等待一段时间（或等你确认训练完成）===
timeout /t 25

REM === 强制关闭子窗口（通过窗口标题精确匹配）===
for %%T in (DRONE_1 DRONE_2 DRONE_3 DRONE_4 DRONE_5 DRONE_6 DRONE_7) do (
    taskkill /FI "WINDOWTITLE eq %%T" /T /F
)
taskkill /FI "WINDOWTITLE eq CENTRAL_SERVER" /T /F
