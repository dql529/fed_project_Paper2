# R4-FedAvg Paper Runbook

本文档记录复现实验与图表生成命令，默认在仓库根目录执行。

## 0) 通用说明

- 本地环境是 PowerShell，逗号列表务必加引号，例如：
  - `--attack-modes "label_flip,stealth_amp,dt_logit_scale"`
  - `--mal-nodes "0,3,5"`
- 常规默认输出文件：
  - `runs.csv`（每个 seed 一行）
  - `summary.csv`（跨 seed 的 mean/std）
  - `rounds.csv`（每轮日志）
  - `nodes.csv`（每节点日志）
  - `fallback_summary.csv`、`passrate_summary.csv`、`sensitivity_summary_ref.csv`、`sensitivity_summary_audit.csv`（新增）

> 当输出文件名通过 `--out-*` 覆盖时，绘图脚本需同步传入同一套路径。

## 1) 基线复现（快速回归）

```bat
python r4_agg_minitest.py --attack-modes "label_flip,stealth_amp,dt_logit_scale" --dt-levels "D0,D1,D2" --mal-nodes "0,3,5" --methods weighted --seeds "0,1,2,3,4"
```

对应图：
```bat
python paper_figs.py --figs 1,2,3 --dt D0 --out-dir paper_figs
```

## 2) DT 支撑度 + fallback 统计（S_DT）

```bat
python r4_agg_minitest.py --attack-modes "label_flip,stealth_amp,dt_logit_scale" --dt-levels D0 --mal-nodes 0,3,5 --methods weighted --seeds 0,1,2,3,4 --ref-size-grid "32,64,128,256,512"
```

检查：
- `rounds.csv` 包含 `S_DT`、`fallback_flag`、`num_masked`、`num_valid`、`tau_gate`、`benign_pass_rate`、`malicious_pass_rate`、`benign_admitted_weight_mass`。
- `fallback_summary.csv` / `runs.csv` 中有 `fallback_rate`。

绘图：
```bat
python paper_figs.py --figs 5 --dt D0 --f-sdt 5 --methods weighted
```

## 3) 统一 τ 门限敏感性（`f=5`，含 3 攻击）

```bat
python r4_agg_minitest.py --attack-modes "label_flip,stealth_amp,dt_logit_scale" --dt-levels D0,D1,D2 --mal-nodes 5 --methods weighted --seeds 0,1,2,3,4 --tau-grid "0.3,0.4,0.5,0.6,0.7,0.8"
```

绘图（清洁宏 F1、`W_mal`、`benign` 误杀率）：
```bat
python paper_figs.py --figs 6 --dt D1 --methods weighted --attacks "label_flip,stealth_amp,dt_logit_scale"
```

也可用：
```bat
python plot_from_csv.py --summary summary.csv --runs runs.csv --tau-sweep "0.3,0.4,0.5,0.6,0.7,0.8" --dt-levels D0,D1,D2
```

## 4) server_validation 基线（与 weighted 对齐）

```bat
python r4_agg_minitest.py --attack-modes "label_flip,stealth_amp,dt_logit_scale" --dt-levels D0,D1,D2 --mal-nodes 0,3,5 --methods "weighted,server_val" --seeds 0,1,2,3,4
```

可与 weighted 同图比较（示例）：
```bat
python paper_figs.py --figs 1 --dt D1 --methods weighted,server_val --attacks "label_flip,stealth_amp,dt_logit_scale"
```

## 5) Krum/Bulyan 条件化启用

### 5.1 仅 f=3

```bat
python r4_agg_minitest.py --attack-modes "label_flip,dt_logit_scale" --dt-levels D1 --mal-nodes 3 --methods krum,bulyan --seeds 0,1,2,3,4 --tau-grid "0.5"
```

### 5.2 f=5 触发 skip/条件跳过

```bat
python r4_agg_minitest.py --attack-modes "label_flip,dt_logit_scale" --dt-levels D1 --mal-nodes 5 --methods krum,bulyan --seeds 0,1,2,3,4 --tau-grid "0.5"
```

检查 `runs.csv`/终端输出有 `skipped=True` 与 `skip_reason`。

## 6) adaptive mimic 攻击（λ 扫描）

```bat
python r4_agg_minitest.py --attack-modes adaptive_mimic --dt-levels D0,D1,D2 --mal-nodes 0,3,5 --methods weighted --seeds 0,1,2,3,4 --adaptive-mimic-lambdas "0.1,1,10"
```

绘图：
```bat
python paper_figs.py --figs 9 --dt D1 --methods weighted --attacks adaptive_mimic --lambda-m 1 --ref-size 128
```

（`--lambda-m` 为过滤参数，未给则绘制全部扫描点。）

## 7) 参考集规模与审计集规模敏感性

```bat
python r4_agg_minitest.py --attack-modes "label_flip,dt_logit_scale" --dt-levels D1 --mal-nodes 3 --methods weighted --seeds 0,1,2,3,4 --ref-size-grid "32,64,128,256,512" --audit-size-grid "0,32,64,128,256"
```

绘图：
```bat
python paper_figs.py --figs 8 --dt D1 --f-fig2 3 --ref-size 128 --audit-size 64 --tau-gate 0.7
```

> 重点检查：`--audit-size-grid 0` 条件下 `R2_source` 在 `nodes.csv`/`runs.csv` 是否为 `"none"`。

## 8) 统计检验说明（论文方法）

- 轮次级别是同一 seed 下的时间序列，不可直接作为独立样本用于参数显著性检验。
- 显著性比较（均值差异/相关性）以 `seed` 作为独立单元。
- `summary/fallback_summary/passrate_summary/sensitivity_summary_*` 在聚合阶段均基于 `runs.csv` 的 seed 级记录（`_m` 与 `_s` 为 seed 维度统计量）。

## 9) 图示总览映射（便于复现）

- Fig.1：`python paper_figs.py --figs 1 --dt D0`
- Fig.2：`python paper_figs.py --figs 2 --dt D0 --f-fig2 5`
- Fig.3：`python paper_figs.py --figs 3 --dt D0 --mal-nodes 5`（若需可在命令中调整 `--mal-nodes`）
- Fig.4：`python paper_figs.py --figs 4 --dt D0 --ablation-attacks "label_flip,dt_logit_scale"`
- Fig.5：`python paper_figs.py --figs 5 --dt D0 --f-sdt 5`
- Fig.6：`python paper_figs.py --figs 6 --dt D0`
- Fig.7：`python paper_figs.py --figs 7 --dt D0 --f-fig2 5`
- Fig.8：`python paper_figs.py --figs 8 --dt D0 --f-fig2 5`
- Fig.9：`python paper_figs.py --figs 9 --dt D0`
