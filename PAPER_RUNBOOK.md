# R4-FedAvg Paper Runbook

This runbook records the paper-facing experiment definitions and the exact commands used to regenerate the final figures and tables.

## Today Locked

- Main method: `weighted = R2 + R4`
- Paper ablation: `weighted_r4only = R4-only`
- `R3` is disabled in the final paper configuration because its contribution is negligible in this task setting
- Default paper setting: `D0`, `tau_gate=0.70`, `fallback=median`
- Mainline attacks: `label_flip`, `stealth_amp`, `dt_logit_scale`

## Output Layout

- Grouped experiment outputs are written under `artifacts/<group>/`
- Paper-facing tables are written under `artifacts/paper_tables/`
- Paper-facing figures are written under `artifacts/paper/<group>/`

## Mainline Base Run

```bat
python r4_agg_minitest.py --attack-modes "label_flip,stealth_amp,dt_logit_scale" --dt-levels D0 --mal-nodes "0,3,5" --methods "weighted,mean,median,trimmed_mean" --seeds "0,1,2,3,4" --ref-size-grid "128" --audit-size-grid "0" --out-root artifacts --exp-group base
```

## Mainline Paper Figures

```bat
python paper_figs.py --figs 1,2,3 --dt D0 --exp-group base --out-dir artifacts/paper/base
```

Figure notes:

- `Fig.1`: clean holdout macro-F1 vs `f`
- `Fig.2`: admitted malicious weight mass `W_mal` vs round
- `Fig.3`: node-level score separation under `label_flip, f=5`, showing `pi_norm` and `R4`

## Supplementary Experiment A: R2+R4 vs R4-only

Three attacks at `f=5`:

```bat
python r4_agg_minitest.py --attack-modes "label_flip,stealth_amp,dt_logit_scale" --dt-levels D0 --mal-nodes "5" --methods "weighted,weighted_r4only" --seeds "0,1,2,3,4" --ref-size-grid "128" --audit-size-grid "0" --out-root artifacts/appendix_ablation --exp-group base
```

Weak-spot check at `label_flip, f=3`:

```bat
python r4_agg_minitest.py --attack-modes "label_flip" --dt-levels D0 --mal-nodes "3" --methods "weighted,weighted_r4only" --seeds "0,1,2,3,4" --ref-size-grid "128" --audit-size-grid "0" --out-root artifacts/appendix_ablation_lf3 --exp-group base
```

## Supplementary Experiment B: Tau Sanity Check

Only run `label_flip, f=3`:

```bat
python r4_agg_minitest.py --attack-modes "label_flip" --dt-levels D0 --mal-nodes "3" --methods "weighted" --seeds "0,1,2,3,4" --tau-grid "0.6,0.7" --ref-size-grid "128" --audit-size-grid "0" --out-root artifacts/appendix_taucheck --exp-group tau
```

Stopping rule:

- If `tau=0.60` improves `polluted_f1` by less than `0.02`, stop tuning
- If `tau=0.60` improves `benign_pass_rate` by less than `0.05`, stop tuning
- Main paper configuration remains `tau=0.70`

## Paper Analysis Tables

```bat
python paper_analysis.py --base-dir artifacts/base --ablation-dir artifacts/appendix_ablation/base --ablation-extra-dir artifacts/appendix_ablation_lf3/base --tau-dir artifacts/appendix_taucheck/tau --out-dir artifacts/paper_tables --dt D0 --main-method weighted --baseline median --node-attack label_flip --node-f 5
```

This command writes:

- `table_main_performance.csv`
- `table_ablation_r4only.csv`
- `table_mechanism.csv`
- `table_stats_weighted_vs_median.csv`
- `table_seed_results.csv`
- `table_tau_sanity.csv`
- `node_plot_data_label_flip_f5.csv`

## Statistical Unit

- All inferential statistics are computed at the `seed` level
- Round-level traces are not treated as independent samples
- The paper-facing significance comparison is `weighted` vs `median`

## Mainline Scope

Keep the main paper focused on:

- One main performance table
- One mechanism table
- One node-level separation figure

Do not reintroduce:

- `R3` as an active paper component
- `weighted_r2only` as a paper ablation
- large tau sweeps or broad parameter grids
