# Paper Mainline Text Pack

This file contains ready-to-paste main-paper text based on the finalized artifacts under `artifacts/paper_tables/` and `artifacts/paper/base/`.

## 1. Locked Method Definition

Under the final configuration, the reputation-weighted aggregation uses `R2` and `R4`, while `R3` is disabled due to negligible contribution in this task setting. The sole paper-facing ablation is `weighted_r4only`, which removes the lightweight performance complement and retains only the semantic component. Unless otherwise stated, the mainline configuration uses `tau_gate=0.70`, coordinate-wise median as the fallback aggregator, and clean holdout macro-F1 as the headline metric.

Source anchors:

- `artifacts/paper_tables/table_ablation_r4only.csv`
- `dt_r4/config.py`
- `r4_agg_minitest.py`

## 2. Experimental Setup Paragraph

We evaluate three attack modes, namely `label_flip`, `stealth_amp`, and `dt_logit_scale`, under the mainline DT setting `D0`. The primary malicious-client settings are `f=0` and `f=5`, while `label_flip, f=3` is treated as a boundary regime because it is the medium-strength case in which the proposed method becomes closest to the strongest robust baseline. We compare the proposed `weighted` aggregation against `mean`, `median`, and `trimmed_mean`, and all reported averages are computed over five random seeds. Any inferential comparison is performed only at the seed level using paired analysis rather than round-level traces.

Source anchors:

- `artifacts/paper_tables/table_main_performance.csv`
- `artifacts/paper_tables/table_stats_weighted_vs_median.csv`

## 3. Main Performance Table Caption

Table X. Main performance summary under the final `D0` setting. Rows report `mean`, `median`, `trimmed_mean`, and the proposed `weighted` aggregation. Columns report clean holdout macro-F1, polluted macro-F1, and `delta_f1 = clean_f1 - polluted_f1`. We include `f=0` to show non-attack behavior rather than to claim universal dominance, and we focus on `f=5` together with `label_flip, f=3` as the main stress and boundary regimes, respectively.

Source anchors:

- `artifacts/paper_tables/table_main_performance.csv`

## 4. Mechanism Table Caption

Table Y. Mechanism summary for the proposed `weighted` aggregation under the final `D0` setting. For benign and malicious clients, we report the mean semantic consistency score `R4`, the unnormalized reputation score `Rep`, the normalized admitted reputation weight, denoted as `$\pi$`, and the gate pass rate. Here, `Rep` is the raw reputation score before normalization, whereas `$\pi$` is the normalized admitted weight used for interpretation. `R3` is intentionally omitted because it is disabled in the final configuration and is not part of the paper-facing mechanism definition.

Source anchors:

- `artifacts/paper_tables/table_mechanism.csv`

## 5. Figure 1/2/3 Captions

Fig.1. Primary evaluation metric: clean holdout macro-F1 versus the number of malicious clients `f` under the final `D0` setting. The three panels correspond to `label_flip`, `stealth_amp`, and `dt_logit_scale`, and each curve compares `weighted` against `mean`, `median`, and `trimmed_mean`.

Fig.2. Mechanism analysis via admitted malicious weight mass `W_mal` over communication rounds at `f=5` under the final `D0` setting. The figure shows the round-wise behavior of the proposed `weighted` aggregation under `label_flip`, `stealth_amp`, and `dt_logit_scale`.

Fig.3. Node-level score separation under `label_flip` at `f=5` and `D0`. The left panel shows the normalized admitted reputation weight `$\pi$`, and the right panel shows the semantic consistency score `R4`, for benign and malicious clients under the proposed `weighted` aggregation. This figure is intended as an interpretation figure rather than a direct pass-rate proof; if a dashed gate reference is displayed, it applies only to the `R4` panel.

Source anchors:

- `artifacts/paper/base/Fig3_node_scores_dtD0_f5.png`
- `paper_figs.py`
- `dt_r4/plotting.py`

## 6. Main Results Paragraph

The mainline results show that the proposed `weighted` aggregation consistently improves robustness in the strongest attack regimes while remaining stable in the easier settings. Under `label_flip, f=5`, `weighted` reaches `clean_f1 = 0.7369` and `polluted_f1 = 0.5568`, clearly outperforming the robust baselines, whose best counterpart in this setting is `median` at `0.3671/0.3661`. Under `dt_logit_scale, f=5`, `weighted` attains `clean_f1 = 0.7712` and `polluted_f1 = 0.7764`, again staying well above `median` at `0.6430/0.6396`. Under `stealth_amp, f=5`, the proposed method remains competitive and stable with `clean_f1 = 0.7637` and `polluted_f1 = 0.7691`; the margin over `median` is smaller than in `label_flip` and `dt_logit_scale`, but the method still retains the strongest overall level. The main boundary regime is `label_flip, f=3`, where `weighted` and `median` are close (`0.7562/0.6308` versus `0.7621/0.6371`), indicating that this medium-strength case is the main limit case rather than a universal failure mode.

Source anchors:

- `artifacts/paper_tables/table_main_performance.csv`

## 7. Ablation Paragraph

The ablation results support a simple interpretation: `R4` is the dominant component, while `R2` acts as a stabilizing complement rather than the main driver. This complement matters most under `label_flip`, especially at `f=5`, where `weighted (R2+R4)` reaches `0.7369/0.5568` in clean/polluted macro-F1, whereas `R4-only` drops to `0.5237/0.4505`. The same pattern appears in the boundary regime `label_flip, f=3`, although the gap is smaller (`0.7562/0.6308` for `R2+R4` versus `0.7363/0.6164` for `R4-only`). Under `dt_logit_scale, f=5`, the full method also remains stronger (`0.7712/0.7764` versus `0.7180/0.7197`), which shows that the lightweight performance complement still contributes when the semantic score alone is not sufficient. By contrast, under `stealth_amp, f=5`, `R4-only` and `R2+R4` are nearly identical (`0.7656/0.7696` versus `0.7637/0.7691`), which is consistent with the view that the semantic term is already doing most of the work in that setting.

Source anchors:

- `artifacts/paper_tables/table_ablation_r4only.csv`

## 8. Tau Sanity + Statistics Paragraph

As a targeted sanity check, we compare `tau=0.60` and `tau=0.70` only under `label_flip, f=3`. The polluted macro-F1 barely changes (`0.6310` at `tau=0.60` versus `0.6308` at `tau=0.70`), while the benign pass rate increases sharply from `0.0343` to `0.4457`. At the same time, the malicious pass rate also rises from `0.0000` to `0.0444`. This pattern suggests that `tau=0.70` is conservative, but it remains the default because lowering the threshold does not produce a meaningful performance gain in the boundary regime.

For statistical comparison, we only compare `weighted` against `median`, using five seed-level paired samples for `clean_f1`, `polluted_f1`, and `delta_f1` under the three `f=5` attack settings plus `label_flip, f=3`. In the strong-attack regimes, the improvements are consistent and the paired effect sizes are often large. For example, under `label_flip, f=5`, the paired effect sizes are `d_z = 3.49` for clean F1 and `d_z = 2.38` for polluted F1, while under `dt_logit_scale, f=5`, they are `d_z = 1.87` and `d_z = 2.06`, respectively. However, with only five seeds, the corresponding `p` values remain in the `0.0625` to `0.1250` range, so these results do not justify blanket claims of statistically significant superiority. This is especially important for `label_flip, f=3`, where the paired differences against `median` are near zero and the setting should therefore be described as a boundary regime rather than a decisive win.

Source anchors:

- `artifacts/paper_tables/table_tau_sanity.csv`
- `artifacts/paper_tables/table_stats_weighted_vs_median.csv`
