# Main-text ready LaTeX snippets

The following blocks are aligned with the current finalized artifacts:

- `artifacts/paper/base/Fig1_cleanF1_vs_f_dtD0.png`
- `artifacts/paper/base/Fig2_Wmal_vs_round_dtD0_f5.png`
- `artifacts/paper/base/Fig3_R4_distribution_dtD0_f5.png`
- `artifacts/paper_tables/table_main_performance.csv`
- `artifacts/paper_tables/table_mechanism.csv`
- `artifacts/paper_tables/table_stats_weighted_vs_median.csv`

## 0. Setup-side reproducibility sentence

Insert this in the FL setup / experimental setup paragraph, ideally near the definitions of $X_{\mathrm{ref}}$, $D_{\mathrm{ref}}^{\ell}$, and $S_{\min}$:

```latex
Unless otherwise stated, the final mainline configuration uses a disjoint semantic reference subset of size $n_{\mathrm{ref}}=128$, no server audit subset ($|D_{\mathrm{ref}}^{\ell}|=0$), and a pooled clean holdout set of size 4291 obtained by concatenating the client-side test partitions. In this setting, the support threshold $S_{\min}=10$ corresponds to approximately $7.8\%$ of the active reference subset.
```

If you already have a compact setup or hyperparameter table, you can also add:

```latex
\begin{tabular}{ll}
\toprule
Reference set size $n_{\mathrm{ref}}$ & 128 \\
Audit set size $|D_{\mathrm{ref}}^{\ell}|$ & 0 \\
Clean holdout size & 4291 \\
\bottomrule
\end{tabular}
```

## 1. Table 1 caption and explanation

```latex
\caption{Main performance summary under the final D0 setting. The headline metric remains clean holdout macro-F1, while polluted macro-F1 and $\Delta F_1=\text{Clean F1}-\text{Polluted F1}$ are reported as supplementary indicators. We focus on three strong-attack settings at $f=5$ together with \texttt{label\_flip}, $f=3$, which serves as the boundary regime.}
```

```latex
Table~\ref{tab:main_performance} summarizes the main quantitative results in a compact form. The proposed \textbf{Weighted} method shows its clearest gains in the two strongest semantic attack settings, namely \texttt{label\_flip}, $f=5$ and \texttt{dt\_logit\_scale}, $f=5$, where it clearly outperforms the geometry-based baselines on both clean and polluted macro-F1. Under \texttt{stealth\_amp}, $f=5$, \textbf{Weighted} still attains the strongest overall performance, although the margin over coordinate-wise median is smaller. By contrast, \texttt{label\_flip}, $f=3$ should be interpreted as a boundary regime: \textbf{Weighted} remains competitive, but its performance is very close to \textbf{Median}, so this setting should not be presented as a decisive win.
```

## 2. Table 2 caption and explanation

```latex
\caption{Mechanism summary for the proposed \texttt{weighted} aggregation under the final D0 setting. For each scenario, we report mean semantic consistency score $R4$, unnormalized reputation score $\mathrm{Rep}$, normalized admitted reputation weight $\pi$, and gate pass rate for benign and malicious clients. Here $\mathrm{Rep}$ denotes the raw reputation score before normalization, whereas $\pi$ denotes the normalized admitted weight used for interpretation. $R3$ is intentionally omitted because it is disabled in the final configuration.}
```

```latex
Table~\ref{tab:mechanism_summary} explains why the proposed aggregation works. Across all retained scenarios, benign clients receive higher semantic consistency scores, larger reputation values, larger admitted weights, and higher gate pass rates than malicious clients. The cleanest suppression pattern appears under \texttt{stealth\_amp}, where malicious clients are nearly eliminated in all four mechanism indicators. At the same time, \texttt{label\_flip}, $f=3$ reveals the main boundary behavior of the current configuration: although malicious clients are still suppressed, the benign pass rate is also relatively low, indicating that this medium-strength case is the clearest instance of benign over-filtering.
```

## 3. Figure 2 caption

```latex
\caption{Mechanism analysis via admitted malicious weight mass $W_{\mathrm{mal}}^{t}$ versus communication round at $f=5$ under DT fidelity D0. Smaller values indicate stronger suppression of malicious contribution by the aggregation mechanism.}
```

## 4. Figure 3 caption

```latex
\caption{Node-level semantic separation under \texttt{dt\_logit\_scale} at $f=5$ and DT fidelity D0. The left panel shows the final semantic consistency score $R4$, while the right panel shows the underlying twin-reference divergence $KL(p_{\mathrm{twin}} \,\|\, p_i)$. Benign clients achieve higher $R4$ and lower divergence than malicious clients. Each panel pools final-round client scores from 5 seeds.}
```

## 5. Replacement for `Overall Robustness on Clean Holdout Data`

```latex
\subsection{Overall Robustness on Clean Holdout Data}\label{sec:res_overall}
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{Fig1_cleanF1_vs_f_dtD0.png}
\caption{Primary evaluation metric: clean holdout macro-F1 versus number of malicious clients $f$ (mean $\pm$ 95\% CI over seeds) under DT fidelity D0. The three panels (left-to-right) correspond to \texttt{label\_flip}, \texttt{stealth\_amp}, and \texttt{dt\_logit\_scale}.}
\label{fig:f1_overall}
\end{figure*}

Fig.~\ref{fig:f1_overall} reports the primary evaluation metric, namely clean holdout macro-F1, under increasing malicious participation in the Byzantine client setting. Overall, the proposed \textbf{Weighted} aggregation is most effective in the strong semantic attack regimes rather than uniformly dominating every setting. Its clearest gains appear under \texttt{label\_flip}, $f=5$, and \texttt{dt\_logit\_scale}, $f=5$, where it substantially outperforms naive mean aggregation and the strongest geometry-based robust baseline. Under \texttt{stealth\_amp}, the proposed method still attains the strongest overall level, but the margin over coordinate-wise median is smaller. The main boundary regime is \texttt{label\_flip}, $f=3$, where \textbf{Weighted} remains competitive yet closely tracks \textbf{Median}; this setting should therefore be interpreted as a limit case rather than a universal win. These patterns are consistent with the design objective of the semantic screening mechanism, which is intended to detect updates that may not be extreme in parameter space but remain inconsistent with a trusted semantic reference.
```

## 6. Replacement for `Mechanism Analysis: Malicious Weight Suppression and Semantic Separation`

```latex
\subsection{Mechanism Analysis: Malicious Weight Suppression and Semantic Separation}\label{sec:res_mechanism}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{Fig2_Wmal_vs_round_dtD0_f5.png}
\caption{Mechanism analysis via admitted malicious weight mass $W_{\mathrm{mal}}^{t}$ versus communication round at $f=5$ under DT fidelity D0. Smaller values indicate stronger suppression of malicious contribution by the aggregation mechanism.}
\label{fig:wmal_round}
\end{figure}

Fig.~\ref{fig:wmal_round} shows the admitted malicious weight mass $W_{\mathrm{mal}}^{t}$ under the most challenging setting $f=5$. Here, $W_{\mathrm{mal}}^{t}$ denotes the total malicious weight that is still admitted into aggregation at communication round $t$. Lower values therefore indicate that the aggregation rule is more effectively suppressing malicious contribution before model updates are merged. This mechanism-level view is important because predictive performance alone does not reveal whether robustness is achieved by genuinely filtering malicious updates or merely by partially averaging them out.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{Fig3_R4_distribution_dtD0_f5.png}
\caption{Node-level semantic separation under \texttt{dt\_logit\_scale} at $f=5$ and DT fidelity D0. The left panel shows the final semantic consistency score $R4$, while the right panel shows the underlying twin-reference divergence $KL(p_{\mathrm{twin}} \,\|\, p_i)$. Benign clients achieve higher $R4$ and lower divergence than malicious clients. Each panel pools final-round client scores from 5 seeds.}
\label{fig:r4_dist}
\end{figure}

Fig.~\ref{fig:r4_dist} provides node-level evidence that the semantic prior induces measurable separation between benign and malicious clients under \texttt{dt\_logit\_scale}. In the left panel, benign clients retain higher $R4$ scores than malicious clients; in the right panel, the same separation appears in the opposite direction through the underlying divergence to the trusted twin reference. This figure should be interpreted as a semantic separation figure rather than a direct pass-rate proof. Taken together with Fig.~\ref{fig:wmal_round}, it supports the claim that the proposed mechanism suppresses malicious influence by assigning lower semantic consistency and lower admitted weight to semantically inconsistent updates.
```

## 7. Main-text paragraph to replace the old polluted supplementary subsection

```latex
The compact results in Table~\ref{tab:main_performance} also clarify how the polluted local evaluation surface should be interpreted. Polluted macro-F1 is not the headline metric of the paper, but it remains informative as a supplementary view of how much malicious influence survives local contamination. Under \texttt{label\_flip}, $f=5$, \textbf{Weighted} substantially improves polluted macro-F1 over the strongest geometry baseline, which is consistent with the strong suppression of admitted malicious weight. Under \texttt{dt\_logit\_scale}, the proposed method again shows the clearest advantage, directly supporting the motivation for semantic consistency screening. Under \texttt{stealth\_amp}, the margin is smaller but remains favorable. We therefore absorb the polluted-surface discussion into the main performance summary rather than treating it as a separate main-text result block.
```

## 8. Replacement for `Effect of DT Fidelity and Discussion`

```latex
\subsection{Discussion}\label{sec:res_discussion}
The main lesson from these results is that robust aggregation for security-oriented federated learning should not be viewed as a purely geometric filtering problem. Mean, median, and trimmed mean can suppress certain extreme updates, but they do not explicitly evaluate whether a client update remains semantically consistent with trusted nominal behavior. The proposed method addresses this gap by integrating a trusted semantic reference into a reputation-weighted aggregation rule. Accordingly, its largest gains appear in attack modes where semantic distortion matters most, especially \texttt{label\_flip}, $f=5$, and \texttt{dt\_logit\_scale}, $f=5$.

At the same time, the method should not be described as universally dominant. The boundary case \texttt{label\_flip}, $f=3$ shows that stronger semantic filtering can also introduce benign over-filtering, which is visible both in the main performance table and in the mechanism summary. This is precisely why the method is best characterized as a semantically informed robust aggregator rather than a complete digital-twin security system. In the present paper, the digital twin plays a narrower and more precise role: it provides a trusted semantic prior that improves aggregation robustness under Byzantine client attacks.
```

## 9. Deletions from the old draft

Delete the following from the old draft:

- the entire subsection `Supplementary Results on the Polluted Local Evaluation Surface`
- the sentence `The $y$-axis is plotted on a log scale.` from the old Fig.2 caption
- any Fig.3 wording that refers to a `DT teacher`
- any Fig.3 wording that says scores are `averaged over 15 rounds`
- the empirical claim `As DT fidelity decreases from D0 to D2...` unless you explicitly add an appendix reference with supporting results
