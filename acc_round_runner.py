from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ATTACKS = ["label_flip", "dt_logit_scale"]
METHODS = ["weighted", "median", "mean", "trimmed_mean"]
SEEDS = ["0", "1", "2", "3", "4"]


def benign_profile_for_f(mal_nodes: int) -> List[str]:
    profiles: Dict[int, List[str]] = {
        1: ["light", "medium", "medium", "medium", "heavy", "heavy", "heavy", "heavy", "clean"],
        2: ["light", "medium", "medium", "heavy", "heavy", "heavy", "heavy", "clean"],
        3: ["light", "medium", "medium", "heavy", "heavy", "heavy", "clean"],
        4: ["light", "medium", "medium", "heavy", "heavy", "clean"],
        5: ["light", "medium", "heavy", "heavy", "clean"],
    }
    if mal_nodes not in profiles:
        raise ValueError(f"Unsupported mal_nodes={mal_nodes}")
    return profiles[mal_nodes]


def _run(cmd: List[str], *, workdir: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=workdir, check=True)


def _base_cmd() -> List[str]:
    return [
        sys.executable,
        "r4_agg_minitest.py",
        "--attack-modes",
        ",".join(ATTACKS),
        "--dt-levels",
        "D0",
        "--methods",
        ",".join(METHODS),
        "--seeds",
        ",".join(SEEDS),
        "--ref-size-grid",
        "128",
        "--audit-size-grid",
        "0",
        "--r2-source",
        "local_test",
        "--beta-r2",
        "0.00",
        "--beta-r4",
        "15",
        "--mix-r4-beta",
        "20",
        "--tau-gate",
        "0.69",
        "--dt-attack-scale-start",
        "3.0",
        "--dt-attack-scale-end",
        "0.3",
        "--dt-attack-scale-step",
        "0.30",
        "--label-flip-grad-scale",
        "0.5",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=str, default=".")
    ap.add_argument(
        "--out-root",
        type=str,
        default="artifacts/99_workspace_archive/acc_rounds_f_ablation",
    )
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    out_root = Path(args.out_root)

    for mal_nodes in [1, 2, 3, 4, 5]:
        profile = benign_profile_for_f(mal_nodes)
        cmd = _base_cmd() + [
            "--mal-nodes",
            str(mal_nodes),
            "--benign-noise-profile",
            ",".join(profile),
            "--benign-noise-light-spec",
            "drift_light",
            "--benign-noise-medium-spec",
            "drift_medium_v2",
            "--benign-noise-heavy-spec",
            "drift_heavy_v2",
            "--out-root",
            str(out_root / f"f{mal_nodes}"),
            "--exp-group",
            "heterobenign",
        ]
        _run(cmd, workdir=workdir)


if __name__ == "__main__":
    main()
