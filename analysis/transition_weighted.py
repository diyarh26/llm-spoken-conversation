"""Frequency-weighted transition JSD (robustness check for dialogue_acts.py).

dialogue_acts.py's `coarse trans` score is the UNWEIGHTED mean of per-row JSDs, so a row for
an act that occurs 3 times in a condition (e.g. Answer in C2-P2) counts as much as Statement
with thousands of units — tiny rows are noisy and can dominate. Here each row's JSD is
weighted by how often that act occurs (mean of the two corpora's act shares), so the score
reflects what the conversations mostly do. Reads the CSVs dialogue_acts.py already wrote.

    python analysis/transition_weighted.py results_v3/dialogue_acts
"""
from __future__ import annotations

import csv
import math
import pathlib
import sys


def _jsd(p: list[float], q: list[float]) -> float:
    sp, sq = sum(p), sum(q)
    if not sp and not sq:
        return 0.0
    if not sp or not sq:
        return 1.0
    p = [x / sp for x in p]; q = [x / sq for x in q]
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl = lambda a, b: sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _matrix(path: pathlib.Path) -> dict[str, list[float]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c != "current_act"]
    return {r["current_act"]: [float(r[c]) for c in cols] for r in rows}


def main(out_dir: pathlib.Path) -> None:
    with (out_dir / "da_distribution_coarse_by_condition.csv").open(encoding="utf-8") as f:
        dist = {r["condition"]: r for r in csv.DictReader(f)}
    ref = _matrix(out_dir / "da_transition_coarse_SB_tagger.csv")   # same tagger as the LLMs
    ref_share = dist["SB-tagger"]
    lines = ["condition,weighted_transition_jsd,unweighted_transition_jsd"]
    for cond in sorted(c for c in dist if not c.startswith("SB")):
        mat = _matrix(out_dir / f"da_transition_coarse_{cond}.csv")
        num = den = 0.0
        unweighted = []
        for act in ref:
            d = _jsd(mat[act], ref[act]) if (sum(mat[act]) or sum(ref[act])) else None
            if d is None:
                continue
            w = (float(dist[cond][act]) + float(ref_share[act])) / 2
            num += w * d; den += w; unweighted.append(d)
        lines.append(f"{cond},{num / den:.4f},{sum(unweighted) / len(unweighted):.4f}")
    (out_dir / "da_transition_weighted.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results_v3/dialogue_acts"))
