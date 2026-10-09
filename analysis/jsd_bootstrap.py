"""Conversation-level bootstrap CIs for the coarse dialogue-act JSD vs humans.

The noise floor in dialogue_acts.py resamples *units*; this resamples whole *conversations*
(units within a conversation are not independent), so its intervals are the honest ones for
"is C1 really closer to humans than A3?". Labels come from the DialogTag cache (no retagging).

    python analysis/jsd_bootstrap.py --cache <dialogtag cache json> [--reps 2000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.dialogue_acts import COARSE_LABELS, FINE_TO_COARSE, js_divergence, load_generated  # noqa: E402

IDX = {c: i for i, c in enumerate(COARSE_LABELS)}


def counts(labels: list[str]) -> np.ndarray:
    v = np.zeros(len(COARSE_LABELS))
    for lab in labels:
        v[IDX[FINE_TO_COARSE[lab]]] += 1
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "results_v3" / "da_jsd_bootstrap.csv"))
    args = ap.parse_args()
    cache = json.load(open(args.cache, encoding="utf-8"))
    memo = cache["generated_unit_labels"]
    human = np.array([counts(r["fine_labels"]) for r in cache["human"]])
    by_cond: dict[str, list[np.ndarray]] = {}
    for conv in load_generated(ROOT / "data" / "generated_v3"):
        labs = [memo[hashlib.sha1(t.encode("utf-8")).hexdigest()] for t in conv.texts]
        by_cond.setdefault(conv.condition, []).append(counts(labs))
    rng = np.random.default_rng(20261009)
    h_full = human.sum(0)
    lines = ["condition,jsd,ci_low,ci_high"]
    boots = {}
    for cond in sorted(by_cond):
        m = np.array(by_cond[cond])
        point = js_divergence(m.sum(0), h_full)
        reps = []
        for _ in range(args.reps):
            g = m[rng.integers(0, len(m), len(m))].sum(0)
            h = human[rng.integers(0, len(human), len(human))].sum(0)
            reps.append(js_divergence(g, h))
        boots[cond] = np.array(reps)
        lo, hi = np.quantile(reps, [0.025, 0.975])
        lines.append(f"{cond},{point:.4f},{lo:.4f},{hi:.4f}")
    # architecture-level contrasts (pooled over prompts): C1 vs each other architecture
    arch = {a: np.mean([boots[f"{a}-{p}"] for p in ("P0", "P1", "P2")], axis=0) for a in ("C1", "C2", "C3", "C4")}
    lines.append("")
    lines.append("contrast,mean_diff,ci_low,ci_high,share_of_reps_C1_closer")
    for a in ("C2", "C3", "C4"):
        d = arch[a] - arch["C1"]
        lo, hi = np.quantile(d, [0.025, 0.975])
        lines.append(f"{a}-minus-C1,{d.mean():.4f},{lo:.4f},{hi:.4f},{(d > 0).mean():.4f}")
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
