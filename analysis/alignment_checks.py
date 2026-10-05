"""Robustness checks on ALIGN conceptual alignment (cosine_semanticL).

Reads the per-turn-pair CSV written by analysis/export_align.py and answers three questions
before any alignment claim goes on the poster:

  1. Earlier vs Later — is the within-conversation change real? Per conversation, mean
     alignment of the first vs second half of its turn pairs; Wilcoxon signed-rank per
     condition. Repeated without the last 2 pairs (closing pleasantries are near-identical
     "thanks, you too" and could create the rise on their own).
  2. Length control — long turns' averaged word vectors converge, so alignment can rise with
     turn length alone. (a) OLS: cosine ~ condition + log(words1) + log(words2), standard
     errors clustered by conversation, effects reported vs SB; (b) a length-matched subset
     (both turns 5–40 words) with plain means.
  3. Echo loops — '<cond>-noloop' rows (export_align.py --loop-sensitivity) vs the original.

    python analysis/alignment_checks.py [--csv data/align/alignment_turns.csv] [--out results_v3]
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent


def earlier_later(df: pd.DataFrame, drop_last: int = 0) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition"):
        diffs, early_means, late_means = [], [], []
        for _, conv in g.groupby("conv_id"):
            conv = conv.sort_values("turn_index")
            if drop_last:
                conv = conv.iloc[:-drop_last]
            if len(conv) < 4:
                continue
            half = len(conv) // 2
            e, l = conv["cosine_semanticL"].iloc[:half].mean(), conv["cosine_semanticL"].iloc[half:].mean()
            early_means.append(e); late_means.append(l); diffs.append(l - e)
        p = stats.wilcoxon(diffs).pvalue if len(diffs) >= 5 and any(diffs) else float("nan")
        rows.append({"condition": cond, "n_conversations": len(diffs),
                     "earlier": np.mean(early_means), "later": np.mean(late_means),
                     "delta": np.mean(diffs), "wilcoxon_p": p})
    return pd.DataFrame(rows)


def length_controlled(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["words1", "words2"]).copy()
    d = d[(d["words1"] > 0) & (d["words2"] > 0)]
    d["lw1"], d["lw2"] = np.log(d["words1"]), np.log(d["words2"])
    groups = pd.factorize(d["condition"].astype(str) + "/" + d["conv_id"].astype(str))[0]
    raw = smf.ols("cosine_semanticL ~ C(condition, Treatment('SB'))", d).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    adj = smf.ols("cosine_semanticL ~ C(condition, Treatment('SB')) + lw1 + lw2", d).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    matched = d[(d["words1"].between(5, 40)) & (d["words2"].between(5, 40))]
    rows = []
    for cond in sorted(d["condition"].unique()):
        key = f"C(condition, Treatment('SB'))[T.{cond}]"
        m = matched[matched["condition"] == cond]["cosine_semanticL"]
        rows.append({
            "condition": cond,
            "pairs_with_length": int((d["condition"] == cond).sum()),
            "raw_diff_vs_SB": raw.params.get(key, 0.0),
            "adjusted_diff_vs_SB": adj.params.get(key, 0.0),
            "adjusted_p": adj.pvalues.get(key, float("nan")),
            "matched_5_40w_mean": m.mean() if len(m) else float("nan"),
            "matched_5_40w_pairs": len(m),
        })
    out = pd.DataFrame(rows)
    out.attrs["length_slopes"] = (adj.params["lw1"], adj.params["lw2"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "align" / "alignment_turns.csv"))
    ap.add_argument("--out", default=str(ROOT / "results_v3"))
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype={"conv_id": str})
    df = df.dropna(subset=["cosine_semanticL"])
    df["words1"] = pd.to_numeric(df.get("words1"), errors="coerce")
    df["words2"] = pd.to_numeric(df.get("words2"), errors="coerce")

    overall = df.groupby("condition")["cosine_semanticL"].agg(["count", "mean", "std"]).reset_index()
    el = earlier_later(df)
    el2 = earlier_later(df, drop_last=2)
    lines = ["# Alignment robustness checks (ALIGN cosine_semanticL)", "",
             "## Overall", overall.to_string(index=False, float_format="%.3f"), "",
             "## 1. Earlier vs Later (per-conversation halves, Wilcoxon signed-rank)",
             el.to_string(index=False, float_format="%.3f"), "",
             "## 1b. Same, without the last 2 turn pairs (closings)",
             el2.to_string(index=False, float_format="%.3f"), ""]
    if df["words1"].notna().any():
        lc = length_controlled(df)
        s1, s2 = lc.attrs["length_slopes"]
        lines += ["## 2. Length control (OLS, SEs clustered by conversation; differences vs SB)",
                  f"log-length slopes: words1 {s1:+.3f}, words2 {s2:+.3f} (alignment per log-word)",
                  lc.to_string(index=False, float_format="%.3f"), ""]
        lc.to_csv(out / "alignment_length_controlled.csv", index=False)
    else:
        lines += ["## 2. Length control: SKIPPED (CSV has no words1/words2 — re-export with the "
                  "current analysis/export_align.py)", ""]
    loops = [c for c in overall["condition"] if c.endswith("-noloop")]
    if loops:
        sub = overall[overall["condition"].isin(loops + [c[:-len("-noloop")] for c in loops])]
        lines += ["## 3. Echo loops removed vs original", sub.to_string(index=False, float_format="%.3f"), ""]
    el.assign(variant="all_pairs").pipe(
        lambda a: pd.concat([a, el2.assign(variant="without_last_2_pairs")])
    ).to_csv(out / "alignment_earlier_later.csv", index=False)
    (out / "alignment_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
