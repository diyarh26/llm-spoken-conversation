"""
Is "alignment" just turn length? Adjacent-turn similarity minus a non-adjacent baseline.

Long turns average many word vectors, which drift toward a generic direction, so two long
turns look similar whether or not the second one responds to the first. This check
follows the advisor's four steps:

  1. Baseline: per conversation, the mean similarity of turns that are NOT next to each
     other (at least MIN_GAP turns apart).
  2. Real alignment = adjacent-turn similarity - baseline.
  3. Length check: steps 1-2 again (a) with every turn cut to its first 15 words, and
     (b) using only pairs of similar length (shorter/longer >= 0.5, both >= 3 words).
  4. Decide: does each LLM condition still align more than humans after 2 and 3?

Similarity: the cosine of the mean word2vec-google-news-300 vectors of two turns (case
folded, in-vocabulary words only). Adjacent and baseline pairs use exactly the same
function, so their difference is a fair comparison; the absolute values are not ALIGN's
cosine_semanticL (no lemmatising or frequency filter).

Baseline pairs are at an odd distance (5, 7, 9, ...), so, like adjacent pairs, they are
always between the two speakers; a same-speaker pair would add self-similarity to the
baseline. Every such pair in a conversation is used (no sampling).

Humans: the 50 topic-matched Switchboard calls, as in every other analysis. Turns come
from the shared cleaners (analysis.evaluate_generated). Only numbers are written.

    python analysis/alignment_baseline.py --w2v <word2vec-google-news-300.gz> \\
        data/generated data/generated_workshop/qwen35-9b --out results_workshop/alignment_baseline

Adapted from analysis/alignment_confound_check.py (branch claude/ced-embedding-dispersion-itgksm).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIN_GAP = 5
TRUNCATE = 15
MIN_RATIO = 0.5
MIN_WORDS = 3
_WORD = re.compile(r"[a-zA-Z']+")
VARIANTS = ("full", "first15", "matched")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def turn_vector(kv, ws: list[str]):
    vecs = [kv[w] for w in ws if w in kv.key_to_index]
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def conversation_scores(kv, turns: list[tuple[str, str]], variant: str) -> dict | None:
    """Adjacent mean, baseline mean and their difference for one conversation."""
    toks = [words(t) for _, t in turns]
    lens = [len(w) for w in toks]
    if variant == "first15":
        toks = [w[:TRUNCATE] for w in toks]
    vecs = [turn_vector(kv, w) for w in toks]

    def ok(i: int, j: int) -> bool:
        if vecs[i] is None or vecs[j] is None:
            return False
        if variant != "matched":
            return True
        a, b = lens[i], lens[j]
        return min(a, b) >= MIN_WORDS and min(a, b) / max(a, b) >= MIN_RATIO

    n = len(vecs)
    adj = [float(vecs[i] @ vecs[i + 1]) for i in range(n - 1) if ok(i, i + 1)]
    base = [
        float(vecs[i] @ vecs[j]) for i in range(n) for j in range(i + MIN_GAP, n, 2) if ok(i, j)
    ]
    if not adj or not base:
        return None
    a, b = float(np.mean(adj)), float(np.mean(base))
    return {"adjacent": a, "baseline": b, "corrected": a - b, "n_adj": len(adj)}


def bootstrap_diff(x: np.ndarray, y: np.ndarray, reps: int, rng) -> tuple[float, float]:
    d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(reps)]
    lo, hi = np.quantile(d, [0.025, 0.975])
    return float(lo), float(hi)


def main(argv: list[str] | None = None) -> None:
    from gensim.models import KeyedVectors
    from scipy.stats import mannwhitneyu

    from analysis.evaluate_generated import load_conversations, load_switchboard_reference

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("data_dirs", nargs="+", type=pathlib.Path)
    ap.add_argument("--w2v", required=True, help="word2vec-google-news-300(.gz), binary")
    ap.add_argument("--labels", default="", help="comma-separated prefix per data dir")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    labels = args.labels.split(",") if args.labels else [d.name for d in args.data_dirs]

    kv = KeyedVectors.load_word2vec_format(args.w2v, binary=True)
    groups: dict[str, list] = {"SB": [c["turns"] for c in load_switchboard_reference()]}
    for lab, root in zip(labels, args.data_dirs, strict=True):
        for conv in load_conversations(root):
            groups.setdefault(f"{lab}:{conv['condition']}", []).append(conv["turns"])

    rng = np.random.default_rng(20261118)
    per_conv, summary = [], []
    for variant in VARIANTS:
        scores = {}
        for name, convs in groups.items():
            rows = [conversation_scores(kv, t, variant) for t in convs]
            scores[name] = [r for r in rows if r]
            for i, r in enumerate(rows):
                if r:
                    per_conv.append({"variant": variant, "group": name, "conv": i, **r})
        sb = np.array([r["corrected"] for r in scores["SB"]])
        for name, rows in scores.items():
            c = np.array([r["corrected"] for r in rows])
            row = {
                "variant": variant,
                "group": name,
                "n_conversations": len(rows),
                "adjacent": np.mean([r["adjacent"] for r in rows]),
                "baseline": np.mean([r["baseline"] for r in rows]),
                "corrected": c.mean(),
            }
            if name != "SB":
                lo, hi = bootstrap_diff(c, sb, args.reps, rng)
                row |= {
                    "diff_vs_SB": c.mean() - sb.mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "mannwhitney_p": mannwhitneyu(c, sb).pvalue,
                }
            summary.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    keys = [
        "variant",
        "group",
        "n_conversations",
        "adjacent",
        "baseline",
        "corrected",
        "diff_vs_SB",
        "ci_low",
        "ci_high",
        "mannwhitney_p",
    ]
    with (args.out / "alignment_baseline_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary:
            w.writerow(
                {k: (f"{r[k]:.4g}" if isinstance(r.get(k), float) else r.get(k, "")) for k in keys}
            )
    with (args.out / "alignment_baseline_per_conversation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_conv[0]))
        w.writeheader()
        w.writerows(per_conv)

    for variant in VARIANTS:
        print(f"\n=== {variant} ===")
        print(
            f"{'group':<26}{'n':>4}{'adjacent':>10}{'baseline':>10}{'corrected':>11}"
            f"{'vs SB':>9}{'95% CI':>18}{'p':>10}"
        )
        for r in summary:
            if r["variant"] != variant:
                continue
            extra = ""
            if "diff_vs_SB" in r:
                extra = (
                    f"{r['diff_vs_SB']:>+9.3f}  [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
                    f"{r['mannwhitney_p']:>10.2g}"
                )
            print(
                f"{r['group']:<26}{r['n_conversations']:>4}{r['adjacent']:>10.3f}"
                f"{r['baseline']:>10.3f}{r['corrected']:>11.3f}{extra}"
            )


if __name__ == "__main__":
    main()
