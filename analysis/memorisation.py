"""
Memorisation check: do generated dialogues reproduce Switchboard text?

Switchboard is public, so it may be in the models' training data, and P2 shows two real
excerpts in the prompt. For every generated conversation this reports the share of its
word n-grams (default n = 8) that also occur

  sb_overlap     : anywhere in the Switchboard corpus (all 1,155 SwDA conversations);
  prompt_overlap : in the two P2 excerpts that this conversation's prompt showed
                   (P2 only; copying from the prompt, not from memory).

n-grams that occur only in the shown excerpts are not counted in sb_overlap, so copying
from the prompt is never reported as memorisation.

Because people share everyday phrases ("what do you think about the", "i don't know
what"), some overlap is expected by chance. The human baseline scores each of the 50
target Switchboard calls against all OTHER Switchboard calls (leave-one-out), with the
same function. Generated overlap at or below this baseline means no evidence of
memorisation beyond ordinary shared phrasing.

    python analysis/memorisation.py data/generated
    python analysis/memorisation.py data/generated_workshop/qwen35-9b --n 6 --show 10

Only numbers are printed by default. --show prints the most frequent matched n-grams
(short Switchboard phrases) for manual inspection; do not commit that output.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.evaluate_generated import load_conversations

_WORD = re.compile(r"[a-z']+")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def overlap(grams: list[tuple[str, ...]], reference) -> float | None:
    """Share of the n-gram occurrences that are in `reference` (None if text too short)."""
    if not grams:
        return None
    return sum(g in reference for g in grams) / len(grams)


class Switchboard:
    """Document frequency of every n-gram over the SwDA conversations."""

    def __init__(self, n: int):
        from analysis.swda import conversation_no_of, iter_conversation_files, parse_conversation

        self.n = n
        self.docs: dict[int, set] = {}
        for fp in iter_conversation_files():
            words = tokens(" ".join(t for _, t in parse_conversation(fp)))
            self.docs[conversation_no_of(fp)] = set(ngrams(words, n))
        self.df: Counter = Counter()
        for grams in self.docs.values():
            self.df.update(grams)

    def __contains__(self, gram) -> bool:
        return self.df[gram] > 0

    def in_others(self, conversation_no: int):
        """Membership test that ignores one conversation (leave-one-out)."""
        own = self.docs.get(conversation_no, set())
        return _Excluding(self.df, own)


class _Excluding:
    def __init__(self, df: Counter, own: set):
        self.df, self.own = df, own

    def __contains__(self, gram) -> bool:
        return self.df[gram] - (gram in self.own) > 0


def shown_examples(conversation_no: int, n: int) -> set:
    from analysis.swda import fewshot_examples

    words = tokens(" ".join(e["text"] for e in fewshot_examples(conversation_no, k=2)))
    return set(ngrams(words, n))


def score_generated(convs: list[dict], sb: Switchboard, show: Counter | None = None) -> list[dict]:
    rows = []
    for conv in convs:
        grams = ngrams(tokens(" ".join(t for _, t in conv["turns"])), sb.n)
        is_p2 = conv.get("prompt_level") == "P2"
        shown = shown_examples(conv["conversation_no"], sb.n) if is_p2 else set()
        memory = [g for g in grams if g not in shown]
        row = {
            "condition": conv.get("condition"),
            "conversation_no": conv.get("conversation_no"),
            "sb_overlap": overlap(memory, sb),
            "prompt_overlap": overlap(grams, shown) if is_p2 else None,
        }
        if show is not None:
            show.update(" ".join(g) for g in memory if g in sb)
        rows.append(row)
    return rows


def human_baseline(sb: Switchboard) -> list[float]:
    from analysis.swda import conversation_no_of, iter_conversation_files, parse_conversation
    from generation.sampling import load_target_ids

    targets = set(load_target_ids())
    out = []
    for fp in iter_conversation_files():
        cno = conversation_no_of(fp)
        if cno in targets:
            grams = ngrams(tokens(" ".join(t for _, t in parse_conversation(fp))), sb.n)
            value = overlap(grams, sb.in_others(cno))
            if value is not None:
                out.append(value)
    return out


def _fmt(values: list[float]) -> str:
    if not values:
        return f"{'-':>9}{'-':>9}"
    return f"{100 * statistics.mean(values):>8.2f}%{100 * max(values):>8.2f}%"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("data_dirs", nargs="+", type=pathlib.Path, help="generated-data roots")
    ap.add_argument("--n", type=int, default=8, help="n-gram length in words (default 8)")
    ap.add_argument("--show", type=int, default=0, help="print the N most frequent matches")
    args = ap.parse_args(argv)

    sb = Switchboard(args.n)
    baseline = human_baseline(sb)
    print(f"{args.n}-gram overlap with Switchboard (mean / max per conversation)")
    print(f"{'source':<34}{'n':>5}{'SB mean':>9}{'SB max':>9}{'prompt mean':>13}{'max':>8}")
    print(f"{'HUMAN baseline (leave-one-out)':<34}{len(baseline):>5}{_fmt(baseline)}")
    show = Counter() if args.show else None
    for root in args.data_dirs:
        rows = score_generated(load_conversations(root), sb, show)
        for cond in sorted({r["condition"] for r in rows}):
            sel = [r for r in rows if r["condition"] == cond]
            sbv = [r["sb_overlap"] for r in sel if r["sb_overlap"] is not None]
            pv = [r["prompt_overlap"] for r in sel if r["prompt_overlap"] is not None]
            prompt = _fmt(pv) if pv else f"{'-':>13}{'-':>8}"
            if pv:
                prompt = f"{100 * statistics.mean(pv):>12.2f}%{100 * max(pv):>7.2f}%"
            name = f"{root.name}:{cond}" if len(args.data_dirs) > 1 else cond
            print(f"{name:<34}{len(sel):>5}{_fmt(sbv)}{prompt}")
    if show:
        print(f"\nMost frequent matched {args.n}-grams (inspect only; do not commit):")
        for gram, count in show.most_common(args.show):
            print(f"  {count:>4}  {gram}")


if __name__ == "__main__":
    main()
