"""
Explicit degeneration score for the decoding dev sweep (§8 redo: "tune decoding
per-architecture on a dev set against an explicit degeneration score, freeze it,
document it").

Scores one or more condition directories of generated-conversation JSONs and reports,
per directory:

  degeneration components (what tuning must minimize):
    dup_turn_rate      near-verbatim repeats of an earlier turn (>=8 words, Jaccard>=0.8
                       — same definition as the runtime guard in generation/quality.py)
    turn_cap_rate      conversations cut off by max_turns instead of ending naturally
    token_cap_rate     generate() calls that hit max_new_tokens (truncated turns)
    dup_kept_rate      loop-guard resamples that still came back duplicated
    empty_retry_rate   turns that needed a retry after an empty emission

  descriptives (NOT degeneration — do not tune these away; they are the measured DVs):
    words/turn mean & median, short_turn_rate (<3 words: backchannel-like turns),
    n_turns mean, multi_turn_emission rate

Composite: degeneration_per_conv = (dup turns + dup_kept + token-cap hits
                                    + turn-cap endings) / conversations.

Stdlib only — runs on the VM or locally, no torch. Usage:
    python generation/degeneration_score.py data/generated_v3/C2-P0 [more dirs ...] [--json]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from generation.quality import is_near_duplicate                          # noqa: E402

_C1_LINE = re.compile(r"^\s*Participants?\s*_?([AB])\s*[:,]\s*(.+)$", re.I)


def _turns_of(rec: dict) -> tuple[list[tuple[str, str]], int]:
    """(turns, seed_turns) — parses C1 raw_output into turns when needed."""
    if "turns" in rec:
        return [tuple(t) for t in rec["turns"]], int(rec.get("seed_turns", 0))
    turns = []
    for line in rec.get("raw_output", "").splitlines():
        m = _C1_LINE.match(line)
        if m:
            turns.append((f"Participant{m.group(1).upper()}", m.group(2).strip()))
    return turns, 0  # C1 seed turns live in the prompt, not the output


def score_dir(cond_dir: pathlib.Path) -> dict:
    convs = sorted(cond_dir.glob("*.json"))
    total_turns = 0
    words_per_turn: list[int] = []
    short_turns = 0
    dup_turns = 0
    n_turns_list: list[int] = []
    ended = Counter()
    counters = Counter()
    token_cap_c1 = 0

    for fp in convs:
        rec = json.loads(fp.read_text(encoding="utf-8"))
        turns, seed = _turns_of(rec)
        gen_turns = turns[seed:]
        n_turns_list.append(len(turns))
        ended[rec.get("ended_by", "n/a")] += 1
        counters.update(rec.get("quality_counters", {}))
        token_cap_c1 += int(bool(rec.get("hit_token_cap")))
        for i, (_, txt) in enumerate(gen_turns):
            total_turns += 1
            nw = len(txt.split())
            words_per_turn.append(nw)
            short_turns += int(nw < 3)
            dup_turns += int(is_near_duplicate(txt, turns[:seed + i]))

    n = len(convs)
    if not n or not total_turns:
        return {"dir": str(cond_dir), "conversations": n, "error": "no data"}
    turn_caps = ended.get("turn_cap", 0)
    token_caps = counters.get("hit_token_cap", 0) + token_cap_c1
    degeneration = dup_turns + counters.get("dup_kept", 0) + token_caps + turn_caps
    return {
        "dir": str(cond_dir),
        "conversations": n,
        # --- degeneration components (minimize) ---
        "dup_turn_rate": round(dup_turns / total_turns, 4),
        "turn_cap_rate": round(turn_caps / n, 4),
        "token_cap_rate": round(token_caps / max(total_turns, 1), 4),
        "dup_kept": counters.get("dup_kept", 0),
        "empty_retries": counters.get("empty_retries", 0),
        "degeneration_per_conv": round(degeneration / n, 3),
        # --- descriptives (DVs — report, don't tune) ---
        "n_turns_mean": round(statistics.mean(n_turns_list), 2),
        "words_per_turn_mean": round(statistics.mean(words_per_turn), 2),
        "words_per_turn_median": statistics.median(words_per_turn),
        "short_turn_rate": round(short_turns / total_turns, 4),
        "multi_turn_emission_rate": round(
            counters.get("multi_turn_emissions", 0) / total_turns, 4),
        "ended_by": dict(ended),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="condition directories of generated JSONs")
    ap.add_argument("--json", action="store_true", help="emit one JSON object per dir")
    args = ap.parse_args()
    for d in args.dirs:
        s = score_dir(pathlib.Path(d))
        if args.json:
            print(json.dumps(s))
        else:
            print(f"\n== {s.pop('dir')} ==")
            for k, v in s.items():
                print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
