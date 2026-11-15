"""
Listener-gap test: how often is a whole turn nothing but a listener response?

Real Switchboard callers give many turns that only signal listening ("uh-huh", "yeah",
"right"); LLM dialogues give almost none. P3 (shown examples) and P4 (explicit permission,
no examples) test whether that gap closes when the models are allowed to listen. If the
rates stay near zero under P4, the gap is not a prompting artefact.

Two per-conversation rates, both over ALL turns (scripted "Hello!" seeds removed by the
shared cleaner):

  listener_only_rate : turns that consist only of listener-feedback tokens
                       (analysis.evaluate_generated.BACKCHANNEL_STANDALONE_FULL).
  tiny_turn_rate     : turns of at most 3 words. Lexicon-free, so it also counts listener
                       turns outside the lexicon ("Wow.", "Really?", "Huh.").

The human reference uses exactly the same functions on the topic-matched Switchboard
conversations. Only numbers are printed, never Switchboard text.

    python analysis/listener_gap.py data/generated_workshop/qwen35-9b
    python analysis/listener_gap.py data/generated runs/listener-p4 --sb dev --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.evaluate_generated import BACKCHANNEL_STANDALONE_FULL, load_conversations

TINY_WORDS = 3


def listener_rates(turns: list[tuple[str, str]]) -> dict:
    texts = [t.strip() for _, t in turns if t.strip()]
    if not texts:
        return {"n_turns": 0, "listener_only_rate": None, "tiny_turn_rate": None}
    listener = sum(bool(BACKCHANNEL_STANDALONE_FULL.match(t)) for t in texts)
    tiny = sum(len(t.split()) <= TINY_WORDS for t in texts)
    return {
        "n_turns": len(texts),
        "listener_only_rate": listener / len(texts),
        "tiny_turn_rate": tiny / len(texts),
    }


def switchboard_reference(which: str) -> list[dict]:
    from analysis.swda import conversation_no_of, iter_conversation_files, parse_conversation
    from generation.sampling import load_dev_ids, load_target_ids

    ids = set(load_target_ids() if which == "targets" else load_dev_ids())
    convs = [
        {"condition": "SB-human", "turns": parse_conversation(fp)}
        for fp in iter_conversation_files()
        if conversation_no_of(fp) in ids
    ]
    if len(convs) != len(ids):
        raise RuntimeError(f"found {len(convs)}/{len(ids)} Switchboard conversations")
    return convs


def summarise(convs: list[dict], label_prefix: str = "") -> list[dict]:
    by: dict[str, list[dict]] = {}
    for conv in convs:
        by.setdefault(label_prefix + str(conv.get("condition")), []).append(
            listener_rates(conv["turns"])
        )
    rows = []
    for cond, recs in sorted(by.items()):
        row = {"condition": cond, "n_conversations": len(recs)}
        for key in ("n_turns", "listener_only_rate", "tiny_turn_rate"):
            vals = [r[key] for r in recs if r[key] is not None]
            row[f"{key}_mean"] = round(statistics.mean(vals), 4) if vals else None
            row[f"{key}_sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("data_dirs", nargs="+", type=pathlib.Path, help="generated-data roots")
    ap.add_argument(
        "--sb",
        choices=("targets", "dev", "none"),
        default="targets",
        help="Switchboard reference: the 50 target ids (default), the 5 dev ids, or none",
    )
    ap.add_argument("--csv", type=pathlib.Path, help="also write the table as CSV")
    args = ap.parse_args(argv)

    rows = []
    if args.sb != "none":
        rows += summarise(switchboard_reference(args.sb))
    for root in args.data_dirs:
        prefix = f"{root.name}:" if len(args.data_dirs) > 1 else ""
        rows += summarise(load_conversations(root), prefix)

    print(f"{'condition':<34}{'n':>4}{'turns':>8}{'listener-only':>15}{'<=3 words':>12}")
    for r in rows:
        print(
            f"{r['condition']:<34}{r['n_conversations']:>4}{r['n_turns_mean'] or 0:>8.1f}"
            f"{(r['listener_only_rate_mean'] or 0) * 100:>14.1f}%"
            f"{(r['tiny_turn_rate_mean'] or 0) * 100:>11.1f}%"
        )
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
