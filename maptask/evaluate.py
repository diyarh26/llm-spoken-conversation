"""
Evaluate Map Task dialogues (generated and human) on behaviour and form.

Two steps:

1. annotate (GPU): one model, the same for every source, reads each finished dialogue and
   (a) draws the route from the follower's map only -> route deviation vs the giver's
   route (task success), and (b) labels every turn with its main HCRC dialogue move.
   Results are written next to each record (generated) or to <out>/human/<id>.json.

       python -m maptask.evaluate annotate --model Qwen/Qwen3.5-9B --revision <sha> \\
           data/generated_maptask/qwen35-9b/C1 data/generated_maptask/qwen35-9b/C3
       python -m maptask.evaluate annotate --model ... --human --out data/generated_maptask

2. report (CPU): one table per source.

       python -m maptask.evaluate report data/generated_maptask/human \\
           data/generated_maptask/qwen35-9b/C1 data/generated_maptask/qwen35-9b/C3

Behaviour: the share of each role's turns per move (e.g. follower check / query, giver
align), and route deviation. Form: turns per dialogue, words per turn, listener-only turns.
For humans the corpus's own gold moves are also reported, and the tagger's agreement with
them (on human turns) tells how far the model labels can be trusted.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.evaluate_generated import BACKCHANNEL_STANDALONE_FULL
from maptask.corpus import load_dialogues, load_maps, load_moves
from maptask.prompts import (
    FOLLOWER,
    GIVER,
    MOVE_DEFINITIONS,
    build_move_tagger,
    build_reconstruction,
)
from maptask.score import parse_points, route_deviation

MOVES = tuple(MOVE_DEFINITIONS)
TAG_WINDOW = 30
ROLE = {"giver": GIVER, "follower": FOLLOWER}


# --- human dialogues ---------------------------------------------------------------------


def human_turns(dialogue: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Turns built from the coded moves (consecutive moves by one speaker = one turn), and
    each turn's gold label = its first move. Built from the moves themselves so turns and
    gold labels line up exactly."""
    turns: list[tuple[str, str]] = []
    gold: list[str] = []
    for mv in load_moves(dialogue):
        spk = ROLE[mv["speaker"]]
        if turns and turns[-1][0] == spk:
            turns[-1] = (spk, f"{turns[-1][1]} {mv['text']}")
        else:
            turns.append((spk, mv["text"]))
            gold.append(mv["label"])
    return turns, gold


def human_records() -> list[dict]:
    out = []
    for d, meta in sorted(load_dialogues().items()):
        turns, gold = human_turns(d)
        out.append(
            {
                "domain": "maptask",
                "architecture": "HUMAN",
                "id": d,
                "map": meta["map"],
                "turns": turns,
                "gold_move_labels": gold,
                "devscore": meta["devscore"],
            }
        )
    return out


# --- annotation (model calls) -------------------------------------------------------------


def parse_labels(text: str, n: int) -> list[str] | None:
    try:
        start, end = text.index("["), text.rindex("]") + 1
        labels = json.loads(text[start:end])
    except ValueError:
        return None
    labels = [str(x).strip().lower() for x in labels]
    if len(labels) != n:
        return None
    return [lab if lab in MOVES else "uncodable" for lab in labels]


def tag_moves(ask, turns: list[tuple[str, str]]) -> list[str]:
    """Label turns in windows of TAG_WINDOW; a window whose answer cannot be parsed is
    labelled 'uncodable' (recorded, never guessed)."""
    labels: list[str] = []
    for i in range(0, len(turns), TAG_WINDOW):
        chunk = turns[i : i + TAG_WINDOW]
        got = parse_labels(ask(build_move_tagger(chunk), 1024), len(chunk))
        labels += got or ["uncodable"] * len(chunk)
    return labels


def annotate(record: dict, ask) -> dict:
    m = load_maps()[record["map"]]
    turns = [tuple(t) for t in record["turns"]]
    answer = ask(build_reconstruction(m, turns), 1024)
    drawn = parse_points(answer)
    record["reconstruction"] = {"raw": answer, "points": drawn}
    record["route_deviation"] = route_deviation(drawn, m["giver"]["route"]) if drawn else None
    record["move_labels"] = tag_moves(ask, turns)
    return record


def _ask_fn(args):
    from generation.model_utils import chat, load_model

    model, tok = load_model(args.model, revision=args.revision, precision=args.precision)

    def ask(messages, max_new_tokens):
        text, _ = chat(model, tok, messages, max_new_tokens=max_new_tokens, do_sample=False)
        return text

    return ask


# --- report -------------------------------------------------------------------------------


def load_records(folder: pathlib.Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]


def summarise(records: list[dict], labels_key: str = "move_labels") -> dict:
    turns = [r["turns"] for r in records]
    texts = [t[1] for ts in turns for t in ts]
    row = {
        "n": len(records),
        "turns_per_dialogue": statistics.mean(len(ts) for ts in turns) if turns else math.nan,
        "words_per_turn": statistics.mean(len(t.split()) for t in texts) if texts else math.nan,
        "listener_only_pct": 100
        * sum(bool(BACKCHANNEL_STANDALONE_FULL.match(t)) for t in texts)
        / max(1, len(texts)),
    }
    devs = [r["route_deviation"] for r in records if r.get("route_deviation") is not None]
    row["route_deviation"] = statistics.mean(devs) if devs else None
    for role in (GIVER, FOLLOWER):
        c = Counter(
            lab
            for r in records
            if r.get(labels_key)
            for (spk, _), lab in zip(r["turns"], r[labels_key], strict=False)
            if spk == role
        )
        total = sum(c.values())
        row[role] = {lab: 100 * c[lab] / total for lab in MOVES} if total else None
    return row


def tagger_agreement(records: list[dict]) -> float | None:
    pairs = [
        (g, p)
        for r in records
        if r.get("move_labels") and r.get("gold_move_labels")
        for g, p in zip(r["gold_move_labels"], r["move_labels"], strict=False)
    ]
    return 100 * sum(g == p for g, p in pairs) / len(pairs) if pairs else None


def print_report(rows: dict[str, dict]) -> None:
    print(
        f"{'source':<28}{'n':>4}{'turns':>8}{'words/turn':>12}{'listener-only':>15}{'route dev':>11}"
    )
    for name, r in rows.items():
        dev = f"{r['route_deviation']:.1f}" if r["route_deviation"] is not None else "-"
        print(
            f"{name:<28}{r['n']:>4}{r['turns_per_dialogue']:>8.1f}{r['words_per_turn']:>12.1f}"
            f"{r['listener_only_pct']:>14.1f}%{dev:>11}"
        )
    for role in (GIVER, FOLLOWER):
        print(f"\n{role} moves (% of {role.lower()} turns)")
        print(f"{'source':<28}" + "".join(f"{m[:9]:>10}" for m in MOVES))
        for name, r in rows.items():
            if r[role]:
                print(f"{name:<28}" + "".join(f"{r[role][m]:>9.1f}%" for m in MOVES))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("annotate")
    an.add_argument("folders", nargs="*", type=pathlib.Path)
    an.add_argument("--human", action="store_true", help="annotate the 128 human dialogues")
    an.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "generated_maptask")
    an.add_argument("--model", required=True)
    an.add_argument("--revision", default="main")
    an.add_argument("--precision", choices=("nf4", "bf16", "fp16"), default="nf4")
    rp = sub.add_parser("report")
    rp.add_argument("folders", nargs="+", type=pathlib.Path)
    args = ap.parse_args(argv)

    if args.cmd == "annotate":
        ask = _ask_fn(args)
        jobs = []
        if args.human:
            for rec in human_records():
                jobs.append((args.out / "human" / f"{rec['id']}.json", rec))
        for folder in args.folders:
            for path in sorted(folder.glob("*.json")):
                jobs.append((path, json.loads(path.read_text(encoding="utf-8"))))
        for path, rec in jobs:
            if rec.get("move_labels") is not None:
                continue  # already annotated
            rec["annotator"] = {"model": args.model, "revision": args.revision}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(annotate(rec, ask), indent=1), encoding="utf-8")
            print(f"  annotated {path.name}: route_deviation={rec['route_deviation']}")
        return

    rows = {}
    for folder in args.folders:
        recs = load_records(folder)
        name = f"{folder.parent.name}/{folder.name}" if folder.name != "human" else "HUMAN"
        rows[name] = summarise(recs)
        if folder.name == "human":
            rows["HUMAN (gold moves)"] = summarise(recs, "gold_move_labels")
            agree = tagger_agreement(recs)
            if agree is not None:
                print(f"Tagger agreement with gold first move on human turns: {agree:.1f}%")
    print_report(rows)


if __name__ == "__main__":
    main()
