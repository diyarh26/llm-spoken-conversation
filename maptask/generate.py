"""
Generate Map Task dialogues with C1-C4 (the second domain).

    python -m maptask.generate --arch c3 --model Qwen/Qwen3.5-9B --revision <sha> \\
        --out-root data/generated_maptask/qwen35-9b
    python -m maptask.generate --arch c4 --model Qwen/Qwen3.5-9B --model-b google/gemma-4-31B-it \\
        --revision <sha> --revision-b <sha> --out-root data/generated_maptask/qwen-gemma

One conversation per (map, repetition): 16 maps x --reps (default 1). Files are
<out-root>/<ARCH>/<map>-r<rep>.json; existing files are skipped, so a stopped run resumes.
C4 gives the giver to --model and the follower to --model-b.

The turn loop reuses the main study's guards (generation.turnwise.generate_turn: one retry
of an empty turn, one resample of a near-verbatim loop) and the same decoding defaults.
Map Task calls have no scripted "Hello!": the giver speaks first, and the call ends at a
mutual goodbye, one turn after a goodbye, an empty turn, or --max-turns.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import random
import subprocess
import sys
from collections.abc import Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from generation.cleaning import looks_like_closing
from generation.config import C1 as C1_DECODING
from generation.config import TURNWISE
from generation.run_context import atomic_json
from generation.turnwise import generate_turn
from maptask.corpus import load_maps
from maptask.prompts import FOLLOWER, GIVER, LABELS, build_agent, build_c1, build_c2

# Turn cap = the median length of the 128 human dialogues (126 turns; built from the coded
# moves as in maptask.evaluate.human_turns). Dialogues normally end on their own; the cap
# stops turn-by-turn runs that loop instead of closing. The same cap applies to C2, C3, C4.
MAX_TURNS = 126
ARCHS = ("c1", "c2", "c3", "c4")

Turn = tuple[str, str]
Speak = Callable[[str, list[Turn]], str]


def run_dialogue(speak: Speak, max_turns: int) -> tuple[list[Turn], str]:
    """Alternate Giver/Follower from an empty call; return (turns, ended_by)."""
    history: list[Turn] = []
    closing_seen = False
    for i in range(max_turns):
        speaker = LABELS[i % 2]
        turn = speak(speaker, history)
        if not turn:
            return history, "empty_turn"
        previous = history[-1][1] if history else ""
        history.append((speaker, turn))
        if looks_like_closing(turn) and looks_like_closing(previous):
            return history, "mutual_closing"
        if closing_seen:
            return history, "closing"
        if looks_like_closing(turn):
            closing_seen = True
    return history, "turn_cap"


def parse_c1(text: str) -> list[Turn]:
    """Giver:/Follower: lines of a C1 transcript (other lines are ignored)."""
    turns = []
    for line in text.splitlines():
        spk, sep, txt = line.strip().partition(":")
        if sep and spk.strip() in LABELS and txt.strip():
            turns.append((spk.strip(), txt.strip()))
    return turns


def conversation_seed(seed: int, arch: str, map_id: str, rep: int) -> int:
    value = f"{seed}:maptask:{arch}:{map_id}:{rep}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arch", choices=ARCHS, required=True)
    ap.add_argument("--model", required=True, help="the model (C4: the giver's model)")
    ap.add_argument("--model-b", help="C4 only: the follower's model")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--revision-b", default="main")
    ap.add_argument("--precision", choices=("nf4", "bf16", "fp16"), default="nf4")
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--maps", default="", help="comma-separated map ids (default: all 16)")
    ap.add_argument("--reps", type=int, default=1, help="conversations per map")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    if args.arch == "c4" and not args.model_b:
        ap.error("--arch c4 needs --model-b")

    maps = load_maps()
    map_ids = args.maps.split(",") if args.maps else sorted(maps, key=lambda k: int(k[1:]))
    out_dir = args.out_root / args.arch.upper()
    todo = [
        (mid, rep)
        for mid in map_ids
        for rep in range(1, args.reps + 1)
        if not (out_dir / f"{mid}-r{rep}.json").exists()
    ]
    print(f"[maptask {args.arch.upper()}] {len(todo)} to generate -> {out_dir}")
    if not todo:
        return

    import torch

    from generation.model_utils import chat, load_model

    models = {GIVER: (args.model, args.revision)}
    models[FOLLOWER] = (
        (args.model_b, args.revision_b) if args.arch == "c4" else (args.model, args.revision)
    )
    loaded = {}
    for name, rev in set(models.values()):
        loaded[(name, rev)] = load_model(name, revision=rev, precision=args.precision)
    engine = {role: loaded[key] for role, key in models.items()}
    decoding = {"temperature": TURNWISE.temperature, "top_p": TURNWISE.top_p}

    for mid, rep in todo:
        m = maps[mid]
        seed = conversation_seed(args.seed, args.arch, mid, rep)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        counters: dict[str, int] = {}
        record = {
            "domain": "maptask",
            "architecture": args.arch.upper(),
            "map": mid,
            "rep": rep,
            "models": {role: {"name": n, "revision": r} for role, (n, r) in models.items()},
            "precision": args.precision,
            "thinking": "off",
            "conversation_seed": seed,
            "git_commit": _git_commit(),
            "max_turns": args.max_turns,
        }
        if args.arch == "c1":
            model, tok = engine[GIVER]
            text, info = chat(
                model,
                tok,
                build_c1(m),
                max_new_tokens=C1_DECODING.max_new_tokens,
                **decoding,
            )
            turns, ended_by = parse_c1(text), "single_generation"
            record |= {"raw_output": text, "hit_token_cap": info["hit_token_cap"]}
            record["decoding"] = {"max_new_tokens": C1_DECODING.max_new_tokens, **decoding}
        else:

            def speak(speaker: str, history: list[Turn], _m=m, _c=counters) -> str:
                model, tok = engine[speaker]
                messages = (
                    build_c2(_m, history, speaker, args.max_turns)
                    if args.arch == "c2"
                    else build_agent(speaker, _m, history, args.max_turns)
                )
                return generate_turn(
                    lambda temp: chat(
                        model,
                        tok,
                        messages,
                        max_new_tokens=TURNWISE.max_new_tokens,
                        temperature=temp,
                        top_p=decoding["top_p"],
                        stop_labels=LABELS,
                    ),
                    history,
                    LABELS,
                    temperature=decoding["temperature"],
                    counters=_c,
                )

            turns, ended_by = run_dialogue(speak, args.max_turns)
            record["decoding"] = {"max_new_tokens": TURNWISE.max_new_tokens, **decoding}
            record["quality_counters"] = counters
        record |= {"turns": turns, "n_turns": len(turns), "ended_by": ended_by}
        atomic_json(out_dir / f"{mid}-r{rep}.json", record)
        print(f"  saved {mid}-r{rep}: {len(turns)} turns, ended_by={ended_by}")


if __name__ == "__main__":
    main()
