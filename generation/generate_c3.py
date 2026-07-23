"""
C3 generation: two independent first-person agents using the same model.

Both participants use Vicuna, but each generation call is built from that speaker's own
point of view: own prior turns are assistant messages, partner turns are user messages.

Targets come from the committed stratified manifest (generation/target_ids.json), decoding
from generation/config.py (DV-safe; see GENERATION_SPEC.md). Resumable; writes
<out-root>/C3-<prompt>/<id>.json. Run in tmux:
    python generation/generate_c3.py --prompt P0 --out-root data/generated_v3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from prompts.templates import build_agent, assign_cards                    # noqa: E402
from analysis.swda import load_metadata, make_personas                     # noqa: E402
from generation.sampling import load_target_ids                            # noqa: E402
from generation.config import (                                            # noqa: E402
    TURNWISE, MAX_TURNS, DUP_MIN_WORDS, DUP_JACCARD, RESAMPLE_TEMP_BUMP,
)
from generation.model_utils import (                                       # noqa: E402
    load_model, generate_turn, looks_like_closing, VICUNA,
)

OUT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "generated"
LABELS = ("ParticipantA", "ParticipantB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="P0", choices=["P0", "P1", "P2"])
    ap.add_argument("--n", type=int, default=0,
                    help="generate only the first N manifest ids (0 = all; prefix is ~stratified)")
    ap.add_argument("--ids", default="",
                    help="comma-separated conversation_no's to (re)generate; overrides the manifest")
    # Decoding defaults live in generation/config.py (frozen after the dev sweep); the CLI
    # exists only for the sweep itself.
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--max-new-tokens", type=int, default=TURNWISE.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=TURNWISE.temperature)
    ap.add_argument("--top-p", type=float, default=TURNWISE.top_p)
    ap.add_argument("--min-new-tokens", type=int, default=TURNWISE.min_new_tokens,
                    help="blocks empty emissions only — a real floor would forbid backchannels (DV)")
    ap.add_argument("--stop-at-sentence", dest="stop_at_sentence", action="store_true",
                    default=TURNWISE.stop_at_sentence,
                    help="OFF by default: forcing sentence ends forbids abandoned turns (DV)")
    ap.add_argument("--no-stop-at-sentence", dest="stop_at_sentence", action="store_false")
    ap.add_argument("--repetition-penalty", type=float, default=TURNWISE.repetition_penalty,
                    help="1.0 = off; loops are handled procedurally (near-duplicate resample)")
    ap.add_argument("--no-repeat-ngram", type=int, default=TURNWISE.no_repeat_ngram_size)
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="output root; use data/generated_v3 for the regeneration run")
    args = ap.parse_args()

    cond = f"C3-{args.prompt}"
    out_dir = pathlib.Path(args.out_root) / cond
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata()
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        ids = load_target_ids()
        if args.n:
            ids = ids[:args.n]
    ids = [c for c in ids if c in meta]
    todo = [c for c in ids if not (out_dir / f"{c}.json").exists()]
    print(f"[{cond}] target={len(ids)} todo={len(todo)} done={len(ids) - len(todo)}")
    if not todo:
        return

    decoding = {
        "max_new_tokens": args.max_new_tokens, "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
        "stop_at_sentence": args.stop_at_sentence, "max_turns": args.max_turns,
    }

    model, tok = load_model(VICUNA)
    for cno in todo:
        a, b, topic, sb_prompt = make_personas(meta[cno])
        assign_cards(a, b, cno)   # seeded persona cards; only the P1/P2 prompts render them
        personas = {a.label: a, b.label: b}
        # Peer-greeting seed (matches the paper's GPT4-1 setup). These two turns are
        # scripted, not generated — `seed_turns` lets Track 2 exclude them from tagging.
        history: list[tuple[str, str]] = [("ParticipantA", "Hello!"), ("ParticipantB", "Hello!")]
        counters: dict[str, int] = {}
        closing_seen = False
        for i in range(args.max_turns):
            spk = LABELS[i % 2]
            me = personas[spk]
            partner = personas[LABELS[(i + 1) % 2]]
            messages = build_agent(args.prompt, me, partner, topic, sb_prompt, history,
                                   max_turns=args.max_turns, conversation_no=cno)
            turn = generate_turn(
                model, tok, messages, history, LABELS,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                top_p=args.top_p, min_new_tokens=args.min_new_tokens,
                stop_at_sentence=args.stop_at_sentence,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram,
                dup_min_words=DUP_MIN_WORDS, dup_jaccard=DUP_JACCARD,
                resample_temp_bump=RESAMPLE_TEMP_BUMP, counters=counters,
            )
            if not turn:                      # model left the conversation (even after retry)
                ended_by = "empty_turn"
                break
            prev_turn = history[-1][1]
            history.append((spk, turn))
            # Natural termination: stop at a mutual goodbye / one reciprocal after a farewell
            # (max_turns is only a cap; the prompt's turn-status line helps the model land it).
            if looks_like_closing(turn) and looks_like_closing(prev_turn):
                ended_by = "mutual_closing"
                break
            if closing_seen:
                ended_by = "closing"
                break
            if looks_like_closing(turn):
                closing_seen = True
        else:
            ended_by = "turn_cap"

        rec = {
            "condition": cond, "architecture": "C3", "prompt_level": args.prompt,
            "models": {a.label: VICUNA, b.label: VICUNA},
            "conversation_no": cno, "topic": topic,
            "sb_prompt": sb_prompt,
            "persona_a": dataclasses.asdict(a), "persona_b": dataclasses.asdict(b),
            "turns": history, "n_turns": len(history),
            "seed_turns": 2,
            "ended_by": ended_by,             # mutual_closing / closing / empty_turn / turn_cap
            "multi_turn_emissions": counters.get("multi_turn_emissions", 0),
            "decoding": decoding,
            "quality_counters": counters,     # also: cap hits, empty retries, dup resamples
        }
        (out_dir / f"{cno}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  saved {cno}  turns={len(history)}  counters={counters}")


if __name__ == "__main__":
    main()
