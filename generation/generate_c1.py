"""
C1 (all-at-once) generation with Vicuna-13B.

One model writes the entire dialogue in a single call. Targets come from the committed
stratified manifest (generation/target_ids.json), decoding from generation/config.py
(DV-safe; see GENERATION_SPEC.md). Each conversation is written to disk IMMEDIATELY as
<out-root>/C1-<prompt>/<conversation_no>.json and already-present ids are skipped, so the
job is resumable after an SSH/kernel drop. Run inside tmux on the VM:

    python generation/generate_c1.py --prompt P0 --out-root data/generated_v3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from prompts.templates import build_c1, assign_cards                      # noqa: E402
from analysis.swda import load_metadata, make_personas                    # noqa: E402
from generation.sampling import load_target_ids                           # noqa: E402
from generation.config import C1 as C1_DECODING                           # noqa: E402
from generation.model_utils import load_model, chat, VICUNA               # noqa: E402

OUT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "generated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="P0", choices=["P0", "P1", "P2"])
    ap.add_argument("--n", type=int, default=0,
                    help="generate only the first N manifest ids (0 = all; prefix is ~stratified)")
    ap.add_argument("--ids", default="",
                    help="comma-separated conversation_no's to (re)generate; overrides the manifest")
    # C1 writes the WHOLE conversation in one generate() call. Its known-good config keeps
    # repetition_penalty/no_repeat_ngram OFF: they punish the speaker-label repetition every
    # line needs and truncate the log (see git history of this file for the full diagnosis).
    # 4096 tokens (was 2048) so the cap can no longer bound conversation length — a P0 run
    # of 40 long turns needs ~3k tokens; hitting the cap is recorded as hit_token_cap.
    ap.add_argument("--max-new-tokens", type=int, default=C1_DECODING.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=C1_DECODING.temperature)
    ap.add_argument("--top-p", type=float, default=C1_DECODING.top_p)
    ap.add_argument("--repetition-penalty", type=float, default=C1_DECODING.repetition_penalty)
    ap.add_argument("--no-repeat-ngram", type=int, default=C1_DECODING.no_repeat_ngram_size)
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="output root; use data/generated_v3 for the regeneration run")
    args = ap.parse_args()

    cond = f"C1-{args.prompt}"
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
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
    }

    model, tok = load_model(VICUNA)
    for cno in todo:
        a, b, topic, sb_prompt = make_personas(meta[cno])
        assign_cards(a, b, cno)   # seeded persona cards; only the P1/P2 prompts render them
        messages = build_c1(args.prompt, a, b, topic, sb_prompt, conversation_no=cno)
        text, info = chat(
            model, tok, messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram,
        )
        rec = {
            "condition": cond,
            "architecture": "C1",
            "prompt_level": args.prompt,
            "model": VICUNA,
            "conversation_no": cno,
            "topic": topic,
            "sb_prompt": sb_prompt,
            "persona_a": dataclasses.asdict(a),
            "persona_b": dataclasses.asdict(b),
            "raw_output": text,
            "seed_turns": 2,                  # the prompt scripts the two Hello! openers
            "hit_token_cap": info["hit_token_cap"],
            "n_new_tokens": info["n_new_tokens"],
            "decoding": decoding,
        }
        (out_dir / f"{cno}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"  saved {cno}  ({len(text.split())} words, cap_hit={info['hit_token_cap']})")


if __name__ == "__main__":
    main()
