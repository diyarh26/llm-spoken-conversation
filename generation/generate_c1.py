"""
C1 — all at once. One model writes the entire conversation in a single generation.

Vicuna-13B receives one prompt containing both personas, the topic and the scripted
"Hello!" greetings, and writes the full call as labelled lines. No turn is ever generated
in isolation, so the model always "knows" where the conversation is going.

Decoding: the frozen C1 settings in generation/config.py. Repetition controls stay off:
every line must repeat a speaker label, and a repetition penalty punishes exactly that
repetition, which ends the log after one or two turns.

    python generation/generate_c1.py --prompt P1            # all 50 manifest ids
    python generation/generate_c1.py --prompt P1 --n 5      # a 5-conversation pilot
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from generation.config import C1 as C1_DECODING
from generation.config import VICUNA
from generation.model_utils import chat, load_model
from generation.runner import base_parser, prepare_job
from prompts.templates import build_c1


def main(argv: list[str] | None = None) -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--max-new-tokens", type=int, default=C1_DECODING.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=C1_DECODING.temperature)
    ap.add_argument("--top-p", type=float, default=C1_DECODING.top_p)
    ap.add_argument("--repetition-penalty", type=float, default=C1_DECODING.repetition_penalty)
    ap.add_argument("--no-repeat-ngram", type=int, default=C1_DECODING.no_repeat_ngram_size)
    args = ap.parse_args(argv)

    job = prepare_job("C1", args)
    if not job.todo:
        return
    decoding = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
    }

    model, tok = load_model(VICUNA)
    for cno in job.todo:
        a, b, topic, sb_prompt = job.speakers(cno)
        messages = build_c1(args.prompt, a, b, topic, sb_prompt, conversation_no=cno)
        text, info = chat(model, tok, messages, **decoding)
        job.save(
            cno,
            {
                "condition": job.condition,
                "architecture": "C1",
                "prompt_level": args.prompt,
                "model": VICUNA,
                "conversation_no": cno,
                "topic": topic,
                "sb_prompt": sb_prompt,
                "persona_a": dataclasses.asdict(a),
                "persona_b": dataclasses.asdict(b),
                "raw_output": text,
                "seed_turns": 2,  # the prompt scripts the two "Hello!" openers
                "hit_token_cap": info["hit_token_cap"],
                "n_new_tokens": info["n_new_tokens"],
                "decoding": decoding,
            },
        )
        print(f"  saved {cno}  ({len(text.split())} words, cap_hit={info['hit_token_cap']})")


if __name__ == "__main__":
    main()
