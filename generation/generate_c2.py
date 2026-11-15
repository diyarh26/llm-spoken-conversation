"""
C2 — one model, turn by turn. A single model plays both speakers, one turn per call.

Vicuna-13B sees the whole labelled transcript and writes the next turn for the named
speaker, alternating A and B. This replicates the original study's turn-by-turn setup.
That study found Vicuna could not keep to a single turn under this prompting, so every
generation is truncated to its first turn and `multi_turn_emissions` records how often
the model ran past it.

    python generation/generate_c2.py --prompt P1
"""

from __future__ import annotations

import functools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from generation.cleaning import LABELS
from generation.config import VICUNA
from generation.run_context import bind_run
from generation.runner import (
    add_turnwise_args,
    base_parser,
    prepare_job,
    run_turnwise_job,
    turnwise_decoding,
)
from prompts.templates import writer_view


def main(argv: list[str] | None = None) -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--model", default=VICUNA)
    add_turnwise_args(ap)
    args = ap.parse_args(argv)

    job = prepare_job("C2", args)
    if args.preflight_only:
        print("Corpus/ID preflight passed; no weights loaded or GPU generation attempted.")
        return

    resolved = bind_run(job, args, dict.fromkeys(LABELS, args.model), turnwise_decoding(args))
    if not job.todo:
        return
    from generation.model_utils import chat, load_model

    model, tok = load_model(
        args.model,
        device=args.device,
        revision=resolved[LABELS[0]]["revision"],
        precision=args.precision,
    )
    engine = functools.partial(chat, model, tok)
    run_turnwise_job(
        job,
        args,
        engines=dict.fromkeys(LABELS, engine),
        build_messages=writer_view(args.prompt, args.max_turns),
        model_info={"model": args.model},
    )


if __name__ == "__main__":
    main()
