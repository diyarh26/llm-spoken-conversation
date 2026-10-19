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
from generation.model_utils import VICUNA, chat, load_model
from generation.runner import (
    add_turnwise_args,
    base_parser,
    prepare_job,
    run_turnwise_job,
)
from prompts.templates import writer_view


def main(argv: list[str] | None = None) -> None:
    ap = base_parser(__doc__)
    add_turnwise_args(ap)
    args = ap.parse_args(argv)

    job = prepare_job("C2", args)
    if not job.todo:
        return

    model, tok = load_model(VICUNA)
    engine = functools.partial(chat, model, tok)
    run_turnwise_job(
        job,
        args,
        engines=dict.fromkeys(LABELS, engine),
        build_messages=writer_view(args.prompt, args.max_turns),
        model_info={"model": VICUNA},
    )


if __name__ == "__main__":
    main()
