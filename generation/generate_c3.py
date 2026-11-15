"""
C3 — two independent agents, same model. Each speaker is its own chat session.

Both participants are Vicuna-13B, but every call is built from one speaker's point of
view: its own earlier turns are `assistant` messages and the partner's are `user`
messages. Neither agent sees the other's persona card, as with real strangers on a call,
so any convergence has to happen in the talk itself.

    python generation/generate_c3.py --prompt P1
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
from prompts.templates import agent_view


def main(argv: list[str] | None = None) -> None:
    ap = base_parser(__doc__)
    ap.add_argument("--model", default=VICUNA)
    add_turnwise_args(ap)
    args = ap.parse_args(argv)

    job = prepare_job("C3", args)
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
        build_messages=agent_view(args.prompt, args.max_turns),
        model_info={"models": dict.fromkeys(LABELS, args.model)},
    )


if __name__ == "__main__":
    main()
