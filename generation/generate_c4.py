"""
C4 — two independent agents, different models. Vicuna-13B talks to Mistral-7B.

The same first-person agent setup as C3, except that the two speakers are different
models: ParticipantA is Vicuna-13B and ParticipantB is Mistral-7B-Instruct. Speakers
therefore differ in model identity as well as in what they can see.

    python generation/generate_c4.py --prompt P1
    python generation/generate_c4.py --prompt P1 --device-a cuda:0 --device-b cuda:1
"""

from __future__ import annotations

import functools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from generation.cleaning import LABELS
from generation.config import MISTRAL, VICUNA
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
    ap.add_argument("--model-a", default=VICUNA)
    ap.add_argument("--model-b", default=MISTRAL)
    # Pin each model to its own GPU when one card cannot hold both. Placement never
    # is recorded. Leave unset on a single large GPU (device_map="auto").
    ap.add_argument("--device-a", default=None, help="e.g. cuda:0 (pin model A to one GPU)")
    ap.add_argument("--device-b", default=None, help="e.g. cuda:1 (pin model B to one GPU)")
    add_turnwise_args(ap)
    args = ap.parse_args(argv)

    if args.model_a == args.model_b:
        ap.error("C4 requires two different model checkpoints; use C3 for same-model agents.")
    job = prepare_job("C4", args)
    if args.preflight_only:
        print("Corpus/ID preflight passed; no weights loaded or GPU generation attempted.")
        return

    resolved = bind_run(
        job, args, {LABELS[0]: args.model_a, LABELS[1]: args.model_b}, turnwise_decoding(args)
    )
    if not job.todo:
        return
    from generation.model_utils import chat, load_model

    print(
        f"loading {args.model_a} for {LABELS[0]}"
        + (f" on {args.device_a}" if args.device_a else "")
    )
    model_a, tok_a = load_model(
        args.model_a,
        device=args.device_a,
        revision=resolved[LABELS[0]]["revision"],
        precision=args.precision,
    )
    print(
        f"loading {args.model_b} for {LABELS[1]}"
        + (f" on {args.device_b}" if args.device_b else "")
    )
    model_b, tok_b = load_model(
        args.model_b,
        device=args.device_b,
        revision=resolved[LABELS[1]]["revision"],
        precision=args.precision,
    )

    run_turnwise_job(
        job,
        args,
        engines={
            LABELS[0]: functools.partial(chat, model_a, tok_a),
            LABELS[1]: functools.partial(chat, model_b, tok_b),
        },
        build_messages=agent_view(args.prompt, args.max_turns),
        model_info={"models": {LABELS[0]: args.model_a, LABELS[1]: args.model_b}},
    )


if __name__ == "__main__":
    main()
