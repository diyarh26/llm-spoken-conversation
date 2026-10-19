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

from generation.cleaning import LABELS  # noqa: E402
from generation.model_utils import VICUNA, chat, load_model  # noqa: E402
from generation.runner import (  # noqa: E402
    add_turnwise_args,
    base_parser,
    prepare_job,
    run_turnwise_job,
)
from prompts.templates import agent_view  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = base_parser(__doc__)
    add_turnwise_args(ap)
    args = ap.parse_args(argv)

    job = prepare_job("C3", args)
    if not job.todo:
        return

    model, tok = load_model(VICUNA)
    engine = functools.partial(chat, model, tok)
    run_turnwise_job(
        job,
        args,
        engines=dict.fromkeys(LABELS, engine),
        build_messages=agent_view(args.prompt, args.max_turns),
        model_info={"models": dict.fromkeys(LABELS, VICUNA)},
    )


if __name__ == "__main__":
    main()
