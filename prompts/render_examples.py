"""Render prompts/PROMPT_EXAMPLES.md: every prompt the models received, for one conversation.

    python prompts/render_examples.py

The prompts come from the real templates, using the personas and topic recorded in the
released data, so the document always matches the code. The two Switchboard excerpts that
P2 shows are replaced by placeholders (the corpus is LDC-licensed); the draw of which
excerpts a conversation gets is reproduced exactly from the committed pool recipe.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import swda
from prompts.templates import Persona, build_agent, build_c1, build_c2

CONVERSATION = 2121
OUT = pathlib.Path(__file__).resolve().parent / "PROMPT_EXAMPLES.md"
SEED = [("ParticipantA", "Hello!"), ("ParticipantB", "Hello!")]
LEVELS = {
    "P0": "the original study's basic prompt (replication anchor)",
    "P1": "spoken style + persona cards + peer framing; never names a measured behaviour",
    "P2": "P1 + two short real Switchboard excerpts on other topics",
}


def placeholder_examples(conversation_no: int, k: int = 2) -> list[dict]:
    """The same seeded draw as analysis.swda.fewshot_examples, without the licensed text."""
    recipe = json.loads((ROOT / "generation" / "fewshot_pool.json").read_text(encoding="utf-8"))
    picks = random.Random(f"fewshot:v3:{conversation_no}").sample(recipe["excerpts"], k)
    return [
        {
            "topic": p["topic"],
            "text": f"[{p['window']} turns of real Switchboard call {p['conversation_no']}, "
            "not reproduced here: the corpus is LDC-licensed]",
        }
        for p in picks
    ]


def block(messages: list[dict]) -> str:
    return "\n\n".join(f"**[{m['role']}]**\n\n```text\n{m['content']}\n```" for m in messages)


def main() -> None:
    swda.fewshot_examples = placeholder_examples
    rec = json.loads(
        (ROOT / "data" / "generated" / "C1-P1" / f"{CONVERSATION}.json").read_text(encoding="utf-8")
    )
    a, b = Persona(**rec["persona_a"]), Persona(**rec["persona_b"])
    topic, sb_prompt = rec["topic"], rec["sb_prompt"]

    parts = [
        "# Prompt examples",
        "",
        f"Every prompt the models received, rendered by `prompts/render_examples.py` from "
        f"`prompts/templates.py` for conversation **#{CONVERSATION} ({topic})** with the persona "
        "cards drawn for it. For the turn-by-turn architectures (C2, C3, C4) this is the prompt "
        'for the first generated turn after the two scripted "Hello!" turns; later turns get the '
        "same prompt with a longer transcript and an updated turn count.",
        "",
        " · ".join(f"**{k}** = {v}" for k, v in LEVELS.items()),
    ]
    for level in LEVELS:
        parts += ["", "---", "", f"## {level}", ""]
        parts += [
            "### C1 · all at once (one call writes the whole conversation)",
            "",
            block(build_c1(level, a, b, topic, sb_prompt, conversation_no=CONVERSATION)),
            "",
            "### C2 · turn by turn (one model writes the next turn and sees both speakers)",
            "",
            block(build_c2(level, a, b, topic, sb_prompt, SEED, "ParticipantA", 40, CONVERSATION)),
            "",
            "### C3 / C4 · agent view (speaker A's own session; in C4 speaker B is Mistral, "
            "with the mirror-image prompt)",
            "",
            block(build_agent(level, a, b, topic, sb_prompt, SEED, 40, CONVERSATION)),
        ]
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
