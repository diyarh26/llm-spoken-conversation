"""Detect reasoning leakage without silently turning it into spoken dialogue."""

from __future__ import annotations

import re


def spoken_response(raw: str, decoded: str) -> tuple[str, bool]:
    # Empty Qwen thinking wrappers are harmless; nonempty reasoning is a failed pilot.
    raw_without_empty = re.sub(r"<think>\s*</think>", "", raw)
    leak = bool(re.search(r"<think>|</think>|<\|channel>thought", raw_without_empty))
    text = re.sub(r"<think>\s*</think>", "", decoded).strip()
    if "<|channel>final" in raw:
        text = re.sub(r"^final\s*\n", "", text).strip()
    return text, leak
