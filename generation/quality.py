"""
Torch-free turn-quality primitives, shared by the runtime loop guard
(generation/model_utils.generate_turn) and the post-hoc dev-sweep scorer
(generation/degeneration_score.py). One definition of "near-duplicate" everywhere.
"""

from __future__ import annotations


def token_jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two utterances (case-folded)."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0.0


def is_near_duplicate(turn: str, history: list[tuple[str, str]],
                      min_words: int = 8, threshold: float = 0.8) -> bool:
    """Is `turn` a near-verbatim repeat of an earlier turn (a degeneration loop)?

    Token-set Jaccard >= threshold against any prior turn of comparable length. Turns
    shorter than min_words are NEVER flagged — backchannels ("Uh-huh." / "Yeah.")
    legitimately repeat many times in human talk, and flagging them would procedurally
    suppress the very acts the dialogue-act metric measures.
    """
    if len(set(turn.lower().split())) < min_words:
        return False
    for _, prev in history:
        if len(set(prev.lower().split())) < min_words:
            continue
        if token_jaccard(turn, prev) >= threshold:
            return True
    return False
