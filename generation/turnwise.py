"""
The turn-by-turn conversation loop shared by C2, C3 and C4 (torch-free).

The three turn-wise architectures differ in only two things: how the prompt for the next
turn is built (one model writing both speakers vs. two first-person agents) and which
model answers for each speaker. Everything else is identical and lives here, so the
architecture contrast cannot be confounded by accidental differences between scripts:

- :func:`generate_turn` turns one model call into one clean turn, with the procedural
  quality guards (retry an empty output once, resample a near-verbatim loop once).
- :func:`run_conversation` alternates speakers from the scripted greeting until a natural
  closing, an empty turn, or the turn cap, and reports why the conversation ended.

Models are reached through a plain callable (``chat_fn(temperature) -> (text, info)``), so
this module never imports torch and is fully unit-tested with stub models.
"""

from __future__ import annotations

from collections.abc import Callable

from generation.cleaning import LABELS, clean_single_turn, looks_like_closing, strip_meta_artifacts
from generation.config import DUP_JACCARD, DUP_MIN_WORDS, RESAMPLE_TEMP_BUMP
from generation.quality import is_near_duplicate

Turn = tuple[str, str]  # (speaker label, utterance)
ChatFn = Callable[[float], tuple[str, dict]]  # temperature -> (raw text, {"hit_token_cap": bool, ...})
Speaker = Callable[[str, list[Turn]], str]  # (speaker label, history) -> next turn ("" = left)

# Every turn-wise conversation opens with a scripted mutual greeting, as in the original
# study. These turns are not generated; `seed_turns` in each output file lets the analysis
# drop them.
SEED_TURNS: tuple[Turn, ...] = (("ParticipantA", "Hello!"), ("ParticipantB", "Hello!"))

# Why a turn-wise conversation stopped (recorded as `ended_by`).
ENDINGS = ("mutual_closing", "closing", "empty_turn", "turn_cap")


def generate_turn(
    chat_fn: ChatFn,
    history: list[Turn],
    labels=LABELS,
    *,
    temperature: float,
    dup_min_words: int = DUP_MIN_WORDS,
    dup_jaccard: float = DUP_JACCARD,
    resample_temp_bump: float = RESAMPLE_TEMP_BUMP,
    counters: dict[str, int] | None = None,
) -> str:
    """Produce one clean conversational turn, with the procedural quality guards.

    Pipeline: generate -> truncate to one turn -> strip chatbot residue, then
      - empty result: ONE fresh retry (models may emit an instant end-of-sequence now
        that there is no token floor); still empty = the model left the conversation,
        return "".
      - near-duplicate of an earlier turn (a loop): ONE resample at temperature + bump.
        This replaces logit-level repetition penalties, which would also suppress the
        repeated short acknowledgments the study measures.

    ``counters`` accumulates, per conversation: hit_token_cap, multi_turn_emissions,
    empty_retries, dup_resamples, dup_kept. They are written to every output file.
    """
    counters = counters if counters is not None else {}

    def _once(temp: float) -> str:
        raw, info = chat_fn(temp)
        counters["hit_token_cap"] = counters.get("hit_token_cap", 0) + int(info["hit_token_cap"])
        turn, ran_past = clean_single_turn(raw, labels)
        counters["multi_turn_emissions"] = counters.get("multi_turn_emissions", 0) + int(ran_past)
        return strip_meta_artifacts(turn)

    turn = _once(temperature)
    if not turn:
        counters["empty_retries"] = counters.get("empty_retries", 0) + 1
        turn = _once(temperature)
        if not turn:
            return ""
    if is_near_duplicate(turn, history, dup_min_words, dup_jaccard):
        counters["dup_resamples"] = counters.get("dup_resamples", 0) + 1
        retry = _once(temperature + resample_temp_bump)
        if retry and not is_near_duplicate(retry, history, dup_min_words, dup_jaccard):
            turn = retry
        else:
            counters["dup_kept"] = counters.get("dup_kept", 0) + 1
            turn = retry or turn
    return turn


def run_conversation(speak: Speaker, max_turns: int, labels=LABELS) -> tuple[list[Turn], str]:
    """Alternate speakers from the scripted greeting; return (turns, ended_by).

    ``max_turns`` is only a safety cap. Conversations normally end on their own: at a
    mutual goodbye, or one reciprocal turn after a farewell. Every turn-wise prompt also
    tells the model how many turns have passed (see ``prompts.templates``), so it can
    close before the cap.
    """
    history = list(SEED_TURNS)
    closing_seen = False
    for i in range(max_turns):
        speaker = labels[i % 2]
        turn = speak(speaker, history)
        if not turn:  # the model left the conversation, even after a retry
            return history, "empty_turn"
        previous = history[-1][1]
        history.append((speaker, turn))
        if looks_like_closing(turn) and looks_like_closing(previous):
            return history, "mutual_closing"
        if closing_seen:
            return history, "closing"
        if looks_like_closing(turn):
            closing_seen = True
    return history, "turn_cap"
