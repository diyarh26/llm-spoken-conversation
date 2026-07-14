"""
Per-architecture decoding configuration — the single documented source of truth (§8 redo).

DV-SAFETY RULE (why these values look permissive): decoding must never bound a quantity we
measure. The dialogue-act headline metric measures backchannels, abandoned turns, and turn
length — so:
  - NO token floor (`min_new_tokens=2` only blocks empty emissions). The old floor of 16
    made backchannels ("Uh-huh.") impossible by construction.
  - NO sentence-boundary stopping. It made abandoned/trailing-off turns (`%` acts)
    impossible.
  - NO logit-level anti-repetition by default (repetition_penalty=1.0, ngram ban off).
    Both apply over the whole context incl. the transcript, so they systematically
    suppress conversation-frequent tokens — exactly the repeated short acknowledgments
    the metric counts. Loops are handled procedurally instead (near-duplicate turn →
    one resample at higher temperature; see model_utils.is_near_duplicate) — except C1,
    whose known-good config never looped (all-at-once has no two-agent echo spiral).
  - GENEROUS caps, logged when hit: per-turn 300 tokens, C1 whole-log 4096, 40 turns.

Sampling params (temperature/top_p) are SHARED across architectures so the architecture
contrast is not confounded by sampling choices.

FREEZE PROTOCOL: these are the pre-sweep defaults. The VM runs the dev-set sweep
(generation/degeneration_score.py over the 5 dev ids), amends ONLY what the score forces
(documented escalation: repetition_penalty 1.0 → 1.05), commits the change with the score
evidence in VM_REPORT.md, and then this file is FROZEN for the v3 run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodingConfig:
    max_new_tokens: int
    min_new_tokens: int = 2          # blocks empty emissions only — NOT a turn-length floor
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.0  # off; escalate to at most 1.05 if the dev sweep forces it
    no_repeat_ngram_size: int = 0    # off
    stop_at_sentence: bool = False


# C1 writes the ENTIRE conversation in one call; 4096 tokens fits ~40 long P0 turns in the
# 16k context, so the cap should never bind (log if it does).
C1 = DecodingConfig(max_new_tokens=4096)

# C2/C3/C4 generate one turn per call. 300 tokens ≈ 225 words — far above the longest
# observed turns (~80 words), so the cap is a safety net, not a bound.
TURNWISE = DecodingConfig(max_new_tokens=300)

# Uniform hard cap on turns for the turn-by-turn conditions (was inconsistent: C2=30,
# C3/C4=50). The prompt tells the model where it is in this budget (supervisor fix) so
# conversations close naturally before the cap; hitting it is logged as `hit_turn_cap`.
MAX_TURNS = 40

# Procedural loop guard (replaces logit penalties): a turn of >= DUP_MIN_WORDS words whose
# token-set Jaccard similarity with any earlier turn is >= DUP_JACCARD triggers ONE resample
# at temperature + RESAMPLE_TEMP_BUMP. Short turns are exempt on purpose — backchannels
# legitimately repeat, and guarding them would re-suppress the phenomenon we measure.
DUP_MIN_WORDS = 8
DUP_JACCARD = 0.8
RESAMPLE_TEMP_BUMP = 0.15
