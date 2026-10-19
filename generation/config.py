"""
Models and decoding settings: the single source of truth for every generation run.

Design rule: decoding must never bound a quantity the study measures. The headline metric
counts backchannels, abandoned turns and turn length, so the settings look permissive on
purpose:

- No real token floor. ``min_new_tokens=2`` only blocks empty replies; a floor of 16 tokens
  (used in an early draft) made a backchannel like "Uh-huh." impossible by construction.
- No sentence-boundary stopping. It would make abandoned, trailing-off turns impossible.
- No logit-level anti-repetition (``repetition_penalty=1.0``, no n-gram ban). Both act over
  the whole context, transcript included, so they suppress conversation-frequent tokens:
  exactly the repeated short acknowledgments being measured. Verbatim loops are handled
  procedurally instead: a near-duplicate turn is resampled once at a higher temperature
  (``generation.quality`` / ``generation.turnwise``). C1 needs no guard, since writing
  the whole conversation at once has no two-agent echo spiral.
- Generous caps, and every cap hit is recorded in the output files.

Sampling (temperature / top-p) is shared by all architectures, so the architecture contrast
is not confounded by sampling choices. The values were frozen after a decoding sweep on five
held-out development conversations (``generation/degeneration_score.py``), before the run.
"""

from __future__ import annotations

from dataclasses import dataclass

VICUNA = "lmsys/vicuna-13b-v1.5-16k"  # all architectures; speaker A in C4
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.2"  # speaker B in C4


@dataclass(frozen=True)
class DecodingConfig:
    max_new_tokens: int
    min_new_tokens: int = 2  # blocks empty replies only; NOT a turn-length floor
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.0  # off
    no_repeat_ngram_size: int = 0  # off
    stop_at_sentence: bool = False


# C1 writes the ENTIRE conversation in one call; 4096 tokens fits ~40 long turns in the 16k
# context, so the cap never binds (0 hits in all 150 C1 conversations).
C1 = DecodingConfig(max_new_tokens=4096)

# C2/C3/C4 generate one turn per call; the cap is a safety net. It was raised from 300 to 512
# tokens before the C3/C4 runs, after Mistral wrote genuine 250-word turns in a pilot. C2 had
# already been generated with 300 (one cap hit in 5,596 turns), so its records keep 300.
TURNWISE = DecodingConfig(max_new_tokens=512)

# Hard cap on turns for the turn-by-turn conditions. Every prompt tells the model how many
# turns have passed, so conversations can close before it; `ended_by` records the reason.
MAX_TURNS = 40

# Procedural loop guard (replaces logit penalties): a turn of >= DUP_MIN_WORDS words whose
# token-set Jaccard similarity with any earlier turn is >= DUP_JACCARD triggers ONE resample
# at temperature + RESAMPLE_TEMP_BUMP. Short turns are exempt on purpose: backchannels
# legitimately repeat, and guarding them would re-suppress the phenomenon being measured.
DUP_MIN_WORDS = 8
DUP_JACCARD = 0.8
RESAMPLE_TEMP_BUMP = 0.15
