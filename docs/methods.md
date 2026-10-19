# Methods

This document describes how the 600 conversations were generated and measured, and every
design decision that could affect the results. The short version is in the
[README](../README.md); the full results are in [`results/RESULTS_REPORT.md`](../results/RESULTS_REPORT.md).

- [1. Design](#1-design)
- [2. Sampling](#2-sampling)
- [3. Prompts](#3-prompts)
- [4. Generation](#4-generation)
- [5. Decoding: never bound what you measure](#5-decoding-never-bound-what-you-measure)
- [6. Turn cleaning](#6-turn-cleaning)
- [7. Measures](#7-measures)
- [8. Statistics](#8-statistics)
- [9. Validation](#9-validation)
- [10. Declared caveats](#10-declared-caveats)
- [11. Reproducing](#11-reproducing)

---

## 1. Design

We extend Mayor, Bietti & Bangerter (2025), who compared LLM-written phone calls with the
Switchboard corpus but varied only the prompt and the model. Here the **generation
architecture** is the main factor:

| | Architecture | What a model sees when it writes a turn |
|---|---|---|
| **C1** | All at once | Nothing turn-by-turn: one call writes the entire conversation. |
| **C2** | One model, turn by turn | Both personas and the full labelled transcript; it writes the next turn for the named speaker. |
| **C3** | Two agents, same model | Only its own side: its persona card, its own turns as `assistant` messages, the partner's as `user` messages. |
| **C4** | Two agents, two models | As C3, with speaker A on Vicuna-13B and speaker B on Mistral-7B-Instruct. |

crossed with three prompt levels (P0 paper prompt, P1 spoken style + persona cards, P2 = P1 +
two real Switchboard excerpts) for **12 conditions × 50 conversations = 600**.

The design is **paired**: all 12 conditions generate the same 50 conversations, each with the
topic and caller demographics of a real Switchboard call and the same seeded persona cards.
Differences between conditions are therefore not differences in topic or speakers.

**Models:** `lmsys/vicuna-13b-v1.5-16k` (all architectures; speaker A in C4) and
`mistralai/Mistral-7B-Instruct-v0.2` (speaker B in C4), both 4-bit NF4 with double
quantization (`generation/model_utils.py`). Vicuna is deliberate: the original study found
stronger models to be *further* from human talk, so claims are scoped to these models.

## 2. Sampling

The 50 conversations are a seeded, topic-stratified random sample of the 1,155 Switchboard
Dialog Act (SwDA) conversations (`generation/sampling.py`, seed `20260714`):

1. topic-proportional allocation over SwDA topics with largest-remainder rounding (48 topics);
2. a seeded draw within each topic, then a seeded shuffle, so any prefix of the list is
   itself roughly stratified (`--n 5` gives a valid pilot);
3. five disjoint **development** conversations (`3325, 3003, 3595, 4333, 3657`) for tuning
   decoding. They are never generation targets and never prompt examples.

The draw is committed as a manifest (`generation/target_ids.json`); generators read it and
never re-sample. The same 50 Switchboard calls are the topic-matched human reference for
turn length, markers and alignment. The dialogue-act reference uses all 1,155 SwDA calls,
because a larger reference only makes it more precise (see §7).

## 3. Prompts

All prompts are built by [`prompts/templates.py`](../prompts/templates.py); a rendered example of
every prompt is in [`prompts/PROMPT_EXAMPLES.md`](../prompts/PROMPT_EXAMPLES.md).

| Level | Content |
|---|---|
| **P0** | The original study's basic prompt, kept as a replication anchor: bare demographics ("a 54-year-old man with some college"), the verbatim Switchboard instruction, "about 30 turns". |
| **P1** | Our prompt: spoken register and brevity, a natural ending, a one-sentence peer framing ("an ordinary person, not an assistant"), and a seeded **persona card** per speaker (occupation, stance toward the topic, one life anchor). |
| **P2** | P1 plus two 12-turn excerpts from real Switchboard calls on *other* topics, drawn deterministically per conversation from a pool of 10 listener-rich excerpts (`generation/fewshot_pool.json`, which stores only ids and offsets, never text). |
| P3 | Exploratory and outside the main design: shows listener reactions by example. Reported separately and never as emergent behaviour. |

**The non-circularity rule.** The original study's *enhanced* prompts asked for "okay", "oh"
and "uh-huh" and then measured exactly those words. No prompt in the main design (P0–P2)
names or quotes a listener behaviour or coordination marker; this is enforced by
[`tests/test_prompts.py`](../tests/test_prompts.py). Persona cards describe attitude and
content only, with no interactional verbs, so disagreement or storytelling can only *emerge*
from the combination of stances (two-thirds of pairs hold different stances).

**Turn countdown.** Every turn-by-turn prompt states how many turns have passed and that the
call cannot exceed 40, so models can end the conversation themselves rather than be cut off.
It says nothing about *how* to end.

## 4. Generation

One resumable script per architecture (`generation/generate_c1.py` … `generate_c4.py`). They
share one turn-by-turn core, so the architecture contrast cannot be confounded by accidental
code differences:

- [`generation/turnwise.py`](../generation/turnwise.py): the conversation loop and the per-turn
  quality guards (torch-free, unit-tested with stub models);
- [`generation/cleaning.py`](../generation/cleaning.py): reducing raw output to exactly one
  spoken turn;
- [`generation/runner.py`](../generation/runner.py): command line, target selection, output
  records;
- [`generation/model_utils.py`](../generation/model_utils.py): model loading and `chat()`.

Every turn-by-turn conversation starts from a scripted mutual "Hello!" (as in the original
study; recorded as `seed_turns: 2` and excluded from analysis) and ends at a mutual goodbye,
one reciprocal turn after a farewell, an empty turn, or the 40-turn cap (recorded as
`ended_by`). Each conversation is written to disk as soon as it finishes, and finished ids are
skipped, so a crash loses at most the conversation in flight.

> The generator scripts were refactored after the run for readability (one shared core
> instead of three copies). The refactor was verified to be behaviour-preserving: original and
> refactored code, driven by the same deterministic stub model through 2,198 calls covering
> every code path, produced byte-identical output files. The code exactly as it ran is in the
> git history (tag `pre-cleanup`).

## 5. Decoding: never bound what you measure

Decoding settings live in one place, [`generation/config.py`](../generation/config.py). The
governing rule: **decoding must never make a measured behaviour impossible.** Auditing an
early draft of the pipeline showed that common anti-degeneration settings did exactly that:

| Early setting | Effect on what we measure | Final choice |
|---|---|---|
| `min_new_tokens = 16` | A backchannel ("Uh-huh.", ~3 tokens) cannot be generated | 2 (blocks empty output only) |
| Stop at sentence end | Abandoned, trailing-off turns cannot occur | off |
| `repetition_penalty`, n-gram ban | Act on the whole context, so they suppress the repeated short acknowledgments we count | off |
| Small caps (200 tokens/turn, 2,048 for C1) | Truncate long turns and conversations | generous caps, every hit logged |
| Turn caps of 30 vs 50 by architecture | Inconsistent bound on conversation length | 40 for all, plus the countdown |

Verbatim loops, the problem penalties are meant to fix, are handled **procedurally** instead:
a turn of ≥ 8 words whose token-set Jaccard similarity with any earlier turn is ≥ 0.8 is
resampled once at temperature + 0.15. Short turns are exempt on purpose, because backchannels
legitimately repeat. An empty turn gets one retry. Temperature (0.8) and top-p (0.95) are shared
by all architectures. The values were frozen after a sweep on the five development
conversations, scored with [`generation/degeneration_score.py`](../generation/degeneration_score.py).

**Caps in practice.** C1 (4,096 tokens per call) never hit its cap. C2 was generated with a
300-token turn cap and hit it once in 5,596 turns. The cap was raised to 512 before the C3/C4
runs, after Mistral wrote genuine 250-word turns in a pilot; C3/C4 hit it in 186 of 4,948 turns
(3.8%). A cap can only shorten turns, so the long-turn finding for agents is conservative.

## 6. Turn cleaning

The same cleaning applies to every condition and to analysis only (generation is unchanged),
via `conversation_turns()` in [`analysis/analyze.py`](../analysis/analyze.py):

- the scripted "Hello!" seed turns are dropped (C1's own model-written openings are kept);
- bracketed stage directions that echo the turn countdown ("(Turn 14)", "[Your turn to
  respond…]") are removed; ordinary parentheses such as "401(k)" are kept;
- **language drift** (Vicuna sometimes appends "번역결과", "translation result", and continues
  in Korean; 10 C3/C4 conversations) is cut where the non-Latin script starts.

Degeneration is **measured, not patched**. Echo loops (16% of turns in C2-P0) and language drift
are reported per condition, and every dialogue-act distance is recomputed without them as a
sensitivity check (`results/dialogue_acts_no_loops/`, `results/dialogue_acts_no_drift/`).

## 7. Measures

**Turn length.** Words per turn: mean, median and IQR, per conversation and per condition.

**Coordination markers.** "oh", "okay", "uh-huh" per 100 words, as in the original study.
P2 marker rates are partly example-driven (its excerpts contain them) and are read as such.

**Backchannels.** An expanded listener lexicon (uh-huh, okay, oh, yeah, right, mm-hm, I see,
sure): overall rate, the share of turns that consist *only* of listener tokens (standalone),
and lexical diversity.

**Dialogue acts (the main extension).** Every unit of talk is labelled with what it *does*
(statement, opinion, backchannel, question, agreement…), using the SwDA DAMSL scheme:

- generated turns are split into sentence units; human units are SwDA's own utterance units;
- labels are mapped to one shared 39-code inventory and 10 coarse classes
  ([`analysis/dialogue_acts.py`](../analysis/dialogue_acts.py));
- **two rulers, one comparison.** The human side has expert gold labels, but comparing gold
  human labels with tagger-labelled LLM text would mix two measurement instruments. The
  primary comparison therefore tags *both* sides with the same automatic tagger, DialogTag
  (DistilBERT, trained on SwDA). The tagger's own error on human speech (gold vs tagger:
  JSD 0.035) is reported as a calibration line;
- distance from humans is the **Jensen–Shannon divergence** between act distributions (and
  between act-transition matrices, weighted by act frequency);
- **noise floor:** 1,000 seeded draws of human units, size-matched to each condition, give
  the distance expected from sampling alone (95th percentile ≤ 0.002 on the coarse classes,
  against observed distances of 0.12–0.24);
- **conversation-level bootstrap:** 2,000 replicates resampling whole conversations, because
  units within a conversation are not independent.

A tagger-independent cross-check counts literal `?` questions and exact short reactions on
both sides; it shows the listener gap (19.1% of human units vs 0.0–0.4% in every LLM
condition) without relying on the tagger at all.

**Conceptual alignment.** Semantic similarity of adjacent turns with
[ALIGN](https://github.com/nickduran/align-linguistic-alignment) (Duran et al., 2019) and the
same pretrained word2vec (Google News, 300-d) vectors as the original study; 17,947 turn
pairs. We compare the first and second half of each call and control for turn length, because
averaged vectors of long turns are similar for reasons unrelated to alignment.

## 8. Statistics

- Words per turn: Kruskal–Wallis across architectures at each prompt level; mixed-effects
  models `words ~ corpus × section + (1 | conversation)` for each condition vs humans
  ([`analysis/stats.py`](../analysis/stats.py)).
- Markers: Mann–Whitney tests per condition vs humans.
- Alignment: Wilcoxon signed-rank tests of the within-conversation change (also with the last
  two turn pairs removed, to rule out closing pleasantries), and OLS with conversation-
  clustered standard errors controlling for log turn length
  ([`analysis/alignment_checks.py`](../analysis/alignment_checks.py)).
- Dialogue acts: noise floor and conversation-level bootstrap CIs, as above
  ([`analysis/jsd_bootstrap.py`](../analysis/jsd_bootstrap.py)).

## 9. Validation

- **Replication before generation.** The pipeline first reproduced the original study's
  human numbers: 14.4 words per turn (paper: 13.97), "uh-huh" 1.03 per 100 words (paper:
  1.03), and first-half conceptual alignment 0.56 (paper: ≈ 0.57).
- **The tagger, checked by hand on LLM text.** Four annotators labelled 100 random LLM units
  (20 shared by all, 80 split among them; [`annotation/`](../annotation/)). The tagger matched
  the human label on 77% (95% Wilson CI 67–84%), compared with 72% on human Switchboard speech;
  the annotators agreed with each other 83% of the time (Fleiss' κ = 0.67). Its main error
  under-counts LLM opinions, so the opinion gap is conservative
  ([`results/tagger_handcheck.txt`](../results/tagger_handcheck.txt)).
- **Robustness.** Removing echo-loop turns or the language-drift conversations changes every
  dialogue-act distance by ≤ 0.01.
- **Engineering checks.** 149 automated tests cover turn cleaning, the conversation loop, the
  generators (with stub models), the prompt rules, the integrity of all 600 released files,
  and the README's headline claims against the committed result tables.

## 10. Declared caveats

1. **Turn length is instructed in P1/P2** ("sometimes a sentence or two, sometimes just a word
   or two"). It is the one instructed dimension; no dialogue act is instructed.
2. **Endings are harness-assisted** in the turn-by-turn conditions (countdown and farewell
   detection), so closing behaviour is not compared as emergent.
3. **P1's peer framing** mentions sharing opinions and asking about the partner's. The excess of
   opinions and questions is already present under P0, which has no such framing.
4. **Wording:** the C1/C2 P1/P2 prompts frame the model as the *writer* but reuse clauses
   addressed to a *participant*. The text is identical across architectures within a prompt
   level, so it cannot drive the architecture contrast.
5. **Scope:** two open model families, 4-bit quantized; 50 conversations per condition;
   strictly alternating turns, unlike real overlapping speech; automatic tagging has
   measurable error (quantified in §9).

## 11. Reproducing

```bash
pip install -e ".[analysis,dev]"
make test       # 149 tests, no GPU or licensed data needed
make figures    # README figures from the committed results
make explorer   # data bundle for the conversation explorer
```

Recomputing metrics needs the licensed SwDA corpus under `data/switchboard/swda/`
(`make analysis`). Generation needs a CUDA GPU (`pip install -e ".[generation]"`, then
`make generate`); the exact package versions of the published run are in
[`environment/generation-lock.txt`](../environment/generation-lock.txt). The dialogue-act tagger
runs in its own environment ([`analysis/requirements-dialogue-acts.txt`](../analysis/requirements-dialogue-acts.txt)),
and ALIGN is installed separately, following the original authors' notebooks.
