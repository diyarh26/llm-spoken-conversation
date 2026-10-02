# Generation Spec — v3 regeneration (Track 1)

**Status:** design frozen locally 2026-07-14 (prompt/sampling/model/N). Decoding is at
*pre-sweep defaults*: the VM dev sweep may amend it once, with evidence, then it freezes
(see "Dev sweep protocol"). Output dir: `data/generated_v3/` (supersedes the v2 draft).

This is the documented redo demanded by the rigor review (`RESEARCH_DIALOGUE_ACTS.md` §8–9).
Design discussion resolved 2026-07-14 (local session): **P1 = register + brevity + persona
cards** (option chosen over "situation-only" and "full licensing"), **N = 50 per condition**.

## 1. Why v2 had to be redone — the DV-suppression audit

The dialogue-act signature is now the headline DV. Auditing the v2 generation stack showed
the old settings *suppressed by construction* the phenomena that metric measures:

| v2 setting | Effect on the DV | v3 fix |
|---|---|---|
| `min_new_tokens=16` (C2/C3/C4) | Backchannels ("Uh-huh." ≈ 3 tokens) impossible; human signature is ~19% backchannels | floor → 2 (blocks empty output only) |
| `stop_at_sentence=True` | Abandoned turns (`%` acts), trailing off impossible | off |
| `repetition_penalty=1.15`, `no_repeat_ngram=6` | Both apply over the whole context incl. the transcript → systematically suppress conversation-frequent tokens (= repeated short acknowledgments) | off; loops handled procedurally (near-duplicate resample) |
| `max_new_tokens=200`/turn, C1 `2048` total | Can truncate long turns / whole conversations (turn length & n_turns are measured) | 300/turn, C1 4096; cap-hits logged |
| `max_turns`: C2=30, C3/C4=50 | Inconsistent cap on a measured quantity | uniform 40 + in-prompt turn countdown |
| first-50 target sample | Topic-convenience sample | seeded topic-stratified manifest |
| P2 excerpt = "skip first 80 files" | Collides with random sampling | excerpt excluded from target/dev ids AND target topics |

## 2. Sampling (decided)

- Manifest: **`generation/target_ids.json`** — seed **20260714**, n=50, method:
  topic-proportional stratification over all 1,155 SwDA conversations with
  largest-remainder rounding, seeded draw within topic, seed-shuffled order (so any prefix
  is ~stratified; `--n 10` gives a valid pilot).
- **All 12 conditions generate the same 50 ids** → architecture×prompt comparisons are
  paired, and the Track-2 topic-matched human comparison uses the same conversations.
- **Dev set** (never targets, never the P2 excerpt): `3325, 3003, 3595, 4333, 3657`.
- Extending N later = a second disjoint stratified draw, unioned; never re-draw.

## 3. Prompts

- **P0 — frozen.** The paper's basic prompt, untouched (replication anchor).
- **P1 — principle: evoke through persona and situation; never instruct a measured
  behavior.** Components: spoken register + brevity line ("often just a sentence or two"),
  natural ending, peer/anti-assistant guard, and a **seeded persona card** per speaker
  (occupation + stance toward the topic + one life anchor; pools in
  `prompts/templates.py`, draw seeded by `(conversation_no, label)`, recorded in the
  output JSON). Stances are drawn independently → ~2/3 of pairs disagree in stance, so
  disagreement/stories/tangents can *emerge* without naming any dialogue act, marker, or
  topic behavior. In C3/C4 the partner's card is invisible (independent sessions).
- **P2** = P1 + one real SB excerpt from outside the target ids *and* target topics.
- **Turn countdown (supervisor fix, all levels, C2/C3/C4):** each turn's prompt states
  "the conversation so far has k turns; it cannot go past N" so the model can land a
  natural ending before the cap. Position only — it never says how to close.

**Declared caveats (carry into the writeup):**
1. Turn *length* under P1/P2 is instructed, not emergent (deliberate, as in the paper's
   enhanced prompt); every dialogue-act quantity remains unprompted in all conditions.
2. Conversation *ending* is harness-assisted everywhere (countdown + closing detection) —
   report closing-act (`fc`) results with that caveat.
3. The two seeded "Hello!" turns are scripted: `seed_turns: 2` in every JSON; Track 2
   must exclude them from tagging (and treat human openings symmetrically).

**Known wording caveat (kept deliberately, decided 2026-09-26):** the C1 (and C2) P1/P2 prompts
frame the model as the *writer* ("Write/You are writing a conversation…") but reuse the shared
P1 clauses that address it as a *participant* ("The two of you were each asked…", "You are
just an ordinary person…"). The model handles it (dev transcripts are coherent); the text is
identical across architectures within a prompt level, so it cannot drive the architecture
contrast. Report as a limitation; do not change it mid-study.

**Echo loops are measured, not patched (decided 2026-09-27):** the re-test showed C2-P0 (paper
prompt, turn-by-turn) falls into near-verbatim echo loops (17% of turns; one retry of the
loop guard is often not enough). The guard is NOT changed mid-study: C2-P1 was already
generated with it, and every architecture must run identical generation code, or the
architecture contrast is confounded. Instead, loops are a reported outcome, and the
dialogue-act analysis runs twice: all turns (primary) and `--exclude-loops` (sensitivity;
same near-duplicate definition applied post hoc to every condition).

**P3 — exploratory, INSTRUCTED condition (added 2026-09-30, outside the P0–P2 design):**
P2's framing + (1) the style line shows short reactions by example ("like 'uh-huh', 'yeah',
'right' or 'oh, really?'") and (2) two FIXED excerpts (pool ids 3352 Fishing, 2909
Immigration — the richest in listener behavior by gold labels) with each turn's gold SwDA act
in plain words ("[backchannel]"), plus a note that the labels are a guide and must not be
written. It deliberately shows the measured forms, so it answers a different question —
*can explicit showing close the structural gap?* — and is never reported as emergent.
P0–P2 prompts are verified byte-identical after adding P3 (180 prompts, all builders).
Copied labels are stripped in analysis (`analysis/analyze.py`, exact label vocabulary only).

**Stage-direction junk cutter extended (2026-10-02), applied before the C3/C4 full runs:** the
C3/C4 test showed agents appending non-speech after their turn — "(Your turn count is N)",
"(End of conversation.)", "(I'll now wait for the other caller's response)", "(Note: …)",
"## Example response:" — in C4-P0 (59/177 turns), C4-P2 (64/120) and C3 (18/312). These
patterns were added to `_META_RE` (the July junk cutter) and generation now also STOPS when
one starts (`MetaArtifactStoppingCriteria`, output-neutral like the speaker-marker stop).
Consistency proof: re-applying the new cutter to every stored turn changes **0** turns in
C2-P1 (2,080), C2-P2 (2,051), C2-P0 (1,002 incl. the archived partial run) and C4-P1; C1
never uses it. So all 12 conditions run identical cleaning. Ordinary parentheses in speech
("(IoT)", "401(k)", "(writing, drawing, …)") are not matched (verified).

## 4. Decoding (FROZEN 2026-09-26 as-is; single source: `generation/config.py`)

Freeze decision: the dev sweep forced no escalation for C1/C2 (dup-turn rate ≤0.005 at
P1/P2; C2-P0's 0.10 is the paper-prompt replication condition, kept as-is). C3's
degeneration was the multi-turn/token-cap waste fixed by the next-speaker stop below.

| | C1 (all-at-once) | C2/C3/C4 (turn-wise) |
|---|---|---|
| max_new_tokens | 4096 (whole log) | 300 (per turn) |
| min_new_tokens | — | 2 |
| temperature / top_p | 0.8 / 0.95 | 0.8 / 0.95 (shared — sampling is not a per-arch factor) |
| repetition_penalty / ngram ban | off (known-good; penalties truncate the required label repetition) | off; **procedural loop guard** instead |
| stop_at_sentence | — | off |
| stop at next-speaker marker | — | **on** (added 2026-09-26; output-neutral — the cleaner cuts at the same marker, this only skips discarded tokens, so `hit_token_cap` drops vs the dev sweep) |
| turn cap | prompt asks ~30 turns | 40, logged (`ended_by: turn_cap`) |

**Procedural loop guard** (`generation/quality.py` + `model_utils.generate_turn`): a turn
of ≥8 words with token-set Jaccard ≥0.8 against any earlier turn triggers ONE resample at
temperature +0.15. Turns under 8 words are exempt *on purpose* — backchannels legitimately
repeat. Empty emissions get one retry, then end the conversation (`ended_by: empty_turn`).

## 5. Dev sweep protocol (VM, before the full run)

1. Generate the 5 dev ids per architecture at these defaults (P0 and P1).
2. Score: `python generation/degeneration_score.py data/dev_sweep/<cond>`.
   Reference point: v2-draft C1-P0 scored `dup_turn_rate 0.011, degeneration_per_conv 0.22`.
3. Read 2–3 transcripts per condition (register, coherence, endings).
4. Acceptance: `dup_turn_rate < 0.05` and `degeneration_per_conv < 1.0` and no runaway
   turn-cap endings. If a condition fails on loops: escalate `repetition_penalty`
   1.0 → **1.05 max**, re-score, commit `config.py` with the two scores in `VM_REPORT.md`.
5. Freeze `config.py`. The full run uses no CLI decoding overrides.

## 6. Output JSON additions (v3)

`seed_turns`, `ended_by` (mutual_closing/closing/empty_turn/turn_cap), `decoding` (the
exact parameters used), `quality_counters` (multi_turn_emissions, hit_token_cap,
empty_retries, dup_resamples, dup_kept), persona-card fields inside `persona_a/b`.

## 7. Unchanged decisions

Model: Vicuna-13b-v1.5-16k (C1–C3 + C4-A), Mistral-7B-Instruct-v0.2 (C4-B), 4-bit NF4.
N=50 per condition. Mutual "Hello!" seeding. Natural-termination logic (mutual goodbye or
one reciprocal turn after a farewell).
