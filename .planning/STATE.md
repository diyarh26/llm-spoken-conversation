---
gsd_state_version: '1.0'
status: in_progress
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 10
  completed_plans: 5
  percent: 55
---

# Project State

## Project Reference

See: .planning/PROJECT.md

**Core value (reframed 2026-07-14):** Characterize how the *interactional structure* of LLM-generated
conversation differs from human conversation (Switchboard), using **dialogue acts** as the instrument,
with generation architecture (C1–C4) × prompt (P0–P2) as the manipulated levers. (Earlier framing —
"isolate architecture effect on realism / assistant-drift" — is now just one component.)
**Current focus:** two parallel tracks (see `RESEARCH_DIALOGUE_ACTS.md`).

## DIRECTION UPDATE (2026-07-14) — read this first

After a harsh supervisor review of analysis rigor, the project was **reframed** and the
**headline extension changed** to a **dialogue-act structural comparison** (human vs LLM).
Full detail + the two-task plan live in **`RESEARCH_DIALOGUE_ACTS.md`**. Summary:

- **Question:** how does LLM conversation differ *structurally/interactionally* from human
  conversation, and what narrows the gap? Dialogue acts (distribution + transition grammar,
  compared by JSD) are the instrument. Assistant-drift is demoted to one component.
- **Headline metric:** dialogue-act signature vs Switchboard's **gold DAMSL `act_tag`s**
  (free human ground truth, already in `data/switchboard/**/*.utt.csv`). Secondary:
  alignment trajectory (over-alignment + slope). TSI/CAS/CED demoted to supporting.
- **Stats decided:** human reference = FULL 1,155 corpus; each LLM condition = its 50;
  significance via a bootstrap **noise floor** (50 random humans vs full reference); plus a
  **topic-matched** comparison. Report distributions, not just means.
- **Sampling decided:** replace the "first-50" convenience sample with a **seeded,
  topic-stratified random** sample of `conversation_no`s (change `target_conversations`).
- **Two tracks:** T1 regeneration = **design DECIDED + implemented locally (2026-07-14)**,
  ball is now with the VM (dev sweep → freeze → regen, see `VM_TASKS.md` +
  `generation/GENERATION_SPEC.md`); T2 measurement (local) = can start now with the human
  dialogue-act baseline.
- **Prior `data/generated_v2` is a draft** — superseded by `data/generated_v3` once T1
  reruns. The generation history below is retained for context.

## Current Position

Phase: 2 of 3 (Main Experiment & Analysis)
Status: Generation was running on the VM in tmux but **crashed at C4-P0** (NVML driver/library
mismatch — C4 loads two models, torch's allocator nvmlInit asserted).
Done & safe on the VM (committed locally `b0ab7a0`, NOT yet pushed): **C1-P0, C2-P0, C3-P0 = 50
each (150 conversations)**. C4-P0 has 1. The P1 set and all of P2 were not started.
Last activity: 2026-06-24 — crash diagnosed; resume plan written to VM_TASKS.md.

## NEXT STEPS

**Track 2 — measurement (local, can start now):**
1. Compute the **human dialogue-act signature** across all 1,155 SwDA conversations from the
   gold `act_tag` columns (no tagger/GPU). This is the reference chart — a real artifact.
2. Tag LLM conversations (DialogTag), build distribution + transition JSD, add the bootstrap
   noise floor and the topic-matched comparison.
3. In parallel: **real main-body replication** (topic-initiation extraction → reproduce the
   paper's 0.57 conceptual alignment + marker rates properly) to close the validation gap.

**Track 1 — regeneration (design DONE 2026-07-14; next actor = VM):**
1. ~~Design~~ DECIDED: P1 = register + brevity + seeded persona cards (never name a measured
   behavior); N = 50; Vicuna kept; decoding made DV-safe (the old settings *suppressed the
   headline DV by construction*: min_new_tokens=16 forbade backchannels, sentence-stop
   forbade abandoned turns, context-wide rep-penalty suppressed repeated acknowledgments).
   Turn countdown added to turn-wise prompts (supervisor fix for non-natural endings).
2. ~~Sampling~~ IMPLEMENTED: committed manifest `generation/target_ids.json` (seed 20260714,
   topic-proportional over all 1,155; 5-id disjoint dev set; P2 excerpt excluded from both).
3. VM: dev sweep (`generation/run_v3_devsweep.sh`) → freeze `generation/config.py` →
   full regen (`generation/run_v3_regen.sh`) into `data/generated_v3` → re-run Track-2
   metrics on it.

Full detail: `generation/GENERATION_SPEC.md` (design), `RESEARCH_DIALOGUE_ACTS.md`
(metric), `VM_TASKS.md` (VM commands).

## Local-session work done (2026-06-24, later)

- **Docs reconciled to the real 12-condition design** (C1-C4 × P0/P1/P2). ROADMAP/REQUIREMENTS/PROJECT
  previously said "6 conditions" (an unfinalized reduction). The actual reduction was convs/condition
  200→50; all 12 conditions are kept. P2 = robustness condition, **non-lexical metrics only** (its
  few-shot Switchboard excerpt makes marker measurement circular).
- **Added `analysis/stats.py`** — Phase 2 statistics (ANLY-01..05): words/turn mixed model
  `DV ~ Corpus*Section + (1|ConvID)`, marker rates vs SB (P2 auto-excluded), Independence Gradient
  trend test, optional figures. Runs on numpy+scipy locally today; `statsmodels`+`matplotlib` are
  optional upgrades (mixed model + PNGs); reads ALIGN output from `data/align/alignment_turns.csv`
  when present. Verified end-to-end on the spot-check data.
- **ALIGN export the VM should produce** (so stats.py picks it up): CSV `data/align/alignment_turns.csv`
  with columns `condition, conv_id, turn_index, n_turns, cosine_semanticL` (`condition`="SB" for the baseline).

## Findings so far (smoke + partial)

- **Turn length:** C1 (all-at-once) and **C2-P1 (our prompt) ≈ Switchboard ~14–15 words/turn**;
  C2-P0 ~64, C3/C4 ~75–80 (long). Clear architecture × prompt effect.
- **Coordination markers (oh/okay/uh-huh): ≈ 0 in all LLM conditions** vs Switchboard (uh-huh 0.85).
  Replicates the paper.
- ALIGN validated on Switchboard (~0.59–0.62 vs paper ~0.57). `analysis/analyze.py` ready.

## Blockers / Concerns

- NVML driver/library mismatch on the VM — reboot to fix; do NOT reboot mid-run.
- GitHub PATs were pasted in plaintext — revoke them.
- API access for the Phase-3 LLM-judge still unconfirmed.

## Session Continuity

Last session (2026-07-14, later): **Track-1 generation design decided AND implemented
locally.** Decisions: P1 = register + brevity + seeded persona cards; N = 50; DV-safe
decoding (penalties off, procedural duplicate-resample guard, no token floor/sentence-stop);
seeded topic-stratified manifest `generation/target_ids.json`; turn countdown in turn-wise
prompts (supervisor request). All four generators rewritten to shared modules; sanity checks
passed locally (no-GPU paths). Next actor: **VM** — dev sweep → freeze config → regen v3
(`VM_TASKS.md`). Local can start Track 2 (human DA baseline) in parallel.
Resume files: `generation/GENERATION_SPEC.md` (design + caveats), `RESEARCH_DIALOGUE_ACTS.md`
(metric + plan), `VM_TASKS.md` (VM steps). Auto-memory: [[project-metric-dialogue-acts]],
[[project-status-phase2-generation]].
