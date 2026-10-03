# LLM Spoken Conversation Simulation - Generation Architecture Study

## What This Is

A Technion course research project (NLP / computational learning) testing whether the
**generation architecture** used to produce LLM conversations - all-at-once vs turn-by-turn
vs two independent agents - changes how closely those conversations match real human
telephone conversations (Switchboard), independent of the prompt and the model. It extends
Mayor, Bietti & Bangerter (2025), whose design confounded model, prompt, and architecture.

## Core Value

Cleanly isolate the effect of **generation architecture** on conversational realism - the
confound the original paper could not separate - measured against the Switchboard corpus.

**Reframe (2026-07-14):** the headline question is now broader — **characterize how the
*interactional structure* of LLM-generated conversation differs from human conversation**,
using **dialogue acts** as the instrument, with architecture (C1–C4) × prompt (P0–P2) as the
manipulated levers. Isolating the architecture effect and detecting assistant-drift are now
*components* of that, not the whole point. See `RESEARCH_DIALOGUE_ACTS.md`.

## Requirements

### Validated

- Switchboard parser reproduces the paper's baseline (~14 words/turn) - Phase 1 (local)
- Coordination-marker detectors reproduce the paper's Table 5 ranking - Phase 1 (local)
- ALIGN reproduces the Switchboard conceptual-alignment baseline (~0.57) - Phase 1 (VM)
- C2 turn-by-turn pilot shows Vicuna can emit one turn at a time (`multi_turn_emissions=0`) - Phase 1 (VM)

### Active

- [x] Final Phase 2 condition matrix confirmed: 12 conditions = C1-C4 × P0/P1/P2, 50 convs each (P2 = non-lexical metrics only)
- [ ] Regenerate conversations across the matrix (fixed prompts/decoding; **seeded topic-stratified sample**, not first-50), 50+ per condition
- [ ] Compute turn length, ALIGN alignment, and marker rates per condition vs Switchboard
- [ ] Test the Independence Gradient hypothesis statistically (C1 >= C2 >= C3 >= C4 -> SB)
- [ ] **Novel extension (headline): dialogue-act structural signature** — human vs LLM DA distribution + transition grammar (JSD) vs Switchboard gold DAMSL tags; secondary: alignment trajectory. See `RESEARCH_DIALOGUE_ACTS.md`. (Old TSI/CAS/CED demoted to supporting.)
- [ ] Final poster / presentation

### Out of Scope

- Human evaluation study (Study 2 replication) - no ethics approval, budget, or time
- Fine-tuning a model - too risky for the semester timeline; changes the project
- Full 4x3 x 200-conversation design (2,400 convs) - too large; kept all 12 conditions, reduced to 50 convs each (600 total)
- P2 few-shot as a headline condition - circular-evaluation risk (prompting for what we measure)

## Context

- Based on Mayor et al. (2025), "Can LLMs Simulate Spoken Human Conversations?" (Cognitive Science 49:e70106).
- Benchmark: Switchboard (SwDA distribution) - transcripts + topic/demographic metadata, local only.
- Compute: Azure NC6s v3 VM, 1x V100 16 GB; HuggingFace local inference (no API budget for generation).
- Models: Vicuna-13B (C1-C3), Mistral-7B-Instruct (C4 second agent).
- Workflow: two machines (local Windows + Linux VM) sharing a private GitHub repo; see CLAUDE.md / VM_TASKS.md.
- Team of 3 - Person A (generation), B (analysis), C (QC / extension / poster).

## Constraints

- **Tech stack**: HuggingFace local generation on a V100 16 GB - course constraint (no API keys for generation)
- **Models**: Vicuna-13B + Mistral-7B - fit 16 GB in 4-bit and match the paper's model family
- **Data**: Switchboard is LDC-licensed - kept local only, never committed or redistributed
- **Evaluation**: avoid circular evaluation - the P1 prompt omits the coordination markers we then measure
- **Timeline**: semester course, phase-based per the syllabus

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep all 12 conditions (C1-C4 × P0/P1/P2), cut convs/condition 200→50 | Preserves the full 4×3 design while staying within V100 compute; 50/condition is enough power | Confirmed — 12 × 50 = 600 convs; P2 analyzed on non-lexical metrics only |
| C3 = two first-person contexts, same model | Cleanly isolates the single-author effect vs C2 | Implemented; VM smoke pending |
| C4 = two first-person contexts, different models | Adds model-identity separation after the C3 comparison | Implemented; VM smoke pending |
| Headline = Independence Gradient hypothesis | Explains the paper's unexplained Vicuna result; one falsifiable prediction | Pending |
| Vicuna-13B as primary generator | Matches the paper, fits the VM | Good - C2 pilot had `multi_turn_emissions=0` |
| Validate-before-generate | Pipeline correctness before spending compute | Good |
| **Reframe to structural human↔LLM difference (2026-07-14)** | Assistant-drift was a fixable artifact, not the point; the deep question is the interactional-structure gap | Headline = dialogue acts; architecture/prompt are levers |
| **Dialogue-act signature = headline metric** | Proven technique + free gold DAMSL baseline + measures interactional structure directly; goes beyond Mayor's 3 markers | TSI/CAS/CED demoted to supporting; see `RESEARCH_DIALOGUE_ACTS.md` |
| **Full 1,155 corpus = human reference; bootstrap noise floor; topic-matched comparison** | Bigger reference is more precise; JSD is sample-size sensitive; topic (not count) is the real confound | Never shrink the baseline to 50 |
| **Topic-stratified random sampling replaces "first-50"** | First-N is a convenience sample, not representative of the corpus | Implement in `target_conversations` at regeneration |
| **Keep Vicuna-13b-v1.5-16k as primary** | Per the paper, stronger models are *further* from human; Vicuna is the right object and matches the paper exactly | Scope claims to "within Vicuna"; modern model only an optional stretch |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? Move to Out of Scope with reason
2. Requirements validated? Move to Validated with phase reference
3. New requirements emerged? Add to Active
4. Decisions to log? Add to Key Decisions
5. "What This Is" still accurate? Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check - still the right priority?
3. Audit Out of Scope - reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-22 after Phase 1 VM validation and Phase 2 prep.*
