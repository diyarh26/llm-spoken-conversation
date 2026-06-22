---
gsd_state_version: '1.0'
status: in_progress
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 10
  completed_plans: 2
  percent: 20
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-22)

**Core value:** Isolate the effect of generation architecture on conversational realism vs Switchboard
**Current focus:** Phase 1 — Pipeline Validation & Generation Pilots

## Current Position

Phase: 1 of 3 (Pipeline Validation & Generation Pilots)
Plan: 2 of 4 complete in current phase
Status: In progress — VM setup + ALIGN + pilots next
Last activity: 2026-06-22 — parser + marker metrics validated locally; C1/C2 generators written; git remote live

Progress: [██░░░░░░░░] 20%

## Accumulated Context

### Decisions

Logged in PROJECT.md Key Decisions. Recent:
- 6-condition design (C1–C4 × P0/P1), 50 conversations each
- C3 = two first-person contexts (same model); C4 = different models
- Vicuna-13B primary — ⚠️ may fail turn-by-turn (C2 pilot will test)
- Headline: Independence Gradient (C1 ≥ C2 ≥ C3 ≥ C4 → SB)

### Pending Todos

None tracked via GSD yet.

### Blockers/Concerns

- [Phase 1] Vicuna may not follow turn-by-turn generation (paper §2.2.2). The C2 pilot
  measures `multi_turn_emissions`; if high, switch C2/C3/C4 to Mistral/Llama-3 or rely on
  truncation.
- API access for the Phase-3 LLM-judge is unconfirmed (course-staff meeting pending).

## Session Continuity

Last session: 2026-06-22
Stopped at: Local validation complete; VM cloned; awaiting VM setup + pilots (VM_TASKS.md)
Resume file: VM_TASKS.md (VM tasks) / VM_REPORT.md (results)
