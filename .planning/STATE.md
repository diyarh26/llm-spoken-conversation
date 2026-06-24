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

**Core value:** Isolate the effect of generation architecture on conversational realism vs Switchboard
**Current focus:** Phase 2 — main generation (12 conditions), RESUMING after a GPU-driver crash

## Current Position

Phase: 2 of 3 (Main Experiment & Analysis)
Status: Generation was running on the VM in tmux but **crashed at C4-P0** (NVML driver/library
mismatch — C4 loads two models, torch's allocator nvmlInit asserted).
Done & safe on the VM (committed locally `b0ab7a0`, NOT yet pushed): **C1-P0, C2-P0, C3-P0 = 50
each (150 conversations)**. C4-P0 has 1. The P1 set and all of P2 were not started.
Last activity: 2026-06-24 — crash diagnosed; resume plan written to VM_TASKS.md.

## NEXT STEPS (resume)

1. **Reboot the VM** (fixes the NVML driver mismatch). Verify `nvidia-smi` + `torch.cuda.is_available()`.
2. **Re-run the generation loop** (resumable; do NOT delete data/generated). Then run the 4 **P2** conditions.
3. **Push** the data, then run `python analysis/analyze.py` for the per-condition table, and add ALIGN + stats.

See VM_TASKS.md for the exact commands.

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

Last session ended with generation crashed at C4-P0. Resume via VM_TASKS.md (reboot → resume loop → P2).
Resume files: VM_TASKS.md (VM steps), analysis/analyze.py (metrics), .planning/PROJECT.md & ROADMAP.md (design).
