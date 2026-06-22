# VM Tasks — current

Owner of this file: **local side** (do not edit on the VM).
VM-side agent: read `CLAUDE.md` and `PROJECT_PLAN.md` first, then work these tasks and
record everything in `VM_REPORT.md`.

> Phase 1 goal: a validated measurement pipeline. **Do NOT generate novel conversations
> until ALIGN reproduces the Switchboard baseline** (a later task, added once setup is done).

## TASK 1 — Environment
- Run `nvidia-smi`; record GPU model, VRAM, driver, and CUDA version in `VM_REPORT.md`.
- Create conda env `convsim` (Python 3.10).
- Install `torch` matching the CUDA version, then `pip install -r requirements.txt`.
- Report: torch version, `torch.cuda.is_available()`, and any install errors.

## TASK 2 — Switchboard data (osf.io/zxwtr)
- Download the OSF project files. In `VM_REPORT.md`, list the **folder tree** and the
  **file formats** (CSV / TXT / JSON?).
- Identify (a) the 200-conversation Switchboard sample, (b) the authors' generated corpora,
  (c) their ALIGN + generation **Colab notebooks**.
- Put the Switchboard sample under `data/switchboard/`. **Do not commit it** (gitignored,
  LDC-licensed).

## TASK 3 — ALIGN install
- Install the ALIGN package (Duran et al., 2019) the way the authors' OSF Colab does.
- Record the exact install commands that worked, plus the version, in `VM_REPORT.md`.
- Note any dependency conflicts with the generation stack (a separate env is fine).

## TASK 4 — Download Vicuna (no generation yet)
- Pre-download `lmsys/vicuna-13b-v1.5-16k` in 4-bit and confirm it loads and produces a
  one-line test completion. Report VRAM used and tokens/sec.
- **Stop there.** Do not run any experimental generation.

## TASK 5 — Pilots: C1 (all-at-once) and C2 (turn-by-turn), in tmux
Both pilot generators are ready. After TASKS 1–4, run:
- C1: `python generation/generate_c1.py --prompt P0 --n 10`            -> data/generated/C1-P0/
- C2: `python generation/generate_c2.py --prompt P0 --n 10 --max-turns 30` -> data/generated/C2-P0/

Report in `VM_REPORT.md`:
- **C1**: did Vicuna emit clean `ParticipantA:`/`ParticipantB:` lines? rough words/turn?
- **C2 (the critical one)**: the `multi_turn_emissions` counts in the C2 JSONs. The paper
  found Vicuna CANNOT do turn-by-turn (it dumps multiple turns). If `multi_turn_emissions`
  is high, Vicuna is a poor fit for the turn-by-turn family (C2/C3/C4) and we'll switch
  those conditions to a stronger instruction-follower (Mistral-7B / Llama-3-8B) or rely on
  truncation. **This is the single most important thing the pilot tells us — report it.**

## TASK 6 — Report and hold before scaling
- Do NOT scale to 50/condition and do NOT run C2–C4 yet.
- The local side still owes the **ALIGN validation** (the other half of the gate) and the
  C2/C3/C4 generators. Confirm TASKS 1–5 in `VM_REPORT.md` and hold.
