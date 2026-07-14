# VM Tasks — Track 1: v3 regeneration (2026-07-14)

Owner: local side. Read `CLAUDE.md`, then **`generation/GENERATION_SPEC.md`** — it is the
frozen design for this run (what changed vs v2 and why). This SUPERSEDES the old C1-only
regen task (2026-07-11); do not run that.

Summary of what's new on `main` (already implemented + sanity-checked locally):
- Targets now come from the committed manifest `generation/target_ids.json`
  (seeded topic-stratified 50 + 5 dev ids). Generators no longer take "first N".
- P1/P2 prompts: seeded persona cards + register/brevity; turn countdown in every
  turn-wise prompt (supervisor fix for non-natural endings).
- Decoding is DV-safe (`generation/config.py`): no token floor, no sentence-stop, logit
  penalties OFF, procedural near-duplicate resample instead, generous logged caps,
  uniform max-turns 40.
- `chat()` now returns `(text, info)` and generators log quality counters — if you patch
  anything, keep that contract.

## TASK 0 — Pull, verify env + GPU (BEFORE anything)
```bash
cd ~/llm-spoken-conversation && git pull --ff-only origin main
conda activate convsim
/anaconda/envs/convsim/bin/python -m py_compile generation/*.py prompts/templates.py && echo "SYNTAX OK"
/anaconda/envs/convsim/bin/python -c "from generation.sampling import load_target_ids, load_dev_ids; t=load_target_ids(); print(len(t), 'targets; dev', load_dev_ids())"
# expect: 50 targets; dev [3325, 3003, 3595, 4333, 3657]
nvidia-smi && /anaconda/envs/convsim/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
```
If `nvidia-smi` shows an **NVML driver/library mismatch**: `sudo reboot` NOW (never
mid-run), reconnect, re-activate, re-check. C4 loads two models — this is where the last
run crashed, so get the driver clean before starting.

## TASK 1 — Dev sweep (tmux; a few hours)
```bash
tmux new-session -d -s devsweep 'cd ~/llm-spoken-conversation && PY=/anaconda/envs/convsim/bin/python bash generation/run_v3_devsweep.sh'
# confirm it started, then leave it:
sleep 30 && tail -n 20 run_v3_devsweep.log
```
When done, judge per GENERATION_SPEC.md §5:
- `dup_turn_rate < 0.05`, `degeneration_per_conv < 1.0`, no runaway `turn_cap` endings
  (v2-draft reference: C1-P0 scored dup 0.011 / degen 0.22).
- READ 2–3 transcripts per condition: register (no helpdesk framing in P1), coherence,
  natural endings, and that short reactive turns are now actually appearing.
- If a condition loops with penalties off: raise `repetition_penalty` to **1.05 max** in
  `generation/config.py` for the TURNWISE config only, rerun that condition's dev ids
  (`rm -rf data/dev_sweep/<COND>` first), and put both scores in `VM_REPORT.md`.

## TASK 2 — Freeze the config
Commit `generation/config.py` (even if unchanged, record "sweep passed at defaults" +
the score table in `VM_REPORT.md`). After this commit the config is FROZEN for the run.
```bash
git add generation/config.py VM_REPORT.md data/dev_sweep && git commit -m "freeze(gen-v3): decoding config after dev sweep" && git push
```

## TASK 3 — Full regeneration (tmux; ~days)
```bash
tmux new-session -d -s regen3 'cd ~/llm-spoken-conversation && PY=/anaconda/envs/convsim/bin/python bash generation/run_v3_regen.sh'
sleep 60 && tail -n 20 run_v3_regen.log   # expect C1-P0 "saved <id>" lines, then STOP babysitting
```
The script goes C1→C2→C3→C4 × P0/P1/P2 into `data/generated_v3/`, scores each condition,
and commits + pushes after each. It resumes safely if the session drops (existing ids are
skipped) — just relaunch the same tmux command.

## TASK 4 — When done
Write to `VM_REPORT.md`: the degeneration-score table for all 12 conditions, wall-clock
per condition, any interventions, and 1–2 example transcripts you'd show the supervisor.
Push. Local then re-runs Track-2 metrics on `data/generated_v3`.

## Do NOT
- Do NOT edit `VM_TASKS.md` (local owns it) — everything you produce goes in `VM_REPORT.md`.
- Do NOT touch `data/generated/` or `data/generated_v2/` (kept as the before/after draft).
- Do NOT change prompts, the manifest, or decoding beyond the single documented
  escalation in TASK 1 — the design is frozen (GENERATION_SPEC.md).
- Do NOT commit Switchboard source data or model weights (`.gitignore` enforces).
- Do NOT reboot mid-run.
