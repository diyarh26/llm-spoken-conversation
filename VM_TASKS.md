# VM Tasks — v3 DEV-SWEEP TEST (2026-07-23)

Owner: local side. This is a **TEST**, not the full run. Goal: confirm the rebuilt prompting
(P0 data fix, P1 rewrite, new P2 few-shot pool) and the DV-safe decoding actually move
generation toward human structure **before** we spend days on the full 600-conversation run.
Read `CLAUDE.md`, then `generation/GENERATION_SPEC.md` for the frozen design.

## What changed on `main` since the last VM pull (all local, sanity-checked)
- **Cleaner fixed** (`analysis/swda.py`): drops non-verbal `.` turns, `(( uncertain ))`
  markup, and `--` dashes. Fixes both the few-shot examples and the human baseline.
- **P2 rebuilt**: draws **2 real backchannel-rich Switchboard excerpts** from a committed
  10-excerpt pool (`generation/fewshot_pool.json` — ids+offsets only, NO transcript text;
  reconstructed from local corpus at run time), seeded per conversation, distinct topics.
- **P1 rewritten**: brevity line now allows "just a word or two" (lets short/reactive turns
  emerge), topic stated once in natural case, anti-assistant guard tightened to one line.
- **P0 data fix**: 7 of 66 SwDA topic prompts were truncated in the source metadata
  (`...IMPORTANT.  ORY`, `TENY`, `YOUY`) — restored to clean text. P0 wording is otherwise
  the untouched replication anchor.
- New: `generation/dev_report.py` (short-turn% + backchannel% vs human) and P2 added to the
  dev sweep.

## TASK 0 — Pull, verify env + GPU (BEFORE anything)
```bash
cd ~/llm-spoken-conversation && git pull --ff-only origin main
conda activate convsim
/anaconda/envs/convsim/bin/python -m py_compile generation/*.py prompts/templates.py analysis/swda.py && echo "SYNTAX OK"
# pool must reconstruct 10 excerpts from the local corpus:
/anaconda/envs/convsim/bin/python -c "from analysis.swda import load_fewshot_pool; p=load_fewshot_pool(); print(len(p),'excerpts:',[e['topic'] for e in p])"
# expect: 10 excerpts: ['FISHING', 'HOME REPAIRS', ...]  (if 0 -> the swda corpus isn't extracted locally)
nvidia-smi && /anaconda/envs/convsim/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
```
If `nvidia-smi` shows an **NVML driver/library mismatch**: `sudo reboot` NOW (never mid-run),
reconnect, re-activate, re-check. C4 loads two models — that is where the last run crashed.

## TASK 1 — Run the dev sweep (tmux)
5 dev ids × 4 architectures × P0/P1/P2 → `data/dev_sweep/`. C1 is fast; C2/C3/C4 are
turn-wise (~1–2 h total). Resumable — existing ids are skipped, so just relaunch if it drops.
```bash
tmux new-session -d -s devsweep 'cd ~/llm-spoken-conversation && PY=/anaconda/envs/convsim/bin/python bash generation/run_v3_devsweep.sh'
sleep 30 && tail -n 20 run_v3_devsweep.log   # confirm C1-P0 "saved <id>" lines, then leave it
```
The script prints two readouts at the end: the degeneration score and the did-it-improve
report. If C4 crashes (NVML) before the readouts, run them by hand on whatever exists:
```bash
/anaconda/envs/convsim/bin/python generation/degeneration_score.py data/dev_sweep/C*-P*
/anaconda/envs/convsim/bin/python generation/dev_report.py data/dev_sweep/C*-P*
```

## TASK 2 — Report back (this is the whole point)
Into **`VM_REPORT.md`**, paste:
1. The full **dev_report** table (short-turn% + backchannel% per condition vs the HUMAN row).
2. The **degeneration_score** table.
3. **2–3 example transcripts each** for C2-P1 and C2-P2 (the clearest test of whether short
   reactive turns / backchannels now appear). Just paste the `turns` from the JSON.
4. One line: did P1/P2 backchannel% and short-turn% rise toward HUMAN vs P0? Any degeneration?
Then:
```bash
git add data/dev_sweep run_v3_devsweep.log VM_REPORT.md && git commit -m "test(gen-v3): dev sweep results" && git push
```
**STOP after this.** Do NOT freeze config.py or start the full regen — local reviews the
sweep first and decides. This is a checkpoint, not the run.

## Do NOT
- Do NOT edit `VM_TASKS.md` (local owns it) — everything you produce goes in `VM_REPORT.md`.
- Do NOT touch `data/generated/`, `data/generated_v2/`, or start `data/generated_v3/`.
- Do NOT change prompts, the pool, the manifest, or decoding — the design is frozen.
- Do NOT commit Switchboard source data or model weights (`.gitignore` enforces).
- Do NOT reboot mid-run.
