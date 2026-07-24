# VM Tasks — v3 DEV-SWEEP TEST (2026-07-23, stopgap added 2026-07-24)

Owner: local side. This is a **TEST**, not the full run. Goal: confirm the rebuilt prompting
(P0 data fix, P1 rewrite, new P2 few-shot pool) and the DV-safe decoding actually move
generation toward human structure **before** we spend days on the full 600-conversation run.
Read `CLAUDE.md`, then `generation/GENERATION_SPEC.md` for the frozen design.

## ⚠️ HARDWARE STATUS (2026-07-24): this VM was DOWNGRADED
`nvidia-smi` + Azure IMDS confirm the VM is now `Standard_NV24s_v3` = **2× Tesla M60
(7.5 GB each)**, not the original `NC6s_v3` V100 (16 GB). The V100 didn't fail — the SKU was
changed. **A resize back to `NC6s_v3` is being requested from the tutor.** The M60 is why the
last run crawled (~10 tok/s) and C3/C4 OOM'd. **The full 600-conv run REQUIRES the V100** —
do NOT attempt it on the M60. This TEST, however, can mostly finish on the M60 (see STOPGAP).

## STOPGAP — finish the TEST on the 2×M60 while waiting for the V100
C1 + C2 are already done (5 each × P0/P1/P2). Two fixes are now on `main`:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces fragmentation OOM (lets **C3**,
  a single Vicuna split across both M60s, finish). Set automatically by the sweep script.
- C4 `--device-a/--device-b` — pins C4's two models to separate GPUs (Vicuna→cuda:0,
  Mistral→cuda:1) so they don't collide on one 7.5 GB card.

**Honest expectation:** C3 should complete. **C4 may still OOM** — Vicuna-13B alone (~7 GB)
on a single 7.5 GB M60 leaves almost no room for the growing KV cache, so long conversations
can die. If C4 fails after a few turns, that's expected: **let it go, C4 waits for the V100.**
Getting C1+C2+C3 (3 of 4 architectures) fully tested now is the win.

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

## TASK 1 — Finish the dev sweep on the M60 (tmux)
Resumable — C1/C2 (and the C3 ids already on disk) are skipped, so this just finishes C3 and
attempts C4. Launch with the M60 stopgap env vars set:
```bash
tmux new-session -d -s devsweep 'cd ~/llm-spoken-conversation && \
  PY=/anaconda/envs/convsim/bin/python \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  C4_DEVICE_A=cuda:0 C4_DEVICE_B=cuda:1 \
  bash generation/run_v3_devsweep.sh'
sleep 30 && tail -n 20 run_v3_devsweep.log   # confirm it's saving, then leave it
```
The script prints two readouts at the end (degeneration score + did-it-improve report). If
C4 OOMs (expected — see STOPGAP), that's fine; run the readouts by hand on what exists:
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
