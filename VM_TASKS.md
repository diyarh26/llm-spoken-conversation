# VM Tasks — C2 fix validation (2026-07-06)

Owner: local side. Read `CLAUDE.md` first. **Small, fire-and-forget job — launch detached,
confirm it started, then stop.** It commits and pushes by itself.

## What & why
C3 is fixed and validated. Now C2. C2 gets the **same** turn-quality + natural-termination fix
as C3 (min-length floor, sentence-boundary stop, softer repetition, stop at goodbye), **plus**
its own fix: a strengthened cleaner that strips the leaked speaker-label variants C2 emits
(`PartB:`, `Partner B:`, `Participants:`) — the single-model-writes-both-speakers path.

This task **regenerates only the 8 worst C2 conversations** with the fix into a **separate test
dir** (`data/generated_test/c2fix/`). It does **not** touch `data/generated/`.

## TASK 1 — Pull and verify GPU
```bash
cd ~/llm-spoken-conversation          # (or wherever this repo lives on the VM)
git pull --ff-only origin main
conda activate convsim
/anaconda/envs/convsim/bin/python -m py_compile generation/*.py && echo "SYNTAX OK"
nvidia-smi && /anaconda/envs/convsim/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
```
If `nvidia-smi` shows an **NVML driver/library mismatch**, `sudo reboot`, reconnect,
`conda activate convsim`, then continue. Never reboot mid-run.

## TASK 2 — Launch detached, then leave
```bash
tmux new-session -d -s c2fix 'cd ~/llm-spoken-conversation && bash generation/run_c2_fix_test.sh'
```

## TASK 3 — Confirm it started, then STOP
```bash
sleep 20
tmux ls                          # expect a "c2fix" session
tail -n 15 run_c2_fix_test.log   # expect cuda True and generation starting
```
If the session exists and generation is running, **you are done — disconnect.** The script
generates ~8 conversations (a few minutes), prints a NEW-vs-OLD comparison (turns / median
words-per-turn / leaked-label count), and **commits + pushes `data/generated_test/c2fix/`** by
itself. Do not wait for it.

## Do NOT
- Do **not** touch or regenerate `data/generated/`.
- Do **not** start the full P0 regeneration yet — that waits until C2 and C4 are validated.
- Do **not** run in the foreground or babysit.

## After it pushes
Local pulls `data/generated_test/c2fix/`, reads the conversations, and checks that turns are
coherent, labels no longer leak, and conversations end naturally — before we move to C4 and the
full regeneration.
