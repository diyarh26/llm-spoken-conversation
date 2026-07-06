# VM Tasks — C3 fragmentation-fix validation (2026-07-06)

Owner: local side. Read `CLAUDE.md` first. **Small, fire-and-forget job — launch detached,
confirm it started, then stop.** It commits and pushes by itself.

## What & why
We diagnosed C3-P0: ~64% of conversations collapse into "word salad" — the two Vicuna agents
stop taking turns and instead finish one run-on sentence a word at a time. Root cause: turns
were allowed to end incomplete (no minimum length, no sentence-boundary stop), and the
anti-repetition settings forced the model to bail early. We changed the C3 turn-quality
settings (`generation/generate_c3.py` defaults now: `--min-new-tokens 16`,
`--stop-at-sentence` on, `--repetition-penalty 1.15`, `--no-repeat-ngram 4`) and made
`min_new_tokens` actually enforced in `chat()`.

This task **regenerates only the 8 worst C3 conversations** with the fix into a **separate
test dir** (`data/generated_test/c3fix/`) so we can read them and judge whether turns now
sound like two people before touching anything else. It does **not** touch `data/generated/`.

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

## TASK 2 — Launch the test detached, then leave
```bash
tmux new-session -d -s c3fix 'cd ~/llm-spoken-conversation && bash generation/run_c3_fix_test.sh'
```

## TASK 3 — Confirm it started, then STOP
```bash
sleep 20
tmux ls                          # expect a "c3fix" session
tail -n 15 run_c3_fix_test.log   # expect cuda True and generation starting
```
If the session exists and generation is running, **you are done — disconnect.** The script
generates ~8 conversations (a few minutes), prints a NEW-vs-OLD median-words/turn comparison,
and **commits + pushes `data/generated_test/c3fix/`** by itself. Do not wait for it.

## Do NOT
- Do **not** touch or regenerate `data/generated/` (the current data stays as the baseline).
- Do **not** start the full P0 regeneration (`run_p0_v2.sh`) — that waits until this C3 fix
  and the other architectures are validated.
- Do **not** run in the foreground or babysit.

## After it pushes
The local side pulls `data/generated_test/c3fix/`, reads the conversations, and decides whether
the fix worked (turns coherent, no word-salad) before rolling it into a full regeneration.
