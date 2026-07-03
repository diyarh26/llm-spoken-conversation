# VM Tasks — Regenerate C2 + C3 (P0) with the repetition fix (2026-07-03)

Owner: local side. Read `CLAUDE.md` and `.planning/STATE.md` first.

## Context
The first 200 conversations (C1–C4 × P0, 50 each) are generated and committed. Review found
**C3-P0 is degenerate** (verbatim repetition loops, self-dialogue in a single turn, never
terminates — pads to the 32-turn cap with "(End of conversation)"). **C2-P0** is coherent but
far too long (~66 words/turn vs Switchboard ~14). C1-P0 and C4-P0 are good — **leave them alone**.

The `repetition_penalty=1.2` + `no_repeat_ngram_size=3` fix in `generation/model_utils.py`
was committed with literal `` `n `` (broken, would not import). **Local has now repaired and
pushed it** (commit `d7997e5`). This task regenerates ONLY C2-P0 and C3-P0 with the working fix.

## TASK 1 — Pull and verify the fix imports
```bash
cd ~/llm-spoken-conversation
git pull --ff-only origin main
conda activate convsim
python -m py_compile generation/model_utils.py && echo "SYNTAX OK"
python -c "from generation.model_utils import chat; print('IMPORT OK')"
```
Both must print OK before continuing. Also confirm the GPU is healthy:
```bash
nvidia-smi && python -c "import torch; print('cuda', torch.cuda.is_available())"
```
If `nvidia-smi` errors with an NVML driver/library mismatch, `sudo reboot`, then retry.

## TASK 2 — Delete the bad C2/C3 data (generators SKIP existing ids, so old files must go)
```bash
rm -f data/generated/C2-P0/*.json data/generated/C3-P0/*.json
```
Do NOT touch C1-P0 or C4-P0.

## TASK 3 — Regenerate C2 + C3 (P0 only) in tmux
```bash
tmux new -s gen
conda activate convsim
python generation/generate_c2.py --prompt P0 --n 50 --max-turns 30
python generation/generate_c3.py --prompt P0 --n 50 --max-turns 30
# detach: Ctrl-b then d
```

## TASK 4 — Push + report
```bash
git add data/generated/C2-P0 data/generated/C3-P0
git commit -m "data: regenerate C2-P0 + C3-P0 with repetition_penalty fix"
git push
```
If push is rejected: `git pull --rebase origin main` then push again.
Write results/blockers into `VM_REPORT.md`. Then tell the local lead to re-run
`python analysis/evaluate_generated.py` so the metrics reflect the fixed C2/C3.

## Note
The fix kills verbatim repetition loops. It will NOT by itself fix C3's "(End of conversation)"
padding or the never-terminates-early behavior — flag in VM_REPORT if those persist after regen.
