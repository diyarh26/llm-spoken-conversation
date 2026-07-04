# VM Tasks — Regenerate C2-P0 once more (clean role labels) (2026-07-04)

Owner: local side. Read `CLAUDE.md` and `.planning/STATE.md` first.

## Context
The C2/C3 regeneration with the repetition fix worked: **C3-P0 is fixed** (14.7 w/turn, no
loops, no "(End of conversation)" padding, natural termination) and **C2-P0 dropped 66→18
w/turn**. But review found **C2-P0 still leaks corrupted role labels on ~79 turns** — the
model writes degraded variants like `ParticipantsA:` (stray 's') and `Participant A:` (space)
that the old cleaner missed. C3 was essentially clean (1 turn); this is a C2-specific issue
because C2 is the single model writing *both* speakers.

Local **widened `clean_single_turn` in `generation/model_utils.py`** to catch those variants
(unit-tested) and pushed it. This task regenerates **only C2-P0** with the fixed cleaner.
Leave C1-P0, C3-P0, C4-P0 alone.

## TASK 1 — Pull and verify
```bash
cd ~/llm-spoken-conversation
git pull --ff-only origin main
conda activate convsim
python -m py_compile generation/model_utils.py && echo "SYNTAX OK"
python -c "from generation.model_utils import clean_single_turn as c; print(c('ParticipantsA: Hi there!')[0])"
# expect: Hi there!
nvidia-smi && python -c "import torch; print('cuda', torch.cuda.is_available())"
```
If `nvidia-smi` shows an NVML driver/library mismatch, `sudo reboot`, then retry.

## TASK 2 — Delete C2-P0 only (generators skip existing ids) and regenerate
```bash
rm -f data/generated/C2-P0/*.json
tmux new -s gen
conda activate convsim
python generation/generate_c2.py --prompt P0 --n 50 --max-turns 30
# detach: Ctrl-b then d
```
Do NOT touch C1-P0, C3-P0, or C4-P0.

## TASK 3 — Sanity-check the leak is gone, then push
```bash
python - <<'PY'
import json, glob
n=0
for f in glob.glob('data/generated/C2-P0/*.json'):
    for s,t in json.load(open(f))['turns']:
        if 'articipant' in t: n+=1
print('C2-P0 turns with leaked participant label:', n)   # expect 0 (or very few)
PY
git add data/generated/C2-P0
git commit -m "data: regenerate C2-P0 with fixed role-label cleaner"
git push
```
If push is rejected: `git pull --rebase origin main` then push again. Record the leak count
in `VM_REPORT.md`. Then tell local to re-run `analysis/evaluate_generated.py` and
`analysis/social_metrics.py`.

## Optional (if time) — sentence-transformers CAS/CED
`pip install sentence-transformers` in `convsim`, then:
```bash
python analysis/social_metrics.py --n-sb 50 --embedding-backend sentence-transformers
```
This makes CAS measure semantic anchoring instead of lexical echo (see SOCIAL_METRICS.md).
Push the refreshed `results/social_metrics/*.csv` (NOT turn_labels.csv — it's gitignored).
