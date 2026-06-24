# VM Tasks — RESUME after the GPU-driver crash (2026-06-24)

Owner: local side. Read `CLAUDE.md` and `.planning/STATE.md` first.

## What happened
The full generation run (12 conditions, 50 each) was running in `tmux` but **crashed at C4-P0**
because of an NVIDIA **NVML driver/library version mismatch** (C4 loads two models; torch's
CUDA allocator called `nvmlInit` and it asserted). Safe on disk on the VM:
`C1-P0`, `C2-P0`, `C3-P0` = 50 each (150 conversations). `C4-P0` has 1. The P1 set and all of
P2 were not started. (These 150 are committed locally on the VM as `b0ab7a0` but NOT pushed.)

## TASK 1 — Reboot to fix the GPU driver
Reboot the VM (Azure portal → your VM → **Restart**, or `sudo reboot`). This is the correct
fix now that nothing is running. After it returns, verify:
```bash
nvidia-smi
cd ~/llm-spoken-conversation && conda activate convsim
python -c "import torch; print('cuda', torch.cuda.is_available())"
```
Expect `nvidia-smi` to work and `cuda True`.

## TASK 2 — Resume generation (DO NOT delete data/generated — it is resumable)
```bash
tmux new -s gen
conda activate convsim
for LV in P0 P1; do
  python generation/generate_c1.py --prompt $LV --n 50
  python generation/generate_c2.py --prompt $LV --n 50 --max-turns 30
  python generation/generate_c3.py --prompt $LV --n 50 --max-turns 30
  python generation/generate_c4.py --prompt $LV --n 50 --max-turns 30
done
```
It skips C1/C2/C3-P0 (done), finishes C4-P0, then does the whole P1 set. Detach: Ctrl-b then d.

## TASK 3 — P2 few-shot conditions (after TASK 2)
```bash
python generation/generate_c1.py --prompt P2 --n 50
python generation/generate_c2.py --prompt P2 --n 50 --max-turns 30
python generation/generate_c3.py --prompt P2 --n 50 --max-turns 30
python generation/generate_c4.py --prompt P2 --n 50 --max-turns 30
```

## TASK 4 — Push + report
```bash
git add data/generated && git commit -m "data: full Phase 2 generation (12 conditions)" && git push
```
If push is rejected: `git pull --rebase origin main` then push again. Then tell the local lead
to run `python analysis/analyze.py` and the ALIGN + stats analysis.

## Note
If the NVML mismatch recurs after reboot, it's an unattended NVIDIA driver update; a reboot
re-syncs it. Generators are resumable, so no finished conversation is ever lost.
