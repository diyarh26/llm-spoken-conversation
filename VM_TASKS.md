# VM Tasks — v3 full run (updated 2026-09-27)

Owner: local side. Diyar now launches runs by hand in tmux (the VM agent's sandbox cannot
see the GPU). This file is the record of what runs, in what order, with which command.

## Status
| Condition | Status |
|---|---|
| C1-P0 / P1 / P2 | ✅ 50 each, on `main` |
| C2-P1 | ✅ 50, on `main` |
| C2-P2 | ⏳ running (tmux `regen`/`regen2`) |
| C2-P0 | next — same frozen code (tested in `data/dev_sweep_v2/C2-P0`) |
| C3 (P0/P1/P2) | after C2 — **5-conversation test first** (the M60 fits one Vicuna) |
| C4 (P0/P1/P2) | needs the V100 or a rented GPU (two models) |

Generation code is FROZEN for the whole study. Do not change prompts, decoding, or the loop
guard — every architecture must run identical code. Loops are handled in analysis.

## Next: C2-P0 (start when C2-P2 is done; ~50 h on the M60)
```bash
cd ~/llm-spoken-conversation && tmux new-session -d -s regen 'cd ~/llm-spoken-conversation && PY=/anaconda/envs/convsim/bin/python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CONDS="C2-P0" bash generation/run_v3_regen.sh'
```

## Then: C3 5-conversation test (same 5 dev ids, all three prompts)
```bash
cd ~/llm-spoken-conversation && tmux new-session -d -s c3test 'cd ~/llm-spoken-conversation && PY=/anaconda/envs/convsim/bin/python OUT_ROOT=data/dev_sweep_v2 LOG=run_v3_retest.log PUSH_EACH=1 ARCHS="c3" bash generation/run_v3_devsweep.sh'
```
Local reviews it (speed, multi-turn emissions, token caps, transcripts) before the full C3 run.

## Check progress (any time, read-only)
```bash
cd ~/llm-spoken-conversation && tmux ls; for d in data/generated_v3/*; do echo "$d $(ls $d | wc -l)"; done; tail -n 2 run_v3_regen.log | cut -c1-120
```

## Do NOT
- Do NOT `git pull` while a run is active (the running script pulls by itself after each condition).
- Do NOT change prompts, the manifest, `generation/config.py`, or decoding.
- Do NOT commit Switchboard source data or model weights.
