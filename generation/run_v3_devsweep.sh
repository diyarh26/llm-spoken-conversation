#!/usr/bin/env bash
# Dev sweep (GENERATION_SPEC.md §5): generate the 5 committed dev ids per architecture at
# the pre-sweep decoding defaults, then score them. Output goes to data/dev_sweep/ (never
# mixed with real conditions). Run inside tmux — takes a few hours for the turn-wise archs.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=run_v3_devsweep.log

DEV_IDS=$($PY -c "from generation.sampling import load_dev_ids; print(','.join(map(str, load_dev_ids())))")
echo "dev ids: $DEV_IDS" | tee -a "$LOG"

for arch in c1 c2 c3 c4; do
  for p in P0 P1; do
    echo "=== dev ${arch}-${p} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    $PY "generation/generate_${arch}.py" --prompt "$p" --ids "$DEV_IDS" \
        --out-root data/dev_sweep 2>&1 | tee -a "$LOG"
  done
done

$PY generation/degeneration_score.py data/dev_sweep/C*-P* | tee -a "$LOG"
echo "SWEEP DONE — check dup_turn_rate < 0.05 and degeneration_per_conv < 1.0 per condition," \
     "then read 2-3 transcripts per condition before freezing config.py." | tee -a "$LOG"
