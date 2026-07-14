#!/usr/bin/env bash
# Full v3 regeneration: 12 conditions × the 50 manifest ids into data/generated_v3/.
# Order: C1 first (fast, single model), C4 last (two models — the historical NVML crash
# point). Resumable: each generator skips ids that already exist on disk. Commits + pushes
# after every condition so nothing is lost. Run inside tmux; takes ~days.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=run_v3_regen.log

for arch in c1 c2 c3 c4; do
  COND_ARCH=$(echo "$arch" | tr '[:lower:]' '[:upper:]')
  for p in P0 P1 P2; do
    echo "=== ${COND_ARCH}-${p} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    $PY "generation/generate_${arch}.py" --prompt "$p" \
        --out-root data/generated_v3 2>&1 | tee -a "$LOG"
    $PY generation/degeneration_score.py "data/generated_v3/${COND_ARCH}-${p}" | tee -a "$LOG"
    git add data/generated_v3 "$LOG" && \
      git commit -m "data(gen-v3): ${COND_ARCH}-${p} complete" && \
      git push || echo "WARN: commit/push failed for ${COND_ARCH}-${p} (continuing)" | tee -a "$LOG"
  done
done
echo "ALL 12 CONDITIONS DONE $(date -u +%FT%TZ)" | tee -a "$LOG"
