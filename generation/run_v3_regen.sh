#!/usr/bin/env bash
# Full v3 regeneration: 12 conditions × the 50 manifest ids into data/generated_v3/.
#
# Order is PROMPT-OUTER (all four architectures at P0, then P1, then P2) — deliberately.
# The run takes days; if it dies or we run out of time, a prompt-outer partial leaves a
# COMPLETE architecture comparison at every prompt level reached, which is the headline
# contrast. Architecture-outer would instead leave some architectures with no data at all.
# Within a level: C1 first (fast, single model), C4 last (two models — the historical NVML
# crash point). Resumable: each generator skips ids that already exist on disk. Commits +
# pushes after every condition so nothing is lost. Run inside tmux.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=run_v3_regen.log

for p in P0 P1 P2; do
  for arch in c1 c2 c3 c4; do
    COND_ARCH=$(echo "$arch" | tr '[:lower:]' '[:upper:]')
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
