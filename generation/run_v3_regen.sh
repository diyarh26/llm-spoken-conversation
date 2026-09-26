#!/usr/bin/env bash
# Full v3 regeneration: 12 conditions × the 50 manifest ids into data/generated_v3/.
#
# Default order is PROMPT-OUTER (all four architectures at P0, then P1, then P2) — deliberately.
# The run takes days; if it dies or we run out of time, a prompt-outer partial leaves a
# COMPLETE architecture comparison at every prompt level reached, which is the headline
# contrast. Architecture-outer would instead leave some architectures with no data at all.
# Within a level: C1 first (fast, single model), C4 last (two models — the historical NVML
# crash point). Resumable: each generator skips ids that already exist on disk. Commits +
# pushes after every condition so nothing is lost. Run inside tmux.
#
# CONDS overrides the order/subset, e.g. on the 2×M60 box (C3/C4 don't fit well):
#   CONDS="C1-P0 C1-P1 C1-P2 C2-P1 C2-P2 C2-P0" bash generation/run_v3_regen.sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=run_v3_regen.log
CONDS=${CONDS:-"C1-P0 C2-P0 C3-P0 C4-P0 C1-P1 C2-P1 C3-P1 C4-P1 C1-P2 C2-P2 C3-P2 C4-P2"}

for cond in $CONDS; do
  arch=$(echo "${cond%-*}" | tr '[:upper:]' '[:lower:]')
  p=${cond#*-}
  echo "=== ${cond} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  $PY "generation/generate_${arch}.py" --prompt "$p" \
      --out-root data/generated_v3 2>&1 | tee -a "$LOG"
  $PY generation/degeneration_score.py "data/generated_v3/${cond}" | tee -a "$LOG"
  git add data/generated_v3 "$LOG" && \
    git commit -m "data(gen-v3): ${cond} complete" && \
    git push || echo "WARN: commit/push failed for ${cond} (continuing)" | tee -a "$LOG"
done
echo "RUN DONE (${CONDS}) $(date -u +%FT%TZ)" | tee -a "$LOG"
