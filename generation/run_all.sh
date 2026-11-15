#!/usr/bin/env bash
# Generation batch into a separate reproduction directory (never the released data).
# C3/A3 and C4/A4 are paused; defaults run only C1/C2 across P0-P2.
#
# Default order is PROMPT-OUTER (all four architectures at P0, then P1, then P2) — deliberately.
# The run takes days; if it dies or we run out of time, a prompt-outer partial leaves a
# COMPLETE architecture comparison at every prompt level reached, which is the headline
# contrast. Architecture-outer would instead leave some architectures with no data at all.
# Within a level: C1 first (fast, single model), C4 last (two models, the most GPU memory).
# Resumable: each generator skips ids that already exist on disk, so a dropped session
# loses at most one conversation. Run inside tmux.
#
# CONDS overrides the order/subset, e.g. only the single-model architectures:
#   CONDS="C1-P0 C1-P1 C1-P2 C2-P0 C2-P1 C2-P2" bash generation/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
MODEL=${MODEL:-lmsys/vicuna-13b-v1.5-16k}
MODEL_B=${MODEL_B:-mistralai/Mistral-7B-Instruct-v0.2}
OUT_ROOT=${OUT_ROOT:-data/generated_reproduction}
LOG=${LOG:-_private/logs/run_all.log}
mkdir -p "$(dirname "$LOG")"
CONDS=${CONDS:-"C1-P0 C2-P0 C1-P1 C2-P1 C1-P2 C2-P2"}

for cond in $CONDS; do
  arch=$(echo "${cond%-*}" | tr '[:upper:]' '[:lower:]')
  p=${cond#*-}
  echo "=== ${cond} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  if [ "$arch" = c4 ]; then
    model_args=(--model-a "$MODEL" --model-b "$MODEL_B")
  else
    model_args=(--model "$MODEL")
  fi
  $PY "generation/generate_${arch}.py" --prompt "$p" "${model_args[@]}" \
      --out-root "$OUT_ROOT" 2>&1 | tee -a "$LOG"
  $PY generation/degeneration_score.py "$OUT_ROOT/${cond}" | tee -a "$LOG"
done
echo "RUN DONE (${CONDS}) $(date -u +%FT%TZ)" | tee -a "$LOG"
