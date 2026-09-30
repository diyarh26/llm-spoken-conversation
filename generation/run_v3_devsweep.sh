#!/usr/bin/env bash
# Dev sweep (GENERATION_SPEC.md §5): generate the 5 committed dev ids per architecture at
# the pre-sweep decoding defaults, then score them. Output goes to data/dev_sweep/ (never
# mixed with real conditions). Run inside tmux — takes a few hours for the turn-wise archs.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python}
# OUT_ROOT/LOG/ARCHS let a re-test (e.g. after a code fix) run beside the original sweep
# on the same 5 dev ids, so before/after is a direct comparison.
OUT_ROOT=${OUT_ROOT:-data/dev_sweep}
LOG=${LOG:-run_v3_devsweep.log}
ARCHS=${ARCHS:-"c1 c2 c3 c4"}

# Reduce CUDA fragmentation OOMs (the allocator's own suggested fix). Safe on any GPU; on
# the tight 2×M60 box it is what lets C3 finish. C4_DEVICE_A/B pin C4's two models to
# separate GPUs on that box (leave unset on the V100 -> device_map="auto").
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
C4_DEVICE_A=${C4_DEVICE_A:-}
C4_DEVICE_B=${C4_DEVICE_B:-}

DEV_IDS=$($PY -c "from generation.sampling import load_dev_ids; print(','.join(map(str, load_dev_ids())))")
echo "dev ids: $DEV_IDS" | tee -a "$LOG"

# P0 (baseline) / P1 (spoken register + word-or-two + persona cards) / P2 (+ few-shot pool).
# All three so the sweep shows whether the prompt changes actually move backchannels/short
# turns UP from P0 -> P1 -> P2 (the whole point of this test).
for arch in $ARCHS; do
  # C4 loads two models — pin them to separate GPUs if C4_DEVICE_A/B are set (M60 stopgap).
  DEV_ARGS=""
  if [ "$arch" = "c4" ] && [ -n "$C4_DEVICE_A" ] && [ -n "$C4_DEVICE_B" ]; then
    DEV_ARGS="--device-a $C4_DEVICE_A --device-b $C4_DEVICE_B"
  fi
  for p in P0 P1 P2; do
    echo "=== dev ${arch}-${p} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    $PY "generation/generate_${arch}.py" --prompt "$p" --ids "$DEV_IDS" \
        --out-root "$OUT_ROOT" $DEV_ARGS 2>&1 | tee -a "$LOG"
  done
  # Push after each architecture so results can be reviewed while the next one runs.
  if [ -n "$PUSH_EACH" ]; then
    git add "$OUT_ROOT" "$LOG" && git commit -m "test(gen-v3): ${arch} re-test done" && \
      git pull --rebase -q origin main && git push \
      || echo "WARN: push failed after ${arch} (continuing)" | tee -a "$LOG"
  fi
done

echo "=== degeneration score ===" | tee -a "$LOG"
$PY generation/degeneration_score.py "$OUT_ROOT"/C*-P* | tee -a "$LOG"
echo "=== did-it-improve report (short turns + backchannels vs human) ===" | tee -a "$LOG"
$PY generation/dev_report.py "$OUT_ROOT"/C*-P* | tee -a "$LOG"
echo "SWEEP DONE — (1) degeneration: dup_turn_rate < 0.05 and degeneration_per_conv < 1.0;" \
     "(2) dev_report: P1/P2 backchannel% and <=3w% should rise toward HUMAN; (3) read 2-3" \
     "transcripts per condition. Then freeze config.py." | tee -a "$LOG"
