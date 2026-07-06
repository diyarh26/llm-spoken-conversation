#!/usr/bin/env bash
# C4 fix VALIDATION (small). Adds natural-termination + the consistent turn-quality knobs
# (min-length, sentence-stop, softer repetition) to the Vicuna<->Mistral condition, which was
# already coherent but padded to the turn cap with repetitive goodbyes / "[End of conversation]".
# Regenerates representative convs into a SEPARATE test dir, then commits + pushes.
# Does NOT touch data/generated/. Launch detached and leave:
#   tmux new-session -d -s c4fix 'cd ~/llm-spoken-conversation && bash generation/run_c4_fix_test.sh'
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || { echo "not in a git repo"; exit 1; }
PY="${PYTHON:-/anaconda/envs/convsim/bin/python}"
OUT="data/generated_test/c4fix"
IDS="4104,4321,4325,4330,4333,4372,4316,4109"
LOG="run_c4_fix_test.log"

echo "=== c4 fix test START $(date -Is) ===" | tee -a "$LOG"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())" 2>&1 | tee -a "$LOG"

# Force a fresh regeneration (generators skip existing ids).
rm -f "$OUT/C4-P0"/*.json

# C4 loads Vicuna + Mistral; make sure both fit (peaks ~15 GB on the 16 GB V100).
"$PY" generation/generate_c4.py --prompt P0 --ids "$IDS" --max-turns 30 --out-root "$OUT" 2>&1 | tee -a "$LOG"

echo "=== NEW (fix) vs OLD (current): turns / median words-per-turn ===" | tee -a "$LOG"
"$PY" - "$OUT/C4-P0" "data/generated/C4-P0" <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, os, statistics, sys
new, old = sys.argv[1], sys.argv[2]
def stat(p):
    b = json.load(open(p, encoding="utf-8")).get("turns", [])[2:]
    if not b: return "empty"
    wpt = [len(t.split()) for _, t in b]
    return f"n={len(b)} med_wpt={round(statistics.median(wpt),1)}"
for f in sorted(glob.glob(os.path.join(new, "*.json"))):
    cid = os.path.basename(f); o = os.path.join(old, cid)
    print(f"  {cid:12} NEW[{stat(f)}]   OLD[{stat(o) if os.path.exists(o) else 'NA'}]")
PY

git add "$OUT" 2>&1 | tee -a "$LOG"
git commit -m "test: C4 fix validation on representative conversations" 2>&1 | tee -a "$LOG"
git pull --rebase origin main 2>&1 | tee -a "$LOG"
git push origin main 2>&1 | tee -a "$LOG" || echo "PUSH FAILED (retry manually)" | tee -a "$LOG"
echo "=== c4 fix test DONE $(date -Is) ===" | tee -a "$LOG"
