#!/usr/bin/env bash
# C2 fix VALIDATION (small, ~8 conversations). Same turn-quality + natural-termination fix as
# C3, PLUS the strengthened leaked-label cleaner (PartB:/Partner B:/Participants:). Regenerates
# the worst C2 conversations into a SEPARATE test dir, then commits + pushes so the local side
# can read them. Does NOT touch data/generated/. Launch detached and leave:
#   tmux new-session -d -s c2fix 'cd ~/llm-spoken-conversation && bash generation/run_c2_fix_test.sh'
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || { echo "not in a git repo"; exit 1; }
PY="${PYTHON:-/anaconda/envs/convsim/bin/python}"
OUT="data/generated_test/c2fix"
# worst label-leak + fragment convs, plus 4325 as a control
IDS="2095,2451,4316,4060,4330,4382,4321,4325"
LOG="run_c2_fix_test.log"

echo "=== c2 fix test START $(date -Is) ===" | tee -a "$LOG"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available())" 2>&1 | tee -a "$LOG"

# Force a fresh regeneration (generators skip existing ids).
rm -f "$OUT/C2-P0"/*.json

"$PY" generation/generate_c2.py --prompt P0 --ids "$IDS" --max-turns 30 --out-root "$OUT" 2>&1 | tee -a "$LOG"

echo "=== NEW (fix) vs OLD (current) : turns / median words-per-turn / leaked-label turns ===" | tee -a "$LOG"
"$PY" - "$OUT/C2-P0" "data/generated/C2-P0" <<'PY' 2>&1 | tee -a "$LOG"
import json, glob, os, re, statistics, sys
new, old = sys.argv[1], sys.argv[2]
def stat(p):
    b = json.load(open(p, encoding="utf-8")).get("turns", [])[2:]
    if not b: return "empty"
    wpt = [len(t.split()) for _, t in b]
    leaks = sum(1 for _, t in b if re.search(r"\bpart(?:ner|icipant)", t, re.I))
    return f"n={len(b)} med_wpt={round(statistics.median(wpt),1)} leaks={leaks}"
for f in sorted(glob.glob(os.path.join(new, "*.json"))):
    cid = os.path.basename(f); o = os.path.join(old, cid)
    print(f"  {cid:12} NEW[{stat(f)}]   OLD[{stat(o) if os.path.exists(o) else 'NA'}]")
PY

git add "$OUT" 2>&1 | tee -a "$LOG"
git commit -m "test: C2 fix validation on worst conversations" 2>&1 | tee -a "$LOG"
git pull --rebase origin main 2>&1 | tee -a "$LOG"
git push origin main 2>&1 | tee -a "$LOG" || echo "PUSH FAILED (retry manually)" | tee -a "$LOG"
echo "=== c2 fix test DONE $(date -Is) ===" | tee -a "$LOG"
