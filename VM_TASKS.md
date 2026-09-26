# VM Tasks — GPU diagnosis, then a C1 + C2 TEST (2026-09-26)

Owner: local side. Read `CLAUDE.md` first. **Deadline context: the poster is on 2026-10-11.**
Every GPU hour counts. The goal today is to (A) find out exactly what is wrong with the GPU and
(B) if it can be made to work, run a small TEST of the latest code: 5 conversations per
condition, **C1 and C2 ONLY**. **No full run yet** — local reviews the test first.

**SCOPE: C3 and C4 are OUT on this GPU. Do not run them at all.**

## TASK 0 — Diagnose the GPU (read-only first, collect everything)
Run these and paste ALL output into `VM_REPORT.md` under a new heading
`## GPU diagnosis (2026-09-26)`. Do not change anything yet.
```bash
cd ~/llm-spoken-conversation && git pull --ff-only origin main
date; uptime; uname -r
# 1. What VM size does Azure say we are? (the email to the tutor needs this)
curl -s -H Metadata:true "http://169.254.169.254/metadata/instance/compute?api-version=2021-02-01" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('vmSize:',d['vmSize'],'| name:',d['name'],'| location:',d['location'])"
# 2. What GPUs does the hardware expose?
lspci | grep -i nvidia
# 3. Driver state
nvidia-smi; echo "exit=$?"
cat /proc/driver/nvidia/version 2>&1
lsmod | grep -i nvidia
dkms status 2>&1
ls /lib/modules/
mokutil --sb-state 2>&1
sudo dmesg | grep -iE "nvidia|NVRM" | tail -n 30
# 4. Did an automatic update change the driver or kernel while we were away?
grep -iE "nvidia|linux-image" /var/log/apt/history.log* 2>/dev/null | tail -n 30
zgrep -ihE "nvidia|linux-image" /var/log/unattended-upgrades/*.log* 2>/dev/null | tail -n 20
dpkg -l | grep -iE "nvidia-(driver|dkms|kernel)|cuda-drivers" | awk '{print $2, $3}'
# 5. Is the rest of the environment still intact?
df -h ~ | tail -1
ls ~/.cache/huggingface/hub | grep -iE "vicuna|mistral"
```
Then write a **plain-language diagnosis** (3–5 lines): which VM size, which GPUs, why the
driver fails (e.g. kernel updated but the NVIDIA module wasn't rebuilt / driver-library
mismatch / Secure Boot blocking the module / no driver installed).

## TASK 1 — Fix the driver (in this order; stop at the first that works)
Nothing is running, so a reboot is safe.
1. `sudo reboot`, reconnect, then `nvidia-smi`. The July NVML mismatch was fixed this way.
2. If it's still broken and `dkms status` shows the nvidia module NOT built for the current
   kernel (`uname -r`): `sudo dkms autoinstall`, then `sudo reboot`, then `nvidia-smi`.
3. If it's still broken: reinstall the **same** driver package that `dpkg -l` shows
   (`sudo apt-get install --reinstall <that nvidia-driver-XXX package>`), reboot, re-check.
   Do NOT switch driver branches or install GRID drivers without reporting first.
4. Once `nvidia-smi` works: **stop automatic updates from breaking it again**:
   `sudo apt-mark hold $(dpkg -l | awk '/^ii/ && $2 ~ /nvidia|linux-image|linux-headers|linux-azure/ {print $2}')`
   and record what was held.

Verify: `nvidia-smi` shows 2 GPUs, and
`/anaconda/envs/convsim/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`
prints `True 2`. Record the result (fixed or not, and how) in `VM_REPORT.md`, then commit and push
the report (`git add VM_REPORT.md && git commit -m "report: GPU diagnosis" && git push`)
**before** starting Task 2, so the local side sees it right away.

If nothing works: report it and STOP. Do not try anything more invasive.

## TASK 2 — If the GPU works: 5-conversation TEST of C1 and C2 ONLY (tmux)
What is being tested: the new **next-speaker stop** on `main` (turn-wise generation, i.e.
C2, now stops as soon as the model starts writing the other speaker's line; the kept text
should be identical, only faster, with fewer token-cap hits). C1 is unaffected by the fix —
its test confirms the environment and the frozen design still work. Same 5 dev ids as the
July sweep, into a NEW folder, so it compares directly with `data/dev_sweep/`.
6 conditions (C1/C2 × P0/P1/P2); results are pushed after C1 and again after C2.
```bash
cd ~/llm-spoken-conversation
/anaconda/envs/convsim/bin/python -m py_compile generation/*.py prompts/templates.py && echo SYNTAX OK
tmux new-session -d -s retest 'cd ~/llm-spoken-conversation && \
  PY=/anaconda/envs/convsim/bin/python \
  OUT_ROOT=data/dev_sweep_v2 LOG=run_v3_retest.log PUSH_EACH=1 ARCHS="c1 c2" \
  bash generation/run_v3_devsweep.sh'
sleep 120 && tail -n 20 run_v3_retest.log
```
Confirm it is loading/generating, then leave it.

## TASK 3 — Report (quick interim after C1 finishes, full report after C2 finishes)
Into `VM_REPORT.md` under `## Re-test with next-speaker stop (2026-09-26)`:
1. **Speed:** minutes per conversation for each condition (from the `=== dev ...` timestamps
   in `run_v3_retest.log`) next to the July numbers from `run_v3_devsweep.log`.
2. **Before/after table** per condition: `hit_token_cap` and `multi_turn_emissions` totals
   (from the JSON `quality_counters`) — old `data/dev_sweep` vs new `data/dev_sweep_v2`.
3. `degeneration_score.py` and `dev_report.py` tables for `data/dev_sweep_v2/C*-P*`
   (the script prints them at the end; run by hand for an interim report).
4. **2 full transcripts each** for C1-P1, C2-P1 and C2-P2 — paste the turns.
5. Anything broken: empty turns, cut-off turns, crashes, OOMs.
6. **Timing forecast** for the full run: minutes/conversation × 50 for each of the 6
   conditions, plus the total in hours.
Then: `git add VM_REPORT.md data/dev_sweep_v2 run_v3_retest.log && git commit -m "test(gen-v3): re-test report" && git push`.
**STOP. Do NOT start the full run (`run_v3_regen.sh`).** Local reviews the test first.

## Do NOT
- Do NOT run C3 or C4 in any form.
- Do NOT edit `VM_TASKS.md` (local owns it). Everything you produce goes in `VM_REPORT.md`.
- Do NOT touch `data/generated/`, `data/generated_v2/`, `data/dev_sweep/`; do NOT create `data/generated_v3/`.
- Do NOT change prompts, the manifest, `generation/config.py`, or decoding.
- Do NOT commit Switchboard source data or model weights.
- Do NOT reboot while the tmux test is generating.
