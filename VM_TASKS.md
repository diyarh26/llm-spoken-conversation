# VM Tasks — GPU diagnosis, then START the v3 run (2026-09-26)

Owner: local side. Read `CLAUDE.md` first. **Deadline context: the poster is on 2026-10-11.**
Every GPU hour counts. The goal today is to (A) find out exactly what is wrong with the GPU and
(B) if it can be made to work, start real v3 generation (C1, then C2) and leave it running.

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

## TASK 2 — If the GPU works: START the v3 run (C1, then C2) in tmux
The design is **FROZEN** (see `generation/GENERATION_SPEC.md` §4). One new change on `main`:
turn-wise generation now stops as soon as the model starts the *next* speaker's line
(output-neutral: the kept turn is identical; it only skips wasted tokens).

Order = C1 (all prompts, ~11 h on the M60) → C2-P1 → C2-P2 → C2-P0 (~90 h total on the
M60). It is resumable and commits + pushes after each condition.
```bash
cd ~/llm-spoken-conversation
/anaconda/envs/convsim/bin/python -m py_compile generation/*.py prompts/templates.py && echo SYNTAX OK
tmux new-session -d -s regen 'cd ~/llm-spoken-conversation && \
  PY=/anaconda/envs/convsim/bin/python \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CONDS="C1-P0 C1-P1 C1-P2 C2-P1 C2-P2 C2-P0" \
  bash generation/run_v3_regen.sh'
sleep 120 && tail -n 20 run_v3_regen.log
```
Confirm the first conversation is saved (`ls data/generated_v3/C1-P0 | wc -l` > 0), write
**one line** into `VM_REPORT.md` ("v3 run started <time UTC>, first file saved"), then commit
and push `VM_REPORT.md` only. **Leave tmux running. Do NOT start C3/C4.** The local side
decides on C3/C4 after the GPU situation with the tutor is clear.

Monitor without disturbing it: `tail -n 5 run_v3_regen.log`,
`for d in data/generated_v3/*; do echo $d $(ls $d | wc -l); done`.

## Do NOT
- Do NOT edit `VM_TASKS.md` (local owns it). Everything you produce goes in `VM_REPORT.md`.
- Do NOT touch `data/generated/`, `data/generated_v2/`, `data/dev_sweep/`.
- Do NOT change prompts, the manifest, `generation/config.py`, or decoding.
- Do NOT commit Switchboard source data or model weights.
- Do NOT reboot while the tmux run is generating.
