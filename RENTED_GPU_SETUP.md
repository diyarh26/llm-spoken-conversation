# Rented GPU (RunPod) — setup and run guide

For: Diyar, and the Claude Code session running on the RunPod machine ("the pod").
Pod Claude: read `CLAUDE.md` first. This file replaces `VM_TASKS.md` for the pod. The pod
plays the VM role: write results and problems into `VM_REPORT.md`, never edit this file.

**Goal:** generate the remaining conversations with the SAME code, models, 4-bit loading and
package versions as the course VM, so all architectures stay comparable:
C3-P0/P1/P2 + C4-P0/P1/P2 (50 each) + a full redo of C2-P0 (50). Poster: 2026-10-11.

**The study design is FROZEN.** Do not change prompts, `generation/config.py`, decoding, the
loop guard, or the manifest. If something looks wrong, report it in `VM_REPORT.md` and stop.

---

## Step 0 — On the course VM: save its exact package list (Diyar, once, ~1 min)
The VM is idle (C2-P0 crashed). This records the exact versions the pod must match.
```bash
cd ~/llm-spoken-conversation && git pull --rebase origin main
/anaconda/envs/convsim/bin/python --version
/anaconda/envs/convsim/bin/pip freeze > requirements-vm-lock.txt
git add requirements-vm-lock.txt && git commit -m "env: exact package versions of the VM generation env" && git push origin main
```

## Step 1 — Create the pod (Diyar, RunPod website)
1. runpod.io → add credit (start with ~$25).
2. On your laptop, make an SSH key if you don't have one: `ssh-keygen -t ed25519`
   (Enter through the questions). Copy `~/.ssh/id_ed25519.pub` into RunPod → Settings → SSH Public Keys.
3. Deploy a **GPU Pod**:
   - GPU: **24 GB or more**. RTX 4090 (24 GB) is enough; RTX A6000 / A40 / L40S (48 GB) is safer for C4.
   - Template: **RunPod PyTorch** (any CUDA 12.x version is fine; we install our own torch).
   - **Container disk 30 GB, Volume disk 100 GB** (mounted at `/workspace`; models ≈ 40 GB).
   - On-demand (not spot/interruptible — an interruption kills the run).
4. When it's running: Connect → copy the **"SSH over exposed TCP"** command
   (looks like `ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519`).

## Step 2 — Copy Switchboard to the pod (Diyar, from the laptop)
Switchboard is LDC-licensed: it goes ONLY to your private pod, never into git.
In PowerShell or Git Bash, from the project folder:
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 swda.zip root@<IP>:/workspace/
```

## Step 3 — Clone the repo on the pod (Diyar, in the SSH session)
Make a GitHub **fine-grained token**: github.com → Settings → Developer settings →
Fine-grained tokens → only repository `llm-spoken-conversation`, permission
**Contents: Read and write**, expiry 30 days. Type it only into the pod terminal — never
paste it into any chat. Revoke it at the end.
```bash
cd /workspace
git clone https://<TOKEN>@github.com/diyarh26/llm-spoken-conversation.git
cd llm-spoken-conversation
git config user.name "Diyar Husayyan" && git config user.email "hasian.diyar@gmail.com"
mkdir -p data/switchboard && cd data/switchboard && unzip -q /workspace/swda.zip && cd /workspace/llm-spoken-conversation
apt-get update -qq && apt-get install -y -qq tmux unzip
```

## Step 4 — Python environment matching the VM (pod)
The VM ran Python 3.10, torch 2.5.1+cu121, transformers 5.12.1, bitsandbytes 0.49.2.
```bash
cd /workspace/llm-spoken-conversation
pip install -q uv
uv venv -p 3.10 /workspace/venv
source /workspace/venv/bin/activate
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
grep -vE "^(torch|nvidia-|triton)" requirements-vm-lock.txt | grep -v " @ " > /tmp/lock.txt
uv pip install -r /tmp/lock.txt
python -c "import torch, transformers, bitsandbytes; print(torch.__version__, transformers.__version__, bitsandbytes.__version__, torch.cuda.is_available())"
```
Expect `2.5.1+cu121 5.12.1 0.49.2 True`. If a line of the lock file fails to install,
drop that package from `/tmp/lock.txt` only if it is not torch/transformers/bitsandbytes/
accelerate, and write what you dropped into `VM_REPORT.md`.

## Step 5 — Models onto the big volume (pod, ~40 GB download)
```bash
echo 'export HF_HOME=/workspace/hf' >> ~/.bashrc && export HF_HOME=/workspace/hf
source /workspace/venv/bin/activate
python -c "from huggingface_hub import snapshot_download as d; d('lmsys/vicuna-13b-v1.5-16k'); d('mistralai/Mistral-7B-Instruct-v0.2')"
```
Mistral may require accepting its license on huggingface.co and an HF token
(`huggingface-cli login`) — if the download says "gated", do that and retry.

## Step 6 — (Optional) Claude Code on the pod
```bash
curl -fsSL https://claude.ai/install.sh | bash
cd /workspace/llm-spoken-conversation && claude
```
Log in with the URL it prints. First message: *"Read CLAUDE.md and RENTED_GPU_SETUP.md.
We are at step 7. Follow the file; report into VM_REPORT.md."*

## Step 7 — Preflight (pod)
```bash
cd /workspace/llm-spoken-conversation && source /workspace/venv/bin/activate && export HF_HOME=/workspace/hf
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m py_compile generation/*.py prompts/templates.py && echo SYNTAX OK
python -c "from analysis.swda import load_fewshot_pool; print(len(load_fewshot_pool()), 'excerpts')"   # expect: 10 excerpts
```

## Step 8 — Archive the 23 partial C2-P0 conversations (pod, once)
They were cut short by the M60 crash; all 50 get regenerated on this pod. Kept, not deleted.
```bash
mkdir -p data/archive && git mv data/generated_v3/C2-P0 data/archive/C2-P0_m60_partial
git commit -m "data: archive the 23 partial C2-P0 convs (M60 OOM); regenerate all 50 on the pod" && git push origin main
```

## Step 9 — TEST: 5 conversations of C3 and C4 (pod, tmux)
Same 5 dev ids as every earlier test, all three prompts, pushed after each architecture.
One big GPU → leave `C4_DEVICE_A/B` unset (both models go on it).
```bash
tmux new-session -d -s test 'cd /workspace/llm-spoken-conversation && source /workspace/venv/bin/activate && export HF_HOME=/workspace/hf && PY=python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OUT_ROOT=data/dev_sweep_v2 LOG=run_v3_retest.log PUSH_EACH=1 ARCHS="c3 c4" bash generation/run_v3_devsweep.sh'
sleep 300; tail -n 5 run_v3_retest.log | cut -c1-120; ls data/dev_sweep_v2/
```
Then write into `VM_REPORT.md` (new heading `## RunPod test (C3/C4)`): GPU model, minutes
per conversation per condition, and the `degeneration_score.py` + `dev_report.py` tables for
`data/dev_sweep_v2/C3-P* C4-P*`. Push. **STOP — local reviews the test before step 10.**

## Step 10 — FULL RUN (only after local says go)
Prompt-outer order, C2-P0 last. Resumable; commits + pushes after every condition.
```bash
tmux new-session -d -s regen 'cd /workspace/llm-spoken-conversation && source /workspace/venv/bin/activate && export HF_HOME=/workspace/hf && PY=python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CONDS="C3-P0 C4-P0 C3-P1 C4-P1 C3-P2 C4-P2 C2-P0" bash generation/run_v3_regen.sh'
```
Check: `tmux ls; for d in data/generated_v3/*; do echo "$d $(ls $d | wc -l)"; done`.
If a condition ends with fewer than 50 files, the generator crashed: read the end of
`run_v3_regen.log`, report it, and re-run the same command (it resumes).

## Step 11 — Finish and clean up (Diyar)
1. Confirm on GitHub: `data/generated_v3/C3-*`, `C4-*`, `C2-P0` have 50 files each.
2. On the pod: `rm -rf data/switchboard /workspace/swda.zip` (licensed data off the pod).
3. RunPod → **Terminate** the pod (Stop still bills the volume).
4. GitHub → revoke the fine-grained token.
