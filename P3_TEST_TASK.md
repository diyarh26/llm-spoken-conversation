# P3 test on a fresh GPU machine — task for the agent

You are a coding agent on a GPU machine that has never run this project. Read `CLAUDE.md`
for context, then do exactly the steps below. **Do not change any code, prompt or setting**:
the study design is frozen. If a step fails in a way not covered here, stop, write what
happened into `P3_TEST_REPORT.md`, push it (step 7), and stop.

**What this test is:** C1 (one model writes a whole phone conversation in one go) with two
prompts on the same 5 conversations: **P2** (style instruction + 2 real Switchboard example
excerpts) vs **P3** (exploratory: P2 + short reactions like "uh-huh" shown by example, and
the example turns labeled with their dialogue act, e.g. `[backchannel]`). Question: does P3
make the model produce more backchannels and short turns?

## 0. Preconditions (the human did these before starting you)
- The repo is cloned (sparse: only code), and `data/switchboard/swda/` exists (the human
  copied `swda.zip` and unzipped it there). Check: `ls data/switchboard/swda | head -3`
  shows `sw00utt ...`. If missing, stop and ask the human — never download it yourself
  from somewhere else, and **never `git add` anything under `data/switchboard/`.**

## 1. GPU check — must pass before anything else
```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```
Needs one NVIDIA GPU with **≥ 16 GB** memory. If `nvidia-smi` fails, check whether you
run inside a sandbox that hides `/dev/nvidia*` (`ls /dev/nvidia*`); if so tell the human to
restart you with GPU access. Do not continue on CPU (the code refuses anyway).

## 2. Python environment matching the reference machine
Python 3.10, torch 2.5.1 (CUDA 12.1 build), transformers 5.12.1, bitsandbytes 0.46.1,
accelerate 1.14.0 — the full list is `requirements-vm-lock.txt`.
```bash
pip install -q uv || python3 -m pip install --user -q uv
uv venv -p 3.10 ~/p3venv && . ~/p3venv/bin/activate
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
grep -vE "^(torch|nvidia-|triton)" requirements-vm-lock.txt | grep -v " @ " > /tmp/lock.txt
uv pip install -r /tmp/lock.txt
python -c "import torch, transformers, bitsandbytes; print(torch.__version__, transformers.__version__, bitsandbytes.__version__, torch.cuda.is_available())"
```
Expect `2.5.1+cu121 5.12.1 0.46.1 True`. If the driver is too old for CUDA 12.1
(`torch.cuda.is_available()` is False with a driver < 530), use
`--index-url https://download.pytorch.org/whl/cu118` instead and record that in the report.
If a non-core package from the lock file fails to install, drop only that line from
`/tmp/lock.txt` and record it. Never change torch/transformers/bitsandbytes/accelerate versions
silently.

## 3. Model (~26 GB download; put the cache on a disk with ≥ 40 GB free)
```bash
df -h ~ /tmp     # pick a disk with space; export HF_HOME=<that disk>/hf if home is small
python -c "from huggingface_hub import snapshot_download as d; d('lmsys/vicuna-13b-v1.5-16k'); print('VICUNA_OK')"
```

## 4. Preflight
```bash
python -m py_compile generation/*.py prompts/templates.py && echo SYNTAX_OK
python -c "from analysis.swda import load_fewshot_pool; p=load_fewshot_pool(); print(len(p), 'excerpts', all('labeled_text' in e for e in p))"
python -c "from prompts.templates import _p3_fewshot_block; print(_p3_fewshot_block()[:400])"
```
Expect `SYNTAX_OK`, `10 excerpts True`, and the start of the labeled examples
(`... [backchannel question]` etc.).

## 5. Run the test (inside tmux or nohup — it must survive a disconnect)
```bash
tmux new-session -d -s p3 'cd '"$PWD"' && . ~/p3venv/bin/activate && export HF_HOME='"${HF_HOME:-$HOME/.cache/huggingface}"' && PY=python PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OUT_ROOT=data/p3_test LOG=run_p3_test.log ARCHS=c1 PROMPTS="P2 P3" bash generation/run_v3_devsweep.sh'
```
(If tmux is missing: `nohup bash -c '<same command>' > /dev/null 2>&1 &`.)
10 conversations total. Check progress: `ls data/p3_test/*/ | wc -l` and
`tail -n 3 run_p3_test.log`. On a modern GPU expect minutes; if after 30 minutes no file
exists, check `nvidia-smi` utilization and the log, and report.

## 6. Report — write `P3_TEST_REPORT.md` (new file at the repo root)
Include:
1. Machine: GPU name + memory, driver, Python/torch/transformers/bitsandbytes versions, and
   anything you deviated from in steps 2–3.
2. Minutes per conversation for C1-P2 and C1-P3 (from the `=== dev` timestamps in the log).
3. The output of:
   ```bash
   python generation/dev_report.py data/p3_test/C1-P2 data/p3_test/C1-P3
   python generation/degeneration_score.py data/p3_test/C1-P2 data/p3_test/C1-P3
   ```
4. **Label copying:** how many C1-P3 conversations contain a bracketed label in the raw
   output: `grep -c "\[backchannel\|\[statement\|\[opinion" data/p3_test/C1-P3/*.json`
5. **One full C1-P2 and one full C1-P3 conversation, same id (3003)**: paste the
   `raw_output` text of each.
6. Two or three sentences: does P3 show more "uh-huh / yeah / right"-type short turns than P2?

## 7. Push to a separate branch (never to main)
```bash
git checkout -b p3-test
git add data/p3_test run_p3_test.log P3_TEST_REPORT.md
git status --short | grep -i switchboard && echo "STOP: switchboard staged" || true
git commit -m "test(P3): C1-P2 vs C1-P3 on <GPU name>"
git push -u origin p3-test
```
Then stop. Do not run anything else.
