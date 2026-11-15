# Aleen's instructions for the workshop generation runs

Your responsibility is **Gemma 4 31B Instruct generation and RunPod environment reporting**. We are extending the existing [llm-spoken-conversation repository](https://github.com/diyarh26/llm-spoken-conversation), keeping its paired topics, personas, prompts, and original results.

## Current pause on agent runs

Diyar has paused **A3/C3 and A4/C4** while the team reviews whether the implementation matches the intended agent experiment. A3/A4 are the poster names for C3/C4 in the code. Do not start, resume, or schedule C3/C4 GPU pilots or production. If either is running, interrupt that process with Ctrl+C in its tmux session, keep completed files and logs, and report the completed IDs to Diyar. Do not delete outputs or stop another person's job.

The active commands below run **C1 and C2 only**. C3/C4 remain available in the code for review and deterministic tests. Resume them only after Diyar explicitly lifts this pause; the decided model pairings remain unchanged.

## Your assignment

| Item | Your setting |
| --- | --- |
| Model | `google/gemma-4-31B-it` |
| Active architectures | C1 and C2 only |
| Paused architectures | A3/C3 and A4/C4; no pilot or production runs yet |
| Prompt levels | P0, P1, P2 |
| Production conversations | Existing 50 IDs in `generation/target_ids.json` |
| Production total | 6 active conditions x 50 = 300 conversations |
| Development conversations | Existing five `dev_ids` in the same manifest |
| Branch | `aleen/gemma4-generation` |
| Production output | `data/generated_workshop/gemma4-31b-it/` |
| Private pilot output | `_private/pilots/gemma4-31b-it/` |
| Run report to commit | `reports/gemma4-31b-it-run.md` |

C1 writes a whole call in one generation. C2 uses one writer to generate alternating turns. C3 gives each speaker an independent chat view using the same model. Their implementations already exist in `generation/generate_c1.py`, `generate_c2.py`, and `generate_c3.py`.

C4 pairings are now decided by the team:

| Owner | Pairing | ParticipantA | ParticipantB |
| --- | --- | --- | --- |
| Diyar | OLMo ↔ Qwen | OLMo 3 Instruct | Qwen3.5-9B |
| Sham | Qwen ↔ Gemma | Qwen3.5-9B | Gemma 4 31B Instruct |
| Aleen | Gemma ↔ OLMo | Gemma 4 31B Instruct | OLMo 3 Instruct |

For execution, the commands use the listed order as A/B, consistent with the earlier proposed assignments. The pairing decision does not itself specify counterbalancing; do not add reversed runs unless the team records that extension. Your C4 uses `google/gemma-4-31B-it` for ParticipantA and `allenai/Olmo-3-7B-Instruct` for ParticipantB. After the pause is lifted, C4 would add 150 conversations across the three prompts. C3 would add another 150. Those 300 agent conversations are excluded from your current active assignment of 300 C1/C2 conversations. The directory slug identifies your assigned batch; C4 records must identify both models explicitly.

The commercial model and base-model test remain separate assignments.

## Before starting a paid generation run

The shared update now provides model selection, native model/processor loading, thinking-off controls, private raw-response traces, immutable model revisions, per-conversation seeds, atomic saves, and run manifests. The original released dataset is protected from new generation writes.

Local automated tests and CPU checks do not verify GPU memory, quantized loading, runtime, or generated conversation quality. Complete the preflight and the active C1/C2 development pilot below before production. A3/C3 and A4/C4 remain paused.

Each condition has a `run_manifest.meta` file. Resuming with a different model revision, precision, seed, source code, environment, input data, or decoding configuration fails rather than mixing results. Preserve this manifest with the condition files. Raw prompts/responses, including retries, are saved under `_private/generation_raw/<run_id>/`; back them up privately because P2 prompts contain licensed examples.


## 1. Connect your GitHub account

1. Sign in to your own GitHub account.
2. Your collaborator access is already arranged. Confirm you can open `diyarh26/llm-spoken-conversation` while signed in.
3. Open the repository and read `README.md`, `docs/methods.md`, and `prompts/PROMPT_EXAMPLES.md`.
4. Use your own branch. Do not push directly to `main`.

Except for the explicitly labelled local connection setup, the commands below are **Bash commands for the Linux RunPod terminal**, not Windows PowerShell commands.

## Connect your local Codex session to RunPod

This is the same SSH approach used for the earlier project. You do not need to give Codex your RunPod account login or an API key. Your own local Codex session can run SSH commands using the key on your own computer.

1. On your computer, run `ssh-keygen -t ed25519` only if you do not already have a suitable key. Do not overwrite an existing key.
2. Add the contents of the public key file `~/.ssh/id_ed25519.pub` to RunPod's SSH public keys. On Windows, print it in PowerShell with `Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"`.
3. In RunPod, open your Pod's **Connect** panel and copy its **SSH over exposed TCP** command. This gives the current host, port, user, and key path. Use your own Pod's details; old project Pod addresses may no longer exist.
4. Test the copied command in your computer's terminal. It should open a shell on the Pod.
5. In your local Codex project chat, provide the connection command and the local key path, then give the task below. Keep the private key on your computer; do not paste its contents into chat.

Example connection command, with placeholders to replace:

```text
ssh root@POD_IP -p POD_PORT -i PATH_TO_YOUR_PRIVATE_KEY
```

Suggested first message to your own Codex session:

```text
Use my working SSH command below to connect to my RunPod Pod.
SSH command: [paste the command with my own host, port, and local key path]
Project path on the Pod: /workspace/llm-spoken-conversation
Read my assigned guide from this repository.
First check the connection, nvidia-smi, disk space, Git branch, and Python environment.
Prepare and run the C1/C2 development pilot after the required preflight checks pass.
C3/A3 and C4/A4 are paused; do not run them.
Use my agreed spending limit. Keep long runs in tmux and preserve all completed files.
```

The Pod also needs its own GitHub authentication to clone/push; that is the separate setup in section 3 below. Do not copy a teammate's private SSH key to solve authentication.

If you prefer opening the remote project directly in the desktop app, follow the [official SSH-host connection flow](https://learn.chatgpt.com/docs/remote-connections#connect-to-an-ssh-host). That flow also requires installing/authenticating Codex on the remote host. The local-session SSH method above does not require that extra remote installation.

## 2. Prepare your RunPod workspace

RunPod access and spending limits are already arranged. Use the agreed budget, GPU, storage, and template. Use a CUDA/PyTorch template compatible with the shared environment. Keep work under the persistent mount, conventionally `/workspace`; verify the actual mount in your Pod configuration.

Use a network volume for files that must survive Pod termination. Container disk is temporary; a Pod volume is deleted on termination. A network volume still needs an off-Pod backup. See [RunPod storage options](https://docs.runpod.io/pods/storage/types).

Open the Pod's Connect panel. Use its web terminal, or copy its SSH connection command to your computer. The SSH key used to connect **to RunPod** is separate from the GitHub authentication configured below.

In the Pod terminal, check:

```bash
nvidia-smi
python --version
git --version
tmux -V
df -h /workspace
```

If a command is missing, have Diyar update the shared template/setup recipe. Record the selected template and GPU in your report. Do not assume closing the browser stops billing.

Use google/gemma-4-31B-it, the instruction-tuned checkpoint. Do not substitute google/gemma-4-31B (base), another size, or a community quantized checkpoint without a documented protocol change.

The official example uses a multimodal model/processor interface. Diyar must verify this model's loading path, native chat template, system-message handling, and reasoning setting for our text-only calls. Do not force Vicuna formatting to get a run started.

Gemma is the largest assigned model. Choose GPU memory and disk capacity after agreeing the precision with Diyar. Test the longest prompt/history, not just a greeting. If memory runs out, report the GPU, precision, and condition; do not independently shorten context, reduce turn limits, or change quantization.

## 3. Authenticate Git on the Pod and clone the repository

Configure GitHub SSH authentication on the Pod. Never copy someone else's private key.

First inspect whether your own key already exists:

```bash
ls -l ~/.ssh/id_ed25519.pub
```

If it does not exist, create it; replace the email with your own. Do not overwrite an existing key.

```bash
ssh-keygen -t ed25519 -C "YOUR_GITHUB_EMAIL"
```

Accept the default filename and use a passphrase. Then:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Copy only the printed **public** key into GitHub → Settings → SSH and GPG keys → New SSH key. Keep the private key outside the repository.

```bash
ssh -T git@github.com
```

Verify GitHub's published host fingerprint if SSH asks about a new host: [GitHub fingerprints](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints). Successful authentication says GitHub does not provide shell access; exit code 1 is normal for that successful test.

For a fresh workspace:

```bash
cd /workspace
git clone git@github.com:diyarh26/llm-spoken-conversation.git
cd llm-spoken-conversation
```

If the repository already exists, enter it instead of cloning over it. Check `git status` before pulling. Preserve any existing edits and do not use `git reset --hard`.

Pull the shared update from GitHub before starting your branch:

```bash
git switch main
git pull --ff-only origin main
git switch -c aleen/gemma4-generation
git config user.name "Aleen"
git config user.email "YOUR_GITHUB_EMAIL_OR_GITHUB_NOREPLY_EMAIL"
git remote -v
git status
git rev-parse HEAD
```

Replace the email placeholder. If your branch already exists, use `git switch aleen/gemma4-generation` instead of `git switch -c`. Record the generation code commit before running. Do not pull or modify code during an active batch.

## 4. Install the agreed environment and verify the data

Use the shared `environment/workshop-requirements.txt` as the starting recipe. It pins PyTorch 2.12.0 with Torchvision 0.27.0, Transformers 5.12.1, Accelerate, and bitsandbytes. Choose CUDA wheels/image compatible with the RunPod driver; do not install CPU-only PyTorch on the Pod. The full GPU environment must still be frozen after the pilot. The original `environment/generation-lock.txt` records the earlier experiment.

Inside the agreed Python environment, install the project:

```bash
python -m pip install -r environment/workshop-requirements.txt
python -m pip install -e ".[generation,analysis,dev]"
python -c "import torch, torchvision; assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print(torch.__version__, torchvision.__version__, torch.version.cuda)"
python -m pytest
```

This editable install is not a version lock. Save `python -m pip freeze` in private diagnostics and record the relevant versions in your report. If the tests fail, report the output before changing shared code. Existing CPU tests do not prove GPU/model compatibility.

If the model download requires authentication, use your own Hugging Face account and the approved client's interactive login. Do not paste tokens into scripts, command arguments, logs, or commits.

The public SwDA text archive is now included in the GitHub repository as `swda.zip`. Read [its source and license notices](SWDA_SOURCE.md). After cloning, extract it from the repository root:

```bash
python -m zipfile -e swda.zip data/switchboard
```

The resulting files must be at:

```text
data/switchboard/swda/swda-metadata.csv
data/switchboard/swda/.../sw_*.utt.csv
```

GitHub contains the archive; the extracted directory stays gitignored to avoid storing it twice. P2 reconstructs its examples from the extracted corpus and fails clearly when they are unavailable. Run this check from the repository root:

```bash
python - <<'PY'
from analysis.swda import load_metadata, fewshot_examples
from generation.sampling import load_manifest

manifest = load_manifest()
metadata = load_metadata()
ids = manifest["target_ids"] + manifest["dev_ids"]
assert len(manifest["target_ids"]) == 50
assert not set(manifest["target_ids"]) & set(manifest["dev_ids"])
assert all(cno in metadata for cno in ids), "Missing conversation metadata"
for cno in ids:
    examples = fewshot_examples(cno, k=2)
    assert len(examples) == 2, f"P2 examples missing for {cno}"
    assert all(e["text"].strip() for e in examples)
print("Metadata and two P2 examples are available for every target and dev ID.")
PY
```

Keep ad-hoc rendered prompts and raw diagnostics private. The included unmodified SwDA archive retains its upstream attribution and terms; do not relicense it under the project's generated-data license. Do not change or regenerate the sampling manifest.

Next check the actual checkpoint configs, native templates, and installed model-class support:

```bash
python generation/preflight.py --owner aleen
```

This downloads small configuration/tokenizer/processor files, not model weights. It also checks the assigned C4 partner's template for future compatibility; it does not run paused C3/C4 conversations. Its private report `_private/preflight-aleen.json` records immutable checkpoint revisions used by the commands below. Keep the same report from pilot through production; do not rerun it to silently change revisions mid-experiment.

## 5. Run the development pilot

Confirm the active scripts expose model and runtime options:

```bash
python generation/generate_c1.py --help
python generation/generate_c2.py --help
```

If `--model` is absent, your checkout is outdated. Update it before running these commands.

Start a persistent terminal for the active C1/C2 pilot:

```bash
tmux new -s gemma4-31b-it
```

Inside tmux, enter the repo, activate the agreed environment, and run the following block. It uses the existing development IDs, not production IDs, and stops on the first failed command.

```bash
(
set -euo pipefail
mkdir -p _private/logs
MODEL_REVISION=$(python -c 'import json; print(json.load(open("_private/preflight-aleen.json"))["models"][0]["revision"])')
DEV_IDS=$(python -c 'from generation.sampling import load_dev_ids; print(",".join(map(str, load_dev_ids())))')
for prompt in P0 P1 P2; do
  for arch in c1 c2; do
    python -u "generation/generate_${arch}.py" \
      --model "google/gemma-4-31B-it" \
      --revision "$MODEL_REVISION" \
      --precision nf4 --thinking off --seed 20261115 \
      --prompt "$prompt" \
      --ids "$DEV_IDS" \
      --out-root "_private/pilots/gemma4-31b-it" \
      2>&1 | tee -a "_private/logs/gemma4-31b-it-pilot.log"
  done

done
)
```

Detach with **Ctrl+B**, then **D**. Reconnect with `tmux attach -t gemma4-31b-it`. Detaching does not stop the run.

The active development pilot is 30 conversations: C1/C2 x three prompts x five development IDs. C3/C4 are paused, including their GPU pilots. Before letting it finish unattended, inspect the first outputs with Diyar:

- The recorded model is your assigned checkpoint.
- C1 contains a parseable full dialogue; C2 contains single spoken turns.
- P2 actually includes both examples, and each architecture has the intended view.
- No reasoning or assistant commentary enters spoken history.
- Token-cap hits, retries, and empty outputs are recorded; record observed peak memory in your report.
- Short acknowledgments remain possible under the shared decoding settings.

A weird conversation is not automatically an implementation failure. Preserve it and report it. If settings or code change after the pilot, use a new pilot output directory so resumability does not reuse the old run.

Send Diyar the code commit, environment, pilot counts, runtime, memory use, and representative issues before production. Agree the frozen configuration and expected cost.

## 6. Generate your 300 active production conversations

Use the same code commit and frozen configuration verified in the pilot. Run inside tmux with the agreed environment active:

```bash
(
set -euo pipefail
mkdir -p _private/logs
MODEL_REVISION=$(python -c 'import json; print(json.load(open("_private/preflight-aleen.json"))["models"][0]["revision"])')
for prompt in P0 P1 P2; do
  for arch in c1 c2; do
    python -u "generation/generate_${arch}.py" \
      --model "google/gemma-4-31B-it" \
      --revision "$MODEL_REVISION" \
      --precision nf4 --thinking off --seed 20261115 \
      --prompt "$prompt" \
      --out-root "data/generated_workshop/gemma4-31b-it" \
      2>&1 | tee -a "_private/logs/gemma4-31b-it-production.log"
  done

done
)
```

Without `--ids` or `--n`, the scripts use all 50 committed targets. Prompt-outer ordering completes an architecture comparison at each prompt level first.

After a disconnect, reattach to tmux before starting another process. If the process actually stopped, rerun the same block with the same settings and output root. Existing completed IDs are skipped. Never run two jobs that write the same output directory.

Do not use `make generate` or the original `run_all.sh` for this assignment unless Diyar has explicitly updated them for these model-specific runs: the original versions target the old experiment.

## 7. Validate and review your results

This checks file coverage, required identity fields, and basic nonempty output using the existing JSON format:

```bash
python - <<'PY'
import json
from pathlib import Path
from generation.sampling import load_target_ids

root = Path("data/generated_workshop/gemma4-31b-it")
targets = set(load_target_ids())
expected_conditions = {f"C{a}-P{p}" for a in (1, 2) for p in (0, 1, 2)}
assert {p.name for p in root.iterdir() if p.is_dir()} == expected_conditions
for condition in sorted(expected_conditions):
    paths = list((root / condition).glob("*.json"))
    assert {int(p.stem) for p in paths} == targets, condition
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["condition"] == condition, path
        assert record["conversation_no"] == int(path.stem), path
        assert record["architecture"] == condition.split("-")[0], path
        assert record["prompt_level"] == condition.split("-")[1], path
        models = {record["model"]} if "model" in record else set(record["models"].values())
        assert models == {"google/gemma-4-31B-it"}, path
        if condition.startswith("C1-"):
            assert record["raw_output"].strip(), path
        else:
            assert record["turns"] and all(turn[1].strip() for turn in record["turns"]), path
    print(condition, len(paths), "files OK")
print("300 active files passed basic validation. Manual review and run-metadata checks still required.")
PY
```

This is a basic check, not the full scientific evaluation. Review a sample from every condition and ask Sham to review some of yours. Report cap hits, retries, parsing problems, and unusual dialogue without replacing difficult cases.

Your extra responsibility is the environment report. Record the Pod template/image, GPU model/count/VRAM, Python, CUDA, PyTorch, Transformers, Accelerate and bitsandbytes versions, model revision, precision, peak GPU memory, disk use, and pilot runtime. Share the report with Diyar and Sham so the environment can be reproduced. Your Gemma memory requirements need not match Sham's Qwen requirements.

Do not overwrite the original `data/generated/` or `results/`. Some existing analysis scripts hardcode those original paths; Diyar will handle adapting and combining the workshop analyses.

## 8. Write your run report and push your work

Create `reports/gemma4-31b-it-run.md` with:

- Your name and assignment.
- The exact generation code commit and model revision.
- Environment, GPU, precision, thinking/reasoning mode, and random-seed policy.
- All decoding values and conversation stopping rules.
- Pilot outcome, production counts per condition, and validation outcome.
- Runtime, observed memory, estimated/actual cost, and remaining problems.
- Retry/cap-hit summary and the location of privately backed-up raw diagnostics.

If some metadata was not recorded by the software, say so; do not invent it. Do not include Switchboard excerpts, tokens, or private keys.

After each completed condition, back up outputs off the Pod. When ready to submit, stage only your assigned outputs and report:

```bash
git status --short
git add data/generated_workshop/gemma4-31b-it/ reports/gemma4-31b-it-run.md
git diff --cached --stat
git diff --cached --name-only
git commit -m "data: add Aleen's gemma4-31b-it workshop generation results"
git push -u origin aleen/gemma4-generation
```

If you change shared code to fix a blocker, coordinate with Diyar and submit that fix separately. Do not stage everything with `git add -A`.

On GitHub, open a pull request from your branch into `main`. Include counts, code commit, model revision, environment, pilot/validation results, and known problems. Request Diyar's review. If a push is rejected, preserve your work and coordinate the branch update; do not force-push.

## 9. Finish the RunPod session

Confirm the generated files and report are backed up outside the Pod and pushed successfully. Keep restricted diagnostics in private storage. Then stop or terminate the GPU resource according to your storage setup. Check the RunPod dashboard: closing the terminal does not stop charges, and persistent storage can keep billing.

## Completion checklist

- [ ] GitHub collaborator access and your own authentication work.
- [ ] Shared code is pulled, CPU preflight passes, and the GPU pilot environment is verified.
- [ ] Metadata and two P2 examples pass the preflight check.
- [ ] Development pilot is reviewed and production settings are frozen.
- [ ] All six active C1/C2 conditions contain the same 50 production IDs.
- [ ] Your 300 active conversations pass basic checks and sample review.
- [ ] Your run report records versions, settings, failures, and cost.
- [ ] Outputs are backed up and a pull request is open.
- [ ] RunPod compute is shut down after backup.
- [ ] No A3/C3 or A4/C4 GPU jobs run while the agent review is pending.

## References

- [Project methods](methods.md)
- [Original prompt examples](../prompts/PROMPT_EXAMPLES.md)
- [Assigned model card](https://huggingface.co/google/gemma-4-31B-it)
- [RunPod storage documentation](https://docs.runpod.io/pods/storage/types)

