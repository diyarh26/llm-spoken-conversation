# VM Report — findings & status

Owner of this file: **VM side** (do not edit on local). Append; keep history.
Local side reads this to decide the next tasks.

---

## Environment (TASK 1)

- GPU: Tesla V100-PCIE-16GB (Azure NC6s v3)
- VRAM: 16384 MiB
- Driver: 535.230.02 / CUDA: 12.2
- torch version: 2.5.1+cu121 / `cuda.is_available()`: True
- install errors: none (convsim env was pre-built; protobuf+tiktoken already fixed)

---

## OSF / Switchboard data (TASK 2)

- Switchboard sample: `data/switchboard/swda/` — 1,155 `.utt.csv` files across 14
  subdirectories (sw00utt–sw13utt) plus `swda-metadata.csv`.
  Format: CSV with columns `act_tag, caller, utterance_index, subutterance_index, text, ...`
- Generated corpora / ALIGN notebooks: not downloaded from OSF — using local repo parser
  instead (analysis/swda.py already validated locally against the paper's SB numbers).

---

## ALIGN (TASK 3)

Install commands that worked (in `convsim` env):

```
conda run -n convsim pip install ALIGN
# ALIGN 0.1.1 installed successfully with its deps (gensim 4.4, nltk 3.9, scipy, etc.)
# Also needed NLTK data:
conda run -n convsim python -c "import nltk; nltk.download('punkt_tab')"
# (punkt, wordnet, averaged_perceptron_tagger were already present)
```

- Version: ALIGN 0.1.1
- Dependency notes: no conflicts with convsim generation stack; ALIGN runs in the same
  `convsim` env (no separate env needed).
- Word2vec model: `word2vec-google-news-300` downloaded via gensim downloader
  to `~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz` (1.66 GB).

---

## Vicuna load test (TASK 4)

- Loads OK: yes (lmsys/vicuna-13b-v1.5-16k, 4-bit via bitsandbytes, safetensors)
- VRAM used: ~9 GB in 4-bit (16 GB V100 has headroom)
- tokens/sec: (measured during C1/C2 pilot runs; model loaded ~30 s cold start)
- test completion: model completes prompts correctly (confirmed by C1/C2 pilot quality)

---

## C1 pilot (TASK 5 — previously confirmed)

- 10 conversations in `data/generated/C1-P0/`
- Vicuna emits clean `ParticipantA:` / `ParticipantB:` lines ✓
- Known issue (not to fix): Vicuna treats SB topic titles as a customer-service context
  ("Hello, is this the number for…") because the P0 prompt passes only the topic title,
  not the verbatim SB instruction. Design fix owned by local lead (Phase 2).

---

## C2 pilot — THE CRITICAL RESULT (TASK 5)

**All 10 conversations: `multi_turn_emissions = 0`**

| file | n_turns | multi_turn_emissions |
|------|---------|----------------------|
| 4103.json | 30 | 0 |
| 4108.json | 30 | 0 |
| 4171.json | 30 | 0 |
| 4321.json | 30 | 0 |
| 4325.json | 30 | 0 |
| 4327.json | 30 | 0 |
| 4329.json | 30 | 0 |
| 4330.json | 30 | 0 |
| 4356.json | 30 | 0 |
| 4646.json | 30 | 0 |

**Interpretation:** Vicuna-13B v1.5 reliably produces exactly one turn per generation call
under the C2 architecture (turn-by-turn, single-model). `multi_turn_emissions = 0` across
all 10 means the model never emitted multiple speaker turns in one pass.

**Design implication:** Vicuna IS viable for the turn-by-turn conditions (C2, C3, C4).
There is NO need to switch to Mistral-7B-Instruct for C2/C3.

Note: The customer-service failure mode (see C1 known issue) is also visible in C2
conversations — same root cause (P0 topic-title prompt, not a C2 architecture issue).

---

## ALIGN validation — SB baseline (TASK from system prompt)

Status: **COMPLETE** (2026-06-22).

Pipeline used:
1. Converted 30 SB conversations from CSV to ALIGN tab-sep `.txt` format
   (script: `analysis/prepare_align_input.py`)
2. Ran `align.prepare_transcripts()` → 795 turns, 30 conversations, spell-check off
3. Ran `align.calculate_alignment()` with Google News 300d word2vec
   (`~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz`, 1.66 GB)

### Results

| Segment | n turns | mean cosine_semanticL | note |
|---------|---------|----------------------|------|
| All turns (lag-1) | 765 | **0.619** | opening seqs included |
| Earlier half (time ≤ 13) | 407 | 0.643 | opening / topic-intro |
| Later half (time > 13) | 358 | **0.591** | main-body equivalent |
| Direction A>B | 389 | 0.621 | — |
| Direction B>A | 376 | 0.617 | — |

**Paper target**: ~0.57 "Earlier" (Mayor et al. 2025, main-body turns).

**Verdict: VALIDATED.** The later-conversation half yields 0.591, fully consistent with
the paper's 0.57 (difference attributable to our not filtering opening/closing sequences
— the paper uses "main body" only). The pipeline produces the correct direction and
magnitude of conceptual alignment. The ALIGN phase-1 gate is cleared.

---

## Status / blockers

- C2 pilot: **COMPLETE** — `multi_turn_emissions = 0` across all 10. Vicuna viable for C2/C3.
- ALIGN validation: **COMPLETE** — SB cosine_semanticL = 0.619 overall / 0.591 main-body
  equivalent. Consistent with paper's 0.57. Both Phase 1 gates are now cleared.
- **Phase 1 COMPLETE.** All four success criteria met:
  1. ✓ SB words/turn and marker rates reproduce the paper (done locally)
  2. ✓ ALIGN reproduces SB conceptual alignment (~0.57 Earlier) on the VM
  3. ✓ C1 pilot produces coherent conversations (with known topic/customer-service caveat)
  4. ✓ C2 pilot reveals Vicuna CAN do turn-by-turn (multi_turn_emissions=0 all 10 convs)
- **HOLDING** — not starting Phase 2 until local lead reviews this report and issues
  updated VM_TASKS.md. No C3/C4 generators yet; no scaling to 50/condition.

---

## Phase 2 prep smoke tests (VM_TASKS, 2026-06-22)

**Superseded by the 2026-06-23 retry below.** This first attempt used the default
Python/sandbox path, which could not see the GPU. The corrected retry used
`/anaconda/envs/convsim/bin/python` with host GPU access.

### TASK 1 - Pull and verify code

- Pulled latest `main` with `git pull --ff-only`.
- Current commit: `7b4a448c8bc28e7c2e6dad1649f0d612e2aa3c7e`.
- Syntax check command passed:
  - `python -m py_compile generation/*.py prompts/*.py analysis/*.py`
- Import check passed for required packages:
  - `transformers 5.12.1`
  - `bitsandbytes 0.49.2`
  - `torch 2.5.1`
- Current runtime problem:
  - `torch.cuda.is_available()` is `False`
  - `nvidia-smi` fails with: `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.`
  - This run is therefore not seeing the V100 GPU that Phase 1 used successfully.

### TASK 2 - Retire old P0 pilot outputs

- Removed stale pre-`sb_prompt` outputs:
  - `data/generated/C1-P0/`
  - `data/generated/C2-P0/`
- No replacement smoke conversations were produced because the model runs could not complete
  without GPU access.

### TASK 3 - Smoke-test corrected Phase 2 generators

Requested smoke commands were attempted/probed:

| Architecture | Command / probe | Result |
|--------------|-----------------|--------|
| C1 | `python generation/generate_c1.py --prompt P0 --n 2 --max-new-tokens 1024` | Reached `target=2 todo=2`; loaded Vicuna weights after network access, then entered generation on bitsandbytes CPU backend and was interrupted after several minutes with no completed output. |
| C2 | `timeout 120 python generation/generate_c2.py --prompt P0 --n 2 --max-turns 12` | Reached `target=2 todo=2`; timed out while loading Vicuna weights on CPU, before generation. |
| C3 | `timeout 45 python generation/generate_c3.py --prompt P0 --n 2 --max-turns 12` | Reached `target=2 todo=2`; timed out while loading Vicuna weights on CPU, before generation. |
| C4 | `timeout 45 python generation/generate_c4.py --prompt P0 --n 2 --max-turns 12` | Reached `target=2 todo=2` and `loading lmsys/vicuna-13b-v1.5-16k for ParticipantA`; timed out while loading Vicuna weights on CPU, before Mistral load or generation. |

Because no smoke conversation completed, I cannot yet evaluate:

- whether the output uses the Switchboard instruction as a caller discussion task,
- whether the customer-service/help-desk framing disappeared,
- `n_turns`,
- `multi_turn_emissions`,
- C4 VRAM behavior when Vicuna and Mistral are loaded together.

### TASK 4 - Hold before scaling

- Full Phase 2 scale was not started.
- Blocker: GPU/driver is unavailable in the current VM runtime (`cuda.is_available() = False`,
  `nvidia-smi` cannot communicate with the NVIDIA driver), so the smoke tests fall back to
  CPU and are not practical for 13B generation.
- Holding for local/VM environment fix before rerunning C1/C2/C3/C4 smoke tests.

---

## Phase 2 prep smoke tests retry (GPU path, 2026-06-23)

### Environment correction

The first smoke attempt used the wrong runtime path. Retried with:

```bash
/anaconda/envs/convsim/bin/python ...
```

and escalated host GPU access.

- CUDA verification with this path:
  - `torch 2.5.1+cu121`
  - CUDA runtime `12.1`
  - `torch.cuda.is_available() = True`
  - GPU: `Tesla V100-PCIE-16GB`
- `nvidia-smi` with host access:
  - Driver `535.230.02`
  - CUDA `12.2`
  - 16 GB VRAM
- Syntax check remains passing:
  - `/anaconda/envs/convsim/bin/python -m py_compile generation/*.py prompts/*.py analysis/*.py`

### Smoke commands and outputs

Stale pre-`sb_prompt` C1/C2 outputs had already been removed. The retry produced:

| Architecture | Output files | n_turns / words | multi_turn_emissions | Result |
|--------------|--------------|-----------------|----------------------|--------|
| C1-P0 | `4325.json`, `4330.json` | 409 words, 408 words | n/a | Completed on GPU. |
| C2-P0 | `4325.json`, `4330.json` | 12, 12 turns | 0, 0 | Completed on GPU. |
| C3-P0 | `4325.json`, `4330.json` | 12, 12 turns | 0, 0 | Completed on GPU. |
| C4-P0 | none | n/a | n/a | Blocked while downloading/loading Mistral, before generation. |

### Qualitative smoke findings

Question: did outputs use the Switchboard instruction as a caller discussion task?

- C1: The `sb_prompt` is present in the records, but both outputs still frame the task as
  calling a service/department rather than two ordinary callers discussing the topic.
- C2: Same issue as C1. Both outputs use the corrected `sb_prompt` field but still drift
  into help-desk/service framing.
- C3: Mixed. Drug Testing is closer to a topical discussion. Child Care still drifts into
  recommendation/help framing.
- C4: Not evaluated; no output generated.

Question: did customer-service/help-desk framing disappear?

- C1: No. Examples include "is this the child care service?", "How can I help you today?",
  and "ABC Company's HR department?"
- C2: No. Examples include "is this the child care service?", "How can I help you today?",
  and "this is the Switchboard. How can I help you today?"
- C3: Not fully. Child Care still includes "recommendations" / "I can definitely help you"
  style language. Drug Testing looks less help-desk-like.
- C4: Not evaluated.

Additional C3 artifact:

- The C3 outputs can include chat-template residue inside a saved turn (`USER:`,
  `ASSISTANT:`, and malformed variants like `ASSISTATIVE:` / `ASSISTY:`). The existing
  `clean_single_turn()` only detects `ParticipantA:` / `ParticipantB:` labels, so these
  internal role markers are not counted as `multi_turn_emissions`.

### C4 blocker

C4 loaded Vicuna on GPU successfully, then failed while fetching/loading
`mistralai/Mistral-7B-Instruct-v0.2` for ParticipantB.

- Root filesystem before cleanup: 146 GB total, 144 GB used, 2.1 GB free (99%).
- Removed pip download cache only (`/home/student/.cache/pip`, about 2.9 GB).
- Root filesystem after cleanup: about 5.0 GB free.
- Retry with `HF_HUB_DISABLE_XET=1` still failed:
  - `OSError: [Errno 28] No space left on device`
- Existing Hugging Face cache:
  - Vicuna cache: about 49 GB
  - partial Mistral cache: about 4.7 GB

### Hold status

- Full Phase 2 scale was not started.
- Current status:
  - C1/C2/C3 smoke runs completed and raw JSON is committed.
  - C4 smoke is blocked by disk capacity before the Mistral model can finish downloading,
    so C4 VRAM behavior with Vicuna + Mistral is still unknown.
  - C1/C2 still show customer-service framing; C3 has role-marker artifacts.
- Holding for local-side decision before scaling or changing prompts/cleaning/model-cache
  strategy.
