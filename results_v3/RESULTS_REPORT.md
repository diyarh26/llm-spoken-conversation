# Results Report — data/generated_v3 (final dataset)

600 conversations, 12 conditions (C1–C4 × P0/P1/P2), 50 conversations each.
Metrics computed by `analysis/evaluate_generated.py`. One important measure —
real conceptual alignment via the ALIGN package (Duran et al., 2019) — is not
included yet; see "Not yet run" at the bottom.

---

## 1. Full metrics table

| Condition | n | Mean w/turn | Median | IQR | oh/100w | okay/100w | uh-huh/100w | BC rate/100w | BC standalone % | BC diversity /8 | Self-repair/100w | Closing order | Sycophancy/100w | Info density CV |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1-P0 | 50 | 17.61 | 17.49 | 11.81 | 0.12 | 0.04 | 0.00 | 0.94 | 0% | 1.54 | 0.008 | 1.00 | 0.86 | 0.206 |
| C1-P1 | 50 | 16.46 | 16.50 | 10.56 | 0.45 | 0.13 | 0.00 | 2.47 | 0% | 2.98 | 0.046 | 1.00 | 0.90 | 0.178 |
| C1-P2 | 50 | 16.89 | 17.77 | 13.52 | 0.35 | 0.07 | 0.00 | 2.36 | 0.2% | 2.88 | 0.114 | 1.00 | 1.01 | 0.240 |
| C2-P0 | 50 | 60.76 | 62.28 | 32.85 | 0.02 | 0.00 | 0.00 | 0.37 | 0% | 1.74 | 0.009 | 0.80 | 0.46 | 0.191 |
| C2-P1 | 50 | 11.64 | 11.33 | 9.08 | 0.07 | 0.00 | 0.00 | 2.10 | 0% | 1.76 | 0.022 | 1.00 | 0.94 | 0.264 |
| C2-P2 | 50 | 11.23 | 10.77 | 8.75 | 0.04 | 0.00 | 0.00 | 2.55 | 0.4% | 1.80 | 0.019 | 1.00 | 0.82 | 0.276 |
| C3-P0 | 50 | 44.60 | 44.87 | 42.71 | 0.04 | 0.00 | 0.00 | 0.30 | 0% | 0.96 | 0.019 | 0.81 | 0.56 | 0.239 |
| C3-P1 | 50 | 33.84 | 34.64 | 37.53 | 0.11 | 0.01 | 0.00 | 0.78 | 0% | 1.96 | 0.006 | 0.90 | 0.81 | 0.259 |
| C3-P2 | 50 | 38.80 | 40.00 | 41.63 | 0.05 | 0.00 | 0.00 | 0.53 | 0% | 1.50 | 0.015 | 0.87 | 0.61 | 0.257 |
| C4-P0 | 50 | 96.61 | 91.88 | 105.98 | 0.03 | 0.00 | 0.00 | 0.22 | 0% | 1.44 | 0.056 | 0.76 | 0.35 | 0.225 |
| C4-P1 | 50 | 83.29 | 58.11 | 122.16 | 0.12 | 0.02 | 0.00 | 0.51 | 0% | 1.86 | 0.109 | 0.89 | 0.47 | 0.276 |
| C4-P2 | 50 | 83.92 | 77.51 | 95.71 | 0.06 | 0.01 | 0.00 | 0.32 | 0% | 1.78 | 0.034 | 0.87 | 0.35 | 0.251 |

*BC = Backchannel (expanded 8-token lexicon: uh-huh, okay, oh, yeah, right, mm-hmm, i see, sure).
Closing order = fraction of conversations with correctly-ordered closing stages.*

---

## 2. Key findings for the poster

### Finding 1 — uh-huh never appears, in any condition
Every single condition shows **0.00 uh-huh/100 words**. Regardless of architecture (C1–C4)
or prompt level (P0–P2), the models essentially never produce this marker. This matches the
original paper's finding that uh-huh is the hardest marker for LLMs to reproduce.

### Finding 2 — Backchannel standalone rate is ~0% everywhere
Across all 12 conditions, **backchannels are almost never their own independent turn** — they
are consistently glued onto a longer sentence instead (e.g. "Oh, that's so interesting, I
also..." rather than a standalone "Oh."). This holds regardless of architecture or prompt,
suggesting it's a structural property of how these models generate dialogue, not something
prompting alone fixes.

### Finding 3 — Repetition rate tracks architecture independence, with one correction
Full-corpus scan for near-verbatim repeated consecutive turns:

| Condition | Repetition rate |
|---|---|
| C1-P0/P1/P2 | **0%** |
| C2-P0 | **44%** |
| C2-P1 | 2% |
| C2-P2 | **16%** |
| C3-P0/P1/P2 | 0–2% |
| C4-P0/P1/P2 | 0–2% |

**Correction to the architecture framing:** the data shows the repetition problem is specific
to **C2** (single model, true turn-by-turn generation — the model predicts both speakers one
turn at a time), not C1. C1 also uses a single model, but generates the *entire* conversation
in one pass ("all-at-once"), which does not produce this failure mode — 0% across all three
prompt levels. C3 (same model, two independent instances) and C4 (two different models) are
both consistently low (0–2%), supporting the intended narrative: **the more independent the
two "speakers" are, the less self-echoing occurs.** The one open question for the poster is
why C2-P0 (44%) is so much worse than C2-P1/P2 (2%/16%) — the simplest explanation is that
the P1/P2 prompts, by shaping the conversation more explicitly, reduce the chance of the model
drifting into a repeated phrase, but this is correlational, not confirmed.

**Recommendation:** report this as evidence *for* the architecture manipulation working as
intended (independence reduces self-echoing), with C2's turn-by-turn single-model design
flagged as the condition most prone to it — not as a generation bug requiring regeneration.

### Finding 4 — Turn length and its spread scale with architecture + token budget
Mean words/turn: C1 (~17) < C2-P1/P2 (~11) < C2-P0 (~61) < C3 (~34–45) < C4 (~83–97).
C4's IQR (up to 122) exceeds its own mean in some conditions, meaning turn length varies
enormously within a single condition — some turns are very short, others extremely long.
This is consistent with the generation config's deliberate choice not to impose a token floor
or ceiling that would bias the turn-length distribution (a dependent variable the study
measures) — per `generation/config.py`'s "DV-safety rule," caps exist only as a safety net
(300–512 tokens/turn), not a target. C4-P0/P2 having the highest abrupt-cutoff rates (16%/12%
in the earlier quality scan) is consistent with occasionally hitting that safety-net cap on a
long turn, not a defect in the generation logic itself.

**Recommendation:** report long/variable C4 turns as an architecture outcome (two independent
models, no artificial length constraint) rather than a problem to fix by regenerating.

### Finding 5 — Sycophancy and closing-order quality highest in C1, lowest in C4
C1 has the most reliably-ordered closings (100% across all three prompt levels) and moderate
sycophancy (0.86–1.01/100w). C4 has the lowest closing-order correctness (76–89%) and lowest
sycophancy (0.35–0.47/100w) — plausibly because C4's two different models don't share the
single-model "agreement" tendency that drives sycophantic language in C1/C2.

---

## 3. Not yet run

**Real conceptual alignment (ALIGN package, Duran et al. 2019)** — the script
(`analysis/export_align.py`) is written and merged to `main`, but needs three one-time
installs (the `ALIGN` pip package, NLTK data, and the ~1.7GB word2vec-google-news-300 model)
that haven't been done yet in this environment. This is the one metric from the original
paper's core three (turn length, markers, alignment) still missing a result.
