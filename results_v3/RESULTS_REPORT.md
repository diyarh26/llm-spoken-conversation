# Results Report — final dataset (`data/generated_v3`)

**Question:** how does the generation architecture shape the structure of LLM conversations,
compared with real human telephone calls (Switchboard)?

- **Architectures:** C1 one model writes the whole conversation at once · C2 one model, turn
  by turn · C3 two separate Vicuna agents · C4 Vicuna ↔ Mistral agents.
- **Prompts:** P0 the paper's basic prompt · P1 spoken register + persona cards · P2 = P1 + a
  real Switchboard excerpt as an example.
- **Data:** 12 conditions × 50 conversations = 600. All 12 conditions use the same 50 topics.
- **Human reference:** the 50 Switchboard conversations those topics come from
  (`generation/target_ids.json`). They reproduce the paper's human numbers: 14.4 words/turn
  (paper 13.97) and uh-huh 1.03 per 100 words (paper 1.03).

First version by Sham (2026-10-03: metrics, real ALIGN, expanded backchannel analysis).
Re-run on 2026-10-05 after the cleaning and fixes listed in §6. Every number below is
reproducible with the commands in §7.

---

## 1. Headline — LLM conversations are missing the listener

Dialogue acts (what each unit of talk *does*) are tagged with DialogTag. Humans are tagged
with the same tagger, so the comparison is like-for-like. Coarse act mix, % of units:

| | Statement | Opinion | **Backchannel** | Yes/no Q | Open Q | Agreement |
|---|---|---|---|---|---|---|
| **Humans (tagger)** | 46.0 | 13.6 | **25.1** | 3.2 | 1.7 | 2.9 |
| C1-P0 / P1 / P2 | 31–33 | 21–22 | **7.7–9.4** | 3.7–6.1 | 9.7–11.5 | 9–14 |
| C2-P0 / P1 / P2 | 26–31 | 32–48 | **1.2–5.1** | 4.2–6.3 | 2.6–17.0 | 4–9 |
| C3-P0 / P1 / P2 | 26–31 | 31–35 | **2.8–5.2** | 5.9–8.9 | 8.0–11.1 | 5–8 |
| C4-P0 / P1 / P2 | 30–41 | 31–40 | **1.1–1.7** | 5.9–7.7 | 6.4–8.8 | ~2 |

**Standalone backchannels**, i.e. a unit that is *only* "uh-huh / yeah / right": humans
**19.1%**, every LLM condition **0.0–0.4%**. The tagger's 8–9% for C1 comes from turns that
*start* with "Yeah, …" and then carry on. These are reaction words, not listener turns.
Sham's lexicon measure agrees: a backchannel is its own turn **51%** of the time for humans
and **0%** in all 12 LLM conditions.

**Distance from humans** (Jensen–Shannon divergence, 0 = identical; lower = more human).
Every condition is far above the noise floor (≈0.001, the distance between two random sets
of human conversations):

| | Act mix (coarse) | Act order (weighted†) | Assistant-style units |
|---|---|---|---|
| C1-P0 | 0.131 | 0.163 | 2.6% |
| C1-P1 | **0.123** | 0.151 | 1.6% |
| C1-P2 | 0.129 | **0.146** | 2.5% |
| C2-P0 | 0.244 | 0.252 | 7.2% |
| C2-P1 | 0.167 | 0.200 | 3.1% |
| C2-P2 | 0.188 | 0.230 | 2.9% |
| C3-P0 | 0.189 | 0.216 | 7.3% |
| C3-P1 | 0.153 | 0.174 | 3.8% |
| C3-P2 | 0.198 | 0.214 | 4.2% |
| C4-P0 | 0.183 | 0.217 | 2.5% |
| C4-P1 | 0.223 | 0.230 | 3.3% |
| C4-P2 | 0.185 | 0.220 | 2.5% |

† Each act's row is weighted by how often that act occurs. The unweighted score
(`da_jsd_vs_sb.csv`) lets acts that occur only 3–6 times dominate; that alone produced C2-P2's
0.416 outlier (0.230 weighted). See `analysis/transition_weighted.py`.

**Reading:** C1 (all-at-once) is closest to humans. Splitting the speakers into separate
agents (C3, C4) does **not** move the structure closer to human talk. It replaces
listening with opinions and questions. The prompt matters less than the architecture: P1 is
usually best, and P0 is worst for C2/C3.

**Robustness:** without echo-loop turns, and without the 10 language-drift conversations,
every score moves by ≤0.01 (`dialogue_acts_no_loops/`, `dialogue_acts_no_drift/`).
**Caveat:** the tagger is 72% accurate against the human gold labels (coarse). A team
hand-check of ~100 LLM units is still pending.

---

## 2. The paper's metrics — humans vs all 12 conditions

Per-conversation means (`results_v3/metrics_summary.csv`):

| Corpus | Words/turn | IQR | Turns/conv | oh | okay | uh-huh | BC rate /100w | BC standalone | BC types /8 |
|---|---|---|---|---|---|---|---|---|---|
| **Switchboard (human)** | **14.4** | 18.2 | 105.6 | 0.66 | 0.22 | **1.03** | 4.24 | **51%** | **5.66** |
| C1-P0 | 18.3 | 11.2 | 20.3 | 0.12 | 0.04 | 0.00 | 0.94 | 0% | 1.54 |
| C1-P1 | 16.7 | 10.3 | 21.0 | 0.46 | 0.13 | 0.00 | 2.47 | 0% | 2.98 |
| C1-P2 * | 18.2 | 12.2 | 21.7 | 0.35 | 0.07 | 0.00 | 2.37 | 0% | 2.88 |
| C2-P0 | 64.8 | 30.6 | 33.3 | 0.02 | 0.00 | 0.00 | 0.37 | 0% | 1.74 |
| C2-P1 | 12.2 | 8.6 | 39.6 | 0.07 | 0.00 | 0.00 | 2.11 | 0% | 1.76 |
| C2-P2 * | 11.8 | 8.4 | 39.0 | 0.04 | 0.00 | 0.00 | 2.56 | 0% | 1.80 |
| C3-P0 | 47.4 | 35.6 | 18.4 | 0.04 | 0.00 | 0.00 | 0.31 | 0% | 0.96 |
| C3-P1 | 39.3 | 34.1 | 17.7 | 0.11 | 0.01 | 0.00 | 0.79 | 0% | 1.96 |
| C3-P2 * | 41.2 | 31.8 | 16.8 | 0.05 | 0.00 | 0.00 | 0.56 | 0% | 1.50 |
| C4-P0 | 103.3 | 93.3 | 19.3 | 0.03 | 0.00 | 0.00 | 0.23 | 0% | 1.44 |
| C4-P1 | 108.6 | 165.0 | 11.3 | 0.12 | 0.02 | 0.00 | 0.52 | 0% | 1.86 |
| C4-P2 * | 95.0 | 83.7 | 15.0 | 0.06 | 0.01 | 0.00 | 0.34 | 0% | 1.78 |

\* The P2 prompt shows the model a real Switchboard excerpt, so P2 marker rates
(oh/okay/uh-huh, BC) partly reflect what it was shown. Even so, uh-huh stays at 0.

**Statistics** (`results_v3/stats_report.txt`):
- Words/turn differ by architecture at every prompt level (Kruskal–Wallis p < 10⁻²⁷).
- Mixed models (words ~ corpus × section + 1|conversation) find every condition significantly
  different from humans **except C1-P1** (+1.1 words, p = 0.35).
- **uh-huh = 0 in all 12 conditions.** oh and okay are far below human levels everywhere
  (p < 0.05 for every P0/P1 condition).

---

## 3. Conceptual alignment (real ALIGN, Duran et al. 2019)

How similar in meaning each turn is to the previous one (word2vec-google-news-300, as in
the paper). 17,947 turn pairs. `results_v3/alignment_checks.txt`.

| | Raw mean | vs human, raw | **vs human, length-controlled** | Earlier → Later | Wilcoxon p |
|---|---|---|---|---|---|
| **Switchboard** | **0.550** | — | — | 0.562 → 0.558 | 0.89 (no change) |
| C1-P0 | 0.608 | +0.06 | **+0.03** | 0.598 → 0.623 | 0.007 |
| C1-P1 | 0.607 | +0.06 | **+0.05** | 0.596 → 0.630 | 0.005 |
| C1-P2 | 0.628 | +0.08 | **+0.05** | 0.621 → 0.642 | 0.040 |
| C2-P0 | 0.888 | +0.34 | **+0.10** | 0.849 → 0.911 | <0.001 |
| C2-P1 | 0.516 | −0.03 | **0.00** (p=0.84) | 0.512 → 0.519 | 0.50 |
| C2-P2 | 0.510 | −0.04 | **0.00** (p=0.79) | 0.497 → 0.523 | 0.056 |
| C3-P0 | 0.772 | +0.22 | **+0.06** | 0.741 → 0.760 | 0.049 |
| C3-P1 | 0.747 | +0.20 | **+0.08** | 0.713 → 0.765 | 0.001 |
| C3-P2 | 0.768 | +0.22 | **+0.08** | 0.756 → 0.774 | 0.093 |
| C4-P0 | 0.872 | +0.32 | **+0.04** | 0.813 → 0.887 | <0.001 |
| C4-P1 | 0.893 | +0.34 | **+0.02** (p=0.12) | 0.843 → 0.876 | 0.004 |
| C4-P2 | 0.866 | +0.32 | **+0.02** | 0.801 → 0.869 | <0.001 |

**What this does and doesn't show:**
1. **Most of the raw gap is turn length.** Alignment rises about 0.08 per log-word for
   every corpus: averaged vectors of long turns look alike. Once length is controlled,
   C4's +0.32 shrinks to +0.02–0.04. C4 is *not* strongly "over-aligned". It just talks in
   very long turns. The architectures that stay over-aligned are **C2-P0 (+0.10, echo
   loops; still +0.09 with the loop turns removed)** and **C3 (+0.06 to +0.08)**.
2. **Earlier → Later is the cleanest alignment result.** Humans show no change (p = 0.89).
   The LLMs rise in 9 of 12 conditions (p < 0.05). It is not caused by the polite closings:
   with the last 2 turn pairs removed the rise never shrinks, and in 11 of 12 conditions it
   gets larger (`results_v3/alignment_earlier_later.csv`). The LLM speakers converge on each other's
   wording over the conversation; humans don't.
3. We do **not** label this "sycophancy". The measure shows semantic similarity, not
   agreement.

---

## 4. Data quality — reported, not patched

| Condition | Echo-loop turns* | Conversations with a loop | Language drift (convs) |
|---|---|---|---|
| C1-P0 / P1 / P2 | 2.5% / 0.1% / 1.5% | 5 / 1 / 5 | 0 |
| C2-P0 | **16.0%** | 37 / 50 | 0 |
| C2-P1 / P2 | 1.1% / 1.0% | 14 / 14 | 0 |
| C3-P0 / P1 / P2 | 5.0% / 1.0% / 5.4% | 5 / 4 / 5 | 4 / 0 / 3 |
| C4-P0 / P1 / P2 | 2.1% / 1.2% / 1.3% | 5 / 1 / 2 | 3 / 0 / 0 |

\* A turn of ≥8 words whose word set overlaps ≥80% with an earlier turn: the same definition
the generator's loop guard uses (`generation/quality.py`). This replaces the earlier
44% / 0% figures, which no code in the repo reproduces.

Language drift = Vicuna appends "번역결과" ("translation result") plus a Korean translation,
and the agents then continue partly in Korean. Turns are cut where the Korean starts; the
English before it is kept.

---

## 5. Poster story in one paragraph

LLM-simulated phone calls lack the listener. Humans spend a quarter of their talk on
backchannels ("uh-huh", "yeah"); every LLM setup produces almost none as standalone
turns and fills the space with opinions and questions. Generating the whole conversation
at once (C1) gets closest to human structure. Giving each speaker its own agent (C3, and
with two different models C4) makes turns 3–7× longer and *further* from human talk, more
like an exchange of messages. LLM speakers also converge on each other's wording as the
conversation goes on, which humans don't. Better prompts (P1/P2) help, but less than the
choice of architecture.

---

## 6. What changed since the 2026-10-03 version

1. **Scripted "Hello!/Hello!" openings were counted** for C2–C4 (not C1) in
   `evaluate_generated.py`. All metrics now use the shared cleaner (`conversation_turns`).
2. **Artifacts cleaned (same rule for all 12):** bracketed stage directions such as "(Turn
   14)" and "[Your turn to respond…]" (88 patterns, all checked by hand, none speech; C1
   unchanged, C2 1 turn, C3/C4 287 turns), plus the Korean drift.
3. **Human sample:** the 50 topic-matched Switchboard conversations, not the first 50 files.
   This, not "main body vs whole conversation", explains the old gap from the paper.
4. **`stats.py` mixed models** reported β = 0, p = 1 because the optimizer stalled
   (`method="lbfgs"`). Fixed.
5. **Alignment re-run** on the cleaned turns and the matched human sample. Length control
   and the significance tests were added.
6. Repetition figures replaced (§4). The "sycophancy" interpretation was dropped (§3).

## 7. Reproduce

```
python analysis/dialogue_acts.py --generated-root data/generated_v3 --out-dir results_v3/dialogue_acts   # + --exclude-loops / --exclude-drift
python analysis/transition_weighted.py results_v3/dialogue_acts
python analysis/evaluate_generated.py --data_dir data/generated_v3 --out_dir results_v3 --switchboard
python analysis/stats.py --data-dir data/generated_v3 > results_v3/stats_report.txt
python analysis/export_align.py --data-dir data/generated_v3 --include-sb --loop-sensitivity C2-P0 C3-P0 C3-P2 --pretrained-vectors <GoogleNews-vectors-negative300.bin>
python analysis/alignment_checks.py --out results_v3
```

Only numbers are committed. Switchboard text never is (LDC license).
