# Advisor checklist: decisions with evidence

All numbers for the published (Vicuna) conditions come from the committed results in
`results/`; Qwen3.5-9B numbers from `results_workshop/`.

## 1. Alignment after length control: keep the figure, report corrected numbers

Source: `results/alignment_checks.txt` (validated ALIGN run; differences vs Switchboard,
OLS with conversation-clustered SEs controlling for log turn length, and a 5-40-word
matched comparison; Switchboard matched mean 0.572).

| Condition | Raw diff | Length-adjusted diff | p | Matched 5-40w mean |
|---|---|---|---|---|
| C1-P0 / P1 / P2 | +0.058 / +0.057 / +0.078 | +0.032 / +0.046 / +0.052 | <=0.001 | 0.607 / 0.613 / 0.630 |
| C2-P0 | +0.339 | +0.103 | <0.001 | 0.719 |
| C2-P1 / P2 | -0.033 / -0.039 | +0.002 / +0.003 | 0.84 / 0.79 | 0.543 / 0.536 |
| C3-P0 / P1 / P2 | +0.22 / +0.20 / +0.22 | +0.057 / +0.078 / +0.082 | <0.001 | 0.653 / 0.698 / 0.701 |
| C4-P0 / P1 / P2 | +0.32 / +0.34 / +0.32 | +0.039 / +0.016 / +0.023 | <0.001 / 0.12 / 0.013 | 0.705 / 0.751 / 0.656 |

Decision: LLMs still align more than humans after length control in 9 of 12 conditions,
so the figure stays, with the length-adjusted numbers. The difference is about a third of
the raw gap, and it disappears for C2-P1/P2 (single writer, turn by turn) and is not
significant for C4-P1 (only 18 length-matched pairs). Qwen3.5-9B (combined ALIGN run,
`results_workshop/alignment_combined/`): every condition aligns more than humans after
length control (+0.09 to +0.17, all p < 0.001).

## 2. Dialogue acts with statement and opinion merged: the result holds

Source: `results/dialogue_acts/da_distribution_coarse_by_condition.csv` (tagger-labelled
human vs tagger-labelled LLM, the primary comparison). Jensen-Shannon distance (log2)
from humans; the recomputation reproduces the published coarse JSDs exactly before merging.

| | P0 | P1 | P2 | Mean |
|---|---|---|---|---|
| C1 original / merged | 0.131 / 0.117 | 0.123 / 0.110 | 0.129 / 0.115 | 0.128 / **0.114** |
| C2 original / merged | 0.244 / 0.158 | 0.167 / 0.122 | 0.188 / 0.149 | 0.200 / 0.143 |
| C3 original / merged | 0.189 / 0.143 | 0.153 / 0.115 | 0.198 / 0.144 | 0.180 / 0.134 |
| C4 original / merged | 0.183 / 0.149 | 0.223 / 0.163 | 0.185 / 0.162 | 0.197 / **0.158** |

Decision: C1 is closest to humans at every prompt level, with and without merging. With
statement and opinion merged, the two-model agents (C4) are the farthest on average and at
P1 and P2; at P0, C2 is farthest (its echo loops). Suggested wording: "the result holds
even when statement and opinion are merged: C1 remains closest to human dialogue-act use
and the C4 agents remain farthest (except at P0, where C2's echo loops dominate)".
Merging shrinks every distance, most for C2 (opinion over-use was a large part of its gap).

## 3. Memorisation: no evidence

Source: `analysis/memorisation.py` (8-gram overlap with all 1,155 SwDA conversations,
excluding n-grams of the P2 excerpts shown in the prompt; leave-one-out human baseline).

| | Mean overlap | Max |
|---|---|---|
| Human baseline (real call vs all other calls) | 0.08% | 0.72% |
| Vicuna, 12 conditions | 0.04-0.41% | 3.8% |
| Qwen3.5-9B, C1/C2 x P0-P3 | 0.00-0.06% | 2.6% |
| Copying from the P2 excerpts (Qwen) | 0.00% | 0.00% |

Matches are generic phrases ("what do you think about the idea of"), not recalled
passages. Decision: no memorisation concern; report the numbers.

## Qwen3.5-9B dialogue acts (C1/C2, P0-P3)

Source: `results_workshop/qwen35-9b/dialogue_acts/` (same DialogTag tagger and human
reference as the published run; the tagger's human distribution and its gold-vs-tagger
calibration, JSD 0.074 fine / 0.035 coarse, reproduce the published values exactly).

Coarse JSD from humans (tagger vs tagger), original / statement+opinion merged:

| | P0 | P1 | P2 | P3 | Mean |
|---|---|---|---|---|---|
| Qwen C1 | 0.100 / 0.081 | 0.057 / 0.057 | 0.038 / 0.038 | 0.054 / 0.053 | 0.062 / **0.057** |
| Qwen C2 | 0.208 / 0.170 | 0.161 / 0.142 | 0.152 / 0.140 | 0.120 / 0.107 | 0.160 / **0.140** |
| Vicuna C1 (published) | 0.131 | 0.123 | 0.129 | - | 0.128 |
| Vicuna C2 (published) | 0.244 | 0.167 | 0.188 | - | 0.200 |

C1 is closer to humans than C2 at every prompt level, with and without merging; the
noise floors (95th percentile, coarse) are <= 0.0014, far below every distance.
