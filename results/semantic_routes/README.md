# Semantic Route Reuse: Provisional `generated_v2` Findings

These are development indicators from the current `data/generated_v2` corpus, not final
project claims. Dialogue acts remain the headline analysis. The generated corpus is expected
to be replaced after the generation issues are resolved, and every table must then be rerun.

## What was run

The canonical run represents each call by eight ordered 20-word landmarks, uses
`all-mpnet-base-v2`, and compares nearest-sibling routes only within authoritative
Switchboard topic. Human reuse is calibrated with 2,000 equally sized subsets from the full
same-topic Switchboard pool. It analyzed all 600 generated calls and 490 human calls.

Two major robustness runs are also present:

- `../semantic_routes_robust_minilm/`: the same landmark analysis with
  `all-MiniLM-L6-v2`;
- `../semantic_routes_robust_full_coverage/`: MPNet with 12 all-content progress bins,
  word-weighted microchunk pooling, a 120-word validity floor, and DTW band 2.

## Robust overall indicator

All 12 conditions have more within-topic nearest-sibling route reuse than equally sized
human groups under all three analyses.

- Canonical MPNet excess reuse ranges from `0.145` (C2-P2) to `0.304` (C3-P1).
- MiniLM excess reuse ranges from `0.123` to `0.274`; its condition ordering agrees closely
  with MPNet landmarks (Spearman `rho = 0.93`).
- Full-coverage MPNet excess reuse ranges from `0.082` to `0.220` despite using a radically
  different transcript representation.
- Every condition-level topic-bootstrap interval is above zero in all three runs.

This supports the narrow provisional statement that same-topic LLM calls in every tested
condition follow more reusable semantic paths than the human reference. It does not yet
establish a population claim because `generated_v2` is provisional and there are only seven
common primary topics.

## Architecture-by-prompt signal that survives robustness

The evidence does **not** support a simple C1-to-C4 ordering. The clearest stable pattern is
specific to C2, the one-model turn-by-turn architecture:

| Comparison (right minus left) | MPNet landmarks | MiniLM landmarks | MPNet full coverage |
|---|---:|---:|---:|
| C2-P0 -> C2-P1 | `-0.092 [-0.148, -0.038]` | `-0.088 [-0.144, -0.033]` | `-0.103 [-0.143, -0.057]` |
| C2-P1 -> C3-P1 | `+0.131 [0.085, 0.168]` | `+0.122 [0.080, 0.154]` | `+0.052 [0.015, 0.090]` |

Thus the P1 spoken-conversation prompt reduces route stereotypy for C2, while C3-P1 is more
route-repetitive than C2-P1. The analogous C2-to-C3 pattern also appears under P2, but P2 is
exploratory because it contains a real Switchboard exemplar.

## What is not robust

The detailed 12-condition ordering changes under full coverage (landmark/full-coverage
Spearman `rho = 0.035`). In particular, the P0 architecture ordering and the apparent P1
increases for C3/C4 depend on representation. They must not be reported as discoveries.

The branching curve remains positive through the body of the call, so the broad LLM/human
gap is not only an opening/closing effect. Those are descriptive pointwise intervals, not a
test of a precise branching-onset time.

## Architecture Route Displacement

Matched-input architecture displacement is large (`0.524` to `0.647`), but same-input
similarity is usually almost the same as different-input similarity from the same topic.
The same-input advantage is only about `-0.014` to `+0.017`. With one stochastic output per
input, this cannot separate architecture change from generation noise. Final regeneration
should create at least two outputs per condition/input so between-architecture displacement
can be calibrated against within-architecture replicate displacement. Every C4 comparison
also changes the second model and is therefore architecture-plus-model.

## Qualitative validity check

The highest-scoring generated pair is C1-P0 Recycling, IDs 4096 and 4358
(`route_similarity = 0.662`). Both calls follow the same ordered outline: local recycling
provision, weak participation/education, expansion of accepted materials, community
initiatives, personal responsibility, and a polite closing. Switchboard's closest routes
still develop through different personal accounts. This is the intended repeated-path
phenomenon, not merely shared topic vocabulary.

## Coverage caveats

Canonical landmarks use 160 word-occurrences per call. Median unique source-position
coverage is 15.5% for humans and 45.6% for generated calls. Eighteen generated calls have
overlapping windows; the shortest C2-P1 call reuses 160 sampled occurrences from only 36
source positions. The full-coverage result is therefore essential, not optional decoration.

All calls are technically embeddable, but only 32 of 50 calls per condition belong to the
seven `n >= 3` primary topics, 44 belong to the thirteen `n >= 2` sensitivity topics, and
six are topic singletons. `route_coverage = 100%` should not be read as 50 calls contributing
to the primary estimate.

See `analysis/SEMANTIC_ROUTES.md` for the complete method, field definitions, robustness
commands, and interpretation limits.
