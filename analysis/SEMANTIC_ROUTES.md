# Semantic Route Reuse (SRR)

`analysis/semantic_routes.py` implements the canonical replacement for the old
Conversation Embedding Dispersion (CED) analysis.

SRR asks a narrow supporting question:

> When independently generated conversations begin from the same Switchboard topic,
> do they repeatedly travel through the same sequence of semantic content, while human
> conversations on that topic branch into more individual routes?

This is a **supporting metric**, not the project's headline measure. The dialogue-act
analysis remains the main account of interactional structure. SRR measures the development
of semantic content across whole conversations; it does not identify questions,
backchannels, repair, grounding, or other conversational functions.

The implementation is deliberately not a renamed version of the old CED. It fixes the two
central validity problems:

- Every comparison is made within an authoritative Switchboard topic. Conversations from
  unrelated topics are never pooled to manufacture dispersion.
- A conversation remains an ordered route of semantic states. It is not collapsed into one
  mean vector before comparison.

There is intentionally no TF-IDF fallback. TF-IDF primarily measures shared topic words,
which is the confound this analysis is designed to control.

## Status of the current data

The script defaults to `data/generated_v2`, which currently contains 50 conversations in
each of the 12 C1-C4 x P0-P2 conditions. These are **provisional development data**. They
come from the first-50 convenience sample and precede the planned regeneration with fixed
prompts, documented decoding, quality control, and seeded topic-stratified sampling.

Accordingly:

- Results from `generated_v2` are pipeline checks and scientific indicators, not final
  project claims.
- The default run label is `provisional-generated-v2`, and the console output also marks
  the results provisional.
- Final numbers must be regenerated from the replacement corpus, written to a fresh output
  directory, and given a final analysis label.
- P2 is exploratory even after regeneration because its prompt contains a real Switchboard
  style example. Semantic or lexical evaluation of P2 is therefore partly circular.

## 1. Data loading, IDs, and topic joins

The join key is Switchboard `conversation_no`, not a display-form topic string.

For every generated JSON record, the loader verifies that:

1. the parent directory and the record's `condition` agree;
2. the filename stem and `conversation_no` agree;
3. `(condition, conversation_no)` is unique;
4. the ID has a row in `swda-metadata.csv`;
5. the transcript contains parseable turns; and
6. any topic stored in the JSON agrees with the authoritative metadata topic after
   normalization.

The authoritative topic is `topic_description` from Switchboard metadata, falling back to
the prompt only if that field is absent. Topic comparison collapses whitespace and uses
case folding, so values such as `BUYING A CAR` and `Buying A Car` join correctly. Display
capitalization is never used as a key.

By default, all generated conditions must contain exactly the same set of conversation IDs.
This protects the architecture/prompt comparison from accidental sample changes. The
`--allow-unbalanced` option exists only for explicitly partial diagnostic runs; it should
not be used for a final comparison.

After finding the generated topics, the loader reads the **full available Switchboard pool**
for those topics. It does not limit the human reference to the 50 source IDs used for
generation. Every generated ID must nevertheless have a corresponding parsed Switchboard
transcript, which catches metadata/corpus mismatches early.

Human and generated conversations are compared as topic groups. The human transcript with
the same `conversation_no` is not treated as a paired target that the generated conversation
is expected to reproduce.

## 2. Turn-invariant, word-count-matched routes

Speaker names and turn boundaries are ignored for route construction. The spoken text is
flattened into one ordered sequence of lexical words. Splitting one utterance into several
turns, or merging adjacent turns without changing their words, therefore cannot change the
route. This keeps SRR distinct from the dialogue-act headline analysis.

### Fixed-mass semantic landmarks (canonical)

Switchboard calls are much longer than the generated calls. Dividing every transcript into
the same number of all-content bins would make one human vector summarize roughly 80-120
words while one generated vector often summarized only 20-50 words. Encoder smoothing could
then masquerade as a human/LLM route difference.

The canonical representation instead samples `K = 8` ordered landmarks, each containing
exactly `W = 20` consecutive lexical words. Landmark starts are evenly spaced from the
opening to the final possible 20-word window. Every route therefore has the same resolution
and the same 160 lexical-word occurrences of encoder input while still spanning the complete
call. Equal word count does not imply equal semantic information; it controls input
granularity rather than solving every representation difference.

Windows overlap when a call is shorter than 160 words. This is intentional: very short
architecture outputs remain part of the result rather than being selected away after
generation. `semantic_route_by_conversation.csv` reports both unique-word coverage and
overlap rate, so a compressed route is visible as an architecture outcome. The current
`generated_v2` corpus has no call shorter than the technical 20-word minimum, so all 600
generated calls enter the canonical analysis.

The tradeoff is explicit: long human calls are sampled across their full span rather than
exhaustively encoded. The former all-content/equal-progress-bin route remains available as
the named `full-coverage` robustness mode. A conclusion should not be presented as robust
if its direction changes between the mass-matched and full-coverage representations.

### MPNet embeddings and word-weighted pooling

The default encoder is the sentence-transformers model `all-mpnet-base-v2`; its output is
L2-normalized. A segment longer than the configured 96-word microchunk limit is split before
encoding, and microchunk vectors are pooled in proportion to their lexical-word counts. A
short residual chunk therefore cannot receive the same weight as a 96-word chunk. The word
limit reduces truncation risk but does not guarantee a fixed subword-token count.

For the canonical 20-word landmarks no split is normally needed. The ordered route is

```text
Z_i = (z_i1, z_i2, ..., z_i8)
```

Sentence-transformers is required. If the package or model cannot be loaded, the script
stops with an error instead of silently changing the construct to TF-IDF.

## 3. Constrained sequence alignment

Routes are compared only inside the same `(condition, topic)` group. For landmarks `k` and `l`,
the local cost is cosine distance:

```text
cost(k, l) = 1 - cosine(z_ik, z_jl)
```

The script uses endpoint-constrained dynamic time warping (DTW). The alignment must begin
at both conversations' first landmark and end at both conversations' last landmark; it cannot select
only a convenient matching subsequence.

Canonical constraints are:

- normalized Sakoe-Chiba band: 1 landmark (about +/-14% of an 8-point route);
- allowed moves: diagonal, advance-left, or advance-right;
- maximum consecutive non-diagonal moves in one direction: 2; and
- penalty for each non-diagonal move: 0.05.

For an admissible path `P`, the optimized accumulated cost is

```text
sum((1 - cosine(z_ik, z_jl)) for (k,l) in P)
    + 0.05 * number_of_non_diagonal_moves(P)
```

The reported route similarity uses a path-independent denominator:

```text
route_similarity = 1 - optimized_cost / max(number_of_left_points,
                                              number_of_right_points)
```

Higher values mean more similar semantic routes. Using fixed route resolution rather than
the selected path length keeps path selection and score reporting coherent: a path cannot
improve its score simply by adding warp steps to enlarge its denominator. The narrow band,
warp-run limit, and penalty prevent DTW from explaining away real route differences by
repeatedly matching many stages to one generic stage. The complete alignment path is stored
with every pairwise score for audit.

## 4. Nearest-sibling route reuse

For every eligible conversation, the script finds its most similar other route within the
same condition and topic:

```text
reuse_i = max(route_similarity(i, j)) for j != i
```

Ties are broken deterministically by the smaller `conversation_no`. A topic-condition's
raw route reuse is the mean of its conversations' nearest-sibling similarities. This is the
primary expression of the near-duplicate question: does each conversation have a sibling
that follows almost the same content route?

The primary analysis requires at least **three** eligible generated conversations per topic.
Topics with exactly two conversations are retained as an explicit sensitivity analysis.
With two conversations there is no genuine nearest-neighbor choice--the only pair must be
each conversation's sibling--so those estimates are less stable.

Condition summaries use the intersection of topics that meet the relevant threshold in
every condition present in the run:

- primary common-topic set: at least 3 eligible generated conversations per condition;
- sensitivity common-topic set: at least 2 eligible generated conversations per condition.

This common-topic rule prevents condition differences from being driven by different topic
mixtures. A partial or unbalanced run may therefore have substantially fewer reportable
topics.

## 5. Full human reference with equal-size subsampling

Nearest-neighbor similarity increases when a conversation has more possible siblings. A
generated topic group of size 3 must not be compared directly with a human topic pool of
size 20 or 30.

For a generated condition-topic group of size `m`, the script therefore:

1. computes all route similarities in the full eligible Switchboard pool for that topic;
2. draws `m` human conversations without replacement;
3. computes nearest-sibling reuse inside that size-`m` subset;
4. repeats this 2,000 times with deterministic seeded random streams; and
5. compares generated reuse with the resulting pool-size-matched human reference.

The topic-level calibrated effect is

```text
excess_route_reuse = generated_route_reuse
                     - mean(pool_matched_human_reuse)
```

Interpretation:

- positive: generated conversations reuse same-topic routes more than equally sized human
  groups;
- zero: generated reuse is at the pool-size-matched human level; and
- negative: generated conversations are less route-repetitive than the human reference.

The same seeded human subsets are reused whenever two conditions have the same topic and
generated pool size. This removes avoidable Monte Carlo noise from architecture contrasts.
The fields `human_subsample_mean_q025` and `human_subsample_mean_q975` are quantiles of this
reference distribution, not confidence intervals for a population parameter.

`human_subsample_upper_tail_fraction` is the add-one-smoothed quantity
`(1 + count(reference >= observed)) / (draws + 1)`. It is a descriptive calibration
position, not a hypothesis-test p-value and not a replacement for the project's inferential
model.

The script also pools individual human nearest-sibling scores from the matched draws and
finds their 95th percentile. `generated_above_human_nearest_p95_rate` is the fraction of
generated conversations whose nearest-sibling score is strictly above that topic- and
pool-size-matched threshold. This is high route reuse motivated by near-duplicate detection;
it is not labelled as a detected duplicate without qualitative validation.

## 6. Aggregation, uncertainty, and contrasts

Condition-level results are macro-averages over common topics: every topic receives equal
weight regardless of how many Switchboard conversations it contains. P0/P1 conditions use
a common primary topic family that excludes P2; the exploratory P2 conditions use their own
common topic family. Thus a P2 quality failure cannot remove a topic from a P0/P1 estimate.
Each architecture or prompt contrast uses the topic intersection for exactly the two
conditions being contrasted.

The reported condition interval is a 2,000-draw percentile bootstrap over topic-level
`excess_route_reuse` values. The script bootstraps topics, not individual conversations.
Bootstrapping conversations would duplicate observations and create artificial exact
nearest neighbors.

`semantic_route_contrasts.csv` reports these prebuilt topic-matched differences:

- adjacent architectures within each prompt: C1->C2, C2->C3, and C3->C4;
- adjacent prompts within each architecture: P0->P1 and P1->P2.

Contrast values are `right - left`, so a negative value means the right-hand condition is
less route-repetitive. Their intervals are also topic bootstraps. Contrasts involving P2 are
marked exploratory. The implementation does not apply a multiplicity correction to this
file; final inferential reporting must distinguish prespecified contrasts from exploratory
ones and apply the project's chosen correction where required.

The default random seed is `20260714`, matching the dialogue-act analysis. Separate stable
random streams are derived from the seed and analysis keys so a change in iteration order
does not silently change results. Human reference streams are keyed only by topic and pool
size, ensuring identical baselines for directly comparable conditions.

## 7. Architecture Route Displacement (exploratory companion)

All 12 generated conditions use the same 50 `conversation_no` inputs. For a fixed prompt
level, matching an ID across two architectures holds its source ID, topic, personas, and
input record fixed; the architecture-specific generation templates still differ. The script
therefore computes a second, explicitly exploratory view:

```text
architecture_route_displacement = 1 - similarity(route_A_i, route_B_i)
```

It also compares each matched-input similarity with different-input pairs from the same
topic. `same_input_similarity_advantage` is positive when two architectures preserve more
of the route for the same input than would be expected from topic alone. This directly asks
whether changing generation architecture redirects the semantic path, while SRR asks
whether a condition repeatedly reuses paths internally.

This companion analysis is not causal in `generated_v2`: there is only one stochastic
output per architecture/input, decoding differs across architectures, and every comparison
to C4 also changes the second agent's model. P2 additionally shares a real exemplar. Final
generation should include repeated outputs per input so between-architecture displacement
can be calibrated against within-architecture stochastic displacement.

## 8. Semantic branching curve

The branching curve is a complementary progress analysis, not a decomposition of the DTW
nearest-sibling score.

At each of the 8 ordered route landmarks, the script computes cosine similarity for
**all** same-topic route pairs in a group and averages those values. It does this separately
for each generated condition and for the full Switchboard topic pool. Because this is an
average over all pairs rather than a nearest-neighbor statistic, it does not have the same
candidate-pool-size bias; the full human pool is used directly.

For each common primary topic and progress point:

```text
excess_progress_similarity = generated mean pair similarity
                             - human mean pair similarity
```

The plotted curve macro-averages this difference over topics and bootstraps topics for its
pointwise interval.

- Positive values mean same-topic LLM conversations remain more semantically coupled at
  that point: a branching deficit relative to humans.
- Negative values mean the generated conversations are more dispersed at that point.

The curve uses like-for-like landmark indices, not DTW-warped positions. Display coordinates
are ordinal midpoints `(k + 0.5) / K`; they are not the exact word-center fractions of each
window, especially in short calls with overlap. Its intervals are pointwise, not
simultaneous; the first isolated point whose interval excludes zero must not be reported as
a confirmed "branching onset."

## 9. Outputs

The default output directory is `results/semantic_routes/`.

### `semantic_route_by_conversation.csv`

One row for every loaded human and generated conversation. It records length, route
representation, sampled-word occurrences, unique-word coverage, overlap, eligibility,
topic-pool size, nearest sibling ID, and nearest-sibling similarity. Use it to audit short
or compressed routes and inspect which conversations form close pairs.
`n_unique_sampled_words` counts distinct source positions covered, not distinct vocabulary
types.

### `semantic_route_pairwise.csv`

Every within-topic pairwise route comparison, including its constrained-DTW similarity and
serialized alignment path. No cross-topic pairs are scored.

### `semantic_route_by_topic_condition.csv`

The calibrated generated-versus-human result for every generated condition-topic group with
at least two eligible conversations. It contains generated reuse, the equally sized human
reference, reference-distribution quantiles, excess reuse, human 95th-percentile threshold,
generated threshold-exceedance rate, and the descriptive upper-tail fraction.

### `semantic_route_by_condition.csv`

The main compact table. It reports common primary/sensitivity topic counts, route coverage,
generated and human reuse, mean excess reuse with its topic-bootstrap interval, the human
95th-percentile exceedance rate, and the `n >= 2` sensitivity result.
`route_coverage` means technically embeddable routes; it does not mean every conversation
belongs to a replicated topic used by the primary estimate. In current `generated_v2`, all
50 calls per condition are embeddable, 32 belong to the seven `n >= 3` primary topics, 44
belong to the thirteen `n >= 2` sensitivity topics, and six have singleton topics.

### `semantic_route_contrasts.csv`

Adjacent architecture and prompt contrasts over the common primary topic set. Values are
right condition minus left condition.

### Architecture displacement outputs

- `semantic_route_architecture_pairs.csv`: all 900 matched-input architecture comparisons.
- `semantic_route_architecture_by_topic.csv`: matched versus different-input similarity by
  topic and architecture pair.
- `semantic_route_architecture_displacement.csv`: equal-topic macro summaries and topic
  bootstrap intervals for all six architecture pairs under each prompt.

These three outputs are exploratory for the current one-output-per-input corpus.

### `semantic_route_progress.csv`

The 8-point branching data: generated similarity, full-pool human similarity, their
difference, and the topic-bootstrap interval.

### Figures

- `semantic_route_by_condition.png`: condition-level excess nearest-sibling reuse.
- `semantic_route_branching.png`: branching-deficit curves, faceted by prompt; P2 is labelled
  exploratory.

Figures are summaries of the high-dimensional calculations. No distance is computed in a
2-D projection.

### `semantic_route_run.json`

The run-provenance record: resolved input paths, embedding backend name, parameters, corpus
and eligibility counts, explicit analysis status, creation time, and a SHA-256 fingerprint
of every generated input JSON. It contains no transcript text or embedding vectors. The
current record does not hash Switchboard, pin the Hugging Face model revision, or capture a
complete package lockfile, so exact archival reproduction also requires the environment and
corpus snapshot used for the run.

## 10. Commands

Install the required embedding package in the analysis environment:

```powershell
py -3 -m pip install sentence-transformers
```

Run the canonical provisional analysis:

```powershell
py -3 analysis/semantic_routes.py --generated-root data/generated_v2
```

Run on a CUDA device and increase the final human-reference Monte Carlo sample:

```powershell
py -3 analysis/semantic_routes.py `
  --generated-root data/generated_v2 `
  --device cuda `
  --human-draws 5000 `
  --bootstrap-draws 5000 `
  --output-dir results/semantic_routes_provisional_5k `
  --analysis-label provisional-generated-v2-5k
```

Skip figures in an environment without matplotlib:

```powershell
py -3 analysis/semantic_routes.py --generated-root data/generated_v2 --no-plots
```

For the eventual regenerated corpus, use its actual path and a fresh output directory:

```powershell
py -3 analysis/semantic_routes.py `
  --generated-root data/generated_final `
  --output-dir results/semantic_routes_final `
  --analysis-label final-regenerated `
  --analysis-status final
```

Do not use `--allow-unbalanced` for that final run. The command should fail if the condition
ID sets differ.

## 11. Required robustness runs

Each robustness run must use a separate output directory so it cannot overwrite the
canonical result.

### Encoder robustness: MiniLM

```powershell
py -3 analysis/semantic_routes.py `
  --generated-root data/generated_v2 `
  --embedding-model all-MiniLM-L6-v2 `
  --output-dir results/semantic_routes_robust_minilm `
  --analysis-label provisional-robust-minilm
```

### Landmark resolution: 6 and 10 points

```powershell
py -3 analysis/semantic_routes.py `
  --generated-root data/generated_v2 `
  --progress-bins 6 `
  --output-dir results/semantic_routes_robust_k6 `
  --analysis-label provisional-robust-k6

py -3 analysis/semantic_routes.py `
  --generated-root data/generated_v2 `
  --progress-bins 10 `
  --output-dir results/semantic_routes_robust_k10 `
  --analysis-label provisional-robust-k10
```

### Landmark semantic mass: 16 and 24 words

```powershell
py -3 analysis/semantic_routes.py `
  --generated-root data/generated_v2 `
  --landmark-words 16 --min-words 16 `
  --output-dir results/semantic_routes_robust_w16 `
  --analysis-label provisional-robust-w16

py -3 analysis/semantic_routes.py `
  --generated-root data/generated_v2 `
  --landmark-words 24 --min-words 24 `
  --output-dir results/semantic_routes_robust_w24 `
  --analysis-label provisional-robust-w24
```

### Full-coverage representation

This deliberately reintroduces unequal words per route point, so it is a construct
sensitivity rather than an interchangeable encoder setting. The 120-word floor should also
be reported as architecture-specific route coverage.

```powershell
py -3 analysis/semantic_routes.py `
  --generated-root data/generated_v2 `
  --route-representation full-coverage `
  --progress-bins 12 --min-words 120 `
  --dtw-band 2 `
  --output-dir results/semantic_routes_robust_full_coverage `
  --analysis-label provisional-robust-full-coverage
```

### Alignment robustness: diagonal and wider DTW

`--dtw-band 0` permits only like-position alignment when both routes have equal length.

```powershell
py -3 analysis/semantic_routes.py `
  --generated-root data/generated_v2 `
  --dtw-band 0 `
  --output-dir results/semantic_routes_robust_diagonal `
  --analysis-label provisional-robust-diagonal

py -3 analysis/semantic_routes.py `
  --generated-root data/generated_v2 `
  --dtw-band 2 `
  --output-dir results/semantic_routes_robust_band2 `
  --analysis-label provisional-robust-band2
```

### Include the `n >= 2` topic sensitivity as the plotted analysis

The canonical output already reports an `n >= 2` sensitivity column. This separate run is
only for checking how the complete summaries and plots change when that threshold is made
primary.

```powershell
py -3 analysis/semantic_routes.py `
  --generated-root data/generated_v2 `
  --primary-min-topic-size 2 `
  --output-dir results/semantic_routes_robust_n2 `
  --analysis-label provisional-robust-n2
```

The direction of the main result should be checked across MPNet/MiniLM, 6/8/10 landmarks,
16/20/24 words per landmark, landmark/full-coverage routes, diagonal/band-1/band-2
alignment, and `n >= 3`/`n >= 2` topics. A sign change is evidence that the result depends
on a modeling choice and must be reported, not hidden.

## 12. Interpretation limits

- SRR still depends on a general-purpose sentence encoder trained mostly on written text.
  Encoder agreement is therefore part of the validity check.
- Within-topic control removes the gross topic-mixture artifact, but a route can still be
  similar because participants answer the same detailed prompt questions. The matched human
  reference receives the same topic prompt, which is the relevant control.
- Fixed-mass landmarks equalize encoder input per route point, but they sample rather than
  exhaust long calls. In short calls the windows overlap. Both unique coverage and overlap
  must be inspected, and the full-coverage robustness result must be reported.
- Generic openings and closings can raise similarity in both corpora. The branching curve
  should show whether excess reuse is confined to endpoints or persists through the body.
- Constrained DTW tolerates modest pacing differences, but its score is not a metric in the
  mathematical sense and should not be interpreted as physical distance.
- Small generated topic groups remain noisy even after pool-size calibration. That is why
  `n >= 3` is primary and stronger future topic replication is desirable.
- The condition intervals quantify variation across the available common topics. They do
  not repair generation artifacts or turn provisional samples into population-level claims.
- SRR measures semantic content development, not interactional function. Any conclusion
  about how conversation is organized must be led by the dialogue-act results.
