# Dialogue-act structural signature — draft-data smoke test

This is a first indicator on `data/generated_v2/`, not a final-dataset result.  It compares
1,155 full Switchboard conversations (223,606
gold utterance units) with 600 generated
conversations (13,805 generated turns) across 12 conditions.

## First signal

On the more tagger-robust coarse inventory, the closest condition is **C1-P2**
(distribution JSD 0.1601) and the farthest is
**C2-P0** (0.3101).  12 of
12 coarse comparisons and 12 of 12 fine comparisons
exceed the 95th-percentile human n=50 sampling-noise floor.  This is evidence
that the structural signature has signal on the draft generations; it is not evidence that
the automatic fine labels are all trustworthy.

The clearest component is listener feedback: Switchboard is
**20.8% backchannels**, while the generated conditions span only
**0.2%**
(C4-P2) to **2.4%**
(C1-P2).  That reproduces the qualitative `sd -> b -> sd` gap in
the spec as an aggregate structural difference rather than a hand-picked example.

Restricting the human reference to the same 50 `conversation_no` values changes coarse JSD
by at most 0.0628 in this run.  That makes topic mix an unlikely sole
explanation for the observed gap, although the current 50 topics are the old first-50
convenience sample rather than the planned stratified sample.

The assistant-register rule layer is complementary to DAMSL, not a new DAMSL act.  Its
highest generated-condition rate is **C1-P1** at
8.1%; the full-SB rule-match rate is
0.0%.  These are literal phrase/list matches and should be read
as an indicator, not a classifier.

## Tagger validation

DialogTag `distilbert-base-uncased` was evaluated on 20 seeded SwDA
conversations (3,351 utterances): fine accuracy **72.1%** and
coarse accuracy **73.9%**.  Largest coarse confusions were:

  - `Abandoned/Other` -> `Statement`: 140
  - `Agreement` -> `Backchannel`: 111
  - `Opinion` -> `Statement`: 108
  - `Statement` -> `Opinion`: 93
  - `Abandoned/Other` -> `Backchannel`: 90

This is an in-domain diagnostic, not a guaranteed model-held-out score: DialogTag was trained
on the 1,155-conversation SwDA distribution and does not publish enough split identifiers to
prove that this seeded sample was absent from its training split.  It also cannot emit gold
`%`, `x`, `ar`, or `no`.  The coarse comparison is therefore the safer headline.

## Method and normalization

- Raw SwDA suffix modifiers are removed (`sd^e -> sd`, `qy^d -> qy`); standard semantic
  exceptions `nn^e -> ng` and `ny^e -> na` are applied first.  Leading-caret base acts such
  as `^q` are preserved.  All direct-CSV rows are retained while `@`, `*`, and parenthesized
  annotation flags are removed.
- Rare comma/semicolon compound tags use the first act, following `cgpotts/swda`.
- `+` continuations inherit the previous normalized act from the **same caller**, not the
  immediately preceding row.  `%` and `x` stay explicit.
- DialogTag's 38 model labels are explicitly mapped into the shared short inventory.  They
  collapse to 35 short codes; gold-only `%`, `x`, `ar`, and `no` complete the 39 columns.
  Grouped rare classes and question/repeat modifiers are collapsed on both sides.  All 39
  fine acts then map exhaustively to the requested ten coarse categories.
- Distribution distance is base-2 Jensen-Shannon divergence.  Transition distance is the
  unweighted mean row JSD of `P(next | current)`; a transition row present on only one side
  has maximal row distance 1.
- The noise floor uses 1,000 seeded draws of 50 whole human
  conversations without replacement against the full human reference (seed 20260714).

## Important limitations

Gold labels are SwDA annotation units, while each generated model turn receives one label.
That granularity mismatch follows the available inputs but can inflate structural differences,
especially around continuations and backchannels.  DialogTag is a legacy TensorFlow model,
was trained on human telephone speech, and forces assistant-like turns into its human taxonomy.
Draft conditions also have unequal generated-turn counts.  Treat rankings as a smoke-test
direction for the regenerated dataset, not a final claim.

The mandated bootstrap matches **conversation count**, not the number of classified units:
an average 50-conversation human draw contains about
9,680 gold annotation
units, versus 823–1,590
turns in a generated condition.  The very small human-only floor is therefore optimistic
for the human/generated granularity mismatch and should not be treated as a fully calibrated
significance threshold, even though the observed JSDs are orders of magnitude larger.

The cached prediction file contains generated-data labels and aggregate gold confusion counts
only—never Switchboard transcript text.  This run reused that cache.

## Files

- `da_distribution_by_condition.csv` and `da_distribution_coarse_by_condition.csv`
- `da_jsd_vs_sb.csv` (fine/coarse, full and topic-matched comparisons)
- `da_transition_*.csv` and `da_transition_coarse_*.csv`
- `da_tagger_validation.csv`, `da_tagger_confusion.csv`
- `assistant_register_by_condition.csv`
- `key_acts_grouped_bar.png`, `sb_transition_heatmap.png`, `jsd_vs_sb.png`

## Reproducing the model-dependent stage

DialogTag is isolated from the main generation environment because its legacy TensorFlow
checkpoint needs older dependency pins.  On Windows with Python 3.11:

```powershell
py -3.11 -m venv .venv-dialogtag
.venv-dialogtag/Scripts/python -m pip install -r analysis/requirements-dialogue-acts.txt
.venv-dialogtag/Scripts/python analysis/dialogue_acts.py --force-retag
```

The first model load downloads DialogTag's checkpoint.  Model-free reruns can then use the
text-free cache with the ordinary project environment.

Run the gold-only path with `python analysis/dialogue_acts.py --human-only`.  A full rerun can
reuse `dialogtag_predictions.json` without importing the model; pass `--force-retag` to invoke
DialogTag again.  See the module docstring and `--help` for paths and statistical controls.
