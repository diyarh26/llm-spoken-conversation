# Social Metrics Script

This document explains what `analysis/social_metrics.py` is supposed to do so the team can
review whether the implementation matches the research idea.

## Goal

The original project already measures:

- words per turn
- alignment
- marker rates such as `oh`, `okay`, and `uh-huh`

This script adds three extension metrics:

```text
TSI / ADI = social-vs-assistant behavior
CAS       = local turn-by-turn context anchoring
CED       = full-conversation diversity/stereotypy
```

The script reads:

- generated conversations from `data/generated/<condition>/*.json`
- Switchboard conversations through `analysis/swda.py`

It writes CSV files to:

```text
results/social_metrics/
```

## How To Run

Dependency-free local run:

```bash
py -3 analysis/social_metrics.py --n-sb 50
```

The default embedding backend is a local TF-IDF fallback. It runs without installing new
packages.

Better semantic embedding run, if `sentence-transformers` is installed:

```bash
py -3 analysis/social_metrics.py --n-sb 50 --embedding-backend sentence-transformers --embedding-model all-MiniLM-L6-v2
```

On Linux or the VM, use `python` instead of `py -3`.

## Metric 1: TSI / ADI

### What It Tries To Measure

TSI/ADI asks:

```text
Does the conversation sound like two humans socially continuing a casual conversation,
or like an assistant trying to help, advise, explain, summarize, or solve a task?
```

### How The Code Does It

The script labels each turn with rule-based proxy labels:

- `A_task_help_advice`
- `B_social_continuation`
- `C_phatic_bonding`
- `D_grounding_backchannel`
- `E_personal_stance_experience`
- `G_repair_clarification`
- `H_assistant_explanation`

The label logic is in:

```text
label_turn()
```

The rule patterns are near the top of `analysis/social_metrics.py`, for example:

- `TASK_PATTERNS`
- `ASSISTANT_EXPLANATION_PATTERNS`
- `PHATIC_PATTERNS`
- `BACKCHANNEL_PATTERNS`
- `PERSONAL_PATTERNS`
- `REPAIR_PATTERNS`

### ADI Formula

```text
ADI = rate(A_task_help_advice) + rate(H_assistant_explanation)
```

The value is capped at `1.0`.

Interpretation:

- higher ADI = more assistant-like behavior
- lower ADI = less assistant-like behavior

### TSI Proxy Formula

The current code uses a rule-based proxy:

```text
TSI_proxy = weighted combination of:
  social_rate
  topic_health
  reciprocity
  1 - ADI
```

This is not the final scientific metric yet. It is a practical first version until the
team creates human annotations.

### Important Limitation

TSI/ADI currently uses rules, not human labels. The team should inspect examples and
adjust the patterns. The strongest final version would use manually annotated turns or an
LLM labeler validated against manual annotations.

## Metric 2: CAS

### What It Tries To Measure

CAS asks:

```text
Does each response fit its real previous turn better than random wrong previous turns?
```

Example:

```text
Real previous turn: I waited three hours at the airport.
Response: Oh no, was your flight delayed?
```

This should be strongly anchored.

Generic response:

```text
Real previous turn: I waited three hours at the airport.
Response: That sounds frustrating.
```

This may be less anchored because it could fit many different contexts.

### How The Code Does It

For each response turn after the first turn:

1. Take the real previous turn.
2. Sample random wrong previous turns from other conversations.
3. Embed the real previous turn, wrong previous turns, and response.
4. Compute cosine similarity.
5. Score the difference.

Formula:

```text
CAS = similarity(real_previous_turn, response)
    - mean(similarity(wrong_previous_turn_i, response))
```

The code is in:

```text
apply_cas()
```

### Negative Sampling

By default, the script prefers same-topic wrong contexts when enough are available. This
helps avoid an easy topic-only shortcut.

To force global random negatives:

```bash
py -3 analysis/social_metrics.py --global-negatives
```

### Interpretation

- higher CAS = response is more specifically anchored to the real context
- lower CAS = response is more generic or movable

## Metric 3: CED

### What It Tries To Measure

CED asks:

```text
Are full conversations in a condition diverse and spread out, or tightly clustered and
stereotyped?
```

Hypothesis:

- Switchboard conversations may be more dispersed.
- LLM conversations may cluster because they repeat similar structures and styles.

### How The Code Does It

For each conversation:

1. Join all turns into one full-conversation text.
2. Embed the full conversation.
3. For each condition, compute the centroid of its conversation embeddings.
4. Compute the average cosine distance from each conversation to the centroid.

Formula:

```text
CED(condition) = mean distance(conversation_embedding_i, condition_centroid)
```

The code is in:

```text
compute_ced()
ced_for_indices()
```

### Interpretation

- higher CED = conversations are more dispersed/diverse
- lower CED = conversations are more clustered/stereotyped

### Important Limitation

CED can accidentally measure topic diversity. The script therefore also writes a
topic-condition version when there are at least two conversations for the same topic and
condition.

## Output Files

### `turn_labels.csv`

One row per turn.

Useful columns:

- `condition`
- `conversation_id`
- `turn_index`
- `speaker`
- `text`
- label columns
- `assistant_like`
- `social_like`
- `topic_code`
- `cas`

Use this file to inspect whether the rules are labeling turns correctly.

### `conversation_metrics.csv`

One row per conversation.

Useful columns:

- `ADI`
- `social_rate`
- `topic_health`
- `reciprocity`
- `TSI_proxy`
- `CAS`

Use this file to compare individual conversations.

### `group_metrics.csv`

One row per condition.

Useful columns:

- `mean_ADI`
- `mean_TSI_proxy`
- `mean_CAS`

Use this file for condition-level comparison.

### `ced_by_condition.csv`

One row per condition.

Useful column:

- `CED`

Use this file to compare global conversation dispersion.

### `ced_by_topic_condition.csv`

One row per topic-condition pair, only when there are at least two conversations in that
topic-condition group.

Use this file to check whether CED patterns remain when topic is controlled.

## What To Check In Code Review

Good review questions:

1. Do the rule patterns for ADI catch real assistant-like turns without overcounting normal
   human advice?
2. Do the social labels catch real social continuation, personal stance, and grounding?
3. Does CAS sample wrong contexts fairly?
4. Should CAS negatives always come from the same topic, or is global sampling acceptable?
5. Does CED mostly measure topic, or does the topic-controlled output help?
6. Should we use TF-IDF for speed or sentence-transformers for better semantic meaning?
7. Are openings/closings distorting the results? If yes, add a main-body-only filter.

## Recommended Next Improvements

1. Inspect `turn_labels.csv` manually for 50-100 turns.
2. Tune the rule patterns.
3. Add a small manual annotation file and compare human labels to rule labels.
4. Use sentence-transformers on the VM for better CAS/CED embeddings.
5. Add plots after the metrics are trusted.

