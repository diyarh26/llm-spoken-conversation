# Is alignment just turn length? Non-adjacent baseline check

Script: `analysis/alignment_baseline.py` (tests: `tests/test_alignment_baseline.py`), adapted
from `analysis/alignment_confound_check.py` on branch `claude/ced-embedding-dispersion-itgksm`,
which was written for this check but never run.

```
python analysis/alignment_baseline.py --w2v word2vec-google-news-300.gz \
    data/generated data/generated_workshop/qwen35-9b --labels Vicuna,Qwen \
    --out results_workshop/alignment_baseline
```

## Method (the advisor's four steps)

1. **Baseline:** per conversation, the mean similarity of turns that are *not* next to each
   other: every pair at least 5 turns apart, at odd distances (5, 7, 9, ...) so that, like
   adjacent pairs, both turns are by different speakers.
2. **Real alignment** = adjacent-turn similarity minus the baseline.
3. **Length check:** steps 1-2 again (a) with every turn cut to its first 15 words, and
   (b) only for pairs of similar length (shorter/longer >= 0.5, both at least 3 words).
4. **Decide:** compare each condition's corrected score with humans: difference, 95%
   bootstrap CI over conversations (2,000 resamples), Mann-Whitney U.

Similarity = cosine of the mean word2vec-google-news-300 vectors of the two turns. Adjacent
and baseline pairs use the same function, so the *difference* is fair. The absolute values
are not ALIGN's `cosine_semanticL` (no lemmatising or frequency filter).

Humans: the 50 topic-matched Switchboard calls. A conversation is scored only if it has at
least one pair 5+ turns apart. This is why n < 50 for some conditions: Vicuna C4-P1 has a
median of 7 turns, and 21 of its 50 conversations have fewer than 6.

## Result: corrected alignment minus humans (95% CI)

| Condition | Full turns | First 15 words | Similar-length pairs |
|---|---|---|---|
| Humans (corrected score) | -0.017 | -0.014 | +0.005 |
| Vicuna C1 (P0/P1/P2) | +0.067 / +0.056 / +0.067 | +0.059 / +0.048 / +0.056 | +0.034 / +0.024 / +0.026 |
| Vicuna C2 | +0.065 / +0.043 / +0.046 | +0.071 / +0.041 / +0.044 | +0.033 / +0.033 / +0.032 |
| Vicuna C3 | +0.108 / +0.088 / +0.095 | +0.083 / +0.068 / +0.074 | +0.072 / +0.040 / +0.056 |
| Vicuna C4 | +0.092 / +0.094 / +0.093 | +0.078 / +0.073 / +0.068 | +0.054 / +0.044 / +0.056 |
| Qwen C1 (P0/P1/P2/P3) | +0.077 / +0.081 / +0.076 / +0.079 | +0.070 / +0.070 / +0.068 / +0.070 | +0.027 / **+0.011 / +0.013 / +0.009** |
| Qwen C2 | +0.042 / +0.047 / +0.039 / +0.052 | +0.053 / +0.043 / +0.037 / +0.048 | +0.017 / +0.023 / +0.017 / +0.028 |

All differences are above zero with a 95% CI that excludes zero, except the three in bold
(Qwen C1-P1/P2/P3 on similar-length pairs, CIs [-0.004, +0.027], [-0.002, +0.028],
[-0.006, +0.024]). Full table with adjacent and baseline values:
`alignment_baseline_summary.csv`; per conversation: `alignment_baseline_per_conversation.csv`.

## Reading

- The raw gap is mostly length. Adjacent similarity alone puts LLMs 0.1-0.4 above humans.
  After subtracting each conversation's own non-adjacent baseline the gap is 0.04-0.11, and
  on similar-length pairs 0.01-0.07.
- For humans, adjacent turns are no more similar than distant ones (corrected score about
  zero, slightly negative on full turns). This is because a short reply such as "uh-huh"
  often follows a long turn. LLM turns are more similar to the turn just before than to
  turns farther away.
- The gap survives every correction in all 12 Vicuna conditions and in Qwen C2. For Qwen
  C1 it survives full turns and the 15-word cut, but on similar-length pairs it shrinks to
  about +0.01 at P1-P3 and is no longer distinguishable from humans.

**Decision:** keep the figure and report the corrected numbers, with the note that the
corrected effect is a fraction of the raw one. Under the strictest check it disappears for
Qwen C1 at P1-P3.
