# Presentation prep: dialogue acts, the tagger, and our results (Diyar)

Part 1 is the 4-minute talk. Part 2 is everything they can ask, with answers. Part 3 is a
cheat sheet of numbers. Every number comes from `results_v3/`.

---

# PART 1: The talk (~480 words, 3.5–4 minutes; Diyar's own version)

"One of the things we measured is the **difference in structure** between LLM and human
conversations. To do this we used **dialogue acts**: the purpose of each sentence in the
conversation, for example a statement, an opinion, a question, or a **backchannel** like
'uh-huh', where the listener signals 'I'm with you'.

We got the idea from the **Switchboard** corpus, where every sentence was labelled by
professional annotators. To label our own conversations we used **DialogTag**, a DistilBERT
model that was already trained on Switchboard. We used it as-is, and we grouped its labels
into **10 main classes**.

We ran the tagger on the **human conversations too**, not only on the LLMs, so that the
tagger's errors affect both sides in the same way. Then we checked how accurate it is. On the
human corpus it agrees with the professional labels **72%** of the time. On LLM text we
labelled **100 LLM sentences by hand**, and the tagger agreed with us **77%** of the time.
Between ourselves we agreed **83%** of the time.

After running the tagger we have a **distribution of acts for each condition**. To measure
how far each one is from the human distribution, we used the **Jensen–Shannon divergence**,
a measure from information theory of how different two distributions are. It goes from **0**,
identical, to **1**, completely different.

To know what counts as 'far', we added **two baselines**:
- **Sampling noise:** for each condition, we drew the same number of units it has from the
  human distribution, 1,000 times, and took the 95th percentile of the JSD. This is how far
  'human vs human' gets just by chance, and it was at most **0.002**.
- **The tagger's own error:** on the same human sentences, the tagger's labels vs the
  professional labels give **0.035**.

*(point to the poster)*

What did we find?

**First, backchannels are almost completely missing in the LLMs.** In human calls, a quarter
of the talk is backchannels, and **19%** is stand-alone 'uh-huh / yeah / right'. In all 12
LLM conditions that's only **0 to 0.4%**. We counted the 19% with a simple word rule, without
the tagger, and it agrees with what the tagger found.

**Second, LLMs fill the backchannel hole with opinions and questions.** Humans: **14%**
opinions and **5%** questions. LLMs: **21–38%** opinions and about **15%** questions.

**Third, the architecture does matter, but not in the direction we expected.** We expected
giving each speaker its own agent to be more human-like. Instead, the closest architecture was
**all-at-once**, with a JSD of **0.12–0.13**. The other architectures were further away,
**0.15–0.24**. This is not chance: when we resampled whole conversations, all-at-once was
closer in **100% of 2,000 resamples**.

**In conclusion:** LLM conversations are missing the listening part, and separating the
speakers doesn't bring it back; it makes things worse."

**Careful:** don't say "the more separation, the worse" as a strict ladder (A3 0.15–0.20 is
sometimes better than C2 0.17–0.24). Say: "all-at-once is the closest; every other
architecture is further away."

---

# PART 2: Questions they can ask (with answers)

## A. Dialogue acts and the label set

**Q: Where do the labels come from?**
Switchboard's dialogue-act annotation, **SWBD-DAMSL** (Jurafsky et al., 1997). Trained
annotators labelled every utterance of 1,155 Switchboard conversations. The original ~220
tags are clustered into **42** standard labels: sd = statement, sv = opinion, b =
backchannel, qy = yes/no question, and so on.

**Q (picky): So exactly how many labels?** Five layers:

| Layer | Labels | What it is |
|---|---|---|
| Raw annotation | ~220 | what annotators wrote, incl. modifiers (`qy^d` declarative question, `sd^e` elaborated) |
| **Standard set** | **42** | Jurafsky et al. (1997) clustering; the number everyone cites |
| DialogTag output | 38 | the tagger's training merged a few of the 42 |
| Our fine level | 39 | one shared list for the experts' and the tagger's labels (a few, like `%` abandoned and `x` non-verbal, exist only on the expert side) |
| **Our main level** | **10** | the categories on the poster |

Short answer: *"About 220 raw tags, clustered into the standard 42; the tagger predicts 38; we
analyse 10 broad categories."*

**Q: Why group into 10 categories?**
1. Several fine labels are rare and the tagger is unreliable on them. The tagger's error on
   humans is **0.074 with the fine labels vs 0.035 with 10**.
2. Our research question is about broad functions (listening, opinion, asking), not fine
   distinctions.
The mapping is fixed in code (`FINE_TO_COARSE` in `analysis/dialogue_acts.py`) and is the
same for every condition.

**Q: What exactly counts as "Backchannel"?**
Four DAMSL labels: **b** (acknowledgement, "uh-huh"), **bk** (response acknowledgement,
"oh okay"), **ba** (appreciation, "that's great"), **bh** (backchannel as a question, "oh
really?"). The common thread: listener feedback that adds no new content.

**Q: What is a "unit"?**
- **Humans:** the annotated utterance (a "slash unit": about one sentence or clause, cut
  by the Switchboard annotators). 223,606 units.
- **LLMs:** we split each turn into sentences at ". ! ?" followed by a capital letter or a
  digit. 41,105 units.

**Q (picky): Isn't that a mismatch?**
Yes, a known limitation. Human units are cut by annotators, LLM units by punctuation. LLM
sentences are, if anything, *longer*, which could hide a backchannel inside a sentence. That
is why we also used (1) a **turn-level rule** (a unit that is only "uh-huh / yeah / right /
okay / mm hm / I see") and (2) Sham's **turn-level** backchannel measure. All three agree.

**Q: Do you remove anything before tagging?**
The two scripted "Hello!" opening turns (we wrote them), bracketed stage directions the
agents echoed from our turn counter ("(Turn 14)"), and Korean text in 10 drifted
conversations. The same rule applies to all 12 conditions.

## B. The tagger (DialogTag)

**Q: What is DialogTag?**
An open-source Python package: **DistilBERT** (a smaller, distilled version of BERT,
`distilbert-base-uncased`) fine-tuned to classify a sentence into Switchboard
dialogue-act labels.

**Q: Did you fine-tune it?**
**No.** We used it **off-the-shelf** (`pip install DialogTag`). Its authors fine-tuned
DistilBERT on Switchboard; we only mapped its 38 output labels to our 10 categories. Instead
of training it we **validated** it: 72% on human speech, 77% on LLM text (hand-check).
*Why not fine-tune it ourselves?* Fine-tuning on LLM text would need thousands of hand-labelled
LLM sentences, which don't exist. Fine-tuning on Switchboard again would just reproduce the
same model.

**Q: Does it use context?**
**No.** Each unit is labelled on its own. Consequence: it can't recognise an *answer* (that
needs the previous question). That's why we make **no claims about Answers**. Our claims
(backchannels, opinions, questions) are visible in the sentence itself.

**Q: How accurate is it?**
- On human Switchboard speech: **70.6% with the fine labels, 72.0% with 10**, against the expert
  labels, over all 223,606 units.
- On LLM text (our hand-check): **77%** (75 of 98; 95% CI 67–84%).

**Q (picky): It was trained on Switchboard, so isn't 72% optimistic?**
Possibly. It has probably seen some of those conversations during training. That's exactly
why we ran an **independent check on LLM text**, which it never saw: 77%.

**Q: Why not use the expert labels for humans instead of the tagger?**
Because the LLM side can *only* be tagged by the tagger. Using the tagger on both sides
keeps the measuring instrument identical, so its biases affect both groups equally. We
still use the expert labels to measure the tagger's own error: the tagger vs expert labels
on humans gives a JSD of **0.035**, our "instrument error" reference line.

**Q: What are its typical errors on LLM text?**
From the hand-check:
- **Opinion → Statement** (7 times): it *under*-counts opinions, so our "too many opinions"
  finding is conservative.
- **Answer → Statement** (5): no context, see above.
- **Open question → Yes/no question** (4): both are questions, so the question total is
  unaffected.

## C. Jensen–Shannon divergence (the one they asked about)

**Definition.** For two distributions P (an LLM condition) and Q (humans) over the 10 acts:
```
M = (P + Q) / 2                         ← the average ("mixture") distribution
JSD(P, Q) = ½ · KL(P ‖ M) + ½ · KL(Q ‖ M)
KL(P ‖ M) = Σ_x P(x) · log2( P(x) / M(x) )
```
So JSD asks: how far is each distribution from their average?

**Q: Why is it between 0 and 1, and not 0 to infinity like KL?**
Three steps. Say them in this order:
1. **Why KL can be infinite:** KL(P‖Q) has log(P(x)/Q(x)). If some act has Q(x) = 0 but
   P(x) > 0, that's log of "something divided by 0", which is infinite.
   *Our example:* if an LLM condition had exactly 0 backchannels, KL(humans ‖ LLM) would be
   infinite, and some of our conditions are close to that.
2. **Why JSD never divides by zero:** JSD compares each distribution with the **average**
   M, and M(x) = (P(x) + Q(x)) / 2 is never 0 where P(x) > 0. So it is always finite.
3. **Why the maximum is exactly 1:** because M(x) ≥ P(x)/2, the ratio P(x)/M(x) is at most
   **2**. With **log base 2**, log2(2) = 1, so every term of KL(P‖M) is at most P(x)·1, and
   KL(P‖M) ≤ Σ P(x) = 1. The same holds for KL(Q‖M), so JSD ≤ ½·1 + ½·1 = **1**.
   It equals 1 exactly when P and Q never overlap (no act occurs in both).
   It equals **0** exactly when P = Q (Gibbs' inequality: KL ≥ 0, with equality only for
   identical distributions).

   *If they ask "and with natural log?":* the maximum is ln 2 ≈ 0.693. We use base 2 so the
   scale is 0–1.

**Q: What does 0.13 actually mean? (an elegant answer)**
JSD has an information meaning. Flip a fair coin to pick "human" or "LLM", draw one act
label from that source, and show it to someone. **JSD is the information (in bits) that one
label gives about which source it came from.** The maximum is 1 bit, the coin itself. So
0.13 means one label tells you 0.13 bits about "human or LLM". For a single sentence that is
a lot, compared with 0.002 for two human samples.

**Q: Why JSD and not KL, chi-square, cosine or Euclidean distance?**
- **KL:** asymmetric (KL(P‖Q) ≠ KL(Q‖P)) and can be infinite (see above).
- **Chi-square:** a *significance test*. With 41,000 units *everything* is significant; it
  measures sample size more than effect size. JSD is an **effect size**.
- **Cosine / Euclidean:** don't respect that these are probability distributions.
- **JSD:** symmetric, bounded 0–1, well defined with zeros, information-theoretic meaning,
  and standard for comparing label distributions. Bonus: √JSD is a true **metric** (it obeys
  the triangle inequality; Endres & Schindelin, 2003).

**Q: Is JSD on the fine labels consistent with 10?**
Partly, and be honest here. With the fine labels (39 codes) all conditions sit at 0.22–0.28 and the ordering
gets mixed (A4-P2 0.22 is lower than C1-P1 0.24), because rare fine labels are dominated by
tagger noise (instrument error 0.074 vs 0.035). That's exactly why the **10-category** result
is the primary one.

## D. Act order (transitions)

**Q: What did you do for order?**
For every pair of consecutive units: "act A followed by act B". This gives a table
("after a question, X% statement, Y% answer…"). Each row is compared with the human row using
JSD.

**Q: Why weighted?**
The plain average treats every row equally, so the row for an act that appears only 3 times
counts as much as the statement row with thousands. That produced a fake outlier (C2-P2
0.416). We weight each row by how often that act occurs: C2-P2 becomes 0.230, in line with
the rest. **Result:** C1 is again closest (0.146–0.163).

## E. Reference lines, statistics and robustness

**Q: How is the "sampling noise" line computed?**
Take as many units as an LLM condition has (about 2,000), drawn at random from the human
distribution; compute the JSD to the full human distribution; repeat **1,000 times**; take
the **95th percentile**. Result: **0.0005–0.0016**. Our distances are about 100× larger.

**Q (picky): That resamples units, but units in one conversation aren't independent.**
Correct, so we also did a **conversation-level bootstrap**: resample whole conversations
(50 LLM and 1,155 human, with replacement), 2,000 times
(`results_v3/da_jsd_bootstrap.csv`). The 95% CIs: C1 0.11–0.16; A3-P1 0.13–0.19; C2-P0
0.21–0.29. **C1 vs each other architecture:** the difference is positive in **100% of 2,000
resamples** (C2 +0.072 [0.049, 0.096]; A3 +0.053 [0.033, 0.074]; A4 +0.070 [0.048, 0.093]).

**Q: And the 0.035 line?**
The tagger vs the expert labels on human speech: how much distance the instrument alone
creates. Our gaps (0.12–0.24) are 3–7× larger, so the tagger's error can't explain them.

**Q: Robustness?**
The whole analysis was re-run (1) without echo-loop turns (a turn of ≥8 words overlapping
≥80% with an earlier turn) and (2) without the 10 language-drift conversations. Every
distance moved by **≤ 0.01**.

**Q: The tagger-free rule check?**
- A unit counts as a backchannel if, after removing punctuation, it is exactly "uh huh",
  "yeah", "right", "okay", "mm hm" or "i see". Humans **19.1%**; LLMs **0.0–0.4%**.
- A unit counts as a question if it contains "?". Humans 3.7%; LLMs 7–24%.

**Q: Why does the tagger give C1 8–9% backchannels but the rule gives about 0%?**
C1 starts turns with "Yeah, I think…": the tagger labels that sentence as a backchannel,
but it's a reaction word on a full turn, not a listener turn. The rule only counts units
that are *only* the reaction.

## F. The hand-check

**Q: How was it designed?**
100 random LLM units, 8–9 from each of the 12 conditions. 20 were labelled by all 4 of us
and 20 by each person alone (40 each), in random order with the tagger's answers hidden.

**Q: How did you score it?**
- **Accuracy:** the tagger vs the human label. For shared units the human label is the
  majority of the 4; 2 ties were excluded, which is why it's 98 units.
- **95% Wilson interval:** the standard confidence interval for a proportion from a small
  sample (better than ±1.96·SE near 0% or 100%). **77% [67%, 84%].**
- **Agreement among us:** 83% pairwise. **Fleiss' kappa 0.67**: agreement corrected for
  chance (0 = chance, 1 = perfect; 0.61–0.80 is called "substantial"). For reference,
  Switchboard's own trained annotators agreed about 84% (κ ≈ 0.80; Stolcke et al., 2000).
- **The tagger vs our majority on the shared units: 15/18 = 83%**, the same rate at which
  we agree with each other.

**Q (picky): 100 sentences is small.**
It's enough to bound the accuracy within about ±9 points, and the key claims don't depend
on the tagger alone: backchannels and questions are confirmed by tagger-free rules.

## G. Limitations to admit calmly

- The tagger ignores context, so there are no claims about answers.
- Unit segmentation differs (annotators vs punctuation); mitigated by turn-level checks.
- Two open 7–13B models; 50 conversations per condition.
- Conversation endings were helped by a turn countdown.

---

# PART 3: Cheat sheet

| What | Number |
|---|---|
| Human units / conversations | 223,606 / 1,155 (all of Switchboard) |
| LLM units / conversations | 41,105 / 600 |
| Tagger accuracy, humans (10 categories / fine labels) | 72.0% / 70.6% |
| Tagger accuracy, LLM text (hand-check) | 77% [67–84], 75/98 |
| Annotator agreement | 83%, Fleiss κ 0.67 |
| Backchannels (tagger): humans / LLMs | 25% / 1–9% |
| Stand-alone backchannels (rule): humans / LLMs | 19.1% / 0.0–0.4% |
| Opinions: humans / LLMs | 14% / 21–38% |
| Questions: humans / LLMs | 5% / ~15% |
| JSD vs humans (10 categories) | C1 0.12–0.13 · C2 0.17–0.24 · A3 0.15–0.20 · A4 0.18–0.22 |
| Sampling noise / tagger error | ≈0.001–0.002 / 0.035 |
| C1 closer than C2 / A3 / A4 | 100% of 2,000 conversation resamples |
| Weighted order distance | C1 0.146–0.163 (lowest) |
| Robustness (no loops / no drift) | every distance changes ≤ 0.01 |
