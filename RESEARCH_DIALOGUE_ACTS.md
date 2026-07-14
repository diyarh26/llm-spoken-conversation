# Extension Metric — Dialogue-Act Structural Signature (current headline)

**Status:** active headline extension, decided 2026-07-14.
**Supersedes as headline:** the three-metric memo in `RESEARCH.md` (TSI/ADI, CAS, CED).
Those are now *supporting/secondary*, not the main story. See "Relationship to the old
extension" at the bottom.

This document is the single source of truth for what the extension metric is, why, how it
is computed, and what is decided vs still open. It exists so a new session or a teammate can
pick up without re-deriving the discussion.

---

## 1. The reframed research question

**OLD framing (too narrow — do not build on this):** "does generation architecture make
conversations more human-like / does it drift into assistant mode / explain the Vicuna
paradox." Assistant-drift was a *generation artifact* we noticed and are actively removing
(peer guard + prompt/decoding redo). Building the headline metric to detect it would mean
measuring a bug we are fixing.

**CURRENT framing (the project):**

> **How does the *interactional structure* of an LLM-generated conversation differ from a
> human one — and which of those differences can be closed by generation choices
> (architecture, prompt), versus which are fundamental and irreducible?**

Key reframing insight: **fixing the prompts does not threaten the project — it purifies it.**
Before the fix, the human↔LLM gap is dominated by cheap, obvious assistant register. After
the fix, what *remains* is the interesting, hard residue: the structural differences that
survive even when the model is explicitly told to be a casual equal. That residue is the
finding.

- **Independent variables (the levers):** generation architecture (C1–C4) × prompt (P0–P2).
  Unchanged from the existing design.
- **Dependent variable (what we measure):** the structural/interactional difference from
  human conversation, via the full dialogue-act inventory. Assistant register is at most
  *one component* of that difference — we do not prejudge it and do not build the metric
  around it.
- **Novelty:** Mayor et al. (2025) measured only surface features (turn length, 3 lexical
  markers oh/okay/uh-huh, alignment). The dialogue-act signature **is** the interactional
  structure — the thing everyone gestures at ("LLMs don't really converse") but nobody has
  measured. Proven technique (DA tagging since the 1990s) + novel question + free gold data.

## 2. What a dialogue act is (one paragraph)

Every utterance in a conversation *does something* — asks, answers, agrees, gives listener
feedback (backchannel), repairs a misunderstanding, etc. A **dialogue act** labels that
*function*, not the words. Switchboard is special: the **SwDA (Switchboard Dialog Act
Corpus)** already has expert hand-labels (DAMSL scheme, ~42 acts) for **every** utterance,
in the `act_tag` column of every `.utt.csv`. That is a **free, gold, expert ground truth**
for the human side — no annotation needed. Key acts to know: `sd` statement-fact, `sv`
statement-opinion, `b` backchannel ("uh-huh"/"yeah"), `aa` agree/accept, `qy` yes-no
question, `qw` wh-question, `bh` backchannel-as-question ("oh, really?"), `%` abandoned,
`+` turn continuation (a turn split *through* the other speaker's backchannel).

The visceral contrast (real conv 4325 vs Vicuna's attempt at 4325): humans are **~19%
backchannels**, short reactive turns, asymmetric roles (storyteller vs listener),
disfluency, abandoned turns, `+` continuations. The LLM version is **~0% backchannels**,
every turn a full 20–40-word `sd`/`sv` statement, symmetric, clean. That gap is the metric.

## 3. The metric (recipe)

1. **Tag every turn with a DAMSL act.**
   - Human side: **free** — read the `act_tag` column.
   - LLM side: run an automatic DA tagger over the generated conversations.
2. **Distribution** — per corpus/condition, the fraction of turns of each act type (a
   fingerprint of *how* the conversation is built).
3. **Transition grammar** — the act→act transition matrix (what follows what). Humans run
   `sd → b → sd` backchannel loops; assistant-register runs `sd → sd → sd` monologue chains.
   This captures sequential rhythm, which counting alone misses. (Stolcke et al. 2000 call
   this a "discourse grammar.")
4. **Headline number** — **Jensen–Shannon divergence (JSD)** between each condition's act
   distribution and the human reference, plus a distance between transition matrices. One
   number per architecture/prompt for "how far is its interactional structure from human."

### Statistical design (decided — see §5 for the "why")

- **Human reference = the FULL corpus** (all 1,155 SwDA conversations, or a large sample).
  Never shrink the reference to match the LLM count — a bigger reference is strictly better
  (more precise). This is the baseline everything is compared against.
- **Each LLM condition = its 50** (that is all we generate per condition).
- **Noise floor (significance ruler):** repeatedly draw 50 *human* conversations at random
  and measure *their* JSD to the full human reference. That band = how big a distance looks
  like even with no real difference, just from n=50. An LLM condition must beat that floor
  to count as a real difference. (Bootstrap.)
- **Topic control (the real fairness lever, not count):** also run a **topic-matched**
  comparison — LLM conditions vs the human conversations on the *same* topics — to rule out
  topic (not architecture) driving the act differences.

### Tools (all local, no paid API, fit the V100)

- `DialogTag` (pip; DistilBERT; ~38 SwDA acts; CPU-fine) — https://github.com/bhavitvyamalik/DialogTag
- Gold human tags: `cgpotts/swda` — https://github.com/cgpotts/swda (we already have the
  `act_tag` column locally in `data/switchboard/swda/**/*.utt.csv`).
- Stronger taggers if needed: SILICONE / RoBERTa (~80–83% on SwDA) — https://arxiv.org/abs/2009.11152
- Comparison: JSD between distributions; Frobenius/JSD distance between transition matrices.
- `ConvoKit` ships SwDA + DA utilities — https://github.com/CornellNLP/ConvoKit

## 4. The one validity catch you MUST own (before the supervisor does)

The tagger is trained on *human* Switchboard. It has **no category for assistant register**
("Here are three things you should consider" → forced into `sd`). So it can **hide** the
drift. Two responses:

1. **Defensive:** measure the tagger's accuracy on held-out gold SB first (report it);
   hand-check a sample of LLM turns it labeled `sd`.
2. **Offensive (the wow):** turns that don't fit the human act inventory at all **are a
   finding** — "LLM conversations contain speech acts with no human equivalent." Add a thin
   rule layer for assistant acts (numbered lists, "you should", "it's important to") and
   report how much of each condition falls *outside* human dialogue-act space.

## 5. Why the stats are set up this way (recorded so we don't relitigate)

- Equal group sizes are **not** required by any test we use. Shrinking the human side to 50
  would only make the reference noisier for no gain; precision is limited by the smaller
  (LLM) side regardless.
- JSD specifically is sample-size sensitive: a 50-conversation distribution is noisier than
  a 1,155 one, which inflates divergence from noise alone. The **noise floor + bootstrap**
  handles this honestly — better than naive "50 = 50 by discarding data."
- Analogy: to judge if someone is unusually tall you compare them to the whole population's
  height distribution, but use the population's spread to decide how many cm counts as
  "unusual." Full data = reference; resampling = what gap is meaningful.

## 6. Secondary metric (cheap, extends the paper): alignment trajectory

Turn Mayor's *descriptive* finding ("LLM alignment is high and rises; humans sit ~0.57 and
stay flat/decrease") into two scalar humanness features per conversation: **over-alignment**
(mean − SB baseline) and **alignment slope** across the 10 turn-pairs. We already own the
ALIGN pipeline. Accommodation is a known bot/human discriminator (Bhatt & Rios 2021);
ConvoKit's `Coordination` module + the Doyle & Frank (2015) confound correction keep the
slope honest.

## 7. Honesty guardrails (do not overclaim)

- Reference-free coherence metrics (USR/GRADE/DynaEval — the CAS family) correlate *weakly*
  with human judgment and were trained on *written* chat → domain shift on spoken
  Switchboard. Use only as **relative** signals across conditions, never absolute humanness
  (Yeh 2021; Durmus 2022).
- An open LLM-judge (Prometheus-2 fits the GPU) is biased *toward* polished assistant
  register → circular per our own CLAUDE.md rule → **validation-only, off the critical path.**

## 8. Lessons from the rigor review (why we pivoted — keep visible)

- Validation was declared "passed" on an **approximation** (later-half proxy, no
  topic-initiation main-body extraction). Redo as a **real replication** of Mayor's
  pipeline (reproduce 0.57 conceptual alignment + the marker rates properly).
- The four architectures did **not** share decoding (C1 ran rep-penalty/ngram OFF; C2/C3/C4
  at 1.15 / ngram 6; C1 got 2048 tokens vs 200/turn). For the redo: tune decoding
  per-architecture on a **dev set against an explicit degeneration score**, freeze it,
  **document it**. Hard exception: never cap a parameter that bounds a measured DV (per-turn
  token budget vs turn length).
- Only one weak model (Vicuna-13b-v1.5-16k). **Keep it** — per the paper, stronger models
  are *further* from human, so Vicuna is the right object; scope claims to "within Vicuna."
  A modern model is an optional stretch second factor, not a fix.
- TSI/CAS/CED were never validated against human labels. The DA metric replaces them as
  headline; if any is kept, validate with Cohen's κ on a hand-coded sample.
- Report distributions, not just means (SB words/turn: median 7 vs mean 14.7 — skewed).
- Correct for multiple comparisons; use the noise-floor for JSD.

## 9. The two work tracks (each > 2 days; map to the two-machine workflow)

**TASK 1 — "Make the conversations real" (VM / generation). STATUS: NOT STARTED — pending
a design discussion of *what* to change.** Open items to settle before running:
- Prompts: keep P0 faithful to the paper's basic prompt (it must stay a replication);
  redesign P1 to fight helpdesk framing, thicken thin personas (give stance/mood, license
  personal stories, tangents, disagreement, short reactions — real SB wanders; topic
  initiation isn't until ~turn 12).
- Decoding: per-architecture tuned + documented (see §8), DV-safe.
- Model: keep Vicuna-13b-v1.5-16k primary; modern model only as optional stretch.
- **Sampling (DECIDED): replace the "first-50" convenience sample with a seeded,
  topic-stratified random sample of `conversation_no`s.** (Current code
  `generation/*/target_conversations` takes the first N in sorted order — change this.)
- Revisit N per condition (50 → maybe more; generation cost permitting).

**TASK 2 — "Build the metric that wows them" (local / measurement). STATUS: CAN START NOW.**
- First concrete step (no tagger, no GPU): compute the **human dialogue-act signature** over
  the full 1,155 SwDA conversations from the gold `act_tag` columns — the reference chart.
- Then: LLM tagging (DialogTag), distribution + transition JSD, noise floor + topic control,
  the assistant-acts-outside-taxonomy layer, the tagger-accuracy validation.
- In parallel: the real main-body replication (fix the validation gap) + alignment trajectory.

## Relationship to the old extension (`RESEARCH.md`)

`RESEARCH.md` proposed TSI/ADI, CAS, CED. Supervisor (Lotem, ~2026-07-05) had approved
TSI/ADI + CED. After the harsh analysis-rigor review, the headline is now the **dialogue-act
signature** (proven technique, free gold baseline, directly measures interactional
structure). CED (within-topic dispersion) and the ADI intuition can live on as *supporting*
axes, validated properly. CAS is demoted (behaved like a lexical-overlap artifact).

## Reading list (start here)

- Stolcke et al. 2000, *Dialogue Act Modeling…* — the 42-tag scheme + transition grammar. https://arxiv.org/abs/cs/0006023
- `cgpotts/swda` — the gold corpus / act definitions. https://github.com/cgpotts/swda
- Jurafsky & Martin, *SLP* 3rd ed., "Discourse & Dialogue" chapter (free online).
- Clark & Brennan 1991, "Grounding in Communication" — why backchannels/repair mean something.
- Yeh et al. 2021 — the honest "reference-free metrics are weak" survey. https://arxiv.org/abs/2106.03706
- Duran, Paxton & Fusaroli 2019, ALIGN — what cosine_semanticL computes.
- Holtzman et al. 2020, "Neural Text Degeneration" — explains the repetition/decoding traps.
