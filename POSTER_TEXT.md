# Poster text — draft v1 (2026-10-05)

Ready to paste onto the poster template. Every number comes from
`results_v3/RESULTS_REPORT.md`; figures are in `results_v3/figures/` (PNG for the poster, PDF
for print). **[brackets]** = still to fill in. Aim: the main message readable in 30 seconds
(title + the four bold finding headlines + Fig 1).

---

## TITLE

**Who's listening? How the way LLMs generate a conversation shapes its structure**

Alternative: *"Splitting the speakers doesn't make LLM phone calls more human"*

[Names] · Technion, course 0950280 · Supervisor: [name] · Extends Mayor et al. (2025),
*Can Large Language Models Simulate Spoken Human Conversations?*, Cognitive Science.

---

## 1. Motivation (left column, top)

LLMs are increasingly used to simulate conversations as synthetic training data, for social
simulations, and as stand-ins for human participants. Mayor et al. (2025) showed that
LLM-written phone calls differ from real ones. But they only varied the **prompt** and the
**model**. Many simulations differ in something else: **how the conversation is generated**.
One model can write the whole dialogue, or two separate agents can talk to each other.

**Our question:** does the *generation architecture* change the structure of LLM
conversations, and does any architecture bring it closer to real human talk?

---

## 2. Setup (left column)

**Four architectures** (a small diagram for each):

| | How the conversation is produced |
|---|---|
| **C1 All at once** | one model writes the whole call in a single pass |
| **C2 Turn by turn** | one model writes one turn at a time, playing both speakers |
| **C3 Two agents** | two separate instances of the same model, each seeing only its own role |
| **C4 Two models** | Vicuna-13B talks to Mistral-7B |

**× three prompts:** P0 the paper's original prompt · P1 spoken style + a persona card per
speaker · P2 = P1 + a real Switchboard excerpt as an example.

**Data:** 12 conditions × 50 conversations = **600 LLM calls**, on the same 50 topics as
**50 real Switchboard telephone calls** (human reference; the dialogue-act analysis uses all 1,155 Switchboard calls). Same models (Vicuna-13B v1.5,
Mistral-7B-Instruct, 4-bit) and the same sampling settings in every condition, so only the
architecture and the prompt change.

**What we measure:**
- **Dialogue acts:** what each sentence *does* (statement, opinion, backchannel, question,
  agreement…), tagged automatically (DialogTag, trained on Switchboard).
- **Distance from humans:** Jensen–Shannon divergence between act distributions.
- **The paper's metrics:** words per turn, the markers *oh / okay / uh-huh*, and conceptual
  alignment between adjacent turns (ALIGN; Duran et al., 2019).

---

## 3. Findings (centre, the large part)

### ① LLM conversations are missing the listener → **Fig 1**
Humans spend a quarter of their talk on backchannels ("uh-huh", "yeah", "right"), and
**19%** of units are *nothing but* a backchannel. In all 12 LLM conditions that is
**0–0.4%**. *uh-huh* never occurs: 0 per 100 words in every condition, vs 1.03 for humans.
The LLMs react with "Yeah, …" at the start of a long turn, but they never just listen.

### ② The listener's space is filled with opinions and questions → **Fig 2**
Opinions: humans 14% of units, LLMs 21–38% (architecture averages). Questions: humans 5%, LLMs ~15%.
The LLM speakers take turns *presenting* rather than *listening*.

### ③ Separate agents do not make it more human; they make it less → **Figs 3, 4**
- Closest to human structure: **C1, all at once** (distance 0.12–0.13; sampling noise ≈ 0.002).
- Two-agent setups (C3, C4) are further away (0.15–0.22) and talk in long, message-like
  turns: **39–47 words** (C3) and **95–109 words** (C4) per turn, vs **14** for humans.
- Prompting helps less than the choice of architecture: P1 is usually the best prompt, but
  the gap between architectures is bigger than the gap between prompts.

### ④ LLM speakers converge on each other; humans don't → **Fig 5**
Conceptual alignment between turns stays **flat over a human call** (p = 0.89) but **rises
in 9 of 12 LLM conditions** (Wilcoxon p < 0.05). Polite closings don't explain it: the rise
holds without the last turns. Most of the LLMs' *extra* alignment, however, is just
**turn length**: long turns look alike. Controlling for length, C4's gap falls from +0.32
to +0.02–0.04.

---

## 4. Are the results trustworthy? (right column)

- **Three independent measures agree** on the missing listener: the tagger, a simple
  word rule, and a backchannel word list.
- **Robust to data problems:** removing echo-loop turns or the 10 conversations that drifted
  into Korean changes every distance by ≤ 0.01.
- **Tagger checked by hand:** 4 annotators labelled 100 LLM sentences. The tagger matches
  them **[X]%** of the time (annotators agree with each other **[Y]%**); on human speech it
  is 72%. Its main error (calling opinions plain statements) makes our opinion gap
  *conservative*.
- **Replication anchor:** our human reference reproduces the paper's numbers (14.4 vs 13.97
  words/turn; uh-huh 1.03 vs 1.03).

---

## 5. Limitations

- Two open 7–13B models, 50 conversations per condition; larger and closed models may differ.
- Conversation endings are helped by the harness (a turn countdown), so we don't claim
  results about openings and closings.
- The P2 prompt shows real Switchboard text, so its marker counts partly reflect the example.
- Strict A/B turn alternation is built into every architecture; humans don't have it.

## 6. Take-home & future work

**Take-home:** *Giving each simulated speaker its own agent does not make LLM conversations
more human. The missing ingredient is the listener, and no architecture or prompt we
tested brings it back.*

**Future work:** let the *listener* decide when to speak and allow short reactions
("listener-decides" turn-taking); test larger and closed models; other conversation types
and languages.

---

### Figure captions (short versions for the poster)

- **Fig 1.** Backchannels (a) as tagged and (b) as stand-alone listener units. Humans:
  black. Bar shade = prompt P0/P1/P2.
- **Fig 2.** What the talk is made of: dialogue-act mix, humans vs each architecture
  (mean over prompts).
- **Fig 3.** Distance of each condition's dialogue-act mix from humans. Dashed: the tagger's
  own error on human talk; dotted: sampling noise (random human samples of the same size).
- **Fig 4.** Mean words per turn (±95% CI); dashed line = humans.
- **Fig 5.** (a) Alignment in the first vs second half of a conversation (prompt P1);
  (b) the gap from humans before (outline) and after (filled) controlling for turn length.
