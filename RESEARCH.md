# Research Memo: Social Continuation vs Assistant-Like Task Completion

> **STATUS (2026-07-14): refocused.** The headline extension is now the **dialogue-act
> structural signature** — see `RESEARCH_DIALOGUE_ACTS.md`. The three metrics below
> (TSI/ADI, CAS, CED) are now *supporting/secondary*, not the main story. Keep this memo for
> the CED (within-topic dispersion) and ADI ideas, but validate any of them properly (human
> κ) before use. CAS is demoted (lexical-overlap artifact).

## Short Version

Our project asks whether the way we generate LLM conversations changes how human-like
they are compared with Switchboard telephone conversations.

The original paper already measures useful things:

- turn length
- conceptual, syntactic, and lexical alignment
- coordination markers like "oh", "okay", and "uh-huh"
- openings and closings
- human judgments of whether excerpts are human or AI-generated

Those metrics are important, but they miss one major difference:

**Human casual conversations often do not try to solve anything.**

People often talk just to keep the social interaction going. They react, agree, lightly
disagree, share experiences, ask back, laugh, drift naturally, or show attention. LLMs,
especially assistant-tuned models, often do something different: they explain, advise,
recommend, summarize, list steps, and try to complete a task.

This memo proposes a three-metric framework for that gap.

Metric 1:

**Transcript Sociality Index (TSI)**

Companion diagnostic:

**Assistant Drift Index (ADI)**

Metric 2:

**Context Anchoring Score (CAS)**

Metric 3:

**Conversation Embedding Dispersion (CED)**

Together, these measure three different things:

```text
TSI / ADI = what kind of social action the turns perform
CAS       = whether each next turn is specifically anchored to the previous context
CED       = whether whole conversations are diverse like human conversations or clustered/stereotyped like LLM outputs
```

The main metric is not "does the text look human?" in a vague way. TSI asks:

> How much is this transcript organized like reciprocal social human talk, and how much
> does it drift into assistant-like task completion?

## The Metric, Explained Simply

The recommendation is to use three complementary metrics.

### 1. Transcript Sociality Index / Assistant Drift Index

TSI and ADI are about **what each turn is doing socially**.

Imagine reading a conversation turn by turn. For each turn, we ask:

**What is this turn doing?**

For example:

- Is it giving advice?
- Is it explaining something in a polished assistant style?
- Is it simply reacting socially?
- Is it sharing a personal experience?
- Is it keeping the topic going?
- Is it asking the other person something back?
- Is it saying "yeah", "right", or "uh-huh" as listener feedback?
- Is it repairing confusion, like "wait, what do you mean?"

Then we count the balance of these behaviors.

A human casual conversation should have lots of:

- social continuation
- personal stance and experience sharing
- grounding and backchannels
- phatic/social bonding
- natural topic maintenance or drift
- occasional repair/clarification

An assistant-like conversation will have more:

- advice
- recommendations
- step-by-step solutions
- generic explanations
- summaries
- list-like answers
- "there are several reasons..." style responses

So the idea is:

```text
Human-social conversation = high social continuation + high reciprocity + low assistantness
```

The easiest first metric is the companion diagnostic:

```text
Assistant Drift Index = proportion of turns that are advice/help/explanation/task-completion turns
```

If a conversation has 40 turns and 12 turns are advice-like or generic assistant-like
explanations:

```text
ADI = 12 / 40 = 0.30
```

That means 30% of the conversation drifted into assistant mode.

Then the fuller metric, TSI, adds the human-social side:

```text
Transcript Sociality Index =
  high social continuation
+ high personal stance / experience sharing
+ high grounding and repair
+ healthy topic maintenance / natural drift
+ balanced reciprocity between the two speakers
- assistant-like advice/explanation drift
```

The exact formula can be finalized after a small annotation pilot. The important part is
the coding framework.

### 2. Context Anchoring Score

CAS is about **whether a response belongs specifically after the previous turn**.

Human conversation is locally dependent. A next turn usually shows that the speaker heard
and understood the previous turn.

Example of a strongly anchored pair:

```text
A: I waited three hours at the airport.
B: Oh no, was your flight delayed?
```

The response clearly belongs after the airport turn.

Example of a weakly anchored pair:

```text
A: I waited three hours at the airport.
B: That can be a frustrating experience for many people.
```

This response is coherent, but generic. It could fit many situations:

```text
My car broke down.
I failed the exam.
My package got lost.
I waited three hours at the airport.
```

CAS measures whether the real previous turn is more semantically related to the response
than random wrong previous turns.

Simple formula:

```text
CAS(turn) = similarity(real previous turn, response)
          - average similarity(random wrong previous turns, same response)
```

High CAS means:

```text
this response belongs here
```

Low CAS means:

```text
this response is generic and could fit many places
```

This can be computed automatically with sentence embeddings and cosine similarity.

### 3. Conversation Embedding Dispersion

CED is about **whether full conversations are varied or stereotyped**.

The idea:

Embed each full conversation as one vector. Then check whether conversations from a group
are spread out or clustered tightly together.

Hypothesis:

```text
Switchboard conversations should be more dispersed.
LLM conversations may cluster more tightly because they repeat similar structures,
styles, openings, explanations, and endings.
```

Simple formula:

```text
CED(group) = average distance from each conversation embedding to the group centroid
```

High CED means:

```text
conversations in this group are diverse
```

Low CED means:

```text
conversations in this group are stereotyped or repetitive
```

Important warning:

CED can accidentally measure topic diversity instead of humanness. To make it valid, we
should compute it within matched topics when possible:

```text
Within the same topic, are LLM conversations more clustered than Switchboard conversations?
```

## Why This Is Novel

Existing dialogue metrics usually ask whether a response is coherent, relevant,
interesting, specific, engaging, or human-like overall.

They usually do not ask:

- Is this conversation trying to complete a task?
- Is it behaving like a helpful assistant?
- Is it socially continuing without needing a goal?
- Are both speakers co-maintaining the interaction?
- Does it feel like two people spending time together rather than an assistant solving
  something?

That is the research gap.

Our proposed contribution is broader than one score. It measures three missing dimensions:

1. **Social action profile**: what each turn is doing interactionally.
2. **Context anchoring**: whether each response depends on the actual previous turn.
3. **Conversation-level diversity**: whether generated conversations are varied or
   stereotyped.

This is different from the original paper's metrics. It can use turn length, markers, and
alignment as supporting information, but it does not repeat them. It measures new axes:

```text
social human interaction <----> assistant-like task completion
context-specific response <----> generic movable response
conversation diversity <----> LLM stereotypy
```

## Project Fit

This metric fits our generation architecture question:

- C1: all-at-once generation
- C2: single-model turn-by-turn generation
- C3: independent same-model interlocutors
- C4: independent different-model interlocutors

Hypothesis:

Independent interlocutors may reduce assistant-like behavior because each speaker has its
own perspective. They may increase reciprocal social continuation, stance sharing, and
natural turn-by-turn interaction.

This also fits our prompt levels:

- P0 basic prompt may produce more assistant-like behavior
- P1 spoken prompt should reduce assistantness and increase social continuation
- P2 few-shot prompt may improve surface human-likeness, but must be handled carefully
  because it may mimic Switchboard examples too directly

## Recommended Metric Names

Best option:

**Transcript Sociality Index (TSI)**

Why: clear, academic-sounding, and says exactly what it measures.

Useful companion:

**Assistant Drift Index (ADI)**

Why: very interpretable. It tells us how much the conversation slips into assistant mode.

Second metric:

**Context Anchoring Score (CAS)**

Why: clear and computational. It tells us whether each next turn is anchored to the actual
previous turn.

Third metric:

**Conversation Embedding Dispersion (CED)**

Why: clear and visual. It tells us whether full conversations are scattered/diverse or
clustered/stereotyped.

Other possible names:

- Conversational Sociality Index
- Interactional Sociality Score
- Social Continuation vs Task Completion Score
- Phatic Continuation Score
- Assistantness vs Sociality Score
- Turn Contingency Score
- Local Responsiveness Score
- Conversation Stereotypy Score
- Conversation Embedding Diversity

## Annotation Labels

The labels should be multi-label. A turn can have more than one label.

Example:

```text
"Oh yeah, I had that happen once too. Did you ever figure out why?"
```

This could be:

- phatic/social bonding
- personal experience sharing
- social continuation

### A. Task / Help / Advice Oriented

Definition:

A turn is task/help/advice oriented when it frames the conversation as solving a problem,
giving instructions, recommending actions, or helping someone reach a goal.

Examples:

- "You should try setting a schedule."
- "Here are three things you can do."
- "The best solution is to call them first."
- "I recommend making a list of pros and cons."

Human-like or assistant-like?

Usually assistant-like, especially when frequent or long.

Edge cases:

A short personal suggestion can still be human:

```text
"I'd probably just call them."
```

This is not necessarily assistant-like if it is brief and personal. It may be better
labeled as personal stance.

### B. Social Continuation

Definition:

A turn keeps the social interaction going without primarily solving anything.

Examples:

- "Yeah, I know what you mean."
- "What about you?"
- "That happened to you too?"
- "Oh, so you were already living there then?"
- "That's kind of what I was thinking."

Human-like or assistant-like?

Human-like.

Why it matters:

This is the core of casual conversation. The speaker is not trying to complete a task.
They are keeping the interaction alive.

### C. Phatic / Social Bonding

Definition:

Small social moves whose main function is connection, warmth, rapport, or channel
maintenance rather than information transfer.

Examples:

- "Oh wow."
- "That's funny."
- "I know."
- "Same here."
- "No kidding."
- "You know?"
- laughter tokens when used socially

Human-like or assistant-like?

Human-like when natural.

Edge cases:

Over-polished empathy can become assistant-like:

```text
"I completely understand how challenging that must be for you."
```

This may be phatic in one context, but in many LLM transcripts it sounds like generic
assistant support.

### D. Grounding / Backchanneling

Definition:

Listener feedback showing attention, receipt, or understanding.

Examples:

- "yeah"
- "right"
- "uh-huh"
- "okay"
- "I see"
- "mm-hm"

Human-like or assistant-like?

Human-like when placed responsively.

Important note:

The original paper already counts some coordination markers. Our use of this label should
not be just another marker count. We should code **function**: is the turn actually acting
as grounding or listener feedback in context?

### E. Personal Stance / Experience Sharing

Definition:

A turn shares a personal opinion, feeling, memory, preference, or experience.

Examples:

- "I never liked flying much."
- "My sister did that once."
- "I feel like schools used to be stricter."
- "I always thought that was strange."
- "That happened to me when I was younger."

Human-like or assistant-like?

Strongly human-like for Switchboard-style casual conversation.

Why it matters:

Humans in casual talk often respond by bringing in their own stance or experience. LLMs
often respond by generalizing instead.

### F. Topic Maintenance / Natural Topic Drift

Definition:

How the turn relates to the current topic.

Suggested sublabels:

- F1: maintains or elaborates the current topic
- F2: smoothly shifts to a related topic
- F3: abruptly changes topic
- F4: loops or repeats without progress

Examples:

F1:

```text
"Yeah, and the traffic around there is terrible too."
```

F2:

```text
"That reminds me of when we moved apartments."
```

F3:

```text
"Anyway, what is your favorite kind of computer?"
```

F4:

```text
"Yes, it is very interesting. It is definitely interesting."
```

Human-like or assistant-like?

F1 and F2 are human-like. F3 and F4 are negative.

Important note:

Human conversation does drift. Perfect topic discipline can feel unnatural. We want
natural topic movement, not rigid task focus.

### G. Repair / Clarification

Definition:

A turn handles trouble in understanding, wording, hearing, reference, or meaning.

Examples:

- "What do you mean?"
- "Wait, which one?"
- "Sorry, I mean my brother."
- "No, not that school, the other one."
- "Say that again?"

Human-like or assistant-like?

Human-like when naturally distributed.

Why it matters:

Repair is central to real interaction. Transcript-only data loses audio timing, but repair
still appears in text.

### H. Generic Assistant-Like Explanation

Definition:

A turn uses polished, generic, explanatory, instructional, or list-like assistant register
that does not feel grounded in the social relationship.

Examples:

- "There are several factors to consider."
- "It is important to remember that..."
- "This topic can be understood in three main ways."
- "Overall, the best approach is..."
- "Many people find that..."

Human-like or assistant-like?

Assistant-like.

Edge cases:

Humans can explain things too. Only use H when the explanation feels generic, polished,
lecture-like, or detached from the two speakers' relationship.

## Which Labels Are Human-Like vs Assistant-Like?

Mostly human-like:

- B: Social Continuation
- C: Phatic / Social Bonding
- D: Grounding / Backchanneling
- E: Personal Stance / Experience Sharing
- F1/F2: Topic Maintenance / Smooth Drift
- G: Repair / Clarification

Mostly assistant-like:

- A: Task / Help / Advice Oriented
- H: Generic Assistant-Like Explanation
- F3/F4: Abrupt Drift / Repetitive Loop

Context-dependent:

- A can be human if brief, personal, and topic-natural
- C can become assistant-like if formulaic
- D can become unnatural if overproduced or badly placed
- H can overlap with ordinary explanation, so annotators need examples

## Possible Formulas

Let conversation `c` have `N` turns.

Let `I_label(t)` be 1 if turn `t` has that label, and 0 otherwise.

### Label Rate

```text
r_label(c) = number of turns with label / total turns
```

Example:

```text
r_A = advice/task/help turns / total turns
```

### Assistant Drift Index

This is the simplest and most understandable subscore.

```text
ADI(c) = r_A(c) + r_H(c)
```

Interpretation:

- low ADI = little assistant-like task completion
- high ADI = much assistant-like advice/explanation behavior

If needed, cap at 1.0 because turns can be multi-label:

```text
ADI(c) = min(1, r_A(c) + r_H(c))
```

### Social Continuation Rate

```text
Social(c) = r_B(c) + r_C(c) + r_E(c) + r_G(c) + 0.5 * r_D(c)
```

Why D gets half weight:

Backchannels are important, but if we weight them too heavily we risk repeating the
original paper's marker-count analysis.

### Topic Health

```text
TopicHealth(c) = r_F1(c) + r_F2(c) - r_F3(c) - r_F4(c)
```

### Reciprocity

For each speaker, compute social-label rate:

```text
SpeakerSocial = rate of B, C, E, or G turns by that speaker
```

Then:

```text
Reciprocity(c) = 1 - abs(SpeakerSocial_A - SpeakerSocial_B)
```

Interpretation:

Human conversations should not have one speaker acting like the only source of social
content while the other behaves like a passive user or assistant.

### Switchboard Similarity

Build a vector for each conversation:

```text
V(c) = [
  r_A, r_B, r_C, r_D, r_E, r_F1, r_F2, r_F3, r_F4, r_G, r_H,
  selected transition rates
]
```

Example transition rates:

- probability of social continuation after personal sharing
- probability of backchannel after personal sharing
- probability of repair after unclear turn
- probability of assistant-like explanation after any user-like statement

Compare each generated conversation to the Switchboard baseline distribution using
Jensen-Shannon divergence:

```text
SB_Sim(c) = 1 - JSD(V(c), V_Switchboard)
```

### Full Transcript Sociality Index

One possible formula:

```text
TSI(c) = 100 * [
  0.35 * SB_Sim(c)
+ 0.20 * Social_norm(c)
+ 0.15 * TopicHealth_norm(c)
+ 0.15 * Reciprocity(c)
+ 0.15 * (1 - ADI_norm(c))
]
```

Important:

The first version does not need to start with this full formula. For the course project,
it may be clearer to report:

- ADI: assistant drift
- social continuation rate
- personal stance rate
- grounding/repair rate
- topic health
- then optionally combine them into TSI

That is easier to explain and less likely to look arbitrary.

## Context Anchoring Score

### What CAS Measures

CAS measures local turn-by-turn responsiveness.

Question:

```text
Does this response fit its real previous turn better than random previous turns?
```

This is different from TSI/ADI. TSI/ADI asks what social function a turn performs. CAS
asks whether the response is specifically tied to the actual local context.

### How CAS Works In Code

For each conversation:

```text
Turn 1: I waited three hours at the airport.
Turn 2: Oh no, was your flight delayed?
Turn 3: Yeah, they said there was bad weather.
Turn 4: That happened to me once in Denver.
```

The real pair is:

```text
context = Turn 1
response = Turn 2
```

Then create wrong pairs automatically by shuffling contexts:

```text
wrong context 1 = My dog hates the rain.
response        = Oh no, was your flight delayed?

wrong context 2 = I started learning piano.
response        = Oh no, was your flight delayed?

wrong context 3 = We went to a restaurant yesterday.
response        = Oh no, was your flight delayed?
```

We do not need a human to label the wrong turns. They are wrong because they come from
other places in the dataset.

Then compute:

```text
real_similarity = embedding_similarity(real_context, response)
wrong_similarity = average embedding_similarity(wrong_context_i, response)
CAS = real_similarity - wrong_similarity
```

Example:

```text
real_similarity = 0.72
wrong_average   = 0.31
CAS             = 0.41
```

This is high. The response is context-specific.

Generic response example:

```text
real context: I waited three hours at the airport.
response: That sounds frustrating.
```

Possible scores:

```text
real_similarity = 0.48
wrong_average   = 0.42
CAS             = 0.06
```

This is low. "That sounds frustrating" can fit many situations.

### CAS Pseudocode

```python
for response_turn in all_turns_with_previous_turn:
    real_context = previous_turn(response_turn)
    response = response_turn.text

    real_score = cosine_similarity(
        embed(real_context.text),
        embed(response)
    )

    wrong_contexts = sample_random_previous_turns(
        exclude_same_conversation=True,
        same_topic_if_possible=True,
        k=20
    )

    wrong_scores = [
        cosine_similarity(embed(ctx.text), embed(response))
        for ctx in wrong_contexts
    ]

    cas_turn = real_score - mean(wrong_scores)
```

Conversation-level CAS:

```text
CAS(conversation) = average CAS over all eligible turns
```

Condition-level CAS:

```text
CAS(condition) = average CAS over conversations in that condition
```

### CAS Expected Results

Expected pattern:

```text
Switchboard: high CAS
C1 all-at-once: lower CAS if it produces scripted/generic turns
C2 turn-by-turn: possibly higher than C1
C3/C4 independent interlocutors: possibly highest among LLM conditions
```

### CAS Literature Support

Conversation analysis has the idea of the "next-turn proof procedure": a next turn shows
how the speaker understood the prior turn. CAS turns that idea into an automatic
transcript-level approximation.

Related dialogue-evaluation work:

- Ghazarian et al. (2019), "Better Automatic Evaluation of Open-Domain Dialogue Systems
  with Contextualized Embeddings"
- Li et al. (2016), "A Diversity-Promoting Objective Function for Neural Conversation
  Models"
- Huang et al. (2020), "GRADE"
- Dziri et al. (2019), "Evaluating Coherence in Dialogue Systems using Entailment"

Useful URLs:

- https://aclanthology.org/W19-2310/
- https://arxiv.org/abs/1510.03055
- https://arxiv.org/abs/2010.03994
- https://aclanthology.org/N19-1381/

### CAS Limitations

CAS may reward topical similarity but miss social fit. For example, a response can be on
topic but socially unnatural.

Mitigations:

- use CAS together with TSI/ADI
- sample wrong contexts from the same topic when possible
- exclude openings/closings or score them separately
- compare CAS to human judgments of local response fit

## Conversation Embedding Dispersion

### What CED Measures

CED measures whether full conversations in a group are diverse or stereotyped.

Question:

```text
Are conversations from this condition spread out in embedding space, or do they cluster
tightly together?
```

This is different from both TSI and CAS:

```text
TSI / ADI = social function of turns
CAS       = local context anchoring between adjacent turns
CED       = global variety of full conversations
```

### How CED Works In Code

For each conversation, create one text:

```text
Speaker A: ...
Speaker B: ...
Speaker A: ...
Speaker B: ...
```

Then embed the whole conversation.

For each group, such as Switchboard or C1-P0:

```text
centroid = average of all conversation embeddings in that group
CED = average distance from each conversation embedding to that centroid
```

Formula:

```text
CED(group) = mean distance(embedding_i, centroid_group)
```

Using cosine distance:

```text
distance = 1 - cosine_similarity(embedding_i, centroid_group)
```

### CED Pseudocode

```python
for group in groups:
    conv_embeddings = [
        embed(full_conversation_text(conv))
        for conv in conversations[group]
    ]

    centroid = mean_vector(conv_embeddings)

    distances = [
        cosine_distance(vec, centroid)
        for vec in conv_embeddings
    ]

    ced_group = mean(distances)
```

### CED Visualization

Use PCA, UMAP, or t-SNE to plot conversation embeddings:

```text
point = one full conversation
color = source or condition
```

Possible visual result:

```text
Switchboard: scattered cloud
LLM C1-P0: tight cluster
LLM C3/C4: maybe more spread out
```

### CED Expected Results

Expected pattern:

```text
Switchboard: higher dispersion
LLM conversations: lower dispersion if they are formulaic
P1/P2: may increase or decrease dispersion depending on prompt strength
C3/C4: may increase dispersion if independent interlocutors create more varied dynamics
```

### CED Important Warning

CED can accidentally measure topic diversity instead of human-likeness.

Bad comparison:

```text
Switchboard has many topics, LLM has fewer topics.
Result: Switchboard is more scattered.
Problem: maybe this is just topic variety.
```

Better comparison:

```text
Within each matched topic, compare Switchboard dispersion vs LLM dispersion.
```

Best reporting:

```text
CED within topic
CED averaged across topics
```

### CED Limitations

- Full-conversation embeddings may mix topic, style, speaker identity, and structure.
- Long conversations may exceed embedding model limits and need chunking.
- PCA/UMAP/t-SNE plots are useful visually but should not be the only evidence.
- Dispersion is not automatically good; very scattered LLM outputs could also be incoherent.

Mitigations:

- topic-match samples
- use numeric dispersion, not only plots
- compare with coherence/CAS
- inspect examples near the cluster center and outliers

## What Existing Metrics Capture

### Mayor, Bietti & Bangerter (2025)

What they measure:

- turn length
- alignment
- coordination markers
- openings and closings
- human discrimination

Why it matters:

This is our baseline paper and the foundation for our replication/extension.

What it misses:

It does not directly measure whether a transcript is socially continuing or trying to
complete a task.

### USR

Source:

Mehri & Eskenazi (2020), "USR: An Unsupervised and Reference Free Evaluation Metric for
Dialog Generation."

What it measures:

Reference-free dialogue quality using several interpretable qualities.

Why it matters:

It shows that dialogue quality can be evaluated without a single gold response.

What it misses:

It is still mostly about response quality, not the social-vs-assistant distinction.

URL:

https://arxiv.org/abs/2005.00456

### FED

Source:

Mehri & Eskenazi (2020), "FED: A Fine-grained Evaluation Dataset for Open-Domain Dialogue."

What it measures:

Fine-grained dialogue qualities such as coherence, relevance, interestingness, diversity,
depth, likeability, informativeness, and error recovery.

Why it matters:

It is close to what we need because it is fine-grained and dialogue-level.

What it misses:

It does not directly separate social continuation from assistant-like task completion.

URL:

https://aclanthology.org/2020.sigdial-1.28/

### SSA / Meena

Source:

Adiwardana et al. (2020), "Towards a Human-like Open-Domain Chatbot."

What it measures:

Sensibleness and specificity.

Why it matters:

It is an influential human-likeness metric for open-domain chatbots.

What it misses:

A response can be sensible and specific while still sounding like an assistant.

URL:

https://research.google/pubs/towards-a-human-like-open-domain-chatbot/

### DialogRPT

Source:

Gao et al. (2020), "Dialogue Response Ranking Training with Large-Scale Human Feedback
Data."

What it measures:

It ranks responses using signals from large-scale human feedback.

Why it matters:

It is useful for engagement and response preference.

What it misses:

Engagement is not the same as non-task-oriented social interaction.

URL:

https://arxiv.org/abs/2009.06978

### GRADE

Source:

Huang et al. (2020), "GRADE: Automatic Graph-Enhanced Coherence Metric for Evaluating
Open-Domain Dialogue Systems."

What it measures:

Dialogue coherence and topic flow.

Why it matters:

Topic maintenance is part of TSI.

What it misses:

It does not distinguish human social continuation from assistant-like explanation.

URL:

https://arxiv.org/abs/2010.03994

### Switchboard Dialog Act / SWBD-DAMSL

What it measures:

Dialogue act categories in Switchboard, such as statements, questions, agreements,
backchannels, and repairs.

Why it matters:

It can support our annotation scheme and gives us a precedent for coding turns by
interactional function.

What it misses:

It was not designed to detect assistant-like behavior because it was created for human
telephone conversations.

URL:

https://convokit.cornell.edu/documentation/switchboard.html

## Literature and Concepts

### Phatic Communication / Phatic Communion

Main idea:

Some talk exists mainly to maintain social connection, not to exchange information.

Why it matters:

This directly supports our claim that human conversation is not always task-oriented.

Supports social continuation vs task completion?

Yes. It is one of the strongest theoretical foundations for the metric.

Key sources:

- Malinowski (1923), "The Problem of Meaning in Primitive Languages"
- Laver (1975), "Communicative Functions of Phatic Communion"

### Casual Conversation

Main idea:

Casual conversation is full of stance-taking, topic drift, humor, repetition, shared
experience, and relational work.

Why it matters:

Switchboard conversations are topic-prompted, but they still contain casual telephone
interaction. We should compare LLMs to this kind of talk, not only to task dialogue.

Supports social continuation vs task completion?

Yes.

Useful source:

- Eggins & Slade (1997), "Analysing Casual Conversation"

### Conversational Grounding

Main idea:

Conversation requires participants to establish that they understand each other well
enough for current purposes.

Why it matters:

Backchannels, acknowledgments, repair, clarification, and confirmations are not just
surface markers. They are part of how people coordinate interaction.

Supports social continuation vs task completion?

Yes, especially labels D and G.

Source:

- Clark & Brennan (1991), "Grounding in Communication"

### Human Interaction Engine

Main idea:

Human conversation is a rapid, structured, cooperative interaction system.

Why it matters:

It supports treating conversation as social coordination, not just alternating generated
texts.

Supports social continuation vs task completion?

Yes.

Source:

- Levinson (2006), "On the Human Interaction Engine"

### From Text to Talk

Main idea:

Language technology often treats dialogue as text, but real conversation includes
sequential organization, turn-taking, timing, and social action.

Why it matters:

Our metric is transcript-only, but it tries to recover some interactional structure by
labeling what turns do.

Supports social continuation vs task completion?

Yes.

Source:

- Dingemanse & Liesenfeld (2022), "From text to talk"

URL:

https://aclanthology.org/2022.acl-long.385/

### Task-Oriented vs Open-Domain Dialogue

Main idea:

Task-oriented dialogue is evaluated by goal completion, slot filling, task success, or
user satisfaction. Open-domain dialogue is usually evaluated by coherence, relevance,
engagement, naturalness, or human-likeness.

Why it matters:

Our LLM conversations are supposed to simulate casual human telephone talk, not complete
tasks. If they behave like task systems, that is a failure mode.

Supports social continuation vs task completion?

Yes.

### LLM Assistant-Like Behavior

Main idea:

Instruction tuning and human preference training often teach models to be helpful,
harmless, explanatory, and solution-oriented.

Why it matters:

This may explain why LLMs drift into advice-giving and generic explanation even when asked
to simulate casual conversation.

Supports social continuation vs task completion?

Yes.

Useful source:

- Ouyang et al. (2022), "Training language models to follow instructions with human
  feedback"

URL:

https://arxiv.org/abs/2203.02155

## Operationalization

### Data Sources

Use:

- Switchboard human transcripts
- original LLM conversations from Mayor et al., if available
- our generated C1-C4 / P0-P2 conversations

Optional extra corpora:

- CALLHOME English
- Santa Barbara Corpus of Spoken American English

Switchboard is enough for a first version, especially because it is already the baseline
for the paper. A second corpus would strengthen the claim but is not required.

### Three-Metric Analysis Plan

Run three metrics on the same conversation set:

1. **TSI/ADI**: manual or LLM-assisted labels for social-vs-assistant turn function.
2. **CAS**: automatic embedding comparison of real adjacent turns vs shuffled wrong
   contexts.
3. **CED**: automatic full-conversation embedding dispersion within each source,
   condition, and ideally topic.

Together:

```text
TSI/ADI tells us what the turns are doing.
CAS tells us whether turns are locally responsive.
CED tells us whether whole conversations are globally diverse or stereotyped.
```

### What to Label

Primary unit:

- turn

Secondary unit:

- adjacent turn pair

Conversation-level:

- aggregate rates and trajectory

Why not only full conversations?

Full-conversation ratings are useful, but they are harder to debug. Turn labels let us
explain exactly why one condition is more assistant-like.

### What Can Be Rule-Based

Some assistant-like signals can be detected automatically:

- numbered lists
- "you should"
- "you could try"
- "I recommend"
- "there are several"
- "it is important to"
- "overall"
- "in conclusion"
- long lecture-like turns
- explicit summaries

Some backchannel tokens can also be detected:

- "yeah"
- "right"
- "uh-huh"
- "okay"
- "I see"

But raw detection is not enough. Context matters.

### What Needs Manual Annotation

Manual annotation is needed for:

- whether advice sounds human or assistant-like
- whether a backchannel is actually grounding
- whether topic drift is natural or abrupt
- whether empathy is phatic or formulaic assistant talk
- whether a personal statement is real stance-sharing or generic filler

### What Can Be LLM-Labeled

LLMs can help label turns after we create a small human-labeled validation set.

Recommended approach:

1. Humans label 200-600 turns.
2. Compute agreement.
3. Ask an LLM to label the same turns.
4. Compare LLM labels to human adjudicated labels.
5. Only then use LLM labels for larger-scale analysis.

Risk:

LLMs may be biased toward polished assistant-like text, so they cannot be the only judge.

## Pilot Experiment Plan

### Pilot Sample

Minimum:

- 100 Switchboard turns
- 100 generated turns

Better:

- 300 Switchboard turns
- 300 generated turns

For condition comparison:

- 3-5 conversations per C1-C4/P0-P2 condition for the pilot
- 10 conversations per condition if time allows

### Annotation Workflow

1. Create a one-page codebook with labels A-H.
2. Two annotators label the same 100-200 turns.
3. Discuss disagreements.
4. Refine edge cases.
5. Label a larger blind sample.
6. Compute agreement.

Agreement metrics:

- Cohen's kappa per label for two annotators
- Krippendorff's alpha if more annotators or missing labels

Target:

- 0.60 is acceptable for a pilot
- 0.70 or higher is stronger

### Validation Plan

Known-groups validation:

- Switchboard should have high TSI and low ADI.
- LLM P0 outputs should have lower TSI and higher ADI.

Condition validation:

- P1 should reduce ADI compared with P0.
- C3/C4 may increase reciprocity compared with C1/C2.
- P2 may look better, but must be interpreted carefully because of few-shot mimicry.

Convergent validation:

- TSI should correlate with human ratings of "sounds like two people socially talking."

Discriminant validation:

- TSI should not just be the same as turn length.
- TSI should not just be the same as marker count.
- TSI should not just be the same as alignment.
- CAS should not just be the same as adjacent-turn embedding similarity; the shuffled
  negative baseline is essential.
- CED should not just be topic diversity; compute it within matched topics if possible.

Optional human survey:

Show participants short transcript excerpts and ask:

- Which sounds more like two people casually talking?
- Which sounds more like an assistant completing a task?
- Which sounds more human?

Then compare their ratings with TSI and ADI.

## Data Requirements

Needed:

- speaker-labeled transcripts
- turn boundaries
- condition labels: C1-C4 and P0-P2
- topic metadata if available
- Switchboard comparison sample
- annotation table with turn IDs and labels
- embedding model for turn-level and conversation-level embeddings
- random negative sampling protocol for CAS
- topic-matched grouping if possible for CED

Recommended annotation table columns:

```text
conversation_id
condition
source
topic
turn_index
speaker
turn_text
label_A_task_help_advice
label_B_social_continuation
label_C_phatic_bonding
label_D_grounding_backchannel
label_E_personal_stance_experience
label_F_topic_code
label_G_repair_clarification
label_H_assistant_explanation
annotator_id
notes
```

Should we use full conversations or snippets?

Use both:

- full conversations for aggregate TSI/ADI
- selected snippets for human survey validation

Should we include openings and closings?

For the main metric, focus on main-body topic talk. Openings and closings are already
studied in the original paper. We can report separate scores:

- main-body TSI
- opening/closing excluded
- optional full-conversation TSI for completeness

Should samples be topic-matched?

Yes, if possible. Some topics naturally invite more advice, so topic matching reduces
confounds.

## Risks and Limitations

### Switchboard Is Not Fully Free Conversation

Switchboard calls are topic-prompted. They are casual telephone conversations, but not
completely spontaneous social hangouts.

Mitigation:

Use Switchboard because it is the paper's baseline, but describe it as topic-prompted
casual telephone conversation.

### Advice Is Not Always Non-Human

Humans give advice too. The problem is not any advice at all. The problem is frequent,
generic, polished, solution-oriented assistant advice.

Mitigation:

Annotate short personal advice differently from generic assistant advice.

### Assistantness Depends On Topic

Some topics naturally cause advice or explanation.

Mitigation:

Topic-match Switchboard and generated samples, or include topic as a control.

### LLM Labeling May Be Biased

LLMs may prefer coherent, polished, assistant-like responses.

Mitigation:

Validate LLM labels against human annotations.

### Labels Are Subjective

Social continuation and phatic bonding require judgment.

Mitigation:

Use a clear codebook, examples, and inter-annotator agreement.

### Transcript-Only Data Misses Audio

Timing, overlap, intonation, pauses, and laughter quality are important in real
conversation.

Mitigation:

State that this is a transcript-level metric. It measures only what is visible in text.

### Marker Double-Counting

Grounding/backchanneling overlaps with the original marker analysis.

Mitigation:

Use D as a functional label in context, not just a count of "yeah" or "uh-huh."

## Suggested Reporting Structure

For the final project/poster, report:

1. Original paper metrics:
   - words per turn
   - alignment
   - marker rates
2. New extension:
   - Assistant Drift Index
   - Social Continuation Rate
   - Personal Stance Rate
   - Topic Health
   - optional combined TSI
   - Context Anchoring Score
   - Conversation Embedding Dispersion
3. Main claim:
   - generation architecture changes not only surface similarity, but also the social
     action profile, local responsiveness, and global diversity/stereotypy of generated
     conversation

## Concrete Next Steps

1. Agree as a team on the metric package: TSI/ADI + CAS + CED.
2. Create a short annotation codebook from labels A-H.
3. Label 200 pilot turns: 100 Switchboard and 100 generated.
4. Check which labels are confusing.
5. Revise definitions for A vs E and C vs H.
6. Compute ADI first because it is easiest to explain.
7. Add social-label rates and optional combined TSI.
8. Compute CAS with shuffled wrong-context baselines.
9. Compute CED with full-conversation embeddings, preferably within matched topics.
10. Compare whether C1-C4/P0-P2 differ on all three dimensions.

## One-Sentence Explanation For Teammates

We are measuring whether LLM conversations behave like two humans socially continuing a
casual conversation, whether each turn is locally anchored to the previous turn, and
whether full conversations are diverse like human conversations or clustered/stereotyped
like LLM outputs.

## One-Sentence Novelty Claim

Unlike prior dialogue metrics that measure coherence, relevance, alignment, or marker
frequency, our metric package measures three missing dimensions of human-like conversation:
social action, local contextual anchoring, and global conversation-level diversity.
