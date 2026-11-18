# Conceptual alignment: combined run (Vicuna + Qwen3.5-9B + Switchboard)

One ALIGN run over all corpora together, so every condition and the human reference
share one vocabulary filter and are directly comparable. Conditions prefixed `Q-` are
Qwen3.5-9B (workshop runs); unprefixed C1-C4 are the original Vicuna data.

Environment check: re-running ALIGN on the original set (Vicuna + Switchboard) with this
environment (align 0.1.1, gensim 4.4.0, nltk 3.10.4, word2vec-google-news-300)
reproduced the published per-pair values (17,947 pairs, mean |diff| 0.00002;
Switchboard 0.550).

Caveat: ALIGN keeps only "content words", chosen by frequency over the whole set
processed together, so absolute values depend on which corpora are included
(Switchboard: 0.550 with Vicuna only, 0.461 here). Compare conditions within one run
(differences vs SB), not across runs.
