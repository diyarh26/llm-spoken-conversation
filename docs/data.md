# Data card: 600 simulated telephone conversations

`data/generated/` holds every conversation generated for this study: **12 conditions × 50
conversations**, one JSON file each. Browse them in the
[conversation explorer](https://diyarh26.github.io/llm-spoken-conversation/).

| | |
|---|---|
| **What** | English telephone conversations between two strangers, written by LLMs |
| **How many** | 600 (4 generation architectures × 3 prompts × the same 50 topics) |
| **Models** | Vicuna-13B v1.5 (16k), Mistral-7B-Instruct v0.2, 4-bit quantized |
| **Grounded in** | 50 real Switchboard calls: their topic instruction and callers' age, gender and education |
| **Size** | 6.3 MB of JSON, about 13,700 turns |
| **License** | CC BY 4.0 (see [License](#license)) |
| **Method** | [`docs/methods.md`](methods.md) |

## Layout

```
data/generated/
├── C1-P0/2121.json … 4876.json      # 50 files per condition
├── C1-P1/ … C4-P2/                  # 12 condition folders
data/align/alignment_turns.csv       # per-turn-pair ALIGN scores (all conditions + humans)
results/figure_data/human_turns.json # turn lengths of the 50 human calls (no text)
```

Condition folders are named `<architecture>-<prompt>`. The 50 file names are Switchboard
conversation numbers, identical in every folder, so a conversation can be compared across all
12 conditions. The list and its sampling seed are in `generation/target_ids.json`.

## Record format

All records share these fields:

| Field | Example | Meaning |
|---|---|---|
| `condition`, `architecture`, `prompt_level` | `"C3-P1"`, `"C3"`, `"P1"` | which of the 12 conditions |
| `conversation_no` | `3508` | the Switchboard call that supplied topic and speakers |
| `topic` | `"Recycling"` | Switchboard topic |
| `sb_prompt` | `"WHAT IS BEING DONE IN YOUR COMMUNITY…"` | the instruction the real callers received |
| `persona_a`, `persona_b` | `{"gender": "woman", "age": 45, "education": "a college degree", "occupation": "a carpenter", "stance": 0, "anchor": 1}` | speaker demographics (from Switchboard) and seeded persona card (`stance`/`anchor` index the pools in `prompts/templates.py`) |
| `seed_turns` | `2` | scripted "Hello!" turns at the start (not generated; dropped in analysis) |
| `decoding` | `{"temperature": 0.8, "top_p": 0.95, …}` | the exact decoding parameters used |

**C1 (all at once)** adds `model`, `raw_output` (the full generated log, one
`ParticipantA: …` line per turn), `hit_token_cap` and `n_new_tokens`.

**C2–C4 (turn by turn)** add:

| Field | Meaning |
|---|---|
| `turns` | `[["ParticipantA", "Hello!"], ["ParticipantB", "Hello!"], ["ParticipantA", "…"], …]` |
| `n_turns` | number of turns, seed turns included |
| `ended_by` | `mutual_closing`, `closing` (one turn after a farewell), `empty_turn` or `turn_cap` |
| `multi_turn_emissions` | how often the model tried to write more than one turn |
| `quality_counters` | token-cap hits, empty-output retries, loop resamples (`dup_resamples`, `dup_kept`) |
| `model` (C2) / `models` (C3, C4) | the model(s) behind each speaker |

## Loading

The analysis reads every record through one cleaner, which drops the scripted greetings,
parses C1's raw log and removes non-speech residue. Use it to see exactly the turns that were
measured:

```python
import json, pathlib
from analysis.analyze import conversation_turns

rec = json.loads(pathlib.Path("data/generated/C4-P1/3508.json").read_text(encoding="utf-8"))
for speaker, text in conversation_turns(rec):
    print(f"{speaker}: {text}")
```

## Statistics and quality

Measured after cleaning. "Ended on their own" counts mutual goodbyes and replies to a farewell.

| Condition | Turns / call | Words / turn | Ended on their own | Hit the 40-turn cap | Echo-loop turns | Language drift |
|---|---:|---:|---:|---:|---:|---:|
| C1-P0 | 20.3 | 18.3 | n/a (one pass) | n/a | 2.5% | 0 |
| C1-P1 | 21.0 | 16.7 | n/a (one pass) | n/a | 0.1% | 0 |
| C1-P2 | 21.7 | 18.2 | n/a (one pass) | n/a | 1.5% | 0 |
| C2-P0 | 33.3 | 64.8 | 15/50 | 35/50 | 16.0% | 0 |
| C2-P1 | 39.6 | 12.2 | 4/50 | 46/50 | 1.1% | 0 |
| C2-P2 | 39.0 | 11.8 | 3/50 | 47/50 | 1.0% | 0 |
| C3-P0 | 18.4 | 47.4 | 43/50 | 7/50 | 5.0% | 4 |
| C3-P1 | 17.7 | 39.3 | 45/50 | 4/50 | 1.0% | 0 |
| C3-P2 | 16.8 | 41.2 | 46/50 | 4/50 | 5.4% | 3 |
| C4-P0 | 19.3 | 103.3 | 40/50 | 10/50 | 2.1% | 3 |
| C4-P1 | 11.3 | 108.6 | 43/50 | 6/50 | 1.2% | 0 |
| C4-P2 | 15.0 | 95.0 | 44/50 | 6/50 | 1.3% | 0 |
| *Humans (50 matched calls)* | 105.6 | 14.4 | | | | |

- **Echo-loop turns:** turns of ≥ 8 words that repeat an earlier turn almost verbatim
  (token-set Jaccard ≥ 0.8; the same rule as the generator's loop guard).
- **Language drift:** conversations in which Vicuna switches into Korean ("번역결과",
  "translation result"). The English before the switch is kept.
- Neither is filtered out of the files: degeneration is part of what is being measured.
  The analysis reports every dialogue-act result with and without them.

## Intended use

Research on simulated conversation: how generation choices shape dialogue structure,
evaluation of conversational agents, and teaching. The conversations are synthetic, are not
human data, and should not be used as stand-ins for human speech without the caveats in
[`docs/methods.md`](methods.md#10-declared-caveats).

## License

The generated conversations and derived numbers in this repository are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the code is MIT-licensed
([`LICENSE`](../LICENSE)). The texts were produced by Vicuna-13B v1.5 (a Llama 2 derivative;
Llama 2 Community License) and Mistral-7B-Instruct v0.2 (Apache 2.0), whose terms may also
apply to downstream use.

**Switchboard is not included.** The SwDA corpus is licensed by the Linguistic Data Consortium.
This repository contains no Switchboard transcript text, only topic instructions and caller
demographics used to seed the prompts, and numbers derived from the corpus (turn lengths,
act frequencies, alignment scores). To recompute the human side, place the SwDA release under
`data/switchboard/swda/`.
