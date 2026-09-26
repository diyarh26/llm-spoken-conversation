# Team tasks until the poster (2026-10-11)

Start here after two months away. `git pull` first. Background reading only if you need it:
`CLAUDE.md` (how the repo works), `RESEARCH_DIALOGUE_ACTS.md` (our headline metric).

## Where we are (2026-09-26)

- **The research question is LOCKED (no more pivots):** *how does the generation
  architecture (C1 all-at-once, C2 turn-by-turn one model, C3 two Vicuna agents, C4 Vicuna ↔
  Mistral) affect the structure of LLM conversations, compared with human Switchboard?*
  Prompts P0/P1/P2 are the second factor. **12 conditions × 50 conversations.**
- **All 4 architectures stay in.** C1 + C2 are generated first on the course VM (2 old
  Tesla M60 cards). C3/C4 load two models and don't fit on the M60; they will run on the V100
  if the tutor returns it, otherwise on a GPU Diyar rents. So **every script must keep C3/C4
  support** and simply work when some condition folders are missing (they arrive later).
- **Generation is running on the VM.** A 5-conversation test is running now. After we review
  it (2026-09-27), the full run starts: about 4 days, done ~2026-10-02. Output:
  `data/generated_v3/<condition>/<id>.json`.
- The July data (`data/generated_v2`) is a **draft**. Its settings suppressed backchannels,
  so its numbers are not final. Everything gets re-run on `generated_v3`.
- Headline story: how far each architecture's conversational structure (dialogue acts:
  backchannels, questions, short reactive turns, what-follows-what) is from human talk, and
  which architecture changes move it closer. Measured under settings that *allow* human-like
  structure, so any gap is the architecture's, not our settings'.

## Rules (do not skip)

- **Test every change** before relying on it. Use the small test data below.
- **Never commit Switchboard text or model weights** (`.gitignore` covers it; don't force it).
- **Don't edit `VM_TASKS.md` or `VM_REPORT.md`** (they're for the VM workflow).
- Pull before you start, commit small, push often. Analysis code goes in `analysis/*.py`.

## Test data you can use today

- `data/dev_sweep/` — July, 5 conversations per condition (C1/C2 × P0/P1/P2 complete).
- `data/dev_sweep_v2/` — the same 5 conversations from the current code (arrives ~2026-09-27).
Both have exactly the same layout as `data/generated_v3/`, so code that works on them will
work on the final data.

Note: every analysis script shares `analysis/analyze.py:conversation_turns()`, which now drops
the scripted "Hello!/Hello!" opening turns from v3 data. Always load turns through it.

---

## Teammate A — the paper's metrics, for whatever architectures exist

Goal: one command per script produces the paper-style numbers for every condition present
vs human — 6 conditions now (C1/C2), 12 once C3/C4 land, with no code change in between.

1. `analysis/original_paper_analysis.py` — words/turn, markers (oh/okay/uh-huh), alignment.
   Keep `ARCHITECTURES = ["C1","C2","C3","C4"]` (line ~67), but make it take the data folder
   as an argument (`--data-dir`, like the other scripts) and **skip missing condition folders
   cleanly** (no crash, no empty rows).
2. `analysis/stats.py` — same: keep `ARCH_ORDER = C1–C4`, use only architectures present.
   The architecture test (the "independence gradient", needs ≥3 architectures) runs once
   C3/C4 exist; until then it should print "skipped: needs ≥3 architectures" and still run
   the **C1 vs C2** comparison per prompt level (and P0 vs P1 vs P2 within each architecture).
3. `analysis/export_align.py` — ALIGN alignment export. Check it runs on
   `--data-dir data/dev_sweep --conditions C1-P0 C1-P1 C1-P2 C2-P0 C2-P1 C2-P2 --include-sb`.
   (ALIGN itself is installed on the VM; if it doesn't run locally, just make sure the script
   is ready, and note it.)
4. Test all three on `data/dev_sweep`. Commit with a short note of what you checked.

**P2 rule:** P2's prompt contains real Switchboard examples, so **never report the
oh/okay/uh-huh marker rates for P2** (circular). Words/turn and alignment are fine.

## Teammate B — route reuse + the poster

1. `analysis/semantic_routes.py` (the "LLM calls follow the same predictable path" metric).
   It hard-codes C1–C4 pairs and colors (lines ~1053, ~1057, ~1199, ~1392, ~1452). Keep
   them, but make it run on `--generated-root data/dev_sweep` where only C1/C2 (+ partial C3)
   exist: compute the contrasts whose conditions are present, **skip** the others with a
   printed note. Test it, commit.
2. **Poster skeleton** (start by ~2026-09-30): the story in 4 panels:
   1. Question + setup: 4 architectures × 3 prompts vs Switchboard; one example human vs
      LLM excerpt.
   2. Replication: words/turn and markers vs the paper's numbers (Teammate A's output).
   3. **Headline:** dialogue-act structure per architecture — backchannel/short-turn gap,
      JSD vs human with the noise floor (Diyar + Claude produce these figures).
   4. Which architecture (and prompt) moves structure closer to human, which doesn't;
      limitations (one base model, tagger trained on human speech).
   Use placeholder figures from the July draft (`results/dialogue_acts/*.png`) until the
   final ones exist, clearly marked DRAFT.

## Diyar (with Claude)

- Runs and watches generation on the VM; reviews the test; starts the full run.
- Dialogue-act analysis (headline): the script is ready (`analysis/dialogue_acts.py
  --generated-root <folder>`); the tagger runs in a separate Python 3.11 environment
  (`analysis/requirements-dialogue-acts.txt`).

## Timeline

| Dates | What |
|---|---|
| 26–27.9 | Test run; A and B fix their scripts on `data/dev_sweep` |
| 27.9 → ~2.10 | Full C1 + C2 run on the VM; A/B test on `data/dev_sweep_v2`; B starts poster |
| **by 30.9** | **Decide C3/C4 GPU:** tutor's V100, or Diyar rents one; C3/C4 run ~30.9 → ~4.10 |
| ~2.10 → 5.10 | Run all analysis on `data/generated_v3`; final figures |
| 5.10 → 9.10 | Poster |
| 10.10 | Buffer |
| **11.10** | **Poster presentation** |
