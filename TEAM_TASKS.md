# Team tasks until the poster (2026-10-11)

Start here after two months away. `git pull` first. Background reading only if you need it:
`CLAUDE.md` (how the repo works), `RESEARCH_DIALOGUE_ACTS.md` (our headline metric).

## Where we are (2026-09-26)

- **Scope is now C1 + C2 only** (all-at-once vs turn-by-turn, one Vicuna model), each with
  prompts P0/P1/P2 = **6 conditions × 50 conversations**, compared with human Switchboard.
  The Azure VM was downgraded to 2 old Tesla M60 cards, and C3/C4 (two models talking) don't
  fit. We asked the tutor for the V100 back; if it returns in time, C3/C4 get added.
- **Generation is running on the VM.** A 5-conversation test is running now. After we review
  it (2026-09-27), the full run starts: about 4 days, done ~2026-10-02. Output:
  `data/generated_v3/<condition>/<id>.json`.
- The July data (`data/generated_v2`) is a **draft**. Its settings suppressed backchannels,
  so its numbers are not final. Everything gets re-run on `generated_v3`.
- Headline story: LLM conversations lack the interactional structure of human talk
  (almost no backchannels like "uh-huh"/"yeah", no short reactive turns), even under settings
  that allow them. Which gaps do the architecture (C1 vs C2) and the prompt (P0→P1→P2) close?

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

## Teammate A — the paper's metrics on C1 + C2

Goal: one command per script produces the paper-style numbers for the 6 conditions vs human.

1. `analysis/original_paper_analysis.py` — words/turn, markers (oh/okay/uh-huh), alignment.
   `ARCHITECTURES = ["C1","C2","C3","C4"]` is hard-coded (line ~67). Make it take the data
   folder as an argument (`--data-dir`, like the other scripts) and work when only C1/C2
   exist (skip missing folders cleanly, no crash, no empty rows).
2. `analysis/stats.py` — same: `ARCH_ORDER` assumes C1–C4, and the "independence gradient"
   test needs ≥3 architectures. With only C1/C2, replace it with a plain **C1 vs C2**
   comparison per prompt level (and P0 vs P1 vs P2 within each architecture).
3. `analysis/export_align.py` — ALIGN alignment export. Check it runs on
   `--data-dir data/dev_sweep --conditions C1-P0 C1-P1 C1-P2 C2-P0 C2-P1 C2-P2 --include-sb`.
   (ALIGN itself is installed on the VM; if it doesn't run locally, just make sure the script
   is ready, and note it.)
4. Test all three on `data/dev_sweep`. Commit with a short note of what you checked.

**P2 rule:** P2's prompt contains real Switchboard examples, so **never report the
oh/okay/uh-huh marker rates for P2** (circular). Words/turn and alignment are fine.

## Teammate B — route reuse + the poster

1. `analysis/semantic_routes.py` (the "LLM calls follow the same predictable path" metric).
   It hard-codes C1–C4 pairs and colors (lines ~1053, ~1057, ~1199, ~1392, ~1452). Make it
   run on `--generated-root data/dev_sweep` with only C1/C2: keep the C1 vs C2 contrast and
   the P0→P1 contrasts, drop C2→C3/C3→C4 when those don't exist. Test it, commit.
2. **Poster skeleton** (start by ~2026-09-30): the story in 4 panels:
   1. Question + setup (6 conditions vs Switchboard; one example human vs LLM excerpt).
   2. Replication: words/turn and markers vs the paper's numbers (Teammate A's output).
   3. **Headline:** dialogue-act structure, i.e. backchannel/short-turn gap, JSD vs human
      with the noise floor (Diyar + Claude produce these figures).
   4. What narrows the gap (prompt, architecture) vs what doesn't; limitations (one model,
      M60 → C3/C4 out, tagger trained on human speech).
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
| ~2.10 → 5.10 | Run all analysis on `data/generated_v3`; final figures |
| 5.10 → 9.10 | Poster |
| 10.10 | Buffer |
| **11.10** | **Poster presentation** |
