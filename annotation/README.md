# Tagger hand-check — instructions (≈30 minutes each)

Our headline result relies on an automatic dialogue-act tagger. It was trained on human
phone calls and has never been checked on LLM text. You are the check.

**Your file:** `sheets/annotator_N.xlsx`

| Annotator | File |
|---|---|
| 1 | `annotator_1.xlsx` |
| 2 | `annotator_2.xlsx` |
| 3 | `annotator_3.xlsx` |
| 4 | `annotator_4.xlsx` |

Each file has **40 sentences**. 20 of them are the same for all four of us (to measure how
much *we* agree), and 20 are yours alone. You can't tell which are which, and that is on purpose.

## How to label

1. Open the file and read the **Label guide** tab first: 10 labels, each with examples.
2. In **Label these**, read the bold sentence in its context (grey = the previous lines). Pick
   **one** label from the dropdown in the yellow column.
3. Label the **sentence**, not the whole turn. "Yeah, I think that's a great idea." is one
   sentence, so pick what it mainly does.
4. Unsure? Pick the best fit and write a note. Don't leave a row empty.
5. **Don't** discuss sentences with each other or compare answers until everyone is done.
6. Fragments like "2." (from numbered lists) or cut-off sentences → **Abandoned/Other**.

Two labels people often get wrong:
- **Backchannel** also covers short appreciations such as "That's great!", "Oh, wow", "That
  sounds nice", as long as they add no new content.
- Greetings, thanks and goodbyes ("Thanks for chatting!", "Have a great day") → **Abandoned/Other**.

## When done

Save the file with the same name and commit it (or send it to Diyar). Diyar scores it with:

```
python analysis/score_annotation.py
```

The answer key (`annotation/_key.csv`) is kept out of git so nobody sees the tagger's answers.
