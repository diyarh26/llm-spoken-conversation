"""Build the conversation explorer's data bundle from the released dataset.

    python explorer/build.py          # -> explorer/data/index.json + explorer/data/<id>.json

Standard library only (the GitHub Pages workflow runs it on every deploy). Turns go through
the same cleaner the analysis uses (analysis.analyze.conversation_turns), so the explorer
shows exactly the turns that were measured. The human calls contribute numbers only (turn
lengths from results/figure_data/human_turns.json); Switchboard text is LDC-licensed and
never published.
"""

from __future__ import annotations

import csv
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.analyze import conversation_turns
from analysis.figures import ARCH_DESC, ARCH_NAME, human_turns, rhythm
from generation.sampling import load_target_ids
from prompts.templates import Persona, naturalize

DATA = ROOT / "data" / "generated"
OUT = pathlib.Path(__file__).resolve().parent / "data"
ARCHS = ("C1", "C2", "C3", "C4")
PROMPTS = {
    "P0": "Paper prompt: the original study's basic instruction.",
    "P1": "Spoken style + persona cards; never names a listener behaviour.",
    "P2": "P1 plus two real Switchboard excerpts as examples.",
}


def speaker(label: str) -> str:
    """'ParticipantA' -> 'A' (C1 occasionally mangles labels; keep what is recoverable)."""
    last = label.strip()[-1:].upper()
    return last if last in ("A", "B") else "?"


def persona(fields: dict) -> dict:
    p = Persona(**{k: fields[k] for k in Persona.__dataclass_fields__})
    card = p.card_third_person().removeprefix(f"{p.label} ").rstrip(".")
    return {"who": p.describe().removeprefix("a "), "card": card}


def conversation(cno: int, human: list[list]) -> dict:
    out: dict = {"id": cno, "conditions": {}}
    for arch in ARCHS:
        for prompt in PROMPTS:
            rec = json.loads(
                (DATA / f"{arch}-{prompt}" / f"{cno}.json").read_text(encoding="utf-8")
            )
            turns = conversation_turns(rec)
            if "topic" not in out:
                out["topic"] = rec["topic"]
                out["instruction"] = naturalize(rec["sb_prompt"])
                out["personas"] = {"A": persona(rec["persona_a"]), "B": persona(rec["persona_b"])}
            out["conditions"][f"{arch}-{prompt}"] = {
                "turns": [[speaker(s), text] for s, text in turns],
                "rhythm": [r[1:] for r in rhythm(turns)],
                "ended_by": rec.get("ended_by", "single_pass"),
            }
    out["human"] = {"rhythm": [[s, w, int(listener)] for s, w, listener in human]}
    return out


def corpus_summary() -> dict:
    metrics = {r["condition"]: r for r in _csv(ROOT / "results" / "metrics_summary.csv")}
    rule = {
        r["condition"]: float(r["rule_backchannel_rate"])
        for r in _csv(ROOT / "results" / "dialogue_acts" / "da_rule_crosscheck.csv")
    }
    return {
        cond: {
            "words_per_turn": round(float(m["mean_words_per_turn_mean"]), 1),
            "turns": round(float(m["n_turns_mean"]), 1),
            "listener_share": round(rule[cond], 4),
        }
        for cond, m in metrics.items()
    }


def _csv(path: pathlib.Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    humans = human_turns()
    topics = []
    for cno in load_target_ids():
        conv = conversation(cno, humans[str(cno)])
        (OUT / f"{cno}.json").write_text(
            json.dumps(conv, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
        )
        llm_wpt = statistics.mean(
            statistics.mean(r[0] for r in c["rhythm"]) for c in conv["conditions"].values()
        )
        topics.append({"id": cno, "topic": conv["topic"], "llm_words_per_turn": round(llm_wpt, 1)})
    index = {
        "architectures": {a: {"name": ARCH_NAME[a], "desc": ARCH_DESC[a]} for a in ARCHS},
        "prompts": PROMPTS,
        "topics": sorted(topics, key=lambda t: (t["topic"], t["id"])),
        "summary": corpus_summary(),
        "featured": 3508,
    }
    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    size = sum(f.stat().st_size for f in OUT.glob("*.json"))
    print(f"wrote {len(topics) + 1} files to {OUT.relative_to(ROOT)} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
