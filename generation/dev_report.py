"""Dev-sweep readout: did the v3 prompting/decoding fixes actually move the generation
toward human structure? Prints, per condition, the two quantities the fixes target —
short-turn rate and backchannel rate — next to the Switchboard human baseline.

This is NOT the formal dialogue-act metric (that lives in analysis/, runs on gold tags).
It is a fast, surface, decision-support readout so we can eyeball the dev sweep before
committing to the full run. Backchannels here are matched by a surface lexicon, so treat
the numbers as indicative, not final.

    python generation/dev_report.py data/dev_sweep/C1-P0 data/dev_sweep/C1-P1 ...
    python generation/dev_report.py data/dev_sweep/C*-P*        # shell-expanded
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.analyze import conversation_turns                       # noqa: E402
from analysis.swda import iter_conversation_files, parse_conversation  # noqa: E402

import json

# Surface backchannel lexicon — short reactive tokens. Matches the pool builder so the dev
# report and the P2 example selection agree on what "a backchannel" looks like.
BC_LEX = re.compile(
    r"^(yeah|yes|yep|right|okay|ok|uh[- ]?huh|huh[- ]?uh|mm+[- ]?h?m*|m-?hm|sure|exactly|"
    r"really|wow|oh|oh yeah|oh really|oh okay|i see|i know|that's right|that's true|true|"
    r"definitely|absolutely|of course|no kidding|gotcha|hmm|nice|jeez|oh no|oh well)\b",
    re.I,
)


def is_backchannel(text: str) -> bool:
    return len(text.split()) <= 5 and bool(BC_LEX.match(text.strip()))


def stats_from_turns(convs: list[list[tuple[str, str]]]) -> dict:
    """convs = list of turn-lists (each [(speaker, text), ...]); the two scripted 'Hello!'
    seed turns are dropped so they don't distort the human-comparison numbers."""
    lens, nturns, bc, total = [], [], 0, 0
    for turns in convs:
        body = [t for t in turns if t[1].strip().lower().rstrip("!.") != "hello"]
        nturns.append(len(turns))
        for _, txt in body:
            lens.append(len(txt.split()))
            total += 1
            if is_backchannel(txt):
                bc += 1
    if not total:
        return {}
    lens.sort()
    return {
        "convs": len(convs),
        "mean_wpt": sum(lens) / len(lens),
        "med_wpt": statistics.median(lens),
        "short_le3": sum(1 for w in lens if w <= 3) / len(lens),
        "backchannel": bc / total,
        "med_turns": statistics.median(nturns),
    }


def load_condition(d: Path) -> list[list[tuple[str, str]]]:
    out = []
    for f in sorted(d.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        out.append(conversation_turns(rec))
    return out


def human_baseline(n: int = 300) -> dict:
    convs = [parse_conversation(fp) for fp in list(iter_conversation_files())[:n]]
    return stats_from_turns(convs)


HDR = f"{'condition':14} {'convs':>5} {'mean_wpt':>8} {'med_wpt':>7} {'<=3w %':>7} {'bckchnl %':>9} {'med_turns':>9}"


def _row(label: str, s: dict) -> str:
    if not s:
        return f"{label:14} (no data)"
    return (f"{label:14} {s['convs']:>5} {s['mean_wpt']:>8.1f} {s['med_wpt']:>7.0f} "
            f"{100*s['short_le3']:>6.1f}% {100*s['backchannel']:>8.1f}% {s['med_turns']:>9.0f}")


def main(argv: list[str]) -> None:
    dirs = [Path(a) for a in argv if Path(a).is_dir()]
    if not dirs:
        print("usage: python generation/dev_report.py data/dev_sweep/C*-P*")
        return
    print(HDR)
    print("-" * len(HDR))
    print(_row("HUMAN (SB)", human_baseline()))
    print("-" * len(HDR))
    for d in sorted(dirs):
        print(_row(d.name, stats_from_turns(load_condition(d))))
    print("\nRead: P1/P2 should move '<=3w %' and 'bckchnl %' UP toward the HUMAN row "
          "vs P0. If P0≈P1 on backchannels, the prompt still isn't letting them emerge.")


if __name__ == "__main__":
    main(sys.argv[1:])
