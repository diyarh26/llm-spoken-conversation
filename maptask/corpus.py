"""
HCRC Map Task corpus: the second domain of the study.

Two speakers each hold a map. The instruction GIVER's map shows a route from START to
FINISH; the FOLLOWER's map shows only START and must have the route drawn on it from the
giver's spoken instructions. The two maps differ on purpose: some landmarks are only on
one map, some appear twice, and neither speaker knows where the differences are. So the
dialogue has to discover and repair them — the private-information setting where
turn-by-turn agents (C3/C4) may behave more like humans than one all-knowing writer (C1).

Data: HCRC Map Task Corpus annotations v2.1 (NXT format), Human Communication Research
Centre, Universities of Edinburgh and Glasgow, licensed CC BY 4.0
(https://groups.inf.ed.ac.uk/maptask/). Download once with

    python -m maptask.corpus --download

which fetches into data/maptask_raw/ (gitignored). This module reads:
  * 128 dialogues: map, giver/follower, eye contact, and the path deviation score
    (devscore: area between the giver's route and the follower's drawn route; lower is
    better) — the human task-success reference;
  * per-map landmarks and on which map(s) each appears (the deliberate differences);
  * every dialogue move (instruct, check, query_yn, acknowledge, ...) with its words and
    start time, in conversation order;
  * plain turns (speaker, text) from the corpus's text transcripts.
The text descriptions of the 16 maps (positions and route) live in
data/maptask/maps.json (see maptask.corpus.load_maps), transcribed from the corpus map images.
"""

from __future__ import annotations

import argparse
import functools
import pathlib
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "maptask_raw"
DATA = RAW / "nxt" / "maptaskv2-1" / "Data"
TRANSCRIPTS = RAW / "tx" / "Transcripts"
NITE = "{http://nite.sourceforge.net/}"

DOWNLOADS = {
    "hcrcmaptask.nxtformatv2-1.zip": "https://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-1.zip",
    "transcripts.zip": "https://groups.inf.ed.ac.uk/maptask/transcripts/transcripts.zip",
    "maps.tar.gz": "https://groups.inf.ed.ac.uk/maptask/interface/maps.tar.gz",
}

# The 12 coded move types (Carletta et al. 1997) plus the corpus's own "uncodable".
MOVE_LABELS = (
    "instruct",
    "explain",
    "check",
    "align",
    "query_yn",
    "query_w",
    "acknowledge",
    "reply_y",
    "reply_n",
    "reply_w",
    "clarify",
    "ready",
    "uncodable",
)
INITIATING = {"instruct", "explain", "check", "align", "query_yn", "query_w"}


def download() -> None:
    """Fetch and unpack the corpus into data/maptask_raw/ (about 13 MB)."""
    import tarfile

    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        dest = RAW / name
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
    with zipfile.ZipFile(RAW / "hcrcmaptask.nxtformatv2-1.zip") as z:
        z.extractall(RAW / "nxt")
    with zipfile.ZipFile(RAW / "transcripts.zip") as z:
        z.extractall(RAW / "tx")
    with tarfile.open(RAW / "maps.tar.gz") as t:
        try:
            t.extractall(RAW / "maps", filter="data")
        except TypeError:  # Python < 3.10.12 has no extraction filters
            t.extractall(RAW / "maps")


def _require() -> None:
    if not DATA.exists():
        raise FileNotFoundError(
            "Map Task corpus not found; run: python -m maptask.corpus --download"
        )


@functools.cache
def load_dialogues() -> dict[str, dict]:
    """{dialogue id: {map, giver, follower, eyecontact, familiar, devscore, quad}}."""
    _require()
    root = ET.parse(DATA / "corpus-resources" / "maptask-corpus.xml").getroot()
    out = {}
    for conv in root.iter("conv"):
        roles = {
            p.get("role"): p.get("href").split("#id(")[1].rstrip(")")
            for p in conv.iter(f"{NITE}pointer")
        }
        dev = conv.get("devscore")
        out[conv.get("id")] = {
            "id": conv.get("id"),
            "map": conv.get("map"),
            "quad": int(conv.get("quad")),
            "eyecontact": conv.get("eyecontact") == "yes",
            "familiar": conv.get("familiar") == "yes",
            "devscore": int(dev) if dev and dev.isdigit() else None,
            "giver": roles.get("g"),
            "follower": roles.get("f"),
        }
    return out


@functools.cache
def load_landmarks() -> dict[str, list[dict]]:
    """{map id: [{name, giver (count on giver map), follower (count on follower map)}]}.

    Counts are 0 (absent), 1, or 2 (the landmark appears twice on that map). A landmark
    whose two counts differ is one of the deliberate map differences."""
    _require()
    root = ET.parse(DATA / "corpus-resources" / "maptask-landmarks.xml").getroot()
    out = {}
    for m in root.iter("map"):
        out[m.get("id")] = [
            {
                "name": lm.get("name"),
                "giver": int(lm.get("giver_map_appears")),
                "follower": int(lm.get("follower_map_appears")),
            }
            for lm in m.iter("landmark")
        ]
    return out


MAPS_JSON = ROOT / "data" / "maptask" / "maps.json"


@functools.cache
def load_maps() -> dict[str, dict]:
    """Text versions of the 16 maps (committed; no download needed):
    {map id: {"giver": {"landmarks": [...], "route": [[x, y], ...]},
              "follower": {"landmarks": [...]}}}.
    Each landmark is {name, x, y} plus "label" when the map's own text differs from name.
    Coordinates: x 0 (west) to 65 (east), y 0 (north) to 90 (south)."""
    import json

    data = json.loads(MAPS_JSON.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def map_differences(map_id: str) -> list[dict]:
    """The landmarks whose presence differs between the giver's and follower's map."""
    return [lm for lm in load_landmarks()[map_id] if lm["giver"] != lm["follower"]]


def _units(dialogue: str, spk: str) -> dict[str, tuple[float, str]]:
    root = ET.parse(DATA / "timed-units" / f"{dialogue}.{spk}.timed-units.xml").getroot()
    return {
        tu.get("id"): (float(tu.get("start")), (tu.text or "").strip())
        for tu in root
        if tu.tag == "tu"
    }


def _span(href: str, order: list[str]) -> list[str]:
    ids = re.findall(r"id\(([^)]+)\)", href)
    if len(ids) == 1:
        return ids
    i, j = order.index(ids[0]), order.index(ids[1])
    return order[i : j + 1]


@functools.cache
def load_moves(dialogue: str) -> list[dict]:
    """Every coded move of both speakers, in time order:
    [{speaker: 'giver'|'follower', label, text, start}]."""
    _require()
    moves = []
    for spk, role in (("g", "giver"), ("f", "follower")):
        units = _units(dialogue, spk)
        tu_root = ET.parse(DATA / "timed-units" / f"{dialogue}.{spk}.timed-units.xml").getroot()
        order = [el.get("id") for el in tu_root]
        root = ET.parse(DATA / "moves" / f"{dialogue}.{spk}.moves.xml").getroot()
        for mv in root.iter("move"):
            words = []
            for child in mv.iter(f"{NITE}child"):
                words += [units[i] for i in _span(child.get("href"), order) if i in units]
            if not words:
                continue
            moves.append(
                {
                    "speaker": role,
                    "label": mv.get("label"),
                    "text": " ".join(w for _, w in words if w),
                    "start": min(t for t, _ in words),
                }
            )
    return sorted(moves, key=lambda m: m["start"])


@functools.cache
def load_turns(dialogue: str) -> list[tuple[str, str]]:
    """(speaker, text) turns from the corpus text transcript; speaker is giver/follower.
    Consecutive lines by the same speaker are merged into one turn."""
    _require()
    turns: list[tuple[str, str]] = []
    names = {"g": "giver", "f": "follower"}
    lines = (TRANSCRIPTS / f"{dialogue}.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 2 or parts[0] not in names:
            continue
        spk, txt = names[parts[0]], parts[1].strip()
        if not txt:
            continue
        if turns and turns[-1][0] == spk:
            turns[-1] = (spk, f"{turns[-1][1]} {txt}")
        else:
            turns.append((spk, txt))
    return turns


def move_distribution(dialogues: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Share of each move label per role, pooled over dialogues (all 128 by default)."""
    from collections import Counter

    counts = {"giver": Counter(), "follower": Counter()}
    for d in dialogues or sorted(load_dialogues()):
        for mv in load_moves(d):
            counts[mv["speaker"]][mv["label"]] += 1
    return {
        role: {lab: c[lab] / max(1, sum(c.values())) for lab in MOVE_LABELS}
        for role, c in counts.items()
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--download", action="store_true", help="fetch the corpus (CC BY 4.0)")
    args = ap.parse_args(argv)
    if args.download:
        download()
    dialogues = load_dialogues()
    turns = [len(load_turns(d)) for d in dialogues]
    print(f"{len(dialogues)} dialogues, {len(load_landmarks())} maps")
    print(
        f"turns per dialogue: mean {sum(turns) / len(turns):.1f}, min {min(turns)}, max {max(turns)}"
    )
    dist = move_distribution()
    print(f"{'move':<12}{'giver':>8}{'follower':>10}")
    for lab in MOVE_LABELS:
        print(f"{lab:<12}{dist['giver'][lab] * 100:>7.1f}%{dist['follower'][lab] * 100:>9.1f}%")


if __name__ == "__main__":
    main()
