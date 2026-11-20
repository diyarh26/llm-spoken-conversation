"""Map Task domain: map data, who-sees-what, non-circular prompts, turn loop, scoring."""

from __future__ import annotations

import re
from collections import Counter

import pytest

from maptask import corpus
from maptask.corpus import load_maps
from maptask.evaluate import parse_labels
from maptask.generate import parse_c1, run_dialogue
from maptask.prompts import (
    FOLLOWER,
    GIVER,
    build_agent,
    build_c1,
    build_c2,
    build_reconstruction,
)
from maptask.score import parse_points, passes, resample, route_deviation

MAPS = load_maps()
needs_corpus = pytest.mark.skipif(
    not corpus.DATA.exists(),
    reason="Map Task corpus not downloaded (python -m maptask.corpus --download)",
)
# Dialogue moves the analysis measures: no generation prompt may ask for them.
MEASURED_MOVES = re.compile(
    r"\b(ask\w*|question\w*|check\w*|confirm\w*|acknowledg\w*|clarif\w*|repeat\w*)\b", re.I
)


def _names(side):
    return Counter(lm["name"] for lm in side["landmarks"])


def test_sixteen_maps_with_routes_from_start_to_finish():
    assert sorted(MAPS, key=lambda k: int(k[1:])) == [f"m{i}" for i in range(16)]
    for m in MAPS.values():
        lms = {lm["name"]: (lm["x"], lm["y"]) for lm in m["giver"]["landmarks"]}
        route = [tuple(p) for p in m["giver"]["route"]]
        assert route[0] == lms["start"]
        assert route[-1] == pytest.approx(lms["finish"], abs=2)
        assert "finish" not in _names(m["follower"])  # followers never see FINISH
        for side in ("giver", "follower"):
            for lm in m[side]["landmarks"]:
                assert 0 <= lm["x"] <= 65 and 0 <= lm["y"] <= 90


@needs_corpus
def test_maps_match_the_corpus_landmark_inventory():
    for mid, m in MAPS.items():
        for side in ("giver", "follower"):
            ref = Counter({lm["name"]: lm[side] for lm in corpus.load_landmarks()[mid] if lm[side]})
            assert _names(m[side]) == ref, (mid, side)


@pytest.mark.parametrize("mid", sorted(MAPS))
def test_each_agent_sees_only_its_own_map(mid):
    m = MAPS[mid]
    giver_only = set(_names(m["giver"])) - set(_names(m["follower"]))
    follower_only = set(_names(m["follower"])) - set(_names(m["giver"]))
    follower_view = build_agent(FOLLOWER, m, [])[0]["content"]
    giver_view = build_agent(GIVER, m, [])[0]["content"]
    for name in giver_only - {"finish"}:
        label = next(lm.get("label", name) for lm in m["giver"]["landmarks"] if lm["name"] == name)
        assert f"- {label} at" not in follower_view, name
    for name in follower_only:
        label = next(
            lm.get("label", name) for lm in m["follower"]["landmarks"] if lm["name"] == name
        )
        assert f"- {label} at" not in giver_view, name
    assert "route is drawn" not in follower_view and "FINISH (marked" not in follower_view
    assert "route is drawn" in giver_view


def test_writer_views_see_both_maps():
    m = MAPS["m0"]
    for prompt in (build_c1(m)[0]["content"], build_c2(m, [], GIVER, 150)[0]["content"]):
        assert "youth hostel" in prompt  # giver only
        assert "flight museum" in prompt  # follower only
        assert "route is drawn" in prompt


def test_generation_prompts_never_ask_for_a_measured_move():
    m = MAPS["m3"]
    history = [(GIVER, "okay so from the start go down"), (FOLLOWER, "down")]
    prompts = [build_c1(m)[0]["content"], build_c2(m, history, FOLLOWER, 150)[0]["content"]]
    prompts += [build_agent(role, m, history, 150)[0]["content"] for role in (GIVER, FOLLOWER)]
    for p in prompts:
        assert not MEASURED_MOVES.search(p), MEASURED_MOVES.search(p).group(0)


def test_reader_sees_only_the_follower_map():
    m = MAPS["m0"]
    prompt = build_reconstruction(m, [(GIVER, "go down past the camera shop")])[0]["content"]
    assert "flight museum" in prompt and "youth hostel" not in prompt
    assert "route is drawn" not in prompt


def test_agent_messages_alternate_roles_from_each_side():
    m, history = MAPS["m0"], [(GIVER, "hi there"), (FOLLOWER, "hi")]
    roles = [msg["role"] for msg in build_agent(GIVER, m, history)]
    assert roles == ["system", "assistant", "user"]
    roles = [msg["role"] for msg in build_agent(FOLLOWER, m, history)]
    assert roles == ["system", "user", "assistant", "user"]


def test_run_dialogue_starts_with_the_giver_and_ends_on_goodbyes():
    script = iter(["okay start at the top", "right", "that's the finish, bye", "bye"])
    turns, ended = run_dialogue(lambda spk, hist: next(script), max_turns=50)
    assert [s for s, _ in turns] == [GIVER, FOLLOWER, GIVER, FOLLOWER]
    assert ended == "mutual_closing"
    turns, ended = run_dialogue(lambda spk, hist: "keep going", max_turns=6)
    assert ended == "turn_cap" and len(turns) == 6


def test_parse_c1_keeps_only_labelled_turns():
    text = "Here it is:\nGiver: go south\nFollower: okay\n\nGiver: now east\nnote: done"
    assert parse_c1(text) == [(GIVER, "go south"), (FOLLOWER, "okay"), (GIVER, "now east")]


def test_route_scoring():
    route = [(0, 0), (10, 0), (10, 10)]
    assert route_deviation(route, route) == pytest.approx(0)
    assert route_deviation([(0, 2), (10, 2), (10, 12)], route) == pytest.approx(2, abs=0.3)
    pts = resample(route, 3)
    assert pts[0] == (0, 0) and pts[-1] == pytest.approx((10, 10))


def test_passes_reports_side_as_seen_by_the_traveller():
    route = [(0, 50), (0, 0)]  # travelling north (y decreases)
    found = passes(route, [{"name": "east", "x": 3, "y": 25}, {"name": "west", "x": -3, "y": 10}])
    assert [(f["name"], f["side"]) for f in found] == [("east", "right"), ("west", "left")]


def test_parse_points_and_labels():
    assert parse_points("[[5, 61], [10, 70.5]]") == [(5, 61), (10, 70.5)]
    assert parse_points("Route: [5,61] then [9, 64]") == [(5, 61), (9, 64)]
    assert parse_points("no idea") == []
    assert parse_labels('["instruct", "Acknowledge", "bogus"]', 3) == [
        "instruct",
        "acknowledge",
        "uncodable",
    ]
    assert parse_labels('["instruct"]', 2) is None
