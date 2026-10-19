"""The degeneration score used to freeze decoding on the development conversations."""

from __future__ import annotations

import json

from generation.degeneration_score import score_dir

LOOP = "we should all drive less and take the bus to work every single day"


def write(path, name, rec):
    (path / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_scores_loops_caps_and_short_turns(tmp_path):
    seed = [["ParticipantA", "Hello!"], ["ParticipantB", "Hello!"]]
    write(
        tmp_path,
        "1",
        {
            "seed_turns": 2,
            "turns": [
                *seed,
                ["ParticipantA", LOOP],
                ["ParticipantB", "Uh-huh."],
                ["ParticipantA", LOOP],
            ],
            "ended_by": "turn_cap",
            "quality_counters": {"hit_token_cap": 1, "dup_kept": 1},
        },
    )
    write(
        tmp_path,
        "2",
        {
            "seed_turns": 2,
            "turns": [*seed, ["ParticipantA", "Bye now!"], ["ParticipantB", "Bye!"]],
            "ended_by": "mutual_closing",
            "quality_counters": {},
        },
    )
    s = score_dir(tmp_path)
    assert s["conversations"] == 2
    assert s["dup_turn_rate"] == round(1 / 5, 4)  # one verbatim repeat among 5 generated turns
    assert s["turn_cap_rate"] == 0.5
    assert s["short_turn_rate"] == round(3 / 5, 4)  # "Uh-huh.", "Bye now!", "Bye!"
    # (1 dup turn + 1 dup kept + 1 token cap + 1 turn-cap ending) / 2 conversations
    assert s["degeneration_per_conv"] == 2.0
    assert s["ended_by"] == {"turn_cap": 1, "mutual_closing": 1}


def test_c1_raw_output_is_parsed(tmp_path):
    write(tmp_path, "1", {"raw_output": "ParticipantA: Hi.\nParticipantB: Hey there!"})
    assert score_dir(tmp_path)["words_per_turn_mean"] == 1.5


def test_empty_directory(tmp_path):
    assert score_dir(tmp_path)["error"] == "no data"
