"""Turn-level metrics shared by the human and generated sides."""

from __future__ import annotations

import pytest

from analysis.analyze import conversation_turns
from analysis.evaluate_generated import m1_turn_length, m_backchannel_analysis
from analysis.metrics import marker_rates, words_per_turn


def test_words_per_turn():
    assert words_per_turn([("A", "one two three"), ("B", "four")]) == [3, 1]


def test_marker_rates_are_per_100_words():
    turns = [("A", "oh okay"), ("B", "uh-huh yeah I see what you mean there")]  # 10 words
    rates = marker_rates(turns)
    assert rates == pytest.approx({"oh": 10.0, "okay": 10.0, "uh-huh": 10.0})


def test_turn_length_summary():
    summary = m1_turn_length([("A", "a b c d"), ("B", "a b"), ("A", "a b c d e f")])
    assert summary["mean_words_per_turn"] == 4
    assert summary["median_words_per_turn"] == 4
    assert summary["n_turns"] == 3


def test_standalone_backchannel_is_a_turn_that_is_only_listening():
    turns = [
        ("A", "So we bought the house in May."),
        ("B", "Uh-huh."),
        ("A", "And then the roof leaked."),
        ("B", "Yeah, that happens a lot with older houses."),
    ]
    stats = m_backchannel_analysis(turns)
    # Two turns contain a listener token; only "Uh-huh." consists of nothing else.
    assert stats["n_backchannel_turns_any"] == 2
    assert stats["backchannel_standalone_rate"] == 0.5


def test_c1_raw_output_is_parsed_into_turns():
    rec = {"raw_output": "ParticipantA: Hi there.\nParticipantB: Hey!\n\nnot a turn"}
    assert conversation_turns(rec) == [("ParticipantA", "Hi there."), ("ParticipantB", "Hey!")]
