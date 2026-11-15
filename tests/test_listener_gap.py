from analysis.listener_gap import listener_rates, summarise


def test_listener_rates_count_whole_turn_listener_responses():
    turns = [
        ("ParticipantA", "So I drive a truck for a living."),
        ("ParticipantB", "Uh-huh."),
        (
            "ParticipantA",
            "Yeah, and the hours are long.",
        ),  # feedback token + content: not listener-only
        ("ParticipantB", "Yeah, right."),
        ("ParticipantA", "Wow."),  # tiny but outside the lexicon
    ]
    r = listener_rates(turns)
    assert r["n_turns"] == 5
    assert r["listener_only_rate"] == 2 / 5
    assert r["tiny_turn_rate"] == 3 / 5


def test_empty_conversation_has_no_rates():
    assert listener_rates([])["listener_only_rate"] is None


def test_summarise_groups_by_condition():
    convs = [
        {"condition": "C2-P4", "turns": [("A", "Uh-huh."), ("B", "I fish every summer.")]},
        {"condition": "C2-P4", "turns": [("A", "Okay."), ("B", "Okay.")]},
    ]
    (row,) = summarise(convs)
    assert row["condition"] == "C2-P4"
    assert row["n_conversations"] == 2
    assert row["listener_only_rate_mean"] == 0.75
