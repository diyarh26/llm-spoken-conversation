"""Turn cleaning: one model output -> exactly one spoken turn.

Every case below is a failure mode observed in real generations."""

from __future__ import annotations

import pytest

from generation.cleaning import (
    clean_single_turn,
    has_meta_artifact,
    looks_like_closing,
    next_speaker_marker,
    strip_lead_labels,
    strip_meta_artifacts,
)


class TestSingleTurn:
    def test_clean_turn_is_untouched(self):
        assert clean_single_turn("I drive a lot for work, so gas prices matter.") == (
            "I drive a lot for work, so gas prices matter.",
            False,
        )

    def test_runs_on_into_partner_turn(self):
        text = "Yeah, I agree. ParticipantB: Totally, and another thing..."
        assert clean_single_turn(text) == ("Yeah, I agree.", True)

    @pytest.mark.parametrize(
        "leak",
        ["ParticipantsA: well", "Participant A: well", "participantb: well"],
    )
    def test_degraded_labels_mid_text_are_caught(self, leak):
        turn, ran_past = clean_single_turn(f"I think so. {leak}")
        assert turn == "I think so."
        assert ran_past

    def test_chat_role_residue_on_new_line(self):
        turn, ran_past = clean_single_turn("Sure thing.\nUSER: what else?")
        assert (turn, ran_past) == ("Sure thing.", True)

    def test_surrounding_quotes_removed(self):
        assert clean_single_turn('"Hi there!"') == ("Hi there!", False)


class TestLeadLabels:
    @pytest.mark.parametrize(
        "raw",
        [
            "ParticipentB: Hello there",
            "ParticipB: Hello there",
            "ParticipANT_A: Hello there",
            "ParticipantB, Hello there",
            "ParticipParticipant B: Hello there",
            "ASSISTANT: Hello there",
        ],
    )
    def test_misspelled_and_repeated_labels_are_peeled(self, raw):
        assert strip_lead_labels(raw) == "Hello there"

    @pytest.mark.parametrize(
        "speech",
        ["Part of the problem is cost.", "Partly, yes.", "Participation matters to me."],
    )
    def test_ordinary_words_are_never_peeled(self, speech):
        assert strip_lead_labels(speech) == speech


class TestMetaArtifacts:
    @pytest.mark.parametrize(
        "residue",
        [
            "(Your turn count is 14)",
            "(End of conversation.)",
            "[End of Response]",
            "\n## Example response:",  # headings start a new line
            "*Session closed.*",
            "**Prompt:** Act like a 42-year-old man",
            "Here's a summary of what we discussed",
            "```",
        ],
    )
    def test_turn_is_cut_at_residue(self, residue):
        assert strip_meta_artifacts(f"That sounds great to me. {residue} trailing") == (
            "That sounds great to me."
        )
        assert has_meta_artifact(f"Fine. {residue}")

    def test_turn_that_is_only_residue_becomes_empty(self):
        assert strip_meta_artifacts("(End of conversation)") == ""

    @pytest.mark.parametrize(
        "speech",
        [
            "I put it in my 401(k) last year.",
            "We use a lot of smart devices (IoT) at home.",
            "My hobbies (writing, drawing, composing music) keep me busy.",
        ],
    )
    def test_ordinary_parentheses_are_speech(self, speech):
        assert strip_meta_artifacts(speech) == speech
        assert not has_meta_artifact(speech)


class TestClosings:
    @pytest.mark.parametrize(
        "farewell",
        [
            "Okay, bye!",
            "It was nice talking to you.",
            "Take care now.",
            "Have a great weekend!",
            "Talk to you later.",
        ],
    )
    def test_farewells(self, farewell):
        assert looks_like_closing(farewell)

    @pytest.mark.parametrize(
        "speech", ["I bought a used car last year.", "Byers is my neighbor's name."]
    )
    def test_ordinary_speech(self, speech):
        assert not looks_like_closing(speech)


def test_stopping_marker_agrees_with_the_cleaner():
    """Generation stops where the cleaner would cut, so stopping never changes a turn."""
    raw = "Well, that is true. ParticipantA: I mean"
    cut = next_speaker_marker(raw)
    assert cut is not None
    assert raw[:cut].strip() == clean_single_turn(raw)[0]
    assert next_speaker_marker("Just one clean turn.") is None
