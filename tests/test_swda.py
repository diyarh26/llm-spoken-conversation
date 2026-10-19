"""Switchboard (SwDA) loading: transcript cleaning, act names, personas from metadata.

All inputs are synthetic; the licensed corpus itself is never needed here."""

from __future__ import annotations

import pathlib

import pytest

from analysis.swda import act_name, clean_text, conversation_no_of, make_personas
from tests.conftest import swda_row


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("{F Uh, } I think [ it's, + it's ] good. /", "Uh, I think it's, it's good."),
        ("{D Well, } {C and } you know /", "Well, and you know"),
        ("((Yeah)) right -/", "Yeah right"),
        ("#Okay.# /", "Okay."),
        ("<Laughter>.", ""),  # non-verbal only: not a turn
    ],
)
def test_clean_text_keeps_the_spoken_words(raw, clean):
    assert clean_text(raw) == clean


@pytest.mark.parametrize(
    ("tag", "name"),
    [
        ("b", "backchannel"),
        ("b^m", "backchannel"),
        ("qy^d", "yes/no question"),
        ("sd(^q)", "statement"),
        ("^2", "completes the other's sentence"),
        ("zz", "other"),
    ],
)
def test_act_names(tag, name):
    assert act_name(tag) == name


def test_conversation_number_from_filename():
    assert conversation_no_of(pathlib.Path("sw00utt/sw_0001_4325.utt.csv")) == 4325


def test_personas_from_metadata():
    a, b, topic, prompt = make_personas(swda_row())
    assert (a.label, a.gender, a.age, a.education) == ("ParticipantA", "man", 54, "some college")
    assert (b.label, b.gender, b.age, b.education) == (
        "ParticipantB",
        "woman",
        33,
        "a college degree",
    )
    assert topic == "Air Pollution"
    assert prompt == "PLEASE DISCUSS AIR POLLUTION."


def test_topic_capitalisation_keeps_apostrophes():
    _, _, topic, _ = make_personas(swda_row(topic_description="WOMEN'S ROLES"))
    assert topic == "Women's Roles"


def test_truncated_source_prompts_are_restored():
    _, _, _, prompt = make_personas(
        swda_row(topic_description="PETS", prompt="FIND OUT WHAT KIND OF PETS ORY")
    )
    assert prompt == "FIND OUT WHAT KIND OF PETS THE OTHER CALLER HAS."


def test_missing_birth_year_falls_back():
    a, _, _, _ = make_personas(swda_row(from_caller_birth_year=""))
    assert a.age == 40
