"""Prompt design rules, enforced.

The study's central safeguard against circular evaluation: no prompt in the main design
(P0-P2) ever names or shows the listener behaviour that is measured. P3 is the one
exploratory condition that shows it on purpose, and is reported separately."""

from __future__ import annotations

import re

import pytest

from analysis import swda
from prompts import templates
from prompts.templates import (
    Persona,
    agent_view,
    assign_cards,
    build_agent,
    build_c1,
    build_c2,
    writer_view,
)

TOPIC, SB_PROMPT = "Air Pollution", "PLEASE DISCUSS AIR POLLUTION."
HISTORY = [("ParticipantA", "Hello!"), ("ParticipantB", "Hello!"), ("ParticipantA", "So, smog?")]

# Listener feedback and coordination markers the study measures.
MEASURED = re.compile(r"\b(uh-?\s?huh|mm-?\s?hm+|backchannel\w*|okay|oh|yeah)\b", re.IGNORECASE)


@pytest.fixture
def personas():
    a = Persona("ParticipantA", "man", 54, "some college")
    b = Persona("ParticipantB", "woman", 33, "a college degree")
    assign_cards(a, b, 2121)
    return a, b


@pytest.fixture
def fake_fewshot(monkeypatch):
    """Stand-in for the licensed Switchboard excerpts shown by P2."""
    example = {"topic": "FISHING", "text": "ParticipantA: uh-huh.\nParticipantB: yeah, okay."}
    monkeypatch.setattr(swda, "fewshot_examples", lambda cno, k=2: [example, example])


def all_prompts(level, a, b):
    """Every prompt text a model can receive at a prompt level, across architectures."""
    texts = [m["content"] for m in build_c1(level, a, b, TOPIC, SB_PROMPT, conversation_no=2121)]
    texts += [
        m["content"]
        for m in build_c2(level, a, b, TOPIC, SB_PROMPT, HISTORY, "ParticipantB", 40, 2121)
    ]
    texts += [
        m["content"]
        for m in build_agent(level, b, a, TOPIC, SB_PROMPT, HISTORY, 40, 2121)
        if m["role"] == "system"
    ]
    return texts


def instruction_text(prompt: str) -> str:
    """The prompt minus P2's quoted Switchboard examples (shown, not instructed)."""
    return re.sub(r"Here are \d+ short excerpts.*?\(End of examples\.\)", "", prompt, flags=re.S)


@pytest.mark.parametrize("level", ["P0", "P1", "P2"])
def test_main_design_never_instructs_measured_behaviour(level, personas, fake_fewshot):
    for prompt in all_prompts(level, *personas):
        assert not MEASURED.search(instruction_text(prompt)), prompt


def test_p2_differs_from_p1_only_by_the_examples(personas, fake_fewshot):
    for p1, p2 in zip(all_prompts("P1", *personas), all_prompts("P2", *personas), strict=True):
        assert "(End of examples.)" in p2
        assert " ".join(instruction_text(p2).split()) == " ".join(p1.split())


def test_p3_is_the_declared_exception(personas, monkeypatch):
    monkeypatch.setattr(templates, "_p3_fewshot_block", lambda: "")
    assert any(MEASURED.search(p) for p in all_prompts("P3", *personas))


def test_persona_cards_are_seeded_and_stable(personas):
    a, b = personas
    again_a = Persona("ParticipantA", "man", 54, "some college")
    again_b = Persona("ParticipantB", "woman", 33, "a college degree")
    assign_cards(again_a, again_b, 2121)
    assert (a, b) == (again_a, again_b)
    assert a.occupation in templates._OCCUPATIONS
    assert 0 <= a.stance < len(templates._STANCES)


def test_p0_keeps_bare_demographics_p1_adds_cards(personas):
    a, b = personas
    p0 = build_c1("P0", a, b, TOPIC, SB_PROMPT)[0]["content"]
    p1 = build_c1("P1", a, b, TOPIC, SB_PROMPT)[0]["content"]
    assert a.occupation not in p0
    assert a.occupation in p1
    assert SB_PROMPT in p0  # P0 quotes the original instruction verbatim (replication)


def test_turnwise_prompts_carry_the_turn_countdown(personas):
    a, b = personas
    prompt = build_c2("P1", a, b, TOPIC, SB_PROMPT, HISTORY, "ParticipantB", 40)[0]["content"]
    assert "has 3 turns of talk" in prompt
    assert "cannot go past 40 turns" in prompt


def test_agent_view_is_first_person_and_private(personas):
    a, b = personas
    messages = agent_view("P1", 40)("ParticipantB", a, b, TOPIC, SB_PROMPT, HISTORY, 2121)
    system, *dialogue = messages
    assert system["role"] == "system"
    assert b.occupation in system["content"]  # sees its own card...
    assert a.occupation not in system["content"]  # ...but not the stranger's
    # B's view: A's turns arrive as `user`, B's own as `assistant`.
    assert [m["role"] for m in dialogue] == ["user", "assistant", "user"]


def test_writer_view_sees_both_speakers(personas):
    a, b = personas
    (message,) = writer_view("P1", 40)("ParticipantA", a, b, TOPIC, SB_PROMPT, HISTORY, 2121)
    assert a.occupation in message["content"]
    assert b.occupation in message["content"]
    assert "ParticipantA: So, smog?" in message["content"]
