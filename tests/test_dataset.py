"""Integrity of the released dataset: 12 conditions x 50 conversations in data/generated/."""

from __future__ import annotations

import json
import pathlib

import pytest

from generation.config import C1, MAX_TURNS, MISTRAL, TURNWISE, VICUNA
from generation.sampling import load_target_ids
from generation.turnwise import ENDINGS

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "generated"
CONDITIONS = [f"C{a}-P{p}" for a in range(1, 5) for p in range(3)]
TARGETS = sorted(load_target_ids())


def records(condition):
    for path in sorted((DATA / condition).glob("*.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def test_exactly_the_twelve_conditions_are_released():
    assert sorted(p.name for p in DATA.iterdir() if p.is_dir()) == CONDITIONS


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_condition_covers_the_same_50_conversations(condition):
    ids = sorted(int(p.stem) for p in (DATA / condition).glob("*.json"))
    assert ids == TARGETS


@pytest.mark.parametrize("condition", CONDITIONS)
def test_records_are_well_formed(condition):
    arch, prompt = condition.split("-")
    for path, rec in records(condition):
        assert rec["condition"] == condition, path
        assert rec["architecture"] == arch
        assert rec["prompt_level"] == prompt
        assert rec["conversation_no"] == int(path.stem)
        assert rec["seed_turns"] == 2
        assert rec["topic"] and rec["sb_prompt"]
        for persona in (rec["persona_a"], rec["persona_b"]):
            assert persona["gender"] in {"man", "woman"}
            assert persona["occupation"]  # every condition shares the seeded cards

        if arch == "C1":
            assert rec["model"] == VICUNA
            assert rec["raw_output"].strip()
            assert rec["decoding"]["max_new_tokens"] == C1.max_new_tokens
        else:
            assert rec["ended_by"] in ENDINGS
            assert rec["n_turns"] == len(rec["turns"]) <= 2 + MAX_TURNS
            assert rec["turns"][:2] == [["ParticipantA", "Hello!"], ["ParticipantB", "Hello!"]]
            assert all(t[0] in {"ParticipantA", "ParticipantB"} and t[1] for t in rec["turns"])
            assert rec["decoding"]["repetition_penalty"] == TURNWISE.repetition_penalty
            # C2 was generated before the per-turn cap was raised from 300 to 512 tokens
            # (it hit the old cap once in 5,596 turns; see generation/config.py).
            expected_cap = 300 if arch == "C2" else TURNWISE.max_new_tokens
            assert rec["decoding"]["max_new_tokens"] == expected_cap


@pytest.mark.parametrize("arch", ["C2", "C3", "C4"])
def test_models_per_architecture(arch):
    for _, rec in records(f"{arch}-P1"):
        models = {rec["model"]} if "model" in rec else set(rec["models"].values())
        assert models == ({VICUNA, MISTRAL} if arch == "C4" else {VICUNA})


def test_conditions_are_paired_on_personas_and_topics():
    """All 12 conditions of one conversation share topic and speakers: the design is paired."""
    reference = {p.stem: rec for p, rec in records("C1-P1")}
    for condition in CONDITIONS:
        for path, rec in records(condition):
            ref = reference[path.stem]
            assert rec["topic"] == ref["topic"]
            assert rec["persona_a"] == ref["persona_a"]
            assert rec["persona_b"] == ref["persona_b"]
