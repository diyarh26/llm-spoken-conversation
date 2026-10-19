"""Shared fixtures. Every test runs without a GPU and without the licensed Switchboard corpus;
the few that need the corpus are marked ``corpus`` and skipped when it is absent."""

from __future__ import annotations

import pathlib

import pytest

from analysis.swda import DATA_ROOT

ROOT = pathlib.Path(__file__).resolve().parent.parent
HAS_CORPUS = (DATA_ROOT / "swda-metadata.csv").exists()


def pytest_collection_modifyitems(config, items):
    if HAS_CORPUS:
        return
    skip = pytest.mark.skip(reason="Switchboard (SwDA) corpus not available (LDC license)")
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(skip)


def swda_row(**overrides) -> dict:
    """A synthetic SwDA metadata row with the fields the generators read."""
    row = {
        "conversation_no": "2121",
        "talk_day": "911108",
        "topic_description": "AIR POLLUTION",
        "prompt": "PLEASE DISCUSS AIR POLLUTION.",
        "from_caller_sex": "MALE",
        "from_caller_birth_year": "1937",
        "from_caller_education": "2",
        "to_caller_sex": "FEMALE",
        "to_caller_birth_year": "1958",
        "to_caller_education": "3",
    }
    row.update(overrides)
    return row


@pytest.fixture
def root() -> pathlib.Path:
    return ROOT
