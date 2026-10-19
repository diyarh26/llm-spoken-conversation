"""End-to-end generator plumbing with a stub model: ids, records, resumability."""

from __future__ import annotations

import argparse
import json

import pytest

from generation import runner
from generation.config import MAX_TURNS, TURNWISE
from generation.turnwise import ENDINGS
from prompts.templates import agent_view, writer_view
from tests.conftest import swda_row

IDS = [2121, 2145, 2155]


@pytest.fixture
def fake_corpus(monkeypatch):
    meta = {cno: swda_row(conversation_no=str(cno)) for cno in IDS}
    monkeypatch.setattr(runner, "load_metadata", lambda: meta)
    monkeypatch.setattr(runner, "load_target_ids", lambda: list(IDS))
    return meta


def parse(*argv):
    ap = runner.base_parser("test")
    runner.add_turnwise_args(ap)
    return ap.parse_args(list(argv))


def stub_engine(name):
    """A fake model: chats for a few turns, then says goodbye."""

    def engine(messages, *, temperature, stop_labels, **decoding):
        assert stop_labels == ("ParticipantA", "ParticipantB")
        assert decoding["max_new_tokens"] == TURNWISE.max_new_tokens
        n = len(json.dumps(messages))
        if n % 7 == 0:
            return "Okay, bye!", {"hit_token_cap": False}
        return f"{name} says something about item {n}.", {"hit_token_cap": False}

    return engine


def test_cli_defaults_are_the_frozen_config():
    args = parse()
    assert args.prompt == "P0"
    assert args.max_turns == MAX_TURNS
    assert args.max_new_tokens == TURNWISE.max_new_tokens
    assert args.repetition_penalty == 1.0
    assert args.no_repeat_ngram == 0
    assert args.stop_at_sentence is False


def test_prepare_job_resolves_ids(fake_corpus, tmp_path):
    job = runner.prepare_job("C2", parse("--prompt", "P1", "--out-root", str(tmp_path)))
    assert job.condition == "C2-P1"
    assert job.todo == IDS
    job = runner.prepare_job("C2", parse("--n", "2", "--out-root", str(tmp_path)))
    assert job.todo == IDS[:2]
    job = runner.prepare_job("C2", parse("--ids", "2155, 9999", "--out-root", str(tmp_path)))
    assert job.todo == [2155]  # unknown ids are dropped


@pytest.mark.parametrize(
    ("arch", "view", "model_info"),
    [
        ("C2", writer_view, {"model": "vicuna"}),
        ("C3", agent_view, {"models": {"ParticipantA": "vicuna", "ParticipantB": "vicuna"}}),
        ("C4", agent_view, {"models": {"ParticipantA": "vicuna", "ParticipantB": "mistral"}}),
    ],
)
def test_turnwise_job_writes_valid_records_and_resumes(
    fake_corpus, tmp_path, arch, view, model_info
):
    args = parse("--prompt", "P1", "--out-root", str(tmp_path), "--max-turns", "8")
    engines = {"ParticipantA": stub_engine("A"), "ParticipantB": stub_engine("B")}
    job = runner.prepare_job(arch, args)
    runner.run_turnwise_job(
        job,
        args,
        engines=engines,
        build_messages=view(args.prompt, args.max_turns),
        model_info=model_info,
    )

    files = sorted((tmp_path / f"{arch}-P1").glob("*.json"))
    assert [int(f.stem) for f in files] == sorted(IDS)
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        assert rec["condition"] == f"{arch}-P1"
        assert rec["architecture"] == arch
        assert rec["conversation_no"] == int(f.stem)
        assert rec["ended_by"] in ENDINGS
        assert rec["n_turns"] == len(rec["turns"]) <= 2 + 8
        assert rec["turns"][:2] == [["ParticipantA", "Hello!"], ["ParticipantB", "Hello!"]]
        assert rec["seed_turns"] == 2
        assert rec["persona_a"]["occupation"]  # seeded persona card was assigned
        assert rec["decoding"]["max_turns"] == 8
        for key, value in model_info.items():
            assert rec[key] == value

    # Resumability: a second run finds nothing left to do.
    assert runner.prepare_job(arch, args).todo == []


def test_job_saves_one_file_per_conversation(tmp_path):
    job = runner.Job("C1-P0", tmp_path, [1], {})
    path = job.save(1, {"x": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}


def test_parser_rejects_unknown_prompt_level():
    with pytest.raises(SystemExit):
        runner.base_parser(None).parse_args(["--prompt", "P9"])


def test_namespace_shape_matches_decoding_record():
    args = argparse.Namespace(
        max_new_tokens=1, min_new_tokens=2, temperature=0.5, top_p=0.9,
        repetition_penalty=1.0, no_repeat_ngram=0, stop_at_sentence=False, max_turns=4,
    )  # fmt: skip
    assert runner.turnwise_decoding(args) == {
        "max_new_tokens": 1,
        "min_new_tokens": 2,
        "temperature": 0.5,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "stop_at_sentence": False,
        "max_turns": 4,
    }
