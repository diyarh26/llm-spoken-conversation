"""Workshop run isolation and an explicit trace of what each agent actually sees."""

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

from analysis.swda import make_personas
from generation import run_context, runner
from generation.response import spoken_response
from prompts.templates import agent_view, assign_cards
from tests.conftest import swda_row


def args(tmp_path, **overrides):
    parser = runner.base_parser(None)
    runner.add_turnwise_args(parser)
    value = parser.parse_args(["--out-root", str(tmp_path), "--prompt", "P1", "--max-turns", "4"])
    for key, item in overrides.items():
        setattr(value, key, item)
    return value


@pytest.mark.parametrize("architecture", ["C3", "C4"])
def test_agent_replies_are_partner_inputs_and_personas_are_private(tmp_path, architecture):
    """Trace four actual loop steps; same-model and mixed-model sessions share this loop."""
    a, b, topic, prompt = make_personas(swda_row())
    assign_cards(a, b, 2121)
    seen = []

    def engine(speaker):
        def call(messages, **kwargs):
            seen.append((speaker, messages))
            return f"{speaker} response {len(seen)}", {"hit_token_cap": False}

        return call

    history, ending, _ = runner._converse(
        args(tmp_path),
        {a.label: engine(a.label), b.label: engine(b.label)},
        agent_view("P1", 4),
        {},
        a,
        b,
        topic,
        prompt,
        2121,
    )
    assert ending == "turn_cap"
    assert [speaker for speaker, _ in seen] == [a.label, b.label, a.label, b.label]
    assert seen[1][1][-1] == {"role": "user", "content": history[2][1]}
    assert seen[2][1][-1] == {"role": "user", "content": history[3][1]}
    for speaker, messages in seen:
        me, partner = (a, b) if speaker == a.label else (b, a)
        assert me.card_first_person() in messages[0]["content"]
        assert partner.card_first_person() not in messages[0]["content"]
        for turn_speaker, text in history[2:]:
            for message in messages[1:]:
                if message["content"] == text:
                    assert message["role"] == ("assistant" if turn_speaker == speaker else "user")


@pytest.fixture
def bound_job(tmp_path, monkeypatch):
    monkeypatch.setattr(run_context, "ROOT", tmp_path)
    monkeypatch.setattr(run_context, "source_hash", lambda: "source")
    monkeypatch.setattr(
        run_context,
        "resolve_models",
        lambda models, args: {
            speaker: {"name": name, "revision": "immutable-sha"} for speaker, name in models.items()
        },
    )
    job = runner.Job("C2-P1", tmp_path / "new" / "C2-P1", [2121], {}, [2121], "inputs")
    job.out_dir.mkdir(parents=True)
    return job


def test_resume_rejects_different_model_settings_and_corrupt_records(bound_job, tmp_path):
    job = bound_job
    options = args(tmp_path)
    run_context.bind_run(job, options, {"ParticipantA": "model-a"}, {"temperature": 0.8})
    job.save(2121, {"conversation_no": 2121, "model": "model-a"})
    run_context.bind_run(job, options, {"ParticipantA": "model-a"}, {"temperature": 0.8})
    for models, decoding in [
        ({"ParticipantA": "model-b"}, {"temperature": 0.8}),
        ({"ParticipantA": "model-a"}, {"temperature": 0.7}),
    ]:
        with pytest.raises(ValueError, match="identity differs"):
            run_context.bind_run(job, options, models, decoding)
    (job.out_dir / "2121.json").write_text('{"conversation_no": 2121}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid/mixed"):
        run_context.bind_run(job, options, {"ParticipantA": "model-a"}, {"temperature": 0.8})


def test_original_dataset_cannot_be_a_new_run_target(bound_job, tmp_path):
    bound_job.out_dir = tmp_path / "data" / "generated" / "C2-P1"
    with pytest.raises(ValueError, match="protected"):
        run_context.bind_run(bound_job, args(tmp_path), {"A": "model"}, {})


def test_private_trace_preserves_reasoning_failure(bound_job):
    bound_job.run_metadata = {"run_id": "run-123"}
    engine = lambda messages, **kw: (
        "",
        {"raw_output": "<think>private analysis</think>", "reasoning_leak": True},
    )
    traced = run_context.traced_engine(engine, bound_job, 2121, "ParticipantA")
    with pytest.raises(RuntimeError, match="Unexpected reasoning"):
        traced([{"role": "user", "content": "private corpus example"}])
    path = run_context.ROOT / "_private/generation_raw/run-123/2121.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["messages"][0]["content"] == "private corpus example"
    assert "private analysis" in record["info"]["raw_output"]


def test_missing_p2_examples_stop_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_metadata", lambda: {2121: swda_row()})
    monkeypatch.setattr(runner, "load_target_ids", lambda: [2121])
    monkeypatch.setattr(runner, "fewshot_examples", lambda cno, k: [])
    with pytest.raises(ValueError, match="P2 examples unavailable"):
        runner.prepare_job("C1", args(tmp_path, prompt="P2"))


@pytest.mark.parametrize(
    ("raw", "decoded", "expected", "leak"),
    [
        ("<think>\n</think>\nYeah.", "<think>\n</think>\nYeah.", "Yeah.", False),
        (
            "<think>reasoning</think>Yes.",
            "<think>reasoning</think>Yes.",
            "<think>reasoning</think>Yes.",
            True,
        ),
        ("<|channel>thought\nanalysis<channel|>", "thought\nanalysis", "thought\nanalysis", True),
        ("<|channel>final\nHello.<channel|>", "final\nHello.", "Hello.", False),
    ],
)
def test_reasoning_never_silently_becomes_speech(raw, decoded, expected, leak):
    assert spoken_response(raw, decoded) == (expected, leak)


@pytest.mark.parametrize("architecture", ["c1", "c2", "c3", "c4"])
def test_generator_entrypoint_selects_models_and_resumes_without_gpu(
    tmp_path, monkeypatch, architecture
):
    """Exercise the public CLIs through saving and resume, not just their parsers."""
    module = importlib.import_module(f"generation.generate_{architecture}")
    loaded = []
    fake = types.ModuleType("generation.model_utils")

    def load(name, **kwargs):
        loaded.append((name, kwargs))
        return name, None

    fake.load_model = load
    fake.chat = lambda model, tok, messages, **kwargs: (
        "ParticipantA: Hello there.\nParticipantB: Good afternoon.",
        {"hit_token_cap": False, "n_new_tokens": 12},
    )
    monkeypatch.setitem(sys.modules, "generation.model_utils", fake)
    monkeypatch.setattr(runner, "load_metadata", lambda: {2121: swda_row()})
    monkeypatch.setattr(runner, "load_target_ids", lambda: [2121])
    monkeypatch.setattr(run_context, "ROOT", tmp_path)
    monkeypatch.setattr(run_context, "source_hash", lambda: "source")
    monkeypatch.setattr(
        run_context,
        "resolve_models",
        lambda models, args: {
            speaker: {"name": name, "revision": "sha-123"} for speaker, name in models.items()
        },
    )
    monkeypatch.setattr(runner, "seed_conversation", lambda *args: 77)
    if architecture == "c1":
        monkeypatch.setattr(module, "seed_conversation", lambda *args: 77)
    cli = ["--prompt", "P1", "--out-root", str(tmp_path / "new")]
    cli += (
        ["--model-a", "org/a", "--model-b", "org/b"]
        if architecture == "c4"
        else ["--model", "org/a"]
    )
    if architecture != "c1":
        cli += ["--max-turns", "2"]
    module.main(cli)
    record = json.loads((tmp_path / "new" / f"{architecture.upper()}-P1" / "2121.json").read_text())
    assert record["run_id"] and record["conversation_seed"] == 77
    assert loaded[0] == ("org/a", {"device": None, "revision": "sha-123", "precision": "nf4"})
    if architecture == "c4":
        assert loaded[1][0] == "org/b"
        assert record["models"] == {"ParticipantA": "org/a", "ParticipantB": "org/b"}
    before = len(loaded)
    module.main(cli)
    assert len(loaded) == before  # no weight loading when the valid run is complete
