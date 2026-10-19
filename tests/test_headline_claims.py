"""The README's headline claims, checked against the committed result tables.

If a result file is regenerated and a claim stops holding, this fails before the README
can drift out of date."""

from __future__ import annotations

import csv
import pathlib
import statistics

import pytest

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
ARCHS, PROMPTS = ("C1", "C2", "C3", "C4"), ("P0", "P1", "P2")


def table(*parts):
    with (RESULTS.joinpath(*parts)).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def jsd():
    rows = table("dialogue_acts", "da_jsd_vs_sb.csv")
    return {
        r["condition"]: float(r["jsd_dist"])
        for r in rows
        if r["label_set"] == "coarse" and r["view"] == "tagger_human_vs_tagger_llm_primary"
    }


def test_all_at_once_generation_is_closest_to_humans(jsd):
    by_arch = {a: statistics.mean(jsd[f"{a}-{p}"] for p in PROMPTS) for a in ARCHS}
    assert min(by_arch, key=by_arch.get) == "C1"
    assert round(by_arch["C1"], 3) == 0.128


def test_architecture_moves_structure_more_than_the_prompt(jsd):
    by_arch = [statistics.mean(jsd[f"{a}-{p}"] for p in PROMPTS) for a in ARCHS]
    by_prompt = [statistics.mean(jsd[f"{a}-{p}"] for a in ARCHS) for p in PROMPTS]
    ratio = (max(by_arch) - min(by_arch)) / (max(by_prompt) - min(by_prompt))
    assert round(ratio, 1) == 3.6


def test_c1_is_closer_in_every_bootstrap_replicate():
    lines = (RESULTS / "da_jsd_bootstrap.csv").read_text(encoding="utf-8").split("\n\n")[1]
    contrasts = list(csv.DictReader(lines.strip().splitlines()))
    assert {r["contrast"] for r in contrasts} == {"C2-minus-C1", "C3-minus-C1", "C4-minus-C1"}
    assert all(float(r["share_of_reps_C1_closer"]) == 1.0 for r in contrasts)


def test_every_condition_is_far_above_the_noise_floor(jsd):
    floors = {
        r["condition"]: float(r["p95_jsd"])
        for r in table("dialogue_acts", "da_noise_floor.csv")
        if r["label_set"] == "coarse"
    }
    assert all(jsd[c] > 10 * floors[c] for c in jsd if c in floors)


def test_the_listener_is_missing():
    rows = {r["condition"]: r for r in table("dialogue_acts", "da_rule_crosscheck.csv")}
    human = float(rows.pop("SB")["rule_backchannel_rate"])
    llm = [float(r["rule_backchannel_rate"]) for r in rows.values()]
    assert round(100 * human) == 19
    assert len(llm) == 12
    assert max(llm) <= 0.004


def test_humans_turn_length_replicates_the_original_study():
    sb = next(r for r in table("metrics_summary.csv") if r["condition"] == "SB")
    assert round(float(sb["mean_words_per_turn_mean"]), 1) == 14.4  # original study: 13.97
    assert round(float(sb["rate_uh_huh_mean"]), 2) == 1.03  # original study: 1.03


def test_agents_talk_in_long_turns():
    m = {r["condition"]: float(r["mean_words_per_turn_mean"]) for r in table("metrics_summary.csv")}
    assert max(m[f"C4-{p}"] for p in PROMPTS) / m["SB"] > 7
