"""Seeded, topic-stratified sampling of the 50 Switchboard conversations."""

from __future__ import annotations

from collections import Counter

import pytest

from generation import sampling


@pytest.fixture
def pool(monkeypatch):
    by_topic = {
        "CARS": list(range(100, 160)),  # 60
        "PETS": list(range(200, 230)),  # 30
        "TAXES": list(range(300, 310)),  # 10
    }
    monkeypatch.setattr(sampling, "_pool_by_topic", lambda meta: by_topic)
    topic_of = {cno: topic for topic, ids in by_topic.items() for cno in ids}
    return topic_of


def test_allocation_is_proportional(pool):
    target, _, allocation = sampling.stratified_sample(10, seed=1, meta={})
    assert allocation == {"CARS": 6, "PETS": 3, "TAXES": 1}
    assert Counter(pool[c] for c in target) == Counter(allocation)
    assert len(set(target)) == 10


def test_largest_remainder_rounding_hits_n_exactly(pool):
    for n in (1, 7, 11, 23):
        target, _, allocation = sampling.stratified_sample(n, seed=3, meta={})
        assert len(target) == n == sum(allocation.values())


def test_draw_is_deterministic_and_seed_dependent(pool):
    first = sampling.stratified_sample(10, seed=20260714, meta={})
    assert first == sampling.stratified_sample(10, seed=20260714, meta={})
    assert first[0] != sampling.stratified_sample(10, seed=7, meta={})[0]


def test_dev_set_is_disjoint_from_targets(pool):
    target, dev, _ = sampling.stratified_sample(10, seed=1, meta={})
    assert len(dev) == sampling.DEFAULT_DEV
    assert not set(dev) & set(target)


def test_committed_manifest():
    manifest = sampling.load_manifest()
    target, dev = manifest["target_ids"], manifest["dev_ids"]
    assert manifest["seed"] == sampling.DEFAULT_SEED
    assert len(target) == len(set(target)) == 50
    assert len(dev) == 5
    assert not set(target) & set(dev)
    assert sum(manifest["topic_allocation"].values()) == 50
