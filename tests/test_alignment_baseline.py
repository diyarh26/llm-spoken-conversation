import numpy as np

from analysis.alignment_baseline import conversation_scores


class FakeVectors:
    """Word "w" + k-th letter points at angle 15*k degrees; "aa"/"bb" are two orthogonal directions."""

    def __init__(self):
        self.key_to_index = dict.fromkeys(["aa", "bb"] + [f"w{chr(97 + k)}" for k in range(24)], 0)

    def __getitem__(self, w):
        if w in ("aa", "bb"):
            return np.array([1.0, 0.0]) if w == "aa" else np.array([0.0, 1.0])
        a = np.radians(15 * (ord(w[1]) - 97))
        return np.array([np.cos(a), np.sin(a)])


def turns(texts):
    return [("A" if i % 2 == 0 else "B", t) for i, t in enumerate(texts)]


def test_adjacent_similarity_above_baseline_is_positive():
    # the topic drifts slowly: neighbours are close, turns 5+ apart are far
    texts = [f"w{chr(97 + k)}" for k in range(10)]
    r = conversation_scores(FakeVectors(), turns(texts), "full")
    assert r["adjacent"] > r["baseline"]
    assert r["corrected"] == r["adjacent"] - r["baseline"]


def test_baseline_uses_only_cross_speaker_pairs_at_least_five_apart():
    # every turn identical -> adjacent and baseline both 1, corrected 0
    r = conversation_scores(FakeVectors(), turns(["aa"] * 8), "full")
    assert r["adjacent"] == r["baseline"] == 1.0 and r["corrected"] == 0.0
    # too short for any pair 5 apart -> no score
    assert conversation_scores(FakeVectors(), turns(["aa"] * 5), "full") is None


def test_first15_truncates_and_matched_drops_unequal_pairs():
    long_turn = " ".join(["aa"] * 15 + ["bb"] * 30)
    texts = [long_turn, "aa"] * 4
    full = conversation_scores(FakeVectors(), turns(texts), "full")
    first = conversation_scores(FakeVectors(), turns(texts), "first15")
    assert first["adjacent"] > full["adjacent"]  # the first 15 words are all "aa"
    assert conversation_scores(FakeVectors(), turns(texts), "matched") is None
