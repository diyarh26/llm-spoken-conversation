from collections import Counter

from analysis.memorisation import _Excluding, ngrams, overlap, tokens


def test_ngrams_and_overlap():
    words = tokens("So, what kind of car are you thinking of?")
    grams = ngrams(words, 3)
    assert grams[0] == ("so", "what", "kind") and len(grams) == len(words) - 2
    assert overlap(grams, {("so", "what", "kind")}) == 1 / len(grams)
    assert overlap([], set()) is None


def test_leave_one_out_ignores_the_conversation_itself():
    df = Counter({("a", "b"): 1, ("c", "d"): 2})
    others = _Excluding(df, own={("a", "b"), ("c", "d")})
    assert ("a", "b") not in others  # only in this conversation
    assert ("c", "d") in others  # also in another conversation
