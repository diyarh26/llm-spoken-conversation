"""Small correctness tests for the dialogue-act normalization and distances."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.dialogue_acts import (
    COARSE_LABELS,
    DIALOGTAG_TO_FINE,
    FINE_LABELS,
    FINE_TO_COARSE,
    dialogtag_to_fine,
    js_divergence,
    normalize_swda_base,
    sentence_units,
    transition_jsd,
)


class NormalizationTests(unittest.TestCase):
    def test_suffix_modifiers_and_annotation_flags(self) -> None:
        cases = {
            "sd^e": "sd",
            "qy^d": "qy",
            "qw^d^t": "qw",
            "b^m": "b",
            "sd(^q)@": "sd",
            "+@": "+",
            "%@*": "%",
            "x@": "x",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_swda_base(raw), expected)

    def test_leading_caret_acts_are_not_suffixes(self) -> None:
        self.assertEqual(normalize_swda_base("^q^t"), "^q")
        self.assertEqual(normalize_swda_base("^2@"), "^2")
        self.assertEqual(normalize_swda_base("^h^r"), "^h")
        self.assertEqual(normalize_swda_base("^g@"), "^g")

    def test_standard_aliases_and_compounds(self) -> None:
        cases = {
            "qr^d": "qy",
            "fe": "ba",
            "co^t": "oo",
            "fx": "sv",
            "am^r": "aap",
            "nd^t": "arp",
            "fw*": "fo",
            "sd;no": "sd",
            "aa,ar": "aa",
            "nn^e": "ng",
            "ny^e": "na",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_swda_base(raw), expected)

    def test_dialogtag_question_collapses(self) -> None:
        self.assertEqual(dialogtag_to_fine("Declarative Yes-No-Question"), "qy")
        self.assertEqual(dialogtag_to_fine("Declarative Wh-Question"), "qw")
        self.assertEqual(dialogtag_to_fine("Repeat-phrase"), "b")
        self.assertEqual(
            dialogtag_to_fine("backchannel in question form"), "bh"
        )

    def test_shared_inventory_is_exhaustive(self) -> None:
        self.assertEqual(len(FINE_LABELS), 39)
        self.assertEqual(set(FINE_LABELS), set(FINE_TO_COARSE))
        self.assertEqual(set(COARSE_LABELS), set(FINE_TO_COARSE.values()))
        self.assertTrue(set(DIALOGTAG_TO_FINE.values()).issubset(FINE_LABELS))


class DistanceTests(unittest.TestCase):
    def test_jsd_bounds_and_symmetry(self) -> None:
        left = np.array([1.0, 0.0])
        right = np.array([0.0, 1.0])
        self.assertEqual(js_divergence(left, left), 0.0)
        self.assertAlmostEqual(js_divergence(left, right), 1.0)
        self.assertAlmostEqual(
            js_divergence(left, right), js_divergence(right, left)
        )

    def test_transition_missing_row_is_maximal(self) -> None:
        left = np.array([[1.0, 0.0], [0.0, 0.0]])
        right = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(transition_jsd(left, right), 0.5)


class GranularityTests(unittest.TestCase):
    def test_sentence_units_preserve_short_reactions(self) -> None:
        self.assertEqual(sentence_units("Yeah. I see what you mean! Right?"),
                         ["Yeah.", "I see what you mean!", "Right?"])

    def test_sentence_units_do_not_split_abbreviations_without_terminal_space(self) -> None:
        self.assertEqual(sentence_units("I met Dr. Smith yesterday."),
                         ["I met Dr.", "Smith yesterday."])


if __name__ == "__main__":
    unittest.main()
