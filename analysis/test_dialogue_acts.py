"""Small correctness tests for the dialogue-act normalization and distances."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.analyze import conversation_turns, has_language_drift
from analysis.dialogue_acts import (
    COARSE_LABELS,
    DIALOGTAG_TO_FINE,
    FINE_LABELS,
    FINE_TO_COARSE,
    dialogtag_to_fine,
    drop_loop_turns,
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


class SeedTurnTests(unittest.TestCase):
    def test_turnwise_seed_greetings_dropped(self) -> None:
        rec = {"seed_turns": 2, "turns": [["A", "Hello!"], ["B", "Hello!"], ["A", "Uh-huh."]]}
        self.assertEqual(conversation_turns(rec), [("A", "Uh-huh.")])

    def test_c1_model_opener_kept(self) -> None:
        # C1 has the greetings only in its prompt; a model-written opener must survive.
        rec = {"seed_turns": 2, "raw_output": "A: Hi, how are you?\nB: Hello! Good."}
        self.assertEqual(len(conversation_turns(rec)), 2)
        echoed = {"seed_turns": 2, "raw_output": "A: Hello!\nB: Hello!\nA: So, pets?"}
        self.assertEqual(conversation_turns(echoed), [("A", "So, pets?")])

    def test_records_without_seed_field_unchanged(self) -> None:
        rec = {"turns": [["A", "Hello!"], ["B", "Hello!"]]}
        self.assertEqual(len(conversation_turns(rec)), 2)


class ArtifactCleaningTests(unittest.TestCase):
    def test_meta_notes_are_stripped_and_speech_kept(self):
        rec = {"turns": [
            ["A", "I like fishing. (This is turn 14)"],
            ["B", "Me too. What about you?\n\n(End of turn)\n\n[The conversation cannot go past 40 turns.]"],
            ["A", "(Conversation ends here)"],
            ["B", "I put it in my 401(k) and talked (briefly) about it."],
        ]}
        self.assertEqual(conversation_turns(rec), [
            ("A", "I like fishing."),
            ("B", "Me too. What about you?"),
            ("B", "I put it in my 401(k) and talked (briefly) about it."),
        ])

    def test_language_drift_is_cut_and_flagged(self):
        rec = {"turns": [["A", "Do you garden?\n\n번역결과 \n안녕"],
                         ["B", "안녕하세요"],
                         ["A", "Café au lait, Bren� Brown."]]}
        self.assertEqual(conversation_turns(rec),
                         [("A", "Do you garden?"), ("A", "Café au lait, Bren Brown.")])
        self.assertTrue(has_language_drift(rec))
        self.assertFalse(has_language_drift({"turns": [["A", "Café, naïve — 50€ ™"]]}))


class LoopFilterTests(unittest.TestCase):
    def test_echo_repeats_dropped_backchannels_kept(self) -> None:
        long_turn = "yes I have test driven the Honda Accord and the Toyota Camry myself"
        turns = [("A", long_turn), ("B", "Uh-huh."), ("A", long_turn + " too"),
                 ("B", "Uh-huh."), ("A", "What about the price of those two cars though?")]
        kept = drop_loop_turns(turns)
        self.assertEqual([t for _, t in kept],
                         [long_turn, "Uh-huh.", "Uh-huh.",
                          "What about the price of those two cars though?"])


class GranularityTests(unittest.TestCase):
    def test_sentence_units_preserve_short_reactions(self) -> None:
        self.assertEqual(sentence_units("Yeah. I see what you mean! Right?"),
                         ["Yeah.", "I see what you mean!", "Right?"])

    def test_sentence_units_do_not_split_abbreviations_without_terminal_space(self) -> None:
        self.assertEqual(sentence_units("I met Dr. Smith yesterday."),
                         ["I met Dr.", "Smith yesterday."])


if __name__ == "__main__":
    unittest.main()
