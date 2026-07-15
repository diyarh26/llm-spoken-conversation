"""Deterministic correctness tests for the Semantic Route Reuse metric."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.semantic_routes import (
    EmbeddedRoute,
    RouteConversation,
    _common_topics,
    architecture_displacement_rows,
    architecture_displacement_summary_rows,
    calibrated_topic_rows,
    condition_summary_rows,
    contrast_rows,
    constrained_dtw_similarity,
    embed_routes,
    landmark_microchunks,
    lexical_words,
    load_route_conversations,
    nearest_sibling_rows,
    normalize_topic,
    pairwise_route_rows,
    progress_microchunks,
    progress_summary_rows,
    topic_condition_rows,
)


class DeterministicEmbedder:
    """Small content-sensitive test double; never downloads a model."""

    name = "deterministic-test-embedder"

    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.seen_texts.extend(texts)
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append([float(value + 1) for value in digest[:8]])
        return np.asarray(rows, dtype=float)


class ChunkLengthEmbedder:
    """Maps full chunks and residual chunks to orthogonal vectors."""

    name = "chunk-length-test-embedder"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if len(text.split()) == 4 else [0.0, 1.0] for text in texts]
        )


def make_conversation(
    condition: str,
    conversation_no: int,
    topic: str,
    turns: tuple[tuple[str, str], ...] = (("A", "some route text"),),
) -> RouteConversation:
    source = "SB" if condition == "SB" else "LLM"
    architecture, prompt_level = (
        ("SB", "SB") if condition == "SB" else tuple(condition.split("-", 1))
    )
    return RouteConversation(
        source=source,
        condition=condition,
        architecture=architecture,
        prompt_level=prompt_level,
        conversation_no=conversation_no,
        topic=topic,
        topic_key=normalize_topic(topic),
        turns=turns,
    )


def make_route(
    condition: str,
    conversation_no: int,
    topic: str,
    vectors: list[list[float]],
) -> EmbeddedRoute:
    matrix = np.asarray(vectors, dtype=float)
    return EmbeddedRoute(
        conversation=make_conversation(condition, conversation_no, topic),
        vectors=matrix,
        n_words=20,
        n_bins=len(matrix),
        n_microchunks=len(matrix),
        eligible=True,
    )


class TopicAndChunkingTests(unittest.TestCase):
    def test_normalize_topic_is_case_and_whitespace_insensitive(self) -> None:
        self.assertEqual(normalize_topic("  TRIAL\n BY   JURY  "), "trial by jury")
        self.assertEqual(normalize_topic("Care\u00a0Of The Elderly"), "care of the elderly")

    def test_equal_mass_microchunks_cover_every_word_once(self) -> None:
        turns = (
            ("A", "One two three."),
            ("B", "four five six seven eight nine ten"),
        )
        n_words, bins = progress_microchunks(
            turns, n_bins=3, max_microchunk_words=2
        )

        self.assertEqual(n_words, 10)
        reconstructed = [
            word
            for progress_bin in bins
            for microchunk in progress_bin
            for word in microchunk.split()
        ]
        self.assertEqual(reconstructed, lexical_words(turns))
        bin_sizes = [sum(len(chunk.split()) for chunk in progress_bin) for progress_bin in bins]
        self.assertEqual(bin_sizes, [4, 3, 3])
        self.assertTrue(
            all(len(chunk.split()) <= 2 for progress_bin in bins for chunk in progress_bin)
        )

    def test_chunking_and_embeddings_ignore_turn_boundaries(self) -> None:
        split_many = (
            ("A", "one two"),
            ("B", "three four"),
            ("A", "five six"),
        )
        split_few = (("A", "one two three four five six"),)

        self.assertEqual(
            progress_microchunks(split_many, n_bins=3, max_microchunk_words=1),
            progress_microchunks(split_few, n_bins=3, max_microchunk_words=1),
        )

        conversations = [
            make_conversation("C1-P0", 1, "Shared Topic", split_many),
            make_conversation("C2-P0", 1, "Shared Topic", split_few),
        ]
        embedder = DeterministicEmbedder()
        routes = embed_routes(
            conversations,
            embedder,
            n_bins=3,
            max_microchunk_words=1,
            min_words=1,
            route_representation="full-coverage",
        )
        self.assertTrue(routes[0].eligible)
        self.assertTrue(routes[1].eligible)
        np.testing.assert_allclose(routes[0].vectors, routes[1].vectors)

    def test_landmarks_equalize_semantic_mass_and_keep_short_calls(self) -> None:
        turns = (("A", " ".join(f"word{i}" for i in range(36))),)
        n_words, bins, unique_words = landmark_microchunks(
            turns, n_landmarks=8, landmark_words=20
        )
        self.assertEqual(n_words, 36)
        self.assertEqual(len(bins), 8)
        self.assertTrue(all(len(progress_bin[0].split()) == 20 for progress_bin in bins))
        self.assertEqual(unique_words, 36)

        route = embed_routes([make_conversation("C1-P0", 1, "Alpha", turns)], DeterministicEmbedder())[0]
        self.assertTrue(route.eligible)
        self.assertEqual(route.n_sampled_word_occurrences, 160)
        self.assertEqual(route.n_unique_sampled_words, 36)

    def test_microchunk_pooling_is_weighted_by_word_count(self) -> None:
        turns = (("A", "one two three four five six seven eight nine ten"),)
        route = embed_routes(
            [make_conversation("C1-P0", 1, "Alpha", turns)],
            ChunkLengthEmbedder(),
            n_bins=2,
            max_microchunk_words=4,
            min_words=1,
            route_representation="full-coverage",
        )[0]
        expected = np.asarray([4.0, 1.0]) / np.sqrt(17.0)
        np.testing.assert_allclose(route.vectors[0], expected)
        np.testing.assert_allclose(route.vectors[1], expected)


class DtwTests(unittest.TestCase):
    def setUp(self) -> None:
        self.route = np.eye(3, dtype=float)

    def test_identical_route_has_unit_similarity(self) -> None:
        similarity, path = constrained_dtw_similarity(self.route, self.route, band=0)
        self.assertAlmostEqual(similarity, 1.0)
        self.assertEqual(path, [(0, 0), (1, 1), (2, 2)])

    def test_reversing_the_route_changes_similarity(self) -> None:
        identical, _ = constrained_dtw_similarity(self.route, self.route, band=2)
        reversed_similarity, _ = constrained_dtw_similarity(
            self.route, self.route[::-1], band=2
        )
        self.assertLess(reversed_similarity, identical)
        self.assertLess(reversed_similarity, 0.8)

    def test_unequal_routes_can_align_when_band_allows_warping(self) -> None:
        repeated_opening = np.vstack([self.route[0], self.route])
        with self.assertRaisesRegex(ValueError, "No DTW path"):
            constrained_dtw_similarity(
                self.route, repeated_opening, band=0, warp_penalty=0.0
            )

        similarity, path = constrained_dtw_similarity(
            self.route, repeated_opening, band=1, warp_penalty=0.0
        )
        self.assertAlmostEqual(similarity, 1.0)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 3))

    def test_score_uses_fixed_route_resolution_not_selected_path_length(self) -> None:
        left = np.eye(4, dtype=float)[:3]
        right = np.asarray(
            [
                [0.2, 0.0, 0.0, np.sqrt(0.96)],
                [0.9, 0.2, 0.0, np.sqrt(0.15)],
                [0.0, 0.2, 0.2, np.sqrt(0.92)],
            ]
        )
        similarity, path = constrained_dtw_similarity(
            left, right, band=2, warp_penalty=0.05
        )
        self.assertEqual(path, [(0, 0), (1, 1), (2, 2)])
        self.assertAlmostEqual(similarity, 0.2)
        reverse_similarity, _ = constrained_dtw_similarity(
            right, left, band=2, warp_penalty=0.05
        )
        self.assertAlmostEqual(reverse_similarity, similarity)


class NearestSiblingTests(unittest.TestCase):
    def test_nearest_sibling_stays_inside_condition_and_topic(self) -> None:
        routes = [
            make_route("C1-P0", 1, "Alpha", [[1, 0], [1, 0]]),
            make_route("C1-P0", 2, "Alpha", [[0.99, 0.1], [0.99, 0.1]]),
            make_route("C1-P0", 3, "Alpha", [[0, 1], [0, 1]]),
            make_route("C2-P0", 1, "Alpha", [[1, 0], [1, 0]]),
            make_route("C2-P0", 2, "Alpha", [[1, 0], [1, 0]]),
            # These are exact semantic copies of C1-P0/1, but a different topic.
            make_route("C1-P0", 4, "Beta", [[1, 0], [1, 0]]),
            make_route("C1-P0", 5, "Beta", [[1, 0], [1, 0]]),
        ]
        pairs = pairwise_route_rows(routes)
        nearest = nearest_sibling_rows(routes, pairs)
        route_lookup = {
            route.conversation.key: route.conversation for route in routes
        }

        row_by_key = {
            (str(row["condition"]), int(row["conversation_no"])): row
            for row in nearest
        }
        self.assertEqual(row_by_key[("C1-P0", 1)]["nearest_sibling_no"], 2)
        for row in nearest:
            self.assertEqual(row["status"], "eligible")
            self.assertNotEqual(row["conversation_no"], row["nearest_sibling_no"])
            sibling = route_lookup[(str(row["condition"]), int(row["nearest_sibling_no"]))]
            current = route_lookup[(str(row["condition"]), int(row["conversation_no"]))]
            self.assertEqual(current.condition, sibling.condition)
            self.assertEqual(current.topic_key, sibling.topic_key)


class AggregationTests(unittest.TestCase):
    @staticmethod
    def calibrated_row(
        condition: str,
        topic: str,
        generated: float,
        human: float,
        *,
        primary: bool = True,
        n_generated: int = 3,
    ) -> dict:
        return {
            "condition": condition,
            "architecture": condition.split("-")[0],
            "prompt_level": condition.split("-")[1],
            "topic": topic.title(),
            "topic_key": normalize_topic(topic),
            "n_generated": n_generated,
            "n_human_pool": 20,
            "primary_eligible": primary,
            "generated_route_reuse": generated,
            "human_pool_matched_reuse": human,
            "excess_route_reuse": generated - human,
            "generated_above_human_nearest_p95_rate": 0.0,
        }

    def test_condition_summary_uses_common_topics_and_macro_average(self) -> None:
        rows = [
            # Very different pool sizes must not change the equal-topic average.
            self.calibrated_row("C1-P0", "Alpha", 0.0, 0.0, n_generated=100),
            self.calibrated_row("C1-P0", "Beta", 1.0, 0.0, n_generated=2),
            self.calibrated_row("C1-P0", "Only C1", 0.9, 0.0),
            self.calibrated_row("C2-P0", "Alpha", 0.2, 0.0),
            self.calibrated_row("C2-P0", "Beta", 0.4, 0.0),
        ]
        self.assertEqual(_common_topics(rows, primary=True), {"alpha", "beta"})

        routes = [
            make_route("C1-P0", 1, "Alpha", [[1, 0], [1, 0]]),
            make_route("C2-P0", 1, "Alpha", [[1, 0], [1, 0]]),
        ]
        summaries = condition_summary_rows(rows, routes, bootstrap_draws=20, seed=3)
        by_condition = {row["condition"]: row for row in summaries}

        self.assertEqual(by_condition["C1-P0"]["n_primary_topics"], 2)
        self.assertAlmostEqual(by_condition["C1-P0"]["mean_excess_route_reuse"], 0.5)
        self.assertAlmostEqual(by_condition["C2-P0"]["mean_excess_route_reuse"], 0.3)

    def test_exploratory_p2_does_not_remove_primary_topics(self) -> None:
        rows = []
        for condition in ("C1-P0", "C2-P0", "C1-P1", "C2-P1"):
            rows.extend(
                [
                    self.calibrated_row(condition, "Alpha", 0.3, 0.1),
                    self.calibrated_row(condition, "Beta", 0.4, 0.1),
                ]
            )
        for condition in ("C1-P2", "C2-P2"):
            rows.append(self.calibrated_row(condition, "Alpha", 0.5, 0.1))
        routes = [
            make_route(condition, index, "Alpha", [[1, 0], [1, 0]])
            for index, condition in enumerate(
                ("C1-P0", "C2-P0", "C1-P1", "C2-P1", "C1-P2", "C2-P2"),
                start=1,
            )
        ]
        summaries = condition_summary_rows(rows, routes, bootstrap_draws=10, seed=4)
        by_condition = {row["condition"]: row for row in summaries}
        self.assertEqual(by_condition["C1-P0"]["n_primary_topics"], 2)
        self.assertEqual(by_condition["C2-P1"]["n_primary_topics"], 2)
        self.assertEqual(by_condition["C1-P2"]["n_primary_topics"], 1)

    def test_contrasts_use_only_the_two_conditions_being_compared(self) -> None:
        rows = [
            self.calibrated_row("C1-P0", "Alpha", 0.2, 0.1),
            self.calibrated_row("C1-P0", "Beta", 0.3, 0.1),
            self.calibrated_row("C2-P0", "Alpha", 0.4, 0.1),
            self.calibrated_row("C2-P0", "Beta", 0.5, 0.1),
            self.calibrated_row("C3-P0", "Alpha", 0.6, 0.1),
        ]
        contrasts = contrast_rows(rows, bootstrap_draws=10, seed=5)
        lookup = {
            (row["left_condition"], row["right_condition"]): row for row in contrasts
        }
        self.assertEqual(lookup[("C1-P0", "C2-P0")]["n_topics"], 2)
        self.assertEqual(lookup[("C2-P0", "C3-P0")]["n_topics"], 1)


class HumanCalibrationTests(unittest.TestCase):
    def test_same_topic_and_pool_size_share_exact_human_reference_draws(self) -> None:
        routes = []
        base_vectors = (
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.9, 0.1], [0.9, 0.1]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.1, 0.9], [0.1, 0.9]],
        )
        for index, vectors in enumerate(base_vectors, start=1):
            routes.append(make_route("SB", index, "Alpha", vectors))
        for condition in ("C1-P0", "C2-P0"):
            for index, vectors in enumerate(base_vectors[:3], start=101):
                routes.append(make_route(condition, index, "Alpha", vectors))

        pairs = pairwise_route_rows(routes)
        nearest = nearest_sibling_rows(routes, pairs)
        topics = topic_condition_rows(nearest, pairs)
        calibrated, references = calibrated_topic_rows(
            routes, pairs, topics, human_draws=25, seed=17
        )
        by_condition = {row["condition"]: row for row in calibrated}
        for field in (
            "human_pool_matched_reuse",
            "human_subsample_mean_q025",
            "human_subsample_mean_q975",
            "human_subsample_nearest_p95",
        ):
            self.assertEqual(by_condition["C1-P0"][field], by_condition["C2-P0"][field])
        self.assertEqual(len(references), 1)
        self.assertNotIn("one_sided_subsampling_p", by_condition["C1-P0"])
        self.assertNotIn("near_duplicate_rate", by_condition["C1-P0"])


class SecondaryOutputTests(unittest.TestCase):
    def test_progress_uses_bin_midpoints(self) -> None:
        vectors = [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]]
        routes = []
        for condition in ("SB", "C1-P0"):
            for conversation_no in (1, 2, 3):
                routes.append(make_route(condition, conversation_no, "Alpha", vectors))
        calibrated = [
            AggregationTests.calibrated_row("C1-P0", "Alpha", 0.5, 0.4)
        ]
        rows = progress_summary_rows(
            routes, calibrated, n_points=4, bootstrap_draws=10, seed=8
        )
        self.assertEqual(
            [row["progress_fraction"] for row in rows],
            [0.125, 0.375, 0.625, 0.875],
        )

    def test_architecture_displacement_uses_matched_inputs(self) -> None:
        vectors_by_id = {
            1: [[1.0, 0.0], [1.0, 0.0]],
            2: [[0.0, 1.0], [0.0, 1.0]],
            3: [[0.7, 0.7], [0.7, 0.7]],
        }
        routes = [
            make_route(condition, conversation_no, "Alpha", vectors)
            for condition in ("C1-P0", "C2-P0")
            for conversation_no, vectors in vectors_by_id.items()
        ]
        matched, topics = architecture_displacement_rows(routes)
        summaries = architecture_displacement_summary_rows(
            topics, bootstrap_draws=10, seed=9
        )
        self.assertEqual(len(matched), 3)
        self.assertEqual(len(topics), 1)
        self.assertAlmostEqual(topics[0]["mean_same_input_route_similarity"], 1.0)
        self.assertGreater(topics[0]["same_input_similarity_advantage"], 0.0)
        self.assertEqual(len(summaries), 1)
        self.assertAlmostEqual(summaries[0]["mean_architecture_route_displacement"], 0.0)


class LoaderFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.generated_root = self.root / "generated"
        self.swda_root = self.root / "swda"
        (self.swda_root / "sw00utt").mkdir(parents=True)

        with (self.swda_root / "swda-metadata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["conversation_no", "topic_description", "prompt"]
            )
            writer.writeheader()
            for conversation_no in (1001, 1002, 1003):
                writer.writerow(
                    {
                        "conversation_no": conversation_no,
                        "topic_description": "SHARED   TOPIC",
                        "prompt": "Discuss the shared topic.",
                    }
                )

        for conversation_no in (1001, 1002, 1003):
            path = self.swda_root / "sw00utt" / f"sw_0001_{conversation_no}.utt.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["caller", "text"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"caller": "A", "text": "Human opening. /"},
                        {"caller": "A", "text": "More detail. /"},
                        {"caller": "B", "text": "Human response. /"},
                    ]
                )

        for condition in ("C1-P0", "C2-P0"):
            (self.generated_root / condition).mkdir(parents=True)
            for conversation_no in (1001, 1002):
                self.write_generated(condition, conversation_no)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_generated(self, condition: str, conversation_no: int) -> None:
        common = {
            "condition": condition,
            "architecture": condition.split("-")[0],
            "prompt_level": condition.split("-")[1],
            "conversation_no": conversation_no,
            # Deliberately differs from authoritative metadata in case/whitespace.
            "topic": "Shared Topic",
        }
        if condition.startswith("C1"):
            common["raw_output"] = (
                "ParticipantA: Generated opening.\n"
                "ParticipantB: Generated response."
            )
        else:
            common["turns"] = [
                ["ParticipantA", "Generated opening."],
                ["ParticipantB", "Generated response."],
            ]
        path = self.generated_root / condition / f"{conversation_no}.json"
        path.write_text(json.dumps(common), encoding="utf-8")

    def test_loader_handles_both_schemas_and_uses_authoritative_topic(self) -> None:
        conversations = load_route_conversations(
            self.generated_root, self.swda_root, strict_balanced=True
        )
        human = [conversation for conversation in conversations if conversation.source == "SB"]
        generated = [
            conversation for conversation in conversations if conversation.source == "LLM"
        ]

        # The whole same-topic human pool is retained, not only the generated seed IDs.
        self.assertEqual(len(human), 3)
        self.assertEqual(len(generated), 4)
        self.assertEqual({conversation.topic_key for conversation in conversations}, {"shared topic"})
        self.assertEqual({conversation.topic for conversation in conversations}, {"Shared Topic"})
        self.assertEqual(
            {len(conversation.turns) for conversation in generated}, {2}
        )
        # Consecutive SwDA annotation rows from caller A are merged into one turn.
        self.assertEqual(len(human[0].turns), 2)

    def test_strict_loader_rejects_unbalanced_condition_ids(self) -> None:
        self.write_generated("C2-P0", 1003)
        with self.assertRaisesRegex(ValueError, "do not share the same conversation IDs"):
            load_route_conversations(
                self.generated_root, self.swda_root, strict_balanced=True
            )

        conversations = load_route_conversations(
            self.generated_root, self.swda_root, strict_balanced=False
        )
        c2_ids = {
            conversation.conversation_no
            for conversation in conversations
            if conversation.condition == "C2-P0"
        }
        self.assertEqual(c2_ids, {1001, 1002, 1003})

    def test_loader_can_require_the_complete_condition_factorial(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected condition factorial"):
            load_route_conversations(
                self.generated_root,
                self.swda_root,
                expected_conditions={"C1-P0", "C2-P0", "C3-P0"},
            )


if __name__ == "__main__":
    unittest.main()
