"""Topic-controlled Semantic Route Reuse (SRR) for whole conversations.

SRR treats a conversation as an ordered route through semantic space.  It is a
supporting metric for the dialogue-act analysis: the question here is whether
independently generated conversations on the same Switchboard topic repeatedly
follow the same content path.

The implementation deliberately avoids the two failure modes of the legacy CED:

* topics are joined through Switchboard ``conversation_no`` metadata and compared
  with a normalized key, rather than pooled across the corpus; and
* an ordered sequence of fixed-word-count route landmarks is retained, rather than mean
  pooling the entire conversation into one vector.

Run the provisional analysis with::

    py -3 analysis/semantic_routes.py --generated-root data/generated_v2

Sentence-transformers is required.  There is intentionally no TF-IDF fallback:
word-overlap vectors would mostly recover the prompted topic that SRR is designed
to control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations, product
from typing import Collection, Protocol, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.analyze import conversation_turns  # noqa: E402
from analysis.swda import (  # noqa: E402
    conversation_no_of,
    iter_conversation_files,
    load_metadata,
    parse_conversation,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_GENERATED_ROOT = ROOT / "data" / "generated_v2"
DEFAULT_SWDA_ROOT = ROOT / "data" / "switchboard" / "swda"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "semantic_routes"
DEFAULT_MODEL = "all-mpnet-base-v2"
EXPECTED_GENERATED_CONDITIONS = frozenset(
    f"C{architecture}-P{prompt}"
    for architecture in range(1, 5)
    for prompt in range(3)
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class RouteConversation:
    source: str
    condition: str
    architecture: str
    prompt_level: str
    conversation_no: int
    topic: str
    topic_key: str
    turns: tuple[tuple[str, str], ...]

    @property
    def key(self) -> tuple[str, int]:
        return self.condition, self.conversation_no


@dataclass
class EmbeddedRoute:
    conversation: RouteConversation
    vectors: np.ndarray
    n_words: int
    n_bins: int
    n_microchunks: int
    eligible: bool
    exclusion_reason: str = ""
    route_representation: str = "landmarks"
    n_sampled_word_occurrences: int = 0
    n_unique_sampled_words: int = 0


class TextEmbedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class SentenceRouteEmbedder:
    """Strict sentence-transformer backend used by the research metric."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the local environment
            raise RuntimeError(
                "Semantic Route Reuse requires sentence-transformers. Install the "
                "requirements; TF-IDF is not a valid fallback for this metric."
            ) from exc

        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # pragma: no cover - model/network dependent
            raise RuntimeError(f"Could not load sentence-transformer {model_name!r}: {exc}") from exc
        self.name = f"sentence-transformers:{model_name}"
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        arr = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(arr, dtype=float)


def normalize_topic(topic: str) -> str:
    """Canonical comparison key; display capitalization is never used for joins."""

    return re.sub(r"\s+", " ", topic).strip().casefold()


def _display_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", topic).strip().title()


def condition_sort_key(condition: str) -> tuple[int, int, str]:
    if condition == "SB":
        return -1, -1, condition
    match = re.fullmatch(r"C(\d+)-P(\d+)", condition)
    if match:
        architecture, prompt = (int(x) for x in match.groups())
        return prompt, architecture, condition
    return 999, 999, condition


def _record_condition(path: pathlib.Path, record: dict) -> str:
    folder_condition = path.parent.name
    condition = str(record.get("condition") or folder_condition)
    if condition != folder_condition:
        raise ValueError(
            f"Condition mismatch in {path}: folder={folder_condition!r}, record={condition!r}"
        )
    return condition


def load_route_conversations(
    generated_root: pathlib.Path = DEFAULT_GENERATED_ROOT,
    swda_root: pathlib.Path = DEFAULT_SWDA_ROOT,
    *,
    strict_balanced: bool = True,
    expected_conditions: Collection[str] | None = None,
) -> list[RouteConversation]:
    """Load generated data and the full topic-matched Switchboard reference.

    Generated records are joined to authoritative Switchboard topic metadata through
    ``conversation_no``.  Every human conversation on one of the generated topics is
    loaded, not just the 50 conversations used to seed generation.  Pool-size-matched
    subsampling later makes nearest-neighbour scores comparable.
    """

    generated_root = pathlib.Path(generated_root)
    swda_root = pathlib.Path(swda_root)
    paths = sorted(generated_root.glob("*/*.json"))
    if not paths:
        raise FileNotFoundError(f"No <condition>/*.json files found under {generated_root}")

    metadata = load_metadata(swda_root)
    raw_records: list[tuple[pathlib.Path, dict, str, int, list[tuple[str, str]]]] = []
    ids_by_condition: dict[str, set[int]] = defaultdict(set)
    seen: set[tuple[str, int]] = set()

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        condition = _record_condition(path, record)
        try:
            conversation_no = int(record.get("conversation_no", path.stem))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid conversation_no in {path}") from exc
        if path.stem != str(conversation_no):
            raise ValueError(
                f"Filename/ID mismatch in {path}: stem={path.stem}, conversation_no={conversation_no}"
            )
        key = (condition, conversation_no)
        if key in seen:
            raise ValueError(f"Duplicate generated conversation {key}")
        seen.add(key)
        if conversation_no not in metadata:
            raise ValueError(f"Generated ID {conversation_no} has no Switchboard metadata row")
        turns = conversation_turns(record)
        if not turns:
            raise ValueError(f"Generated conversation has no parseable turns: {path}")
        raw_records.append((path, record, condition, conversation_no, turns))
        ids_by_condition[condition].add(conversation_no)

    if expected_conditions is not None:
        observed_conditions = set(ids_by_condition)
        expected = set(expected_conditions)
        if observed_conditions != expected:
            raise ValueError(
                "Generated corpus does not contain the expected condition factorial: "
                f"missing={sorted(expected - observed_conditions)} "
                f"unexpected={sorted(observed_conditions - expected)}"
            )

    if strict_balanced:
        ordered_conditions = sorted(ids_by_condition, key=condition_sort_key)
        reference_condition = ordered_conditions[0]
        reference_ids = ids_by_condition[reference_condition]
        mismatches = []
        for condition in ordered_conditions[1:]:
            missing = sorted(reference_ids - ids_by_condition[condition])
            extra = sorted(ids_by_condition[condition] - reference_ids)
            if missing or extra:
                mismatches.append(
                    f"{condition}: missing={missing[:5]} extra={extra[:5]}"
                )
        if mismatches:
            raise ValueError(
                "Generated conditions do not share the same conversation IDs; use "
                "--allow-unbalanced only for an explicitly partial diagnostic run. "
                + "; ".join(mismatches)
            )

    generated: list[RouteConversation] = []
    generated_topic_keys: set[str] = set()
    for path, record, condition, conversation_no, turns in raw_records:
        meta = metadata[conversation_no]
        authoritative_topic = str(meta.get("topic_description") or meta.get("prompt") or "")
        topic_key = normalize_topic(authoritative_topic)
        if not topic_key:
            raise ValueError(f"Switchboard ID {conversation_no} has no usable topic")
        record_topic = normalize_topic(str(record.get("topic") or ""))
        if record_topic and record_topic != topic_key:
            raise ValueError(
                f"Topic mismatch for {condition}/{conversation_no}: "
                f"generated={record.get('topic')!r}, Switchboard={authoritative_topic!r}"
            )
        architecture = str(record.get("architecture") or condition.split("-")[0])
        prompt_level = str(record.get("prompt_level") or condition.split("-")[-1])
        generated.append(
            RouteConversation(
                source="LLM",
                condition=condition,
                architecture=architecture,
                prompt_level=prompt_level,
                conversation_no=conversation_no,
                topic=_display_topic(authoritative_topic),
                topic_key=topic_key,
                turns=tuple((str(speaker), str(text)) for speaker, text in turns),
            )
        )
        generated_topic_keys.add(topic_key)

    human: list[RouteConversation] = []
    found_human_ids: set[int] = set()
    for path in iter_conversation_files(swda_root):
        conversation_no = conversation_no_of(path)
        meta = metadata.get(conversation_no)
        if not meta:
            continue
        authoritative_topic = str(meta.get("topic_description") or meta.get("prompt") or "")
        topic_key = normalize_topic(authoritative_topic)
        if topic_key not in generated_topic_keys:
            continue
        turns = parse_conversation(path)
        if not turns:
            continue
        human.append(
            RouteConversation(
                source="SB",
                condition="SB",
                architecture="SB",
                prompt_level="SB",
                conversation_no=conversation_no,
                topic=_display_topic(authoritative_topic),
                topic_key=topic_key,
                turns=tuple((str(speaker), str(text)) for speaker, text in turns),
            )
        )
        found_human_ids.add(conversation_no)

    generated_ids = {conv.conversation_no for conv in generated}
    missing_human = sorted(generated_ids - found_human_ids)
    if missing_human:
        raise ValueError(
            f"Generated IDs have no matching parsed Switchboard transcript: {missing_human[:10]}"
        )

    human.sort(key=lambda conv: (conv.topic_key, conv.conversation_no))
    generated.sort(key=lambda conv: (condition_sort_key(conv.condition), conv.conversation_no))
    return human + generated


def lexical_words(turns: Sequence[tuple[str, str]]) -> list[str]:
    """Flatten spoken content while deliberately ignoring speaker/turn boundaries."""

    words: list[str] = []
    for _, text in turns:
        words.extend(match.group(0) for match in WORD_RE.finditer(text))
    return words


def progress_microchunks(
    turns: Sequence[tuple[str, str]],
    *,
    n_bins: int = 12,
    max_microchunk_words: int = 96,
) -> tuple[int, list[list[str]]]:
    """Split all lexical content into equal-mass progress bins.

    Every word appears exactly once.  Long bins are split into microchunks before
    embedding to reduce late-content truncation risk; the normalized microchunk vectors are
    pooled back into one vector per progress bin. The word limit is not a guarantee about
    subword-token length, so robustness settings should remain below the encoder limit.
    """

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if max_microchunk_words < 1:
        raise ValueError("max_microchunk_words must be at least 1")
    words = lexical_words(turns)
    if not words:
        return 0, []
    actual_bins = min(n_bins, len(words))
    base, remainder = divmod(len(words), actual_bins)
    bins: list[list[str]] = []
    cursor = 0
    for index in range(actual_bins):
        size = base + (1 if index < remainder else 0)
        bin_words = words[cursor: cursor + size]
        cursor += size
        chunks = [
            " ".join(bin_words[start: start + max_microchunk_words])
            for start in range(0, len(bin_words), max_microchunk_words)
        ]
        bins.append(chunks)
    assert cursor == len(words)
    return len(words), bins


def landmark_microchunks(
    turns: Sequence[tuple[str, str]],
    *,
    n_landmarks: int = 8,
    landmark_words: int = 20,
    max_microchunk_words: int = 96,
) -> tuple[int, list[list[str]], int]:
    """Sample equal-word-count landmarks across the complete conversation span.

    Each landmark contains exactly ``landmark_words`` consecutive lexical words.  Start
    positions are evenly spaced from the opening to the final possible window.  Windows
    may overlap in short calls: this keeps short architecture outputs in the analysis and
    makes their limited semantic span auditable instead of silently excluding them.

    Returns total words, landmark microchunks, and the number of distinct source-word
    positions covered by at least one landmark.
    """

    if n_landmarks < 2:
        raise ValueError("n_landmarks must be at least 2")
    if landmark_words < 1:
        raise ValueError("landmark_words must be at least 1")
    if max_microchunk_words < 1:
        raise ValueError("max_microchunk_words must be at least 1")
    words = lexical_words(turns)
    n_words = len(words)
    if n_words < landmark_words:
        return n_words, [], 0

    span = n_words - landmark_words
    starts = [round(index * span / (n_landmarks - 1)) for index in range(n_landmarks)]
    bins: list[list[str]] = []
    covered: set[int] = set()
    for start in starts:
        window = words[start: start + landmark_words]
        covered.update(range(start, start + landmark_words))
        bins.append(
            [
                " ".join(window[offset: offset + max_microchunk_words])
                for offset in range(0, len(window), max_microchunk_words)
            ]
        )
    return n_words, bins, len(covered)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_routes(
    conversations: Sequence[RouteConversation],
    embedder: TextEmbedder,
    *,
    n_bins: int = 8,
    max_microchunk_words: int = 96,
    min_words: int = 20,
    route_representation: str = "landmarks",
    landmark_words: int = 20,
) -> list[EmbeddedRoute]:
    """Build and embed routes in one batch; short conversations are explicitly flagged."""

    if route_representation not in {"landmarks", "full-coverage"}:
        raise ValueError("route_representation must be 'landmarks' or 'full-coverage'")
    preparations: list[tuple[int, list[list[str]], bool, str, int, int]] = []
    all_texts: list[str] = []
    all_text_weights: list[int] = []
    spans: list[list[tuple[int, int]]] = []
    for conversation in conversations:
        if route_representation == "landmarks":
            n_words, bins, n_unique_sampled_words = landmark_microchunks(
                conversation.turns,
                n_landmarks=n_bins,
                landmark_words=landmark_words,
                max_microchunk_words=max_microchunk_words,
            )
            technical_minimum = max(min_words, landmark_words)
        else:
            n_words, bins = progress_microchunks(
                conversation.turns,
                n_bins=n_bins,
                max_microchunk_words=max_microchunk_words,
            )
            n_unique_sampled_words = n_words
            technical_minimum = min_words
        n_sampled_word_occurrences = sum(
            len(chunk.split()) for progress_bin in bins for chunk in progress_bin
        )
        eligible = n_words >= technical_minimum and len(bins) >= 2
        reason = "" if eligible else f"too_short:{n_words}<{technical_minimum}"
        preparations.append(
            (
                n_words,
                bins,
                eligible,
                reason,
                n_sampled_word_occurrences,
                n_unique_sampled_words,
            )
        )
        route_spans: list[tuple[int, int]] = []
        if eligible:
            for microchunks in bins:
                start = len(all_texts)
                all_texts.extend(microchunks)
                all_text_weights.extend(len(chunk.split()) for chunk in microchunks)
                route_spans.append((start, len(all_texts)))
        spans.append(route_spans)

    if not all_texts:
        raise ValueError("No conversations are long enough to construct semantic routes")
    micro_vectors = _l2_normalize(embedder.encode(all_texts))
    if len(micro_vectors) != len(all_texts):
        raise ValueError(
            f"Embedder returned {len(micro_vectors)} rows for {len(all_texts)} texts"
        )
    if not np.isfinite(micro_vectors).all():
        raise ValueError("Embedding matrix contains non-finite values")

    routes: list[EmbeddedRoute] = []
    for conversation, preparation, route_spans in zip(
        conversations, preparations, spans
    ):
        (
            n_words,
            bins,
            eligible,
            reason,
            n_sampled_word_occurrences,
            n_unique_sampled_words,
        ) = preparation
        if eligible:
            pooled = []
            for start, end in route_spans:
                pooled.append(
                    np.average(
                        micro_vectors[start:end],
                        axis=0,
                        weights=np.asarray(all_text_weights[start:end], dtype=float),
                    )
                )
            vectors = _l2_normalize(np.asarray(pooled, dtype=float))
        else:
            vectors = np.empty((0, micro_vectors.shape[1]), dtype=float)
        routes.append(
            EmbeddedRoute(
                conversation=conversation,
                vectors=vectors,
                n_words=n_words,
                n_bins=len(bins),
                n_microchunks=sum(len(chunks) for chunks in bins),
                eligible=eligible,
                exclusion_reason=reason,
                route_representation=route_representation,
                n_sampled_word_occurrences=n_sampled_word_occurrences,
                n_unique_sampled_words=n_unique_sampled_words,
            )
        )
    return routes


def _within_band(i: int, j: int, n: int, m: int, band: int) -> bool:
    if band < 0:
        return True
    if n <= 1 or m <= 1:
        return True
    normalized_gap = abs(i / (n - 1) - j / (m - 1))
    return normalized_gap <= band / max(n - 1, m - 1) + 1e-12


def constrained_dtw_similarity(
    left: np.ndarray,
    right: np.ndarray,
    *,
    band: int = 1,
    warp_penalty: float = 0.05,
    max_warp_run: int = 2,
) -> tuple[float, list[tuple[int, int]]]:
    """Cosine DTW with fixed endpoints, a narrow band, and bounded warping."""

    left = _l2_normalize(left)
    right = _l2_normalize(right)
    n, m = len(left), len(right)
    if n == 0 or m == 0:
        raise ValueError("Cannot align an empty route")
    if max_warp_run < 1:
        raise ValueError("max_warp_run must be at least 1")
    if warp_penalty < 0:
        raise ValueError("warp_penalty must be non-negative")
    local_cost = 1.0 - np.clip(left @ right.T, -1.0, 1.0)

    # state: 0 diagonal/start, 1 vertical (advance left), 2 horizontal (advance right)
    start = (0, 0, 0, 0)
    costs: dict[tuple[int, int, int, int], float] = {start: float(local_cost[0, 0])}
    lengths: dict[tuple[int, int, int, int], int] = {start: 1}
    previous: dict[tuple[int, int, int, int], tuple[int, int, int, int]] = {}

    def update(
        current: tuple[int, int, int, int],
        nxt: tuple[int, int, int, int],
        added: float,
    ) -> None:
        candidate = costs[current] + added
        candidate_length = lengths[current] + 1
        incumbent = costs.get(nxt, math.inf)
        incumbent_length = lengths.get(nxt, 10**9)
        if candidate < incumbent - 1e-12 or (
            abs(candidate - incumbent) <= 1e-12 and candidate_length < incumbent_length
        ):
            costs[nxt] = candidate
            lengths[nxt] = candidate_length
            previous[nxt] = current

    for i in range(n):
        for j in range(m):
            cell_states = [state for state in list(costs) if state[0] == i and state[1] == j]
            for state in cell_states:
                _, _, last_move, run = state
                if i + 1 < n and j + 1 < m and _within_band(i + 1, j + 1, n, m, band):
                    update(state, (i + 1, j + 1, 0, 0), float(local_cost[i + 1, j + 1]))
                if i + 1 < n and _within_band(i + 1, j, n, m, band):
                    next_run = run + 1 if last_move == 1 else 1
                    if next_run <= max_warp_run:
                        update(
                            state,
                            (i + 1, j, 1, next_run),
                            float(local_cost[i + 1, j]) + warp_penalty,
                        )
                if j + 1 < m and _within_band(i, j + 1, n, m, band):
                    next_run = run + 1 if last_move == 2 else 1
                    if next_run <= max_warp_run:
                        update(
                            state,
                            (i, j + 1, 2, next_run),
                            float(local_cost[i, j + 1]) + warp_penalty,
                        )

    end_states = [state for state in costs if state[0] == n - 1 and state[1] == m - 1]
    if not end_states:
        raise ValueError(
            f"No DTW path inside band={band}; increase the band for route lengths {n} and {m}"
        )
    end = min(end_states, key=lambda state: (costs[state], lengths[state]))
    path: list[tuple[int, int]] = []
    cursor = end
    while True:
        path.append((cursor[0], cursor[1]))
        if cursor == start:
            break
        cursor = previous[cursor]
    path.reverse()
    # The path minimizes cumulative regularized cost.  Normalize by a fixed route
    # resolution rather than the selected path length; otherwise adding warp steps can
    # improve the reported score merely by making its denominator larger.
    similarity = 1.0 - costs[end] / max(n, m)
    return float(similarity), path


def _route_groups(routes: Sequence[EmbeddedRoute]) -> dict[tuple[str, str], list[EmbeddedRoute]]:
    groups: dict[tuple[str, str], list[EmbeddedRoute]] = defaultdict(list)
    for route in routes:
        if route.eligible:
            groups[(route.conversation.condition, route.conversation.topic_key)].append(route)
    return groups


def pairwise_route_rows(
    routes: Sequence[EmbeddedRoute],
    *,
    band: int = 1,
    warp_penalty: float = 0.05,
    max_warp_run: int = 2,
) -> list[dict]:
    rows: list[dict] = []
    for (condition, _topic_key), group in sorted(
        _route_groups(routes).items(), key=lambda item: (condition_sort_key(item[0][0]), item[0][1])
    ):
        if len(group) < 2:
            continue
        group.sort(key=lambda route: route.conversation.conversation_no)
        head = group[0].conversation
        for left, right in combinations(group, 2):
            similarity, path = constrained_dtw_similarity(
                left.vectors,
                right.vectors,
                band=band,
                warp_penalty=warp_penalty,
                max_warp_run=max_warp_run,
            )
            rows.append(
                {
                    "source": head.source,
                    "condition": condition,
                    "architecture": head.architecture,
                    "prompt_level": head.prompt_level,
                    "topic": head.topic,
                    "topic_key": head.topic_key,
                    "conversation_a": left.conversation.conversation_no,
                    "conversation_b": right.conversation.conversation_no,
                    "route_similarity": similarity,
                    "route_distance": 1.0 - similarity,
                    "alignment_path": json.dumps(path, separators=(",", ":")),
                }
            )
    return rows


def nearest_sibling_rows(
    routes: Sequence[EmbeddedRoute], pair_rows: Sequence[dict]
) -> list[dict]:
    best: dict[tuple[str, int], tuple[float, int]] = {}
    group_sizes = {
        key: len(group)
        for key, group in _route_groups(routes).items()
    }
    for row in pair_rows:
        condition = str(row["condition"])
        left = int(row["conversation_a"])
        right = int(row["conversation_b"])
        similarity = float(row["route_similarity"])
        for current, sibling in ((left, right), (right, left)):
            key = (condition, current)
            incumbent = best.get(key)
            if incumbent is None or similarity > incumbent[0] + 1e-12 or (
                abs(similarity - incumbent[0]) <= 1e-12 and sibling < incumbent[1]
            ):
                best[key] = similarity, sibling

    rows: list[dict] = []
    for route in sorted(
        routes,
        key=lambda item: (condition_sort_key(item.conversation.condition), item.conversation.conversation_no),
    ):
        conv = route.conversation
        group_size = group_sizes.get((conv.condition, conv.topic_key), 0)
        sibling = best.get(conv.key)
        if not route.eligible:
            status = route.exclusion_reason
        elif group_size < 2:
            status = "singleton_topic"
        else:
            status = "eligible"
        rows.append(
            {
                "source": conv.source,
                "condition": conv.condition,
                "architecture": conv.architecture,
                "prompt_level": conv.prompt_level,
                "conversation_no": conv.conversation_no,
                "topic": conv.topic,
                "topic_key": conv.topic_key,
                "n_words": route.n_words,
                "n_bins": route.n_bins,
                "n_microchunks": route.n_microchunks,
                "route_representation": route.route_representation,
                "n_sampled_word_occurrences": route.n_sampled_word_occurrences,
                "n_unique_sampled_words": route.n_unique_sampled_words,
                "sampled_unique_coverage": (
                    route.n_unique_sampled_words / route.n_words if route.n_words else 0.0
                ),
                "sampled_overlap_rate": (
                    1.0 - route.n_unique_sampled_words / route.n_sampled_word_occurrences
                    if route.n_sampled_word_occurrences
                    else 0.0
                ),
                "status": status,
                "topic_pool_size": group_size,
                "nearest_sibling_no": sibling[1] if sibling else "",
                "nearest_sibling_similarity": sibling[0] if sibling else "",
            }
        )
    return rows


def topic_condition_rows(
    nearest_rows: Sequence[dict], pair_rows: Sequence[dict]
) -> list[dict]:
    nearest_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    pair_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in nearest_rows:
        if row["status"] == "eligible":
            nearest_groups[(str(row["condition"]), str(row["topic_key"]))].append(row)
    for row in pair_rows:
        pair_groups[(str(row["condition"]), str(row["topic_key"]))].append(row)

    rows: list[dict] = []
    for key, group in sorted(
        nearest_groups.items(), key=lambda item: (condition_sort_key(item[0][0]), item[0][1])
    ):
        if len(group) < 2:
            continue
        condition, topic_key = key
        similarities = [float(row["nearest_sibling_similarity"]) for row in group]
        pair_similarities = [
            float(row["route_similarity"]) for row in pair_groups.get(key, [])
        ]
        head = group[0]
        rows.append(
            {
                "source": head["source"],
                "condition": condition,
                "architecture": head["architecture"],
                "prompt_level": head["prompt_level"],
                "topic": head["topic"],
                "topic_key": topic_key,
                "n_conversations": len(group),
                "mean_route_reuse": statistics.fmean(similarities),
                "median_route_reuse": statistics.median(similarities),
                "mean_pairwise_similarity": statistics.fmean(pair_similarities),
                "max_pairwise_similarity": max(pair_similarities),
            }
        )
    return rows


def _stable_rng(seed: int, *parts: object) -> random.Random:
    material = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _similarity_matrix(
    group: Sequence[EmbeddedRoute], pair_lookup: dict[tuple[str, str, int, int], float]
) -> np.ndarray:
    size = len(group)
    matrix = np.full((size, size), -np.inf, dtype=float)
    condition = group[0].conversation.condition
    topic_key = group[0].conversation.topic_key
    for i, left in enumerate(group):
        for j in range(i + 1, size):
            right = group[j]
            a, b = sorted((left.conversation.conversation_no, right.conversation.conversation_no))
            value = pair_lookup[(condition, topic_key, a, b)]
            matrix[i, j] = matrix[j, i] = value
    return matrix


def calibrated_topic_rows(
    routes: Sequence[EmbeddedRoute],
    pair_rows: Sequence[dict],
    topic_rows: Sequence[dict],
    *,
    human_draws: int = 2000,
    seed: int = 20260714,
    primary_min_topic_size: int = 3,
) -> tuple[list[dict], dict[tuple[str, int], list[float]]]:
    """Calibrate generated nearest-sibling reuse to an equally sized human pool."""

    if human_draws < 1:
        raise ValueError("human_draws must be at least 1")
    groups = _route_groups(routes)
    pair_lookup: dict[tuple[str, str, int, int], float] = {}
    for row in pair_rows:
        a, b = sorted((int(row["conversation_a"]), int(row["conversation_b"])))
        pair_lookup[(str(row["condition"]), str(row["topic_key"]), a, b)] = float(
            row["route_similarity"]
        )
    topic_lookup = {
        (str(row["condition"]), str(row["topic_key"])): row for row in topic_rows
    }

    rows: list[dict] = []
    reference_draws: dict[tuple[str, int], list[float]] = {}
    human_matrix_cache: dict[str, np.ndarray] = {}
    human_reference_cache: dict[tuple[str, int], tuple[list[float], list[float]]] = {}
    for (condition, topic_key), generated_group in sorted(
        groups.items(), key=lambda item: (condition_sort_key(item[0][0]), item[0][1])
    ):
        if condition == "SB" or len(generated_group) < 2:
            continue
        human_group = sorted(
            groups.get(("SB", topic_key), []),
            key=lambda route: route.conversation.conversation_no,
        )
        pool_size = len(generated_group)
        if len(human_group) < pool_size:
            raise ValueError(
                f"Human topic pool {topic_key!r} has {len(human_group)} conversations, "
                f"fewer than {condition}'s {pool_size}"
            )
        if topic_key not in human_matrix_cache:
            human_matrix_cache[topic_key] = _similarity_matrix(human_group, pair_lookup)
        reference_key = (topic_key, pool_size)
        if reference_key not in human_reference_cache:
            human_matrix = human_matrix_cache[topic_key]
            rng = _stable_rng(seed, "human-reference", topic_key, pool_size)
            draw_scores: list[float] = []
            individual_scores: list[float] = []
            human_indices = list(range(len(human_group)))
            for _ in range(human_draws):
                chosen = rng.sample(human_indices, pool_size)
                subset = human_matrix[np.ix_(chosen, chosen)]
                nearest = subset.max(axis=1)
                draw_scores.append(float(nearest.mean()))
                individual_scores.extend(float(value) for value in nearest)
            human_reference_cache[reference_key] = (draw_scores, individual_scores)
        draw_scores, individual_scores = human_reference_cache[reference_key]
        reference_draws[reference_key] = draw_scores

        generated_topic = topic_lookup[(condition, topic_key)]
        observed = float(generated_topic["mean_route_reuse"])
        human_mean = statistics.fmean(draw_scores)
        human_p95 = _quantile(individual_scores, 0.95)
        generated_nearest = []
        generated_matrix = _similarity_matrix(generated_group, pair_lookup)
        generated_nearest.extend(float(value) for value in generated_matrix.max(axis=1))
        upper_tail_fraction = (1 + sum(score >= observed for score in draw_scores)) / (
            human_draws + 1
        )
        rows.append(
            {
                "condition": condition,
                "architecture": generated_topic["architecture"],
                "prompt_level": generated_topic["prompt_level"],
                "topic": generated_topic["topic"],
                "topic_key": topic_key,
                "n_generated": pool_size,
                "n_human_pool": len(human_group),
                "primary_eligible": pool_size >= primary_min_topic_size,
                "generated_route_reuse": observed,
                "human_pool_matched_reuse": human_mean,
                "human_subsample_mean_q025": _quantile(draw_scores, 0.025),
                "human_subsample_mean_q975": _quantile(draw_scores, 0.975),
                "excess_route_reuse": observed - human_mean,
                "human_subsample_nearest_p95": human_p95,
                "generated_above_human_nearest_p95_rate": statistics.fmean(
                    [float(score > human_p95) for score in generated_nearest]
                ),
                "human_subsample_upper_tail_fraction": upper_tail_fraction,
            }
        )
    return rows, reference_draws


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    draws: int,
    rng: random.Random,
) -> tuple[float, float]:
    values = [float(value) for value in values]
    if not values or draws < 1:
        return float("nan"), float("nan")
    sampled_means = [
        statistics.fmean(rng.choice(values) for _ in range(len(values)))
        for _ in range(draws)
    ]
    return _quantile(sampled_means, 0.025), _quantile(sampled_means, 0.975)


def _common_topics(
    calibrated_rows: Sequence[dict],
    *,
    primary: bool,
    conditions: Collection[str] | None = None,
) -> set[str]:
    available = {str(row["condition"]) for row in calibrated_rows}
    selected = available if conditions is None else available.intersection(conditions)
    selected_conditions = sorted(selected, key=condition_sort_key)
    topic_sets = []
    for condition in selected_conditions:
        topic_sets.append(
            {
                str(row["topic_key"])
                for row in calibrated_rows
                if row["condition"] == condition
                and (bool(row["primary_eligible"]) or not primary)
            }
        )
    return set.intersection(*topic_sets) if topic_sets else set()


def _analysis_families(calibrated_rows: Sequence[dict]) -> dict[str, set[str]]:
    conditions = {str(row["condition"]) for row in calibrated_rows}
    return {
        "primary": {condition for condition in conditions if not condition.endswith("-P2")},
        "exploratory_p2": {condition for condition in conditions if condition.endswith("-P2")},
    }


def condition_summary_rows(
    calibrated_rows: Sequence[dict],
    routes: Sequence[EmbeddedRoute],
    *,
    bootstrap_draws: int = 2000,
    seed: int = 20260714,
) -> list[dict]:
    families = _analysis_families(calibrated_rows)
    family_topics = {
        family: {
            "primary": _common_topics(
                calibrated_rows, primary=True, conditions=family_conditions
            ),
            "sensitivity": _common_topics(
                calibrated_rows, primary=False, conditions=family_conditions
            ),
        }
        for family, family_conditions in families.items()
        if family_conditions
    }
    conditions = sorted({str(row["condition"]) for row in calibrated_rows}, key=condition_sort_key)
    route_counts: dict[str, tuple[int, int]] = {}
    for condition in conditions:
        condition_routes = [route for route in routes if route.conversation.condition == condition]
        route_counts[condition] = (
            sum(route.eligible for route in condition_routes),
            len(condition_routes),
        )

    rows: list[dict] = []
    for condition in conditions:
        family = "exploratory_p2" if condition.endswith("-P2") else "primary"
        primary_topics = family_topics[family]["primary"]
        sensitivity_topics = family_topics[family]["sensitivity"]
        all_rows = [row for row in calibrated_rows if row["condition"] == condition]
        primary_rows = [row for row in all_rows if row["topic_key"] in primary_topics]
        sensitivity_rows = [row for row in all_rows if row["topic_key"] in sensitivity_topics]
        if not primary_rows:
            continue
        primary_excess = [float(row["excess_route_reuse"]) for row in primary_rows]
        sensitivity_excess = [float(row["excess_route_reuse"]) for row in sensitivity_rows]
        ci_low, ci_high = bootstrap_mean_ci(
            primary_excess,
            draws=bootstrap_draws,
            rng=_stable_rng(seed, "condition", condition, "primary"),
        )
        sensitivity_low, sensitivity_high = bootstrap_mean_ci(
            sensitivity_excess,
            draws=bootstrap_draws,
            rng=_stable_rng(seed, "condition", condition, "sensitivity"),
        )
        eligible_count, total_count = route_counts[condition]
        head = primary_rows[0]
        rows.append(
            {
                "condition": condition,
                "architecture": head["architecture"],
                "prompt_level": head["prompt_level"],
                "analysis_role": "exploratory" if head["prompt_level"] == "P2" else "primary",
                "n_primary_topics": len(primary_rows),
                "n_sensitivity_topics": len(sensitivity_rows),
                "n_eligible_conversations": eligible_count,
                "n_total_conversations": total_count,
                "route_coverage": eligible_count / total_count if total_count else float("nan"),
                "mean_generated_route_reuse": statistics.fmean(
                    float(row["generated_route_reuse"]) for row in primary_rows
                ),
                "mean_pool_matched_human_reuse": statistics.fmean(
                    float(row["human_pool_matched_reuse"]) for row in primary_rows
                ),
                "mean_excess_route_reuse": statistics.fmean(primary_excess),
                "excess_ci_low": ci_low,
                "excess_ci_high": ci_high,
                "mean_generated_above_human_nearest_p95_rate": statistics.fmean(
                    float(row["generated_above_human_nearest_p95_rate"])
                    for row in primary_rows
                ),
                "sensitivity_excess_route_reuse_n_ge_2": statistics.fmean(sensitivity_excess),
                "sensitivity_ci_low": sensitivity_low,
                "sensitivity_ci_high": sensitivity_high,
            }
        )
    return rows


def contrast_rows(
    calibrated_rows: Sequence[dict],
    *,
    bootstrap_draws: int = 2000,
    seed: int = 20260714,
) -> list[dict]:
    lookup = {
        (str(row["condition"]), str(row["topic_key"])): float(row["excess_route_reuse"])
        for row in calibrated_rows
    }
    conditions = {str(row["condition"]) for row in calibrated_rows}
    specifications: list[tuple[str, str, str]] = []
    for prompt in ("P0", "P1", "P2"):
        for left_arch, right_arch in (("C1", "C2"), ("C2", "C3"), ("C3", "C4")):
            specifications.append(
                ("architecture", f"{left_arch}-{prompt}", f"{right_arch}-{prompt}")
            )
    for architecture in ("C1", "C2", "C3", "C4"):
        specifications.extend(
            [
                ("prompt", f"{architecture}-P0", f"{architecture}-P1"),
                ("prompt", f"{architecture}-P1", f"{architecture}-P2"),
            ]
        )

    rows: list[dict] = []
    for factor, left, right in specifications:
        if left not in conditions or right not in conditions:
            continue
        common = _common_topics(
            calibrated_rows, primary=True, conditions={left, right}
        )
        if not common:
            continue
        deltas = [lookup[(right, topic)] - lookup[(left, topic)] for topic in sorted(common)]
        ci_low, ci_high = bootstrap_mean_ci(
            deltas,
            draws=bootstrap_draws,
            rng=_stable_rng(seed, "contrast", left, right),
        )
        rows.append(
            {
                "factor": factor,
                "left_condition": left,
                "right_condition": right,
                "n_topics": len(deltas),
                "delta_excess_reuse_right_minus_left": statistics.fmean(deltas),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "interpretation": "negative means the right condition is less route-repetitive",
                "analysis_role": "exploratory" if "P2" in {left.split("-")[-1], right.split("-")[-1]} else "primary",
            }
        )
    return rows


def architecture_displacement_rows(
    routes: Sequence[EmbeddedRoute],
    *,
    band: int = 1,
    warp_penalty: float = 0.05,
    max_warp_run: int = 2,
    primary_min_topic_size: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Compare architectures on the same generated input, within prompt and topic.

    Matching ``conversation_no`` means the source ID, topic, personas, prompt level, and
    input record are held fixed; architecture-specific templates still differ. The
    same-input similarity is contextualized by all
    different-input cross-architecture pairs from that topic.  With one stochastic output
    per input this remains exploratory: it cannot isolate architecture from sampling noise.
    """

    groups = _route_groups(routes)
    conditions = {
        condition for condition, _topic in groups if condition != "SB"
    }
    matched_rows: list[dict] = []
    topic_rows: list[dict] = []
    for prompt in ("P0", "P1", "P2"):
        prompt_conditions = sorted(
            [condition for condition in conditions if condition.endswith(f"-{prompt}")],
            key=condition_sort_key,
        )
        for left_condition, right_condition in combinations(prompt_conditions, 2):
            left_architecture = left_condition.split("-")[0]
            right_architecture = right_condition.split("-")[0]
            left_topics = {
                topic for condition, topic in groups if condition == left_condition
            }
            right_topics = {
                topic for condition, topic in groups if condition == right_condition
            }
            for topic_key in sorted(left_topics.intersection(right_topics)):
                left_group = sorted(
                    groups[(left_condition, topic_key)],
                    key=lambda route: route.conversation.conversation_no,
                )
                right_group = sorted(
                    groups[(right_condition, topic_key)],
                    key=lambda route: route.conversation.conversation_no,
                )
                matched_scores: list[float] = []
                unmatched_scores: list[float] = []
                display_topic = left_group[0].conversation.topic
                for left_route, right_route in product(left_group, right_group):
                    similarity, path = constrained_dtw_similarity(
                        left_route.vectors,
                        right_route.vectors,
                        band=band,
                        warp_penalty=warp_penalty,
                        max_warp_run=max_warp_run,
                    )
                    left_id = left_route.conversation.conversation_no
                    right_id = right_route.conversation.conversation_no
                    if left_id == right_id:
                        matched_scores.append(similarity)
                        matched_rows.append(
                            {
                                "prompt_level": prompt,
                                "left_condition": left_condition,
                                "right_condition": right_condition,
                                "left_architecture": left_architecture,
                                "right_architecture": right_architecture,
                                "conversation_no": left_id,
                                "topic": display_topic,
                                "topic_key": topic_key,
                                "same_input_route_similarity": similarity,
                                "architecture_route_displacement": 1.0 - similarity,
                                "alignment_path": json.dumps(path, separators=(",", ":")),
                                "analysis_role": "exploratory",
                            }
                        )
                    else:
                        unmatched_scores.append(similarity)
                if not matched_scores or not unmatched_scores:
                    continue
                matched_mean = statistics.fmean(matched_scores)
                unmatched_mean = statistics.fmean(unmatched_scores)
                topic_rows.append(
                    {
                        "prompt_level": prompt,
                        "left_condition": left_condition,
                        "right_condition": right_condition,
                        "left_architecture": left_architecture,
                        "right_architecture": right_architecture,
                        "topic": display_topic,
                        "topic_key": topic_key,
                        "n_left": len(left_group),
                        "n_right": len(right_group),
                        "n_matched_inputs": len(matched_scores),
                        "n_unmatched_pairs": len(unmatched_scores),
                        "primary_eligible": len(matched_scores) >= primary_min_topic_size,
                        "mean_same_input_route_similarity": matched_mean,
                        "mean_different_input_route_similarity": unmatched_mean,
                        "architecture_route_displacement": 1.0 - matched_mean,
                        "same_input_similarity_advantage": matched_mean - unmatched_mean,
                        "analysis_role": "exploratory",
                        "c4_also_changes_second_model": (
                            left_architecture == "C4" or right_architecture == "C4"
                        ),
                    }
                )
    return matched_rows, topic_rows


def architecture_displacement_summary_rows(
    topic_rows: Sequence[dict],
    *,
    bootstrap_draws: int = 2000,
    seed: int = 20260714,
) -> list[dict]:
    """Macro-average matched architecture displacement over eligible topics."""

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in topic_rows:
        if row["primary_eligible"]:
            groups[
                (
                    str(row["prompt_level"]),
                    str(row["left_condition"]),
                    str(row["right_condition"]),
                )
            ].append(row)

    summaries: list[dict] = []
    for (prompt, left, right), rows in sorted(groups.items()):
        displacement = [float(row["architecture_route_displacement"]) for row in rows]
        advantage = [float(row["same_input_similarity_advantage"]) for row in rows]
        displacement_ci = bootstrap_mean_ci(
            displacement,
            draws=bootstrap_draws,
            rng=_stable_rng(seed, "architecture-displacement", left, right),
        )
        advantage_ci = bootstrap_mean_ci(
            advantage,
            draws=bootstrap_draws,
            rng=_stable_rng(seed, "same-input-advantage", left, right),
        )
        head = rows[0]
        summaries.append(
            {
                "prompt_level": prompt,
                "left_condition": left,
                "right_condition": right,
                "left_architecture": head["left_architecture"],
                "right_architecture": head["right_architecture"],
                "n_topics": len(rows),
                "mean_same_input_route_similarity": statistics.fmean(
                    float(row["mean_same_input_route_similarity"]) for row in rows
                ),
                "mean_different_input_route_similarity": statistics.fmean(
                    float(row["mean_different_input_route_similarity"]) for row in rows
                ),
                "mean_architecture_route_displacement": statistics.fmean(displacement),
                "displacement_ci_low": displacement_ci[0],
                "displacement_ci_high": displacement_ci[1],
                "mean_same_input_similarity_advantage": statistics.fmean(advantage),
                "advantage_ci_low": advantage_ci[0],
                "advantage_ci_high": advantage_ci[1],
                "analysis_role": "exploratory",
                "c4_also_changes_second_model": head["c4_also_changes_second_model"],
            }
        )
    return summaries


def resample_route(vectors: np.ndarray, n_points: int) -> np.ndarray:
    vectors = _l2_normalize(vectors)
    if n_points < 1:
        raise ValueError("n_points must be at least 1")
    if len(vectors) == n_points:
        return vectors
    if len(vectors) == 1:
        return np.repeat(vectors, n_points, axis=0)
    old_positions = (np.arange(len(vectors), dtype=float) + 0.5) / len(vectors)
    new_positions = (np.arange(n_points, dtype=float) + 0.5) / n_points
    out = []
    for position in new_positions:
        upper = int(np.searchsorted(old_positions, position, side="left"))
        if upper <= 0:
            out.append(vectors[0])
            continue
        if upper >= len(vectors):
            out.append(vectors[-1])
            continue
        lower = upper - 1
        width = old_positions[upper] - old_positions[lower]
        alpha = (position - old_positions[lower]) / width if width else 0.0
        out.append((1.0 - alpha) * vectors[lower] + alpha * vectors[upper])
    return _l2_normalize(np.asarray(out, dtype=float))


def progress_summary_rows(
    routes: Sequence[EmbeddedRoute],
    calibrated_rows: Sequence[dict],
    *,
    n_points: int = 8,
    bootstrap_draws: int = 2000,
    seed: int = 20260714,
) -> list[dict]:
    families = _analysis_families(calibrated_rows)
    family_topics = {
        family: _common_topics(
            calibrated_rows, primary=True, conditions=family_conditions
        )
        for family, family_conditions in families.items()
        if family_conditions
    }
    reportable_topics = set().union(*family_topics.values()) if family_topics else set()
    groups = _route_groups(routes)
    topic_progress: dict[tuple[str, str], np.ndarray] = {}
    for key, group in groups.items():
        condition, topic_key = key
        if topic_key not in reportable_topics or len(group) < 2:
            continue
        sampled = [resample_route(route.vectors, n_points) for route in group]
        pair_values = [
            np.sum(left * right, axis=1)
            for left, right in combinations(sampled, 2)
        ]
        topic_progress[(condition, topic_key)] = np.mean(pair_values, axis=0)

    conditions = sorted(
        {str(row["condition"]) for row in calibrated_rows}, key=condition_sort_key
    )
    rows: list[dict] = []
    for condition in conditions:
        family = "exploratory_p2" if condition.endswith("-P2") else "primary"
        common = family_topics[family]
        if not common:
            continue
        for point in range(n_points):
            llm_values = []
            human_values = []
            for topic in sorted(common):
                llm_values.append(float(topic_progress[(condition, topic)][point]))
                human_values.append(float(topic_progress[("SB", topic)][point]))
            deltas = [llm - human for llm, human in zip(llm_values, human_values)]
            ci_low, ci_high = bootstrap_mean_ci(
                deltas,
                draws=bootstrap_draws,
                rng=_stable_rng(seed, "progress", condition, point),
            )
            rows.append(
                {
                    "condition": condition,
                    "architecture": condition.split("-")[0],
                    "prompt_level": condition.split("-")[-1],
                    "analysis_role": "exploratory" if condition.endswith("-P2") else "primary",
                    "progress_index": point,
                    "progress_fraction": (point + 0.5) / n_points,
                    "n_topics": len(common),
                    "mean_generated_pair_similarity": statistics.fmean(llm_values),
                    "mean_human_pair_similarity": statistics.fmean(human_values),
                    "excess_progress_similarity": statistics.fmean(deltas),
                    "excess_ci_low": ci_low,
                    "excess_ci_high": ci_high,
                }
            )
    return rows


def write_csv(path: pathlib.Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if path.exists():
            path.unlink()
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_condition_summary(rows: Sequence[dict], out_path: pathlib.Path) -> pathlib.Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = sorted(rows, key=lambda row: condition_sort_key(str(row["condition"])))
    x = np.arange(len(rows))
    means = np.array([float(row["mean_excess_route_reuse"]) for row in rows])
    lower = means - np.array([float(row["excess_ci_low"]) for row in rows])
    upper = np.array([float(row["excess_ci_high"]) for row in rows]) - means
    colors = {"C1": "#4C78A8", "C2": "#F58518", "C3": "#54A24B", "C4": "#E45756"}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    p2_positions = [index for index, row in enumerate(rows) if row["prompt_level"] == "P2"]
    if p2_positions:
        ax.axvspan(
            min(p2_positions) - 0.45,
            max(p2_positions) + 0.45,
            color="#F2F2F2",
            zorder=-2,
        )
        ax.text(
            statistics.fmean(p2_positions),
            0.98,
            "P2 exploratory",
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            color="#666666",
            fontsize=9,
        )
    for index, row in enumerate(rows):
        architecture = str(row["architecture"])
        ax.errorbar(
            index,
            means[index],
            yerr=[[lower[index]], [upper[index]]],
            fmt="o",
            markersize=7,
            capsize=3,
            color=colors.get(architecture, "#666666"),
        )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x, [str(row["condition"]) for row in rows], rotation=45, ha="right")
    ax.set_ylabel("Excess semantic-route reuse vs humans")
    ax.set_title("Within-topic nearest-sibling route reuse (positive = more formulaic)")
    ax.grid(axis="y", alpha=0.2)
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color, label=architecture)
        for architecture, color in colors.items()
    ]
    legend_handles.append(
        Line2D(
            [0], [0], color="black", linestyle="--", label="pool-matched human reference"
        )
    )
    ax.legend(handles=legend_handles, loc="best", ncols=3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_branching(rows: Sequence[dict], out_path: pathlib.Path) -> pathlib.Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"C1": "#4C78A8", "C2": "#F58518", "C3": "#54A24B", "C4": "#E45756"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for axis, prompt in zip(axes, ("P0", "P1", "P2")):
        prompt_rows = [row for row in rows if row["prompt_level"] == prompt]
        for architecture in ("C1", "C2", "C3", "C4"):
            condition = f"{architecture}-{prompt}"
            series = sorted(
                [row for row in prompt_rows if row["condition"] == condition],
                key=lambda row: int(row["progress_index"]),
            )
            if not series:
                continue
            x = np.array([float(row["progress_fraction"]) for row in series])
            y = np.array([float(row["excess_progress_similarity"]) for row in series])
            lo = np.array([float(row["excess_ci_low"]) for row in series])
            hi = np.array([float(row["excess_ci_high"]) for row in series])
            axis.plot(x, y, marker="o", markersize=3, label=architecture, color=colors[architecture])
            axis.fill_between(x, lo, hi, color=colors[architecture], alpha=0.10)
        axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
        axis.set_title(f"{prompt}{' (exploratory)' if prompt == 'P2' else ''}")
        axis.set_xlabel("Ordered route landmark (display position)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Excess same-topic similarity vs humans")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Semantic branching deficit (positive = LLM routes branch less)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def generated_fingerprint(root: pathlib.Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(pathlib.Path(root).glob("*/*.json")):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_run_metadata(
    path: pathlib.Path,
    *,
    args: argparse.Namespace,
    embedder_name: str,
    conversations: Sequence[RouteConversation],
    routes: Sequence[EmbeddedRoute],
    summary_rows: Sequence[dict],
) -> None:
    metadata = {
        "analysis": "Semantic Route Reuse",
        "analysis_label": args.analysis_label,
        "status": args.analysis_status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generated_root": str(pathlib.Path(args.generated_root).resolve()),
        "generated_data_sha256": generated_fingerprint(pathlib.Path(args.generated_root)),
        "swda_root": str(pathlib.Path(args.swda_root).resolve()),
        "embedding_backend": embedder_name,
        "parameters": {
            "progress_bins": args.progress_bins,
            "route_representation": args.route_representation,
            "landmark_words": args.landmark_words,
            "max_microchunk_words": args.max_microchunk_words,
            "min_words": args.min_words,
            "dtw_band": args.dtw_band,
            "warp_penalty": args.warp_penalty,
            "max_warp_run": args.max_warp_run,
            "primary_min_topic_size": args.primary_min_topic_size,
            "human_draws": args.human_draws,
            "bootstrap_draws": args.bootstrap_draws,
            "seed": args.seed,
        },
        "counts": {
            "human_conversations_loaded": sum(conv.source == "SB" for conv in conversations),
            "generated_conversations_loaded": sum(conv.source == "LLM" for conv in conversations),
            "eligible_human_routes": sum(
                route.eligible and route.conversation.source == "SB" for route in routes
            ),
            "eligible_generated_routes": sum(
                route.eligible and route.conversation.source == "LLM" for route in routes
            ),
            "conditions_reported": len(summary_rows),
        },
        "interpretation": {
            "mean_excess_route_reuse": "positive means more nearest-sibling route reuse than equally sized human groups",
            "human_subsample_quantiles": "reference-distribution quantiles, not confidence intervals",
            "architecture_route_displacement": "exploratory because current data have one stochastic output per input",
            "P2": "exploratory because the prompt contains a real Switchboard style example",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def print_summary(
    rows: Sequence[dict], embedder_name: str, *, analysis_label: str, analysis_status: str
) -> None:
    print("=" * 102)
    print(f"SEMANTIC ROUTE REUSE — {analysis_status.upper()}: {analysis_label}")
    print(f"embedding backend: {embedder_name}")
    print("positive excess = more same-topic route reuse than pool-size-matched humans")
    print("=" * 102)
    print(
        f"{'condition':10} {'topics':>6} {'coverage':>9} {'reuse':>8} "
        f"{'human':>8} {'excess':>8} {'CI':>19} {'>human95':>9}"
    )
    print("-" * 102)
    for row in sorted(rows, key=lambda item: condition_sort_key(str(item["condition"]))):
        print(
            f"{str(row['condition']):10} {int(row['n_primary_topics']):6d} "
            f"{float(row['route_coverage']):9.2%} "
            f"{float(row['mean_generated_route_reuse']):8.3f} "
            f"{float(row['mean_pool_matched_human_reuse']):8.3f} "
            f"{float(row['mean_excess_route_reuse']):8.3f} "
            f"[{float(row['excess_ci_low']):.3f}, {float(row['excess_ci_high']):.3f}] "
            f"{float(row['mean_generated_above_human_nearest_p95_rate']):9.1%}"
        )
    print("-" * 102)
    if analysis_status == "provisional":
        print("P2 rows are exploratory; these results are development indicators, not final claims.")
    else:
        print("P2 rows remain exploratory even in a final corpus analysis.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", default=str(DEFAULT_GENERATED_ROOT))
    parser.add_argument("--swda-root", default=str(DEFAULT_SWDA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="sentence-transformers device; default auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--route-representation",
        choices=["landmarks", "full-coverage"],
        default="landmarks",
        help="landmarks equalizes lexical words per route state; full-coverage is a robustness mode",
    )
    parser.add_argument("--progress-bins", type=int, default=8)
    parser.add_argument(
        "--landmark-words",
        type=int,
        default=20,
        help="lexical words in each fixed-mass landmark",
    )
    parser.add_argument("--max-microchunk-words", type=int, default=96)
    parser.add_argument(
        "--min-words",
        type=int,
        default=20,
        help="minimum lexical words; landmark mode also requires at least --landmark-words",
    )
    parser.add_argument("--dtw-band", type=int, default=1)
    parser.add_argument("--warp-penalty", type=float, default=0.05)
    parser.add_argument("--max-warp-run", type=int, default=2)
    parser.add_argument("--primary-min-topic-size", type=int, default=3)
    parser.add_argument("--human-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--analysis-label", default="provisional-generated-v2")
    parser.add_argument(
        "--analysis-status",
        choices=["provisional", "final"],
        default="provisional",
        help="explicit reporting status; final must be requested deliberately",
    )
    parser.add_argument("--allow-unbalanced", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = pathlib.Path(args.output_dir)

    conversations = load_route_conversations(
        pathlib.Path(args.generated_root),
        pathlib.Path(args.swda_root),
        strict_balanced=not args.allow_unbalanced,
        expected_conditions=None if args.allow_unbalanced else EXPECTED_GENERATED_CONDITIONS,
    )
    embedder = SentenceRouteEmbedder(
        args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    routes = embed_routes(
        conversations,
        embedder,
        n_bins=args.progress_bins,
        max_microchunk_words=args.max_microchunk_words,
        min_words=args.min_words,
        route_representation=args.route_representation,
        landmark_words=args.landmark_words,
    )
    pairs = pairwise_route_rows(
        routes,
        band=args.dtw_band,
        warp_penalty=args.warp_penalty,
        max_warp_run=args.max_warp_run,
    )
    nearest = nearest_sibling_rows(routes, pairs)
    topics = topic_condition_rows(nearest, pairs)
    calibrated, _reference_draws = calibrated_topic_rows(
        routes,
        pairs,
        topics,
        human_draws=args.human_draws,
        seed=args.seed,
        primary_min_topic_size=args.primary_min_topic_size,
    )
    summaries = condition_summary_rows(
        calibrated,
        routes,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    contrasts = contrast_rows(
        calibrated,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    architecture_pairs, architecture_topics = architecture_displacement_rows(
        routes,
        band=args.dtw_band,
        warp_penalty=args.warp_penalty,
        max_warp_run=args.max_warp_run,
        primary_min_topic_size=args.primary_min_topic_size,
    )
    architecture_displacement = architecture_displacement_summary_rows(
        architecture_topics,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    progress = progress_summary_rows(
        routes,
        calibrated,
        n_points=args.progress_bins,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )

    write_csv(output_dir / "semantic_route_by_conversation.csv", nearest)
    write_csv(output_dir / "semantic_route_pairwise.csv", pairs)
    write_csv(output_dir / "semantic_route_by_topic_condition.csv", calibrated)
    write_csv(output_dir / "semantic_route_by_condition.csv", summaries)
    write_csv(output_dir / "semantic_route_contrasts.csv", contrasts)
    write_csv(output_dir / "semantic_route_architecture_pairs.csv", architecture_pairs)
    write_csv(output_dir / "semantic_route_architecture_by_topic.csv", architecture_topics)
    write_csv(
        output_dir / "semantic_route_architecture_displacement.csv",
        architecture_displacement,
    )
    write_csv(output_dir / "semantic_route_progress.csv", progress)
    if not args.no_plots:
        plot_condition_summary(summaries, output_dir / "semantic_route_by_condition.png")
        plot_branching(progress, output_dir / "semantic_route_branching.png")
    write_run_metadata(
        output_dir / "semantic_route_run.json",
        args=args,
        embedder_name=embedder.name,
        conversations=conversations,
        routes=routes,
        summary_rows=summaries,
    )
    print_summary(
        summaries,
        embedder.name,
        analysis_label=args.analysis_label,
        analysis_status=args.analysis_status,
    )
    print(f"\nWrote Semantic Route Reuse outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
