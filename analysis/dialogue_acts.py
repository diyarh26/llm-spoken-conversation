"""Dialogue-act structural signatures for Switchboard and generated dialogue.

The human side uses the gold ``act_tag`` field in the local SwDA CSV files.  The
generated side is tagged with DialogTag and then mapped back to exactly the same
short-code inventory.  The module deliberately keeps model imports inside the
tagging adapter: ``--human-only`` and cached statistical reruns do not require
DialogTag, TensorFlow, or torch.

Usage::

    python analysis/dialogue_acts.py --human-only
    python analysis/dialogue_acts.py
    python analysis/dialogue_acts.py --force-retag --validation-conversations 20

The default output directory is ``results/dialogue_acts/``.  No transcript text
is written: the optional cache contains only generated-data labels and aggregate
gold validation counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.analyze import conversation_turns  # noqa: E402
from analysis.swda import (  # noqa: E402
    DATA_ROOT as SWDA_ROOT,
    clean_text,
    conversation_no_of,
    iter_conversation_files,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEN_ROOT = ROOT / "data" / "generated_v2"
OUT_DIR = ROOT / "results" / "dialogue_acts"
CACHE_VERSION = 1

# DialogTag is trained on the 38-class simplification in cgpotts/swda.  The task
# additionally requires every suffix modifier to be removed (notably qy^d -> qy,
# qw^d -> qw, and b^m -> b), while retaining the four leading-caret base acts.
# Gold-only %, x, ar, and no remain real columns even though DialogTag cannot emit
# them.  The result is one shared 39-code inventory, with structural zeros where
# the automatic tagger lacks a class.
FINE_LABELS = [
    "sd", "b", "sv", "%", "aa", "ba", "qy", "x", "ny", "fc",
    "qw", "nn", "bk", "h", "fo", "bh", "^q", "bf", "na", "ad",
    "^2", "qo", "qh", "^h", "ng", "br", "fp", "qrr", "arp", "t3",
    "oo", "aap", "bd", "t1", "^g", "fa", "ft", "ar", "no",
]

COARSE_LABELS = [
    "Statement",
    "Opinion",
    "Backchannel",
    "YesNoQuestion",
    "WhQuestion/OpenQuestion",
    "Agreement",
    "Answer",
    "Directive",
    "Repair/Hedge",
    "Abandoned/Other",
]

# A deliberately exhaustive coarse map.  Some rare DAMSL acts have no perfect
# ten-way home; the choices below preserve their broad interactional function.
FINE_TO_COARSE = {
    "sd": "Statement",
    "^q": "Statement",       # quotation
    "sv": "Opinion",
    "b": "Backchannel",
    "bk": "Backchannel",     # response acknowledgement
    "bh": "Backchannel",     # backchannel in question form
    "ba": "Backchannel",     # appreciation/listener response
    "qy": "YesNoQuestion",
    "^g": "YesNoQuestion",   # tag question
    "qrr": "YesNoQuestion",  # or-clause
    "qw": "WhQuestion/OpenQuestion",
    "qo": "WhQuestion/OpenQuestion",
    "qh": "WhQuestion/OpenQuestion",
    "aa": "Agreement",
    "aap": "Agreement",      # maybe/accept-part
    "bd": "Agreement",       # downplayer
    "ny": "Answer",
    "nn": "Answer",
    "na": "Answer",
    "ng": "Answer",
    "ar": "Answer",          # reject
    "no": "Answer",          # other answer
    "arp": "Answer",         # dispreferred answer
    "ad": "Directive",
    "oo": "Directive",       # offers/options/commits
    "h": "Repair/Hedge",
    "bf": "Repair/Hedge",    # summarize/reformulate
    "br": "Repair/Hedge",    # signal non-understanding
    "^h": "Repair/Hedge",    # hold before answer/agreement
    "^2": "Repair/Hedge",    # collaborative completion
    "t1": "Repair/Hedge",    # self-talk/word-search turn management
    "%": "Abandoned/Other",
    "x": "Abandoned/Other",
    "fo": "Abandoned/Other",
    "fc": "Abandoned/Other",
    "fp": "Abandoned/Other",
    "fa": "Abandoned/Other",
    "ft": "Abandoned/Other",
    "t3": "Abandoned/Other", # third-party talk, outside the dyadic exchange
}

# Exact long strings from DialogTag 1.1.3's downloaded label_map.txt.  Several
# labels correspond to grouped SwDA codes in that model; we choose one canonical
# short code for the group.  Declarative questions and repeat phrases follow the
# requested suffix-stripping rule, so both sides collapse identically.
DIALOGTAG_TO_FINE = {
    "Statement-non-opinion": "sd",
    "Acknowledge (Backchannel)": "b",
    "Statement-opinion": "sv",
    "Agree/Accept": "aa",
    "Appreciation": "ba",
    "Yes-No-Question": "qy",
    "Yes Answers": "ny",
    "Conventional-closing": "fc",
    "Wh-Question": "qw",
    "No Answers": "nn",
    "Response Acknowledgement": "bk",
    "Hedge": "h",
    "Declarative Yes-No-Question": "qy",
    "Backchannel in Question Form": "bh",
    "Quotation": "^q",
    "Summarize/Reformulate": "bf",
    "Other": "fo",
    "Affirmative Non-yes Answers": "na",
    "Action-directive": "ad",
    "Collaborative Completion": "^2",
    "Repeat-phrase": "b",
    "Open-Question": "qo",
    "Rhetorical-Question": "qh",
    "Hold Before Answer/Agreement": "^h",
    "Negative Non-no Answers": "ng",
    "Signal-non-understanding": "br",
    "Conventional-opening": "fp",
    "Or-Clause": "qrr",
    "Dispreferred Answers": "arp",
    "3rd-party-talk": "t3",
    "Offers, Options Commits": "oo",
    "Maybe/Accept-part": "aap",
    "Downplayer": "bd",
    "Self-talk": "t1",
    "Tag-Question": "^g",
    "Declarative Wh-Question": "qw",
    "Apology": "fa",
    "Thanking": "ft",
}

# cgpotts/swda's standard grouping, expressed with ordinary short codes rather
# than compound column names.  Applied after modifiers and annotation flags are
# removed.  For comma/semicolon multi-tags, the first tag is used, matching the
# reference loader's documented policy.
GOLD_BASE_ALIASES = {
    "qr": "qy",
    "fe": "ba",
    "oo": "oo",
    "co": "oo",
    "cc": "oo",
    "fx": "sv",
    "aap": "aap",
    "am": "aap",
    "arp": "arp",
    "nd": "arp",
    "fo": "fo",
    "o": "fo",
    "fw": "fo",
    '"': "fo",
    "by": "fo",
    "bc": "fo",
}

ASSISTANT_PATTERNS = {
    "numbered_list": re.compile(r"(?:^|\n)\s*\d{1,2}[.)]\s+", re.I),
    "you_should": re.compile(r"\byou\s+should\b", re.I),
    "you_could_try": re.compile(r"\byou\s+could\s+try\b", re.I),
    "i_recommend": re.compile(r"\bi\s+recommend\b", re.I),
    "important_to": re.compile(r"\bit(?:'s|\s+is)\s+important\s+to\b", re.I),
    "several": re.compile(r"\bthere\s+are\s+several\b", re.I),
    "conclusion": re.compile(r"(?:^|[.!?]\s+)(?:overall|in\s+conclusion)\b", re.I),
}


@dataclass
class Conversation:
    source: str
    condition: str
    conversation_no: int
    speakers: list[str]
    texts: list[str]
    fine_labels: list[str] = field(default_factory=list)

    def coarse_labels(self) -> list[str]:
        return [FINE_TO_COARSE[label] for label in self.fine_labels]


@dataclass
class ValidationResult:
    conversation_ids: list[int]
    fine_confusion: Counter[tuple[str, str]]

    @property
    def n_items(self) -> int:
        return sum(self.fine_confusion.values())


def normalize_swda_base(raw_tag: str) -> str:
    """Collapse one non-continuation raw SwDA tag to the shared inventory.

    Policy, in order:
      * take the first tag in rare comma/semicolon compounds;
      * preserve ``%`` (abandoned/uninterpretable) and ``x`` (non-verbal);
      * discard parenthesized, ``@``, and ``*`` annotation modifiers;
      * strip suffix ``^`` modifiers, while retaining leading ``^q/^2/^h/^g``;
      * apply the standard grouped-class aliases listed above.

    ``+`` requires caller-local history and is therefore resolved by
    :func:`load_switchboard`, not here.
    """

    tag = re.split(r"[,;]", raw_tag.strip(), maxsplit=1)[0].strip()
    if tag.startswith("+"):
        return "+"
    if tag.startswith("%"):
        return "%"
    if tag.startswith("x"):
        return "x"

    # These two elaboration codes are semantic classes in the standard SwDA
    # reduction, not disposable style modifiers.  Handle them before the task's
    # general suffix collapse.
    if tag == "nn^e":
        return "ng"
    if tag == "ny^e":
        return "na"

    tag = re.sub(r"\([^)]*\)", "", tag)
    tag = tag.replace("@", "").replace("*", "")
    if tag.startswith("^"):
        tag = tag[:2]
    elif "^" in tag:
        tag = tag.split("^", 1)[0]
    tag = GOLD_BASE_ALIASES.get(tag, tag)
    if tag not in FINE_LABELS:
        raise ValueError(f"Unmapped SwDA act_tag {raw_tag!r} -> {tag!r}")
    return tag


def load_switchboard(root: pathlib.Path = SWDA_ROOT) -> list[Conversation]:
    """Read gold SwDA annotations without using the turn-merging parser."""

    conversations: list[Conversation] = []
    for path in iter_conversation_files(root):
        speakers: list[str] = []
        texts: list[str] = []
        labels: list[str] = []
        previous_by_caller: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                caller = row["caller"].strip()
                label = normalize_swda_base(row["act_tag"])
                if label == "+":
                    if caller not in previous_by_caller:
                        raise ValueError(
                            f"Continuation without prior act for caller {caller!r} "
                            f"at {path}:{row_number}"
                        )
                    label = previous_by_caller[caller]
                else:
                    previous_by_caller[caller] = label
                speakers.append(caller)
                texts.append(row["text"])
                labels.append(label)
        if labels:
            conversations.append(
                Conversation(
                    source="SB",
                    condition="SB",
                    conversation_no=conversation_no_of(path),
                    speakers=speakers,
                    texts=texts,
                    fine_labels=labels,
                )
            )
    if not conversations:
        raise FileNotFoundError(f"No SwDA .utt.csv files found below {root}")
    return conversations


def load_generated(root: pathlib.Path = GEN_ROOT) -> list[Conversation]:
    conversations: list[Conversation] = []
    for path in sorted(root.glob("*/*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        turns = conversation_turns(record)
        if not turns:
            continue
        condition = str(record.get("condition", path.parent.name))
        conversation_no = int(record.get("conversation_no", path.stem))
        conversations.append(
            Conversation(
                source="LLM",
                condition=condition,
                conversation_no=conversation_no,
                speakers=[speaker for speaker, _ in turns],
                texts=[text for _, text in turns],
            )
        )
    if not conversations:
        raise FileNotFoundError(f"No generated JSON conversations found below {root}")
    return conversations


def _normal_label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.casefold())


_DIALOGTAG_NORMALIZED = {
    _normal_label_key(long_label): short_label
    for long_label, short_label in DIALOGTAG_TO_FINE.items()
}


def dialogtag_to_fine(label: str) -> str:
    """Map a DialogTag long label to the shared short-code inventory."""

    if label in DIALOGTAG_TO_FINE:
        return DIALOGTAG_TO_FINE[label]
    key = _normal_label_key(label)
    if key in _DIALOGTAG_NORMALIZED:
        return _DIALOGTAG_NORMALIZED[key]
    raise ValueError(
        f"Unknown DialogTag label {label!r}. Expected one of: "
        + ", ".join(sorted(DIALOGTAG_TO_FINE))
    )


class DialogTagAdapter:
    """Lazy DialogTag wrapper with a batched fast path and public-API fallback."""

    def __init__(self, model_name: str):
        try:
            from dialog_tag import DialogTag
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "DialogTag is required only when predictions are not cached. "
                "Install the isolated tagger environment from "
                "analysis/requirements-dialogue-acts.txt, or run --human-only."
            ) from exc

        self.model_name = model_name
        self.tagger = DialogTag(model_name)
        self._batch_available = False
        try:
            self._tokenizer = getattr(self.tagger, "_DialogTag__tokenizer")
            self._model = getattr(self.tagger, "_DialogTag__model")
            class_helper = getattr(self.tagger, "_DialogTag__classhelper")
            logits_class, class_expanded = class_helper()
            self._index_to_long = {
                int(index): class_expanded[short_code]
                for index, short_code in logits_class.items()
            }
            self._batch_available = True
        except (AttributeError, KeyError, TypeError, ValueError):
            # DialogTag exposes only predict_tag publicly.  Keeping this fallback
            # makes the adapter robust if the package changes its private fields.
            self._batch_available = False

    def predict_many(self, texts: Sequence[str], batch_size: int = 32) -> list[str]:
        if not texts:
            return []
        if not self._batch_available:
            return [self.tagger.predict_tag(text) for text in texts]

        predictions: list[str] = []
        try:
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start:start + batch_size])
                encoded = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="tf",
                )
                output = self._model(**encoded, training=False)
                logits = output.logits if hasattr(output, "logits") else output[0]
                indices = np.asarray(logits).argmax(axis=1).tolist()
                predictions.extend(self._index_to_long[int(index)] for index in indices)
            return predictions
        except Exception as exc:  # pragma: no cover - package-version fallback
            print(f"DialogTag batch path failed ({exc}); falling back to predict_tag().")
            self._batch_available = False
            return [self.tagger.predict_tag(text) for text in texts]


def fingerprint_conversations(conversations: Sequence[Conversation]) -> str:
    """Content fingerprint used to reject stale label caches (never stores text)."""

    digest = hashlib.sha256()
    for conversation in sorted(
        conversations, key=lambda item: (item.condition, item.conversation_no)
    ):
        digest.update(conversation.condition.encode("utf-8"))
        digest.update(str(conversation.conversation_no).encode("ascii"))
        for speaker, text in zip(conversation.speakers, conversation.texts):
            digest.update(b"\0")
            digest.update(speaker.encode("utf-8"))
            digest.update(b"\1")
            digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def validation_sample(
    gold: Sequence[Conversation], n_conversations: int, seed: int
) -> list[Conversation]:
    if n_conversations <= 0:
        raise ValueError("--validation-conversations must be positive")
    if n_conversations > len(gold):
        raise ValueError(
            f"Cannot sample {n_conversations} validation conversations from {len(gold)}"
        )
    rng = random.Random(seed)
    return sorted(
        rng.sample(list(gold), n_conversations), key=lambda item: item.conversation_no
    )


def _validation_texts(sample: Sequence[Conversation]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for conversation in sample:
        for raw_text, label in zip(conversation.texts, conversation.fine_labels):
            # DialogTag was trained on cleaned SwDA utterance text.  Keep a raw
            # fallback for pure non-verbal rows that clean to an empty string.
            text = clean_text(raw_text) or raw_text.strip() or "[non-verbal]"
            texts.append(text)
            labels.append(label)
    return texts, labels


def _load_cache(
    cache_path: pathlib.Path,
    generated: Sequence[Conversation],
    sample: Sequence[Conversation],
    model_name: str,
) -> ValidationResult | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open(encoding="utf-8") as handle:
            cache = json.load(handle)
        expected_ids = [conversation.conversation_no for conversation in sample]
        if (
            cache.get("version") != CACHE_VERSION
            or cache.get("model_name") != model_name
            or cache.get("generated_fingerprint") != fingerprint_conversations(generated)
            or cache.get("validation_ids") != expected_ids
            or cache.get("validation_fingerprint") != fingerprint_conversations(sample)
        ):
            return None

        by_key = {
            (row["condition"], int(row["conversation_no"])): row["fine_labels"]
            for row in cache["generated"]
        }
        for conversation in generated:
            labels = list(by_key[(conversation.condition, conversation.conversation_no)])
            if len(labels) != len(conversation.texts) or any(
                label not in FINE_LABELS for label in labels
            ):
                return None
            conversation.fine_labels = labels
        confusion = Counter(
            {
                (row["gold"], row["predicted"]): int(row["count"])
                for row in cache["validation_confusion"]
            }
        )
        return ValidationResult(expected_ids, confusion)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cache(
    cache_path: pathlib.Path,
    generated: Sequence[Conversation],
    sample: Sequence[Conversation],
    model_name: str,
    validation: ValidationResult,
) -> None:
    payload = {
        "version": CACHE_VERSION,
        "model_name": model_name,
        "generated_fingerprint": fingerprint_conversations(generated),
        "validation_ids": validation.conversation_ids,
        "validation_fingerprint": fingerprint_conversations(sample),
        "generated": [
            {
                "condition": conversation.condition,
                "conversation_no": conversation.conversation_no,
                "fine_labels": conversation.fine_labels,
            }
            for conversation in sorted(
                generated, key=lambda item: (item.condition, item.conversation_no)
            )
        ],
        "validation_confusion": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in sorted(validation.fine_confusion.items())
        ],
        "privacy": "No transcript text is stored; validation is aggregate counts only.",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def tag_generated_and_validate(
    generated: Sequence[Conversation],
    gold: Sequence[Conversation],
    *,
    model_name: str,
    batch_size: int,
    validation_conversations: int,
    seed: int,
    cache_path: pathlib.Path,
    force_retag: bool,
    use_cache: bool,
) -> tuple[ValidationResult, bool]:
    sample = validation_sample(gold, validation_conversations, seed)
    if use_cache and not force_retag:
        cached = _load_cache(cache_path, generated, sample, model_name)
        if cached is not None:
            return cached, True

    llm_texts = [text for conversation in generated for text in conversation.texts]
    validation_texts, gold_labels = _validation_texts(sample)
    print(
        f"Loading DialogTag {model_name!r}; tagging {len(llm_texts):,} generated "
        f"turns and {len(validation_texts):,} gold validation utterances..."
    )
    tagger = DialogTagAdapter(model_name)
    long_predictions = tagger.predict_many(
        llm_texts + validation_texts, batch_size=batch_size
    )
    fine_predictions = [dialogtag_to_fine(label) for label in long_predictions]
    if len(fine_predictions) != len(llm_texts) + len(validation_texts):
        raise RuntimeError("DialogTag returned the wrong number of predictions")

    cursor = 0
    for conversation in generated:
        next_cursor = cursor + len(conversation.texts)
        conversation.fine_labels = fine_predictions[cursor:next_cursor]
        cursor = next_cursor
    predicted_gold = fine_predictions[len(llm_texts):]
    validation = ValidationResult(
        conversation_ids=[conversation.conversation_no for conversation in sample],
        fine_confusion=Counter(zip(gold_labels, predicted_gold)),
    )
    if use_cache:
        _write_cache(cache_path, generated, sample, model_name, validation)
    return validation, False


def grouped_by_condition(
    conversations: Sequence[Conversation],
) -> dict[str, list[Conversation]]:
    grouped: defaultdict[str, list[Conversation]] = defaultdict(list)
    for conversation in conversations:
        grouped[conversation.condition].append(conversation)
    return dict(sorted(grouped.items()))


def labels_for(conversation: Conversation, label_set: str) -> list[str]:
    if label_set == "fine":
        return conversation.fine_labels
    if label_set == "coarse":
        return conversation.coarse_labels()
    raise ValueError(f"Unknown label set: {label_set}")


def label_inventory(label_set: str) -> list[str]:
    return FINE_LABELS if label_set == "fine" else COARSE_LABELS


def distribution(
    conversations: Sequence[Conversation], label_set: str
) -> np.ndarray:
    inventory = label_inventory(label_set)
    index = {label: i for i, label in enumerate(inventory)}
    counts = np.zeros(len(inventory), dtype=float)
    for conversation in conversations:
        for label in labels_for(conversation, label_set):
            counts[index[label]] += 1
    total = counts.sum()
    return counts / total if total else counts


def transition_matrix(
    conversations: Sequence[Conversation], label_set: str
) -> np.ndarray:
    inventory = label_inventory(label_set)
    index = {label: i for i, label in enumerate(inventory)}
    counts = np.zeros((len(inventory), len(inventory)), dtype=float)
    for conversation in conversations:
        sequence = labels_for(conversation, label_set)
        for current, following in zip(sequence, sequence[1:]):
            counts[index[current], index[following]] += 1
    row_totals = counts.sum(axis=1, keepdims=True)
    return np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts),
        where=row_totals != 0,
    )


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Base-2 Jensen-Shannon divergence in [0, 1]."""

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_total = float(p.sum())
    q_total = float(q.sum())
    if p_total == 0 and q_total == 0:
        return 0.0
    if p_total == 0 or q_total == 0:
        return 1.0
    p = p / p_total
    q = q / q_total
    midpoint = 0.5 * (p + q)

    def _kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * _kl(p, midpoint) + 0.5 * _kl(q, midpoint)


def transition_jsd(left: np.ndarray, right: np.ndarray) -> float:
    """Unweighted mean row JSD; a one-sided missing row has distance 1."""

    row_distances = [
        js_divergence(left[row], right[row])
        for row in range(left.shape[0])
        if left[row].sum() or right[row].sum()
    ]
    return float(np.mean(row_distances)) if row_distances else 0.0


def bootstrap_noise_floor(
    gold: Sequence[Conversation],
    label_set: str,
    *,
    sample_size: int,
    repetitions: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    if sample_size > len(gold):
        raise ValueError(
            f"Bootstrap sample size {sample_size} exceeds {len(gold)} gold conversations"
        )
    inventory = label_inventory(label_set)
    index = {label: i for i, label in enumerate(inventory)}
    per_conversation = np.zeros((len(gold), len(inventory)), dtype=float)
    for row, conversation in enumerate(gold):
        for label in labels_for(conversation, label_set):
            per_conversation[row, index[label]] += 1
    full = per_conversation.sum(axis=0)
    full /= full.sum()
    rng = random.Random(seed)
    distances = np.zeros(repetitions, dtype=float)
    population = list(range(len(gold)))
    for repetition in range(repetitions):
        chosen = rng.sample(population, sample_size)
        sampled = per_conversation[chosen].sum(axis=0)
        sampled /= sampled.sum()
        distances[repetition] = js_divergence(sampled, full)
    return float(np.quantile(distances, 0.95)), distances


def assistant_register_counts(
    conversations: Sequence[Conversation],
) -> dict[str, float | int]:
    matches = Counter({name: 0 for name in ASSISTANT_PATTERNS})
    any_match = 0
    n_turns = 0
    for conversation in conversations:
        for text in conversation.texts:
            n_turns += 1
            matched = [
                name for name, pattern in ASSISTANT_PATTERNS.items()
                if pattern.search(text)
            ]
            if matched:
                any_match += 1
                matches.update(matched)
    result: dict[str, float | int] = {
        "n_turns": n_turns,
        "assistant_register_turns": any_match,
        "assistant_register_fraction": any_match / n_turns if n_turns else 0.0,
    }
    for name in ASSISTANT_PATTERNS:
        result[f"{name}_fraction"] = matches[name] / n_turns if n_turns else 0.0
    return result


def confusion_for_label_set(
    fine_confusion: Counter[tuple[str, str]], label_set: str
) -> Counter[tuple[str, str]]:
    if label_set == "fine":
        return Counter(fine_confusion)
    confusion: Counter[tuple[str, str]] = Counter()
    for (gold, predicted), count in fine_confusion.items():
        confusion[(FINE_TO_COARSE[gold], FINE_TO_COARSE[predicted])] += count
    return confusion


def validation_accuracy(confusion: Counter[tuple[str, str]]) -> float:
    total = sum(confusion.values())
    correct = sum(count for (gold, predicted), count in confusion.items() if gold == predicted)
    return correct / total if total else 0.0


def write_csv(path: pathlib.Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def distribution_rows(
    groups: dict[str, Sequence[Conversation]], label_set: str
) -> list[dict]:
    inventory = label_inventory(label_set)
    rows: list[dict] = []
    for condition, conversations in groups.items():
        values = distribution(conversations, label_set)
        row = {
            "condition": condition,
            "n_conversations": len(conversations),
            "n_units": sum(len(labels_for(c, label_set)) for c in conversations),
        }
        row.update({label: f"{values[i]:.10f}" for i, label in enumerate(inventory)})
        rows.append(row)
    return rows


def write_transition_csv(
    path: pathlib.Path, matrix: np.ndarray, label_set: str
) -> None:
    inventory = label_inventory(label_set)
    rows = []
    for row_index, current in enumerate(inventory):
        row = {"current_act": current}
        row.update(
            {following: f"{matrix[row_index, col]:.10f}"
             for col, following in enumerate(inventory)}
        )
        rows.append(row)
    write_csv(path, rows, ["current_act", *inventory])


def safe_condition_name(condition: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", condition)


def build_comparison_rows(
    gold: Sequence[Conversation],
    generated_groups: dict[str, list[Conversation]],
    noise_floors: dict[str, float],
) -> tuple[list[dict], dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    gold_by_id = {conversation.conversation_no: conversation for conversation in gold}
    matrices: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict] = []
    for label_set in ("fine", "coarse"):
        reference_distribution = distribution(gold, label_set)
        reference_transition = transition_matrix(gold, label_set)
        for condition, conversations in generated_groups.items():
            ids = {conversation.conversation_no for conversation in conversations}
            missing = sorted(ids.difference(gold_by_id))
            if missing:
                raise ValueError(
                    f"{condition} has conversation_no values absent from SwDA: {missing}"
                )
            matched_gold = [gold_by_id[conversation_no] for conversation_no in sorted(ids)]
            condition_distribution = distribution(conversations, label_set)
            condition_transition = transition_matrix(conversations, label_set)
            matched_distribution = distribution(matched_gold, label_set)
            matched_transition = transition_matrix(matched_gold, label_set)
            assistant = assistant_register_counts(conversations)
            dist_jsd = js_divergence(condition_distribution, reference_distribution)
            rows.append(
                {
                    "condition": condition,
                    "label_set": label_set,
                    "jsd_dist": f"{dist_jsd:.10f}",
                    "jsd_transition": f"{transition_jsd(condition_transition, reference_transition):.10f}",
                    "noise_floor_p95": f"{noise_floors[label_set]:.10f}",
                    "exceeds_floor": str(dist_jsd > noise_floors[label_set]).lower(),
                    "topic_matched_jsd_dist": f"{js_divergence(condition_distribution, matched_distribution):.10f}",
                    "topic_matched_jsd_transition": f"{transition_jsd(condition_transition, matched_transition):.10f}",
                    "assistant_register_fraction": f"{assistant['assistant_register_fraction']:.10f}",
                    "n_conversations": len(conversations),
                    "n_units": sum(len(labels_for(c, label_set)) for c in conversations),
                    "n_topic_matched_sb_conversations": len(matched_gold),
                }
            )
            matrices[(condition, label_set)] = (
                condition_transition,
                matched_transition,
            )
    return rows, matrices


def write_validation_outputs(
    output_dir: pathlib.Path, validation: ValidationResult
) -> list[dict]:
    summary_rows: list[dict] = []
    confusion_rows: list[dict] = []
    for label_set in ("fine", "coarse"):
        confusion = confusion_for_label_set(validation.fine_confusion, label_set)
        total_by_gold: Counter[str] = Counter()
        for (gold, _), count in confusion.items():
            total_by_gold[gold] += count
        accuracy = validation_accuracy(confusion)
        summary_rows.append(
            {
                "label_set": label_set,
                "n_conversations": len(validation.conversation_ids),
                "n_utterances": sum(confusion.values()),
                "accuracy": f"{accuracy:.10f}",
            }
        )
        for (gold, predicted), count in sorted(
            confusion.items(), key=lambda item: (-item[1], item[0])
        ):
            confusion_rows.append(
                {
                    "label_set": label_set,
                    "gold": gold,
                    "predicted": predicted,
                    "count": count,
                    "fraction_of_gold": f"{count / total_by_gold[gold]:.10f}",
                }
            )
    write_csv(
        output_dir / "da_tagger_validation.csv",
        summary_rows,
        ["label_set", "n_conversations", "n_utterances", "accuracy"],
    )
    write_csv(
        output_dir / "da_tagger_confusion.csv",
        confusion_rows,
        ["label_set", "gold", "predicted", "count", "fraction_of_gold"],
    )
    return summary_rows


def print_validation(validation: ValidationResult, top_n: int = 12) -> None:
    print("\nDialogTag validation against gold SwDA")
    for label_set in ("fine", "coarse"):
        confusion = confusion_for_label_set(validation.fine_confusion, label_set)
        accuracy = validation_accuracy(confusion)
        print(
            f"  {label_set:6} accuracy: {accuracy:.3%} "
            f"({sum(confusion.values()):,} utterances; "
            f"{len(validation.conversation_ids)} conversations)"
        )
        errors = [
            (count, gold, predicted)
            for (gold, predicted), count in confusion.items()
            if gold != predicted
        ]
        errors.sort(reverse=True)
        print(f"  {label_set:6} largest confusions (gold -> predicted):")
        for count, gold, predicted in errors[:top_n]:
            print(f"           {gold:25} -> {predicted:25} {count:5d}")


def write_assistant_output(
    output_dir: pathlib.Path,
    gold: Sequence[Conversation],
    generated_groups: dict[str, list[Conversation]],
) -> list[dict]:
    rows: list[dict] = []
    for condition, conversations in [("SB", list(gold)), *generated_groups.items()]:
        row: dict = {"condition": condition}
        row.update(assistant_register_counts(conversations))
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = f"{value:.10f}"
        rows.append(row)
    fields = [
        "condition",
        "n_turns",
        "assistant_register_turns",
        "assistant_register_fraction",
        *[f"{name}_fraction" for name in ASSISTANT_PATTERNS],
    ]
    write_csv(output_dir / "assistant_register_by_condition.csv", rows, fields)
    return rows


def _import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_key_acts(
    groups: dict[str, Sequence[Conversation]], output_path: pathlib.Path
) -> None:
    plt = _import_plotting()
    conditions = list(groups)
    keys = ["Backchannel", "Statement", "Question", "Agreement"]
    values: dict[str, list[float]] = {key: [] for key in keys}
    for condition in conditions:
        coarse = distribution(groups[condition], "coarse")
        by_label = dict(zip(COARSE_LABELS, coarse))
        values["Backchannel"].append(by_label["Backchannel"])
        values["Statement"].append(by_label["Statement"] + by_label["Opinion"])
        values["Question"].append(
            by_label["YesNoQuestion"] + by_label["WhQuestion/OpenQuestion"]
        )
        values["Agreement"].append(by_label["Agreement"])

    x = np.arange(len(conditions))
    width = 0.19
    fig, axis = plt.subplots(figsize=(14, 6))
    for offset, key in enumerate(keys):
        axis.bar(x + (offset - 1.5) * width, values[key], width, label=key)
    axis.set_xticks(x, conditions, rotation=45, ha="right")
    axis.set_ylabel("Proportion of dialogue units")
    axis.set_title("Key dialogue-act groups: Switchboard vs generated conditions")
    axis.legend(ncols=4, frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_sb_heatmap(matrix: np.ndarray, output_path: pathlib.Path) -> None:
    plt = _import_plotting()
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="magma", vmin=0, vmax=max(0.5, float(matrix.max())))
    axis.set_xticks(range(len(COARSE_LABELS)), COARSE_LABELS, rotation=55, ha="right")
    axis.set_yticks(range(len(COARSE_LABELS)), COARSE_LABELS)
    axis.set_xlabel("Next act")
    axis.set_ylabel("Current act")
    axis.set_title("Switchboard transition grammar (coarse acts)")
    fig.colorbar(image, ax=axis, label="P(next | current)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_jsd(
    comparison_rows: Sequence[dict], output_path: pathlib.Path
) -> None:
    plt = _import_plotting()
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for axis, label_set in zip(axes, ("fine", "coarse")):
        rows = [row for row in comparison_rows if row["label_set"] == label_set]
        conditions = [row["condition"] for row in rows]
        values = [float(row["jsd_dist"]) for row in rows]
        floor = float(rows[0]["noise_floor_p95"])
        colors = ["#b43c39" if value > floor else "#4c78a8" for value in values]
        axis.bar(range(len(rows)), values, color=colors)
        axis.axhline(
            floor,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"human n=50 p95 = {floor:.4f}",
        )
        axis.set_ylabel("JSD vs full SB")
        axis.set_title(f"{label_set.capitalize()} dialogue-act distribution")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False)
        axis.set_xticks(range(len(rows)), conditions, rotation=45, ha="right")
    fig.suptitle("Dialogue-act structural distance and the human sampling-noise floor")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def top_confusions(
    validation: ValidationResult, label_set: str, n: int = 5
) -> list[tuple[int, str, str]]:
    confusion = confusion_for_label_set(validation.fine_confusion, label_set)
    errors = [
        (count, gold, predicted)
        for (gold, predicted), count in confusion.items()
        if gold != predicted
    ]
    return sorted(errors, reverse=True)[:n]


def write_readme(
    output_dir: pathlib.Path,
    *,
    gold: Sequence[Conversation],
    generated_groups: dict[str, list[Conversation]],
    comparison_rows: Sequence[dict],
    validation: ValidationResult,
    assistant_rows: Sequence[dict],
    bootstrap_repetitions: int,
    bootstrap_size: int,
    seed: int,
    model_name: str,
    cache_used: bool,
) -> None:
    fine_accuracy = validation_accuracy(
        confusion_for_label_set(validation.fine_confusion, "fine")
    )
    coarse_accuracy = validation_accuracy(
        confusion_for_label_set(validation.fine_confusion, "coarse")
    )
    coarse_rows = [row for row in comparison_rows if row["label_set"] == "coarse"]
    fine_rows = [row for row in comparison_rows if row["label_set"] == "fine"]
    closest = min(coarse_rows, key=lambda row: float(row["jsd_dist"]))
    farthest = max(coarse_rows, key=lambda row: float(row["jsd_dist"]))
    coarse_exceed = sum(row["exceeds_floor"] == "true" for row in coarse_rows)
    fine_exceed = sum(row["exceeds_floor"] == "true" for row in fine_rows)
    assistant_by_condition = {
        row["condition"]: float(row["assistant_register_fraction"])
        for row in assistant_rows
    }
    top_assistant = max(
        generated_groups,
        key=lambda condition: assistant_by_condition[condition],
    )
    matched_deltas = [
        abs(float(row["topic_matched_jsd_dist"]) - float(row["jsd_dist"]))
        for row in coarse_rows
    ]
    largest_matched_delta = max(matched_deltas) if matched_deltas else 0.0
    sb_coarse = dict(zip(COARSE_LABELS, distribution(gold, "coarse")))
    generated_backchannels = {
        condition: dict(zip(COARSE_LABELS, distribution(conversations, "coarse")))[
            "Backchannel"
        ]
        for condition, conversations in generated_groups.items()
    }
    highest_backchannel_condition = max(
        generated_backchannels, key=generated_backchannels.get
    )
    lowest_backchannel_condition = min(
        generated_backchannels, key=generated_backchannels.get
    )
    confusion_lines = [
        f"  - `{gold_label}` -> `{predicted}`: {count:,}"
        for count, gold_label, predicted in top_confusions(validation, "coarse")
    ]
    generated_units = sum(
        len(conversation.fine_labels)
        for conversations in generated_groups.values()
        for conversation in conversations
    )
    text = f"""# Dialogue-act structural signature — draft-data smoke test

This is a first indicator on `data/generated_v2/`, not a final-dataset result.  It compares
{len(gold):,} full Switchboard conversations ({sum(len(c.fine_labels) for c in gold):,}
gold utterance units) with {sum(len(v) for v in generated_groups.values()):,} generated
conversations ({generated_units:,} generated turns) across {len(generated_groups)} conditions.

## First signal

On the more tagger-robust coarse inventory, the closest condition is **{closest['condition']}**
(distribution JSD {float(closest['jsd_dist']):.4f}) and the farthest is
**{farthest['condition']}** ({float(farthest['jsd_dist']):.4f}).  {coarse_exceed} of
{len(coarse_rows)} coarse comparisons and {fine_exceed} of {len(fine_rows)} fine comparisons
exceed the 95th-percentile human n={bootstrap_size} sampling-noise floor.  This is evidence
that the structural signature has signal on the draft generations; it is not evidence that
the automatic fine labels are all trustworthy.

The clearest component is listener feedback: Switchboard is
**{sb_coarse['Backchannel']:.1%} backchannels**, while the generated conditions span only
**{generated_backchannels[lowest_backchannel_condition]:.1%}**
({lowest_backchannel_condition}) to **{generated_backchannels[highest_backchannel_condition]:.1%}**
({highest_backchannel_condition}).  That reproduces the qualitative `sd -> b -> sd` gap in
the spec as an aggregate structural difference rather than a hand-picked example.

Restricting the human reference to the same 50 `conversation_no` values changes coarse JSD
by at most {largest_matched_delta:.4f} in this run.  That makes topic mix an unlikely sole
explanation for the observed gap, although the current 50 topics are the old first-50
convenience sample rather than the planned stratified sample.

The assistant-register rule layer is complementary to DAMSL, not a new DAMSL act.  Its
highest generated-condition rate is **{top_assistant}** at
{assistant_by_condition[top_assistant]:.1%}; the full-SB rule-match rate is
{assistant_by_condition['SB']:.1%}.  These are literal phrase/list matches and should be read
as an indicator, not a classifier.

## Tagger validation

DialogTag `{model_name}` was evaluated on {len(validation.conversation_ids)} seeded SwDA
conversations ({validation.n_items:,} utterances): fine accuracy **{fine_accuracy:.1%}** and
coarse accuracy **{coarse_accuracy:.1%}**.  Largest coarse confusions were:

{chr(10).join(confusion_lines) if confusion_lines else '  - none'}

This is an in-domain diagnostic, not a guaranteed model-held-out score: DialogTag was trained
on the 1,155-conversation SwDA distribution and does not publish enough split identifiers to
prove that this seeded sample was absent from its training split.  It also cannot emit gold
`%`, `x`, `ar`, or `no`.  The coarse comparison is therefore the safer headline.

## Method and normalization

- Raw SwDA suffix modifiers are removed (`sd^e -> sd`, `qy^d -> qy`); standard semantic
  exceptions `nn^e -> ng` and `ny^e -> na` are applied first.  Leading-caret base acts such
  as `^q` are preserved.  All direct-CSV rows are retained while `@`, `*`, and parenthesized
  annotation flags are removed.
- Rare comma/semicolon compound tags use the first act, following `cgpotts/swda`.
- `+` continuations inherit the previous normalized act from the **same caller**, not the
  immediately preceding row.  `%` and `x` stay explicit.
- DialogTag's 38 model labels are explicitly mapped into the shared short inventory.  They
  collapse to 35 short codes; gold-only `%`, `x`, `ar`, and `no` complete the 39 columns.
  Grouped rare classes and question/repeat modifiers are collapsed on both sides.  All 39
  fine acts then map exhaustively to the requested ten coarse categories.
- Distribution distance is base-2 Jensen-Shannon divergence.  Transition distance is the
  unweighted mean row JSD of `P(next | current)`; a transition row present on only one side
  has maximal row distance 1.
- The noise floor uses {bootstrap_repetitions:,} seeded draws of {bootstrap_size} whole human
  conversations without replacement against the full human reference (seed {seed}).

## Important limitations

Gold labels are SwDA annotation units, while each generated model turn receives one label.
That granularity mismatch follows the available inputs but can inflate structural differences,
especially around continuations and backchannels.  DialogTag is a legacy TensorFlow model,
was trained on human telephone speech, and forces assistant-like turns into its human taxonomy.
Draft conditions also have unequal generated-turn counts.  Treat rankings as a smoke-test
direction for the regenerated dataset, not a final claim.

The mandated bootstrap matches **conversation count**, not the number of classified units:
an average 50-conversation human draw contains about
{bootstrap_size * sum(len(c.fine_labels) for c in gold) / len(gold):,.0f} gold annotation
units, versus {min(sum(len(c.fine_labels) for c in conversations) for conversations in generated_groups.values()):,}–{max(sum(len(c.fine_labels) for c in conversations) for conversations in generated_groups.values()):,}
turns in a generated condition.  The very small human-only floor is therefore optimistic
for the human/generated granularity mismatch and should not be treated as a fully calibrated
significance threshold, even though the observed JSDs are orders of magnitude larger.

The cached prediction file contains generated-data labels and aggregate gold confusion counts
only—never Switchboard transcript text.  This run {'reused that cache' if cache_used else 'created fresh predictions'}.

## Files

- `da_distribution_by_condition.csv` and `da_distribution_coarse_by_condition.csv`
- `da_jsd_vs_sb.csv` (fine/coarse, full and topic-matched comparisons)
- `da_transition_*.csv` and `da_transition_coarse_*.csv`
- `da_tagger_validation.csv`, `da_tagger_confusion.csv`
- `assistant_register_by_condition.csv`
- `key_acts_grouped_bar.png`, `sb_transition_heatmap.png`, `jsd_vs_sb.png`

## Reproducing the model-dependent stage

DialogTag is isolated from the main generation environment because its legacy TensorFlow
checkpoint needs older dependency pins.  On Windows with Python 3.11:

```powershell
py -3.11 -m venv .venv-dialogtag
.venv-dialogtag/Scripts/python -m pip install -r analysis/requirements-dialogue-acts.txt
.venv-dialogtag/Scripts/python analysis/dialogue_acts.py --force-retag
```

The first model load downloads DialogTag's checkpoint.  Model-free reruns can then use the
text-free cache with the ordinary project environment.

Run the gold-only path with `python analysis/dialogue_acts.py --human-only`.  A full rerun can
reuse `dialogtag_predictions.json` without importing the model; pass `--force-retag` to invoke
DialogTag again.  See the module docstring and `--help` for paths and statistical controls.
"""
    with (output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(text)


def print_summary(comparison_rows: Sequence[dict]) -> None:
    by_condition: defaultdict[str, dict[str, dict]] = defaultdict(dict)
    for row in comparison_rows:
        by_condition[row["condition"]][row["label_set"]] = row
    print("\nDialogue-act structural-signature summary")
    header = (
        f"{'condition':12} {'fine JSD':>9} {'coarse JSD':>10} "
        f"{'coarse trans':>12} {'floor':>8} {'>floor':>7} "
        f"{'topic JSD':>10} {'assistant':>10}"
    )
    print(header)
    print("-" * len(header))
    for condition in sorted(by_condition):
        fine = by_condition[condition]["fine"]
        coarse = by_condition[condition]["coarse"]
        print(
            f"{condition:12} {float(fine['jsd_dist']):9.4f} "
            f"{float(coarse['jsd_dist']):10.4f} "
            f"{float(coarse['jsd_transition']):12.4f} "
            f"{float(coarse['noise_floor_p95']):8.4f} "
            f"{coarse['exceeds_floor']:>7} "
            f"{float(coarse['topic_matched_jsd_dist']):10.4f} "
            f"{float(coarse['assistant_register_fraction']):9.2%}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute fine and coarse dialogue-act structural signatures."
    )
    parser.add_argument(
        "--generated-root", type=pathlib.Path, default=GEN_ROOT,
        help=f"generated condition directory (default: {GEN_ROOT})",
    )
    parser.add_argument(
        "--swda-root", type=pathlib.Path, default=SWDA_ROOT,
        help=f"extracted SwDA directory (default: {SWDA_ROOT})",
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=OUT_DIR,
        help=f"output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--cache", type=pathlib.Path, default=None,
        help="prediction cache path (default: <out-dir>/dialogtag_predictions.json)",
    )
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-conversations", type=int, default=20)
    parser.add_argument("--bootstrap-size", type=int, default=50)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--force-retag", action="store_true",
        help="ignore a valid cache and invoke DialogTag again",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="do not read or write the prediction cache",
    )
    parser.add_argument(
        "--human-only", action="store_true",
        help="write the gold reference/noise floor without importing DialogTag",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="skip matplotlib figures",
    )
    return parser.parse_args(argv)


def validate_configuration() -> None:
    if set(FINE_LABELS) != set(FINE_TO_COARSE):
        missing = set(FINE_LABELS).difference(FINE_TO_COARSE)
        extra = set(FINE_TO_COARSE).difference(FINE_LABELS)
        raise RuntimeError(f"Invalid coarse map; missing={missing}, extra={extra}")
    if set(FINE_TO_COARSE.values()) != set(COARSE_LABELS):
        raise RuntimeError("Coarse map does not cover exactly COARSE_LABELS")
    if not set(DIALOGTAG_TO_FINE.values()).issubset(FINE_LABELS):
        raise RuntimeError("DialogTag mapping emits a label outside FINE_LABELS")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_configuration()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("--bootstrap-repetitions must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    output_dir = args.out_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (args.cache or output_dir / "dialogtag_predictions.json").resolve()

    print(f"Loading full gold SwDA reference from {args.swda_root}...")
    gold = load_switchboard(args.swda_root)
    print(
        f"Loaded {len(gold):,} conversations and "
        f"{sum(len(c.fine_labels) for c in gold):,} annotated units."
    )

    noise_floors: dict[str, float] = {}
    noise_rows: list[dict] = []
    for label_set in ("fine", "coarse"):
        floor, distances = bootstrap_noise_floor(
            gold,
            label_set,
            sample_size=args.bootstrap_size,
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )
        noise_floors[label_set] = floor
        noise_rows.append(
            {
                "label_set": label_set,
                "sample_size": args.bootstrap_size,
                "repetitions": args.bootstrap_repetitions,
                "seed": args.seed,
                "mean_jsd": f"{float(distances.mean()):.10f}",
                "p95_jsd": f"{floor:.10f}",
            }
        )
    write_csv(
        output_dir / "da_noise_floor.csv",
        noise_rows,
        ["label_set", "sample_size", "repetitions", "seed", "mean_jsd", "p95_jsd"],
    )

    gold_groups: dict[str, Sequence[Conversation]] = {"SB": gold}
    fine_rows = distribution_rows(gold_groups, "fine")
    coarse_rows = distribution_rows(gold_groups, "coarse")
    write_csv(
        output_dir / "da_distribution_by_condition.csv",
        fine_rows,
        ["condition", "n_conversations", "n_units", *FINE_LABELS],
    )
    write_csv(
        output_dir / "da_distribution_coarse_by_condition.csv",
        coarse_rows,
        ["condition", "n_conversations", "n_units", *COARSE_LABELS],
    )
    fine_sb_transition = transition_matrix(gold, "fine")
    coarse_sb_transition = transition_matrix(gold, "coarse")
    write_transition_csv(
        output_dir / "da_transition_SB.csv", fine_sb_transition, "fine"
    )
    write_transition_csv(
        output_dir / "da_transition_coarse_SB.csv", coarse_sb_transition, "coarse"
    )

    if args.human_only:
        if not args.no_figures:
            plot_key_acts(gold_groups, output_dir / "key_acts_grouped_bar.png")
            plot_sb_heatmap(coarse_sb_transition, output_dir / "sb_transition_heatmap.png")
        print("\nHuman-only run complete (DialogTag was not imported).")
        print(f"  fine noise floor p95  : {noise_floors['fine']:.6f}")
        print(f"  coarse noise floor p95: {noise_floors['coarse']:.6f}")
        print(f"Wrote gold outputs to {output_dir}")
        return

    generated = load_generated(args.generated_root)
    generated_groups = grouped_by_condition(generated)
    print(
        f"Loaded {len(generated):,} generated conversations across "
        f"{len(generated_groups)} conditions."
    )
    validation, cache_used = tag_generated_and_validate(
        generated,
        gold,
        model_name=args.model,
        batch_size=args.batch_size,
        validation_conversations=args.validation_conversations,
        seed=args.seed,
        cache_path=cache_path,
        force_retag=args.force_retag,
        use_cache=not args.no_cache,
    )
    print(f"Prediction source: {'validated cache' if cache_used else 'DialogTag model'}")
    print_validation(validation)

    all_groups: dict[str, Sequence[Conversation]] = {"SB": gold, **generated_groups}
    fine_rows = distribution_rows(all_groups, "fine")
    coarse_rows = distribution_rows(all_groups, "coarse")
    write_csv(
        output_dir / "da_distribution_by_condition.csv",
        fine_rows,
        ["condition", "n_conversations", "n_units", *FINE_LABELS],
    )
    write_csv(
        output_dir / "da_distribution_coarse_by_condition.csv",
        coarse_rows,
        ["condition", "n_conversations", "n_units", *COARSE_LABELS],
    )

    for condition, conversations in generated_groups.items():
        safe_name = safe_condition_name(condition)
        write_transition_csv(
            output_dir / f"da_transition_{safe_name}.csv",
            transition_matrix(conversations, "fine"),
            "fine",
        )
        write_transition_csv(
            output_dir / f"da_transition_coarse_{safe_name}.csv",
            transition_matrix(conversations, "coarse"),
            "coarse",
        )

    comparison_rows, _ = build_comparison_rows(
        gold, generated_groups, noise_floors
    )
    write_csv(
        output_dir / "da_jsd_vs_sb.csv",
        comparison_rows,
        [
            "condition", "label_set", "jsd_dist", "jsd_transition",
            "noise_floor_p95", "exceeds_floor", "topic_matched_jsd_dist",
            "topic_matched_jsd_transition", "assistant_register_fraction",
            "n_conversations", "n_units", "n_topic_matched_sb_conversations",
        ],
    )
    write_validation_outputs(output_dir, validation)
    assistant_rows = write_assistant_output(
        output_dir, gold, generated_groups
    )

    if not args.no_figures:
        plot_key_acts(all_groups, output_dir / "key_acts_grouped_bar.png")
        plot_sb_heatmap(coarse_sb_transition, output_dir / "sb_transition_heatmap.png")
        plot_jsd(comparison_rows, output_dir / "jsd_vs_sb.png")

    write_readme(
        output_dir,
        gold=gold,
        generated_groups=generated_groups,
        comparison_rows=comparison_rows,
        validation=validation,
        assistant_rows=assistant_rows,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_size=args.bootstrap_size,
        seed=args.seed,
        model_name=args.model,
        cache_used=cache_used,
    )
    print_summary(comparison_rows)
    print(f"\nWrote dialogue-act outputs to {output_dir}")


if __name__ == "__main__":
    main()
