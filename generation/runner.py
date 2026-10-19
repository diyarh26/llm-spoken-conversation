"""
Command-line interface and resumable-job plumbing shared by the four generators.

Each generator script only defines what makes its architecture unique (who writes which
turn, from which point of view). Target selection, persona assignment, resumability and
the output format are defined once, here.

Resumability: every conversation is written to ``<out-root>/<condition>/<id>.json`` as
soon as it finishes, and ids whose file already exists are skipped. A crashed or
disconnected multi-day run therefore loses at most the conversation in flight.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import pathlib
from collections.abc import Callable
from typing import Any

from analysis.swda import load_metadata, make_personas
from generation.cleaning import LABELS
from generation.config import MAX_TURNS, TURNWISE
from generation.sampling import load_target_ids
from generation.turnwise import SEED_TURNS, Turn, generate_turn, run_conversation
from prompts.templates import Persona, assign_cards

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "generated"
PROMPT_LEVELS = ("P0", "P1", "P2", "P3")

# (speaker label, persona A, persona B, topic, SB prompt, history, conversation_no) -> messages
MessageBuilder = Callable[[str, Persona, Persona, str, "str | None", list[Turn], int], list[dict]]
# A model bound to its tokenizer: (messages, **decoding) -> (raw text, info)
Engine = Callable[..., tuple[str, dict]]


# --- command line --------------------------------------------------------------------


def base_parser(doc: str | None) -> argparse.ArgumentParser:
    """Arguments every generator accepts: prompt level, which ids, where to write."""
    ap = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prompt", default="P0", choices=PROMPT_LEVELS)
    ap.add_argument(
        "--n",
        type=int,
        default=0,
        help="generate only the first N manifest ids (0 = all; any prefix is ~stratified)",
    )
    ap.add_argument(
        "--ids",
        default="",
        help="comma-separated conversation_no's to (re)generate; overrides the manifest",
    )
    ap.add_argument("--out-root", default=str(OUT_ROOT), help="output root directory")
    return ap


def add_turnwise_args(ap: argparse.ArgumentParser) -> None:
    """Decoding flags shared by C2/C3/C4. Defaults are the frozen generation/config.py.

    The flags exist for the decoding dev sweep; the full run used the defaults unchanged.
    """
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--max-new-tokens", type=int, default=TURNWISE.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=TURNWISE.temperature)
    ap.add_argument("--top-p", type=float, default=TURNWISE.top_p)
    ap.add_argument(
        "--min-new-tokens",
        type=int,
        default=TURNWISE.min_new_tokens,
        help="blocks empty emissions only; a real floor would forbid backchannels",
    )
    ap.add_argument(
        "--stop-at-sentence",
        dest="stop_at_sentence",
        action="store_true",
        default=TURNWISE.stop_at_sentence,
        help="off by default: forcing sentence ends would forbid abandoned turns",
    )
    ap.add_argument("--no-stop-at-sentence", dest="stop_at_sentence", action="store_false")
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=TURNWISE.repetition_penalty,
        help="1.0 = off; loops are handled procedurally (near-duplicate resample)",
    )
    ap.add_argument("--no-repeat-ngram", type=int, default=TURNWISE.no_repeat_ngram_size)


def turnwise_decoding(args: argparse.Namespace) -> dict[str, Any]:
    """The decoding parameters recorded in every turn-wise output file."""
    return {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
        "stop_at_sentence": args.stop_at_sentence,
        "max_turns": args.max_turns,
    }


# --- jobs ----------------------------------------------------------------------------


@dataclasses.dataclass
class Job:
    """One condition's run: which conversations are still missing and where they go."""

    condition: str
    out_dir: pathlib.Path
    todo: list[int]
    meta: dict[int, dict]

    def speakers(self, conversation_no: int) -> tuple[Persona, Persona, str, str]:
        """Personas (demographics from Switchboard + seeded persona cards) and the topic."""
        a, b, topic, sb_prompt = make_personas(self.meta[conversation_no])
        assign_cards(a, b, conversation_no)  # only the P1/P2 prompts render the cards
        return a, b, topic, sb_prompt

    def save(self, conversation_no: int, record: dict) -> pathlib.Path:
        path = self.out_dir / f"{conversation_no}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return path


def prepare_job(architecture: str, args: argparse.Namespace) -> Job:
    """Resolve the target ids for one condition and skip those already on disk."""
    condition = f"{architecture}-{args.prompt}"
    out_dir = pathlib.Path(args.out_root) / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata()
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        ids = load_target_ids()
        if args.n:
            ids = ids[: args.n]
    ids = [c for c in ids if c in meta]
    todo = [c for c in ids if not (out_dir / f"{c}.json").exists()]
    print(f"[{condition}] target={len(ids)} todo={len(todo)} done={len(ids) - len(todo)}")
    return Job(condition, out_dir, todo, meta)


def run_turnwise_job(
    job: Job,
    args: argparse.Namespace,
    *,
    engines: dict[str, Engine],
    build_messages: MessageBuilder,
    model_info: dict[str, Any],
) -> None:
    """Generate and save every pending conversation of a turn-wise condition.

    ``engines`` maps each speaker label to the model that speaks for it (the same engine
    for both speakers in C2/C3, two different models in C4). ``build_messages`` is the
    architecture: it decides what each speaker's model sees.
    """
    decoding = {
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "min_new_tokens": args.min_new_tokens,
        "stop_at_sentence": args.stop_at_sentence,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram,
    }
    for cno in job.todo:
        a, b, topic, sb_prompt = job.speakers(cno)
        history, ended_by, counters = _converse(
            args, engines, build_messages, decoding, a, b, topic, sb_prompt, cno
        )
        record = {
            "condition": job.condition,
            "architecture": job.condition.split("-")[0],
            "prompt_level": args.prompt,
            **model_info,
            "conversation_no": cno,
            "topic": topic,
            "sb_prompt": sb_prompt,
            "persona_a": dataclasses.asdict(a),
            "persona_b": dataclasses.asdict(b),
            "turns": history,
            "n_turns": len(history),
            "seed_turns": len(SEED_TURNS),
            "ended_by": ended_by,  # mutual_closing / closing / empty_turn / turn_cap
            "multi_turn_emissions": counters.get("multi_turn_emissions", 0),
            "decoding": turnwise_decoding(args),
            "quality_counters": counters,  # also: cap hits, empty retries, dup resamples
        }
        job.save(cno, record)
        print(f"  saved {cno}  turns={len(history)}  counters={counters}")


def _converse(args, engines, build_messages, decoding, a, b, topic, sb_prompt, cno):
    """One conversation: returns (turns, ended_by, quality counters)."""
    counters: dict[str, int] = {}

    def speak(speaker: str, history: list[Turn]) -> str:
        messages = build_messages(speaker, a, b, topic, sb_prompt, history, cno)
        chat_fn = functools.partial(engines[speaker], messages, stop_labels=LABELS, **decoding)
        return generate_turn(
            lambda temperature: chat_fn(temperature=temperature),
            history,
            LABELS,
            temperature=args.temperature,
            counters=counters,
        )

    history, ended_by = run_conversation(speak, args.max_turns)
    return history, ended_by, counters
