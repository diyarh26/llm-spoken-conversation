"""
Seeded, topic-stratified sampling of Switchboard conversation_no's for generation.

Replaces the old "first-N in sorted file order" convenience sample (which over-represented
whatever topics happen to sort first). The draw is:

  1. Pool = the SwDA conversations that exist on disk AND have metadata (1,155).
  2. Proportional stratification by `topic_description` with largest-remainder rounding,
     so the sample's topic mix mirrors the full corpus (this is what makes the
     topic-matched human comparison in Track 2 fair).
  3. Seeded RNG within each topic; the final list is seed-shuffled so any PREFIX of it is
     itself a roughly stratified subsample (useful for pilots: `--n 10` = first 10 ids).
  4. A small disjoint DEV set is drawn from the leftover pool for decoding tuning.
     Dev conversations are never generation targets and never the P2 few-shot excerpt.

The draw is written ONCE to the committed manifest `generation/target_ids.json`; every
condition (C1–C4 × P0–P2) generates the SAME ids so architecture×prompt comparisons stay
paired. Generators read the manifest via `load_manifest()` — they do not re-sample.

Regenerate the manifest (only if the design changes; the file is committed):
    python generation/sampling.py --n 50 --dev 5 --seed 20260714 --write
Inspect without writing:
    python generation/sampling.py --n 50 --dev 5 --seed 20260714
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.swda import (                                              # noqa: E402
    load_metadata, iter_conversation_files, conversation_no_of,
)

MANIFEST = pathlib.Path(__file__).resolve().parent / "target_ids.json"
DEFAULT_SEED = 20260714   # date the sampling design was decided; do not change casually
DEFAULT_N = 50
DEFAULT_DEV = 5


def _pool_by_topic(meta: dict) -> dict[str, list[int]]:
    """topic_description -> sorted conversation_no's that exist on disk with metadata."""
    by_topic: dict[str, list[int]] = defaultdict(list)
    for fp in iter_conversation_files():
        cno = conversation_no_of(fp)
        if cno in meta:
            by_topic[meta[cno]["topic_description"]].append(cno)
    return {t: sorted(ids) for t, ids in sorted(by_topic.items())}


def stratified_sample(n: int, seed: int, meta: dict) -> tuple[list[int], list[int], dict]:
    """Return (target_ids, dev_ids, topic_allocation). Deterministic for a given seed."""
    by_topic = _pool_by_topic(meta)
    total = sum(len(v) for v in by_topic.values())
    rng = random.Random(seed)

    # Largest-remainder proportional allocation of n across topics (ties broken by name).
    quotas = {t: n * len(ids) / total for t, ids in by_topic.items()}
    alloc = {t: int(q) for t, q in quotas.items()}
    short = n - sum(alloc.values())
    for t in sorted(quotas, key=lambda t: (-(quotas[t] - alloc[t]), t))[:short]:
        alloc[t] += 1

    target: list[int] = []
    for t, k in alloc.items():
        if k > 0:
            target.extend(rng.sample(by_topic[t], k))
    rng.shuffle(target)  # so a prefix of the list is still ~stratified (pilot runs)

    leftovers = sorted(set().union(*by_topic.values()) - set(target))
    dev = rng.sample(leftovers, DEFAULT_DEV) if leftovers else []

    allocation = {t: k for t, k in alloc.items() if k > 0}
    return target, dev, allocation


def load_manifest() -> dict:
    """The committed sampling manifest. Generators call this instead of sampling."""
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_target_ids() -> list[int]:
    return load_manifest()["target_ids"]


def load_dev_ids() -> list[int]:
    return load_manifest()["dev_ids"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--dev", type=int, default=DEFAULT_DEV)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--write", action="store_true",
                    help="write generation/target_ids.json (otherwise just print the draw)")
    args = ap.parse_args()

    meta = load_metadata()
    target, dev, allocation = stratified_sample(args.n, args.seed, meta)
    dev = dev[:args.dev]

    manifest = {
        "seed": args.seed,
        "n": len(target),
        "method": "topic-proportional stratified, largest-remainder rounding, "
                  "seeded RNG within topic, seed-shuffled order (prefix ~ stratified)",
        "target_ids": target,
        "dev_ids": dev,
        "topic_allocation": allocation,
        "target_topics": sorted({meta[c]["topic_description"] for c in target}),
    }
    print(f"pool topics={len(allocation)} target={len(target)} dev={dev}")
    for t, k in sorted(allocation.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {k:2d}  {t}")
    if args.write:
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"wrote {MANIFEST}")
    else:
        print("(dry run — pass --write to save the manifest)")


if __name__ == "__main__":
    main()
