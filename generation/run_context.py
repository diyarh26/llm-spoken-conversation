"""Run identity, private diagnostics and deterministic resume for new generation runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import random
import subprocess
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent


def add_runtime_args(parser):
    parser.add_argument("--precision", choices=("nf4", "bf16", "fp16"), default="nf4")
    parser.add_argument("--revision", default="main", help="model A/single-model revision")
    parser.add_argument("--revision-b", default="main", help="model B revision (C4)")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--thinking", choices=("off",), default="off")
    parser.add_argument("--device", default=None, help="single-model placement, e.g. cuda:0")
    parser.add_argument(
        "--preflight-only", action="store_true", help="check corpus without loading weights"
    )


def atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def source_hash():
    digest = hashlib.sha256()
    for folder in ("generation", "prompts"):
        for path in sorted((ROOT / folder).glob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    for relative in (
        "generation/target_ids.json",
        "generation/fewshot_pool.json",
        "analysis/swda.py",
    ):
        digest.update((ROOT / relative).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def package_versions():
    versions = {}
    for name in ("torch", "transformers", "accelerate", "bitsandbytes", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def resolve_models(models, args):
    """Resolve mutable HF refs once and load weights using the returned immutable SHA."""
    from huggingface_hub import HfApi

    api = HfApi()
    result = {}
    cache = {}
    for index, (speaker, name) in enumerate(models.items()):
        revision = (
            args.revision if index == 0 or len(set(models.values())) == 1 else args.revision_b
        )
        key = (name, revision)
        if key not in cache:
            cache[key] = api.model_info(name, revision=revision).sha
        result[speaker] = {"name": name, "revision": cache[key]}
    return result


def bind_run(job, args, models, decoding):
    """Bind a condition directory to its exact model/settings/environment before loading."""
    if job.out_dir.resolve().is_relative_to((ROOT / "data" / "generated").resolve()):
        raise ValueError("Original released data is protected. Pass a different --out-root.")
    resolved = resolve_models(models, args)
    # For same-model agents use one revision, even when --revision-b was not specified.
    if len(set(models.values())) == 1:
        first = next(iter(resolved.values()))
        resolved = dict.fromkeys(models, first)
    identity = {
        "condition": job.condition,
        "models": resolved,
        "precision": args.precision,
        "thinking": args.thinking,
        "sampling_defaults": {"top_k": 0, "min_p": 0.0, "do_sample": True},
        "seed": args.seed,
        "decoding": decoding,
        "source_sha256": source_hash(),
        "packages": package_versions(),
        "devices": {
            "single": args.device,
            "a": getattr(args, "device_a", None),
            "b": getattr(args, "device_b", None),
        },
        "input_sha256": job.input_hash,
    }
    path = job.out_dir / "run_manifest.meta"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError(f"Run identity differs in {path}; use a new output root.")
        for cno in job.target_ids:
            output = job.out_dir / f"{cno}.json"
            if output.exists():
                record = json.loads(output.read_text(encoding="utf-8"))
                if (
                    record.get("run_id") != previous["run_id"]
                    or record.get("conversation_no") != cno
                ):
                    raise ValueError(f"Invalid/mixed completed record: {output}")
        metadata = previous
    else:
        if any(p.stem.isdigit() for p in job.out_dir.glob("*.json")):
            raise ValueError("Existing conversations have no run manifest; use a new output root.")
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        metadata = {
            "run_id": run_id,
            "identity": identity,
            "git_commit": commit,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(path, metadata)
    job.run_metadata = metadata
    return resolved


def seed_conversation(job, args, conversation_no):
    import torch

    value = f"{args.seed}:{job.condition}:{conversation_no}"
    seed = int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if "hardware" not in job.run_metadata:
        job.run_metadata["hardware"] = {
            "cuda": torch.version.cuda,
            "gpus": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    return seed


def traced_engine(engine, job, conversation_no, speaker):
    """Keep every call and retry, including prompts, in gitignored private storage."""
    run_id = job.run_metadata["run_id"]
    path = ROOT / "_private" / "generation_raw" / run_id / f"{conversation_no}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    def call(messages, **kwargs):
        response, info = engine(messages, **kwargs)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "speaker": speaker,
                        "messages": messages,
                        "decoding": kwargs,
                        "response": response,
                        "info": info,
                    }
                )
                + "\n"
            )
        if info.get("reasoning_leak"):
            raise RuntimeError(
                f"Unexpected reasoning in non-thinking run; raw response saved to {path}"
            )
        return response, info

    return call
