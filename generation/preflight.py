"""Check workshop data and native model templates on CPU, without downloading weights."""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analysis.swda import fewshot_examples, load_metadata, make_personas
from generation.run_context import ROOT, atomic_json, package_versions
from generation.sampling import load_manifest
from prompts.templates import assign_cards, build_agent, build_c1, build_c2

OWNERS = {
    "diyar": ("allenai/Olmo-3-7B-Instruct", "Qwen/Qwen3.5-9B"),
    "sham": ("Qwen/Qwen3.5-9B", "google/gemma-4-31B-it"),
    "aleen": ("google/gemma-4-31B-it", "allenai/Olmo-3-7B-Instruct"),
}


def check_data():
    manifest = load_manifest()
    metadata = load_metadata()
    ids = manifest["target_ids"] + manifest["dev_ids"]
    if len(manifest["target_ids"]) != 50 or set(manifest["target_ids"]) & set(manifest["dev_ids"]):
        raise ValueError("Expected 50 production IDs disjoint from development IDs.")
    for cno in ids:
        if cno not in metadata:
            raise ValueError(f"Missing metadata: {cno}")
        examples = fewshot_examples(cno, k=2)
        if len(examples) != 2 or not all(e["text"].strip() for e in examples):
            raise ValueError(f"Missing P2 examples: {cno}")
    return manifest, metadata


def check_template(name, manifest, metadata):
    import transformers
    from huggingface_hub import HfApi

    revision = HfApi().model_info(name).sha
    config = transformers.AutoConfig.from_pretrained(name, revision=revision)
    multimodal = config.model_type in ("qwen3_5", "gemma4")
    loader = (
        transformers.AutoModelForMultimodalLM if multimodal else transformers.AutoModelForCausalLM
    )
    if type(config) not in loader._model_mapping:
        raise ValueError(f"Installed Transformers cannot load {name}: {type(config).__name__}")
    processor = (
        transformers.AutoProcessor if multimodal else transformers.AutoTokenizer
    ).from_pretrained(name, revision=revision)
    if not processor.chat_template:
        raise ValueError(f"Missing native template: {name}")
    cno = manifest["dev_ids"][0]
    a, b, topic, prompt = make_personas(metadata[cno])
    assign_cards(a, b, cno)
    history = [("ParticipantA", "Hello!"), ("ParticipantB", "Hello!")]
    counts = []
    for level in ("P0", "P1", "P2"):
        views = [
            build_c1(level, a, b, topic, prompt, conversation_no=cno),
            build_c2(level, a, b, topic, prompt, history, "ParticipantA", 40, cno),
            build_agent(level, a, b, topic, prompt, history, 40, cno),
            build_agent(level, b, a, topic, prompt, history, 40, cno),
        ]
        for messages in views:
            kwargs = {"enable_thinking": False} if multimodal else {}
            tokens = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **kwargs,
            )
            counts.append(tokens["input_ids"].shape[-1])
    return {
        "name": name,
        "revision": revision,
        "template_checks": len(counts),
        "max_prompt_tokens": max(counts),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", choices=OWNERS, required=True)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args(argv)
    manifest, metadata = check_data()
    report = {"owner": args.owner, "data_ok": True, "packages": package_versions(), "models": []}
    if not args.data_only:
        for name in OWNERS[args.owner]:
            report["models"].append(check_template(name, manifest, metadata))
            print(f"Native template and model-class checks passed: {name}")
    destination = ROOT / "_private" / f"preflight-{args.owner}.json"
    atomic_json(destination, report)
    print(f"Preflight passed; private report: {destination}")
    print(
        "No weights downloaded. GPU loading, memory, and generation still require a development pilot."
    )


if __name__ == "__main__":
    main()
