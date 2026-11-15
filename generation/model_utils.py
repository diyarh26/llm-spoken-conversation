"""
Model loading and chat generation (the only generation module that needs a GPU).

Supports NF4, BF16 and FP16; use a GPU pilot to determine memory requirements.
Qwen3.5/Gemma4 use their native multimodal model and processor for text-only calls.
Only Vicuna may use the legacy ``USER:/ASSISTANT:`` fallback.

Requires: torch, transformers, accelerate, bitsandbytes (``pip install -e ".[generation]"``).
"""

from __future__ import annotations

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from generation.cleaning import has_meta_artifact, next_speaker_marker
from generation.response import spoken_response


def load_model(name: str, device: str | None = None, *, revision="main", precision="nf4"):
    """Load the selected model and its native tokenizer/processor onto the GPU.

    ``device=None`` uses ``device_map="auto"`` (the whole model on a single GPU, or split
    across several). ``device="cuda:N"`` pins the entire model to one GPU, which lets C4
    put its two models on separate cards. Placement is recorded in the run identity.
    """
    # Never generate on the CPU: if the GPU driver is lost, transformers silently loads the
    # model into RAM and a single conversation takes hours. Fail fast instead.
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible (check nvidia-smi); refusing to run on CPU.")
    if precision not in ("nf4", "bf16", "fp16"):
        raise ValueError(f"Unsupported precision: {precision}")
    config = AutoConfig.from_pretrained(name, revision=revision)
    multimodal = config.model_type in ("qwen3_5", "gemma4")
    if multimodal:
        loader = getattr(transformers, "AutoModelForMultimodalLM", None)
        if loader is None:
            raise RuntimeError(
                "This checkpoint needs AutoModelForMultimodalLM; install the workshop environment."
            )
        tok = transformers.AutoProcessor.from_pretrained(name, revision=revision)
    else:
        loader = AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if not getattr(tok, "chat_template", None) and not name.startswith("lmsys/vicuna-"):
        raise ValueError(
            f"No native chat template for {name}; plain base models need a separate experiment."
        )
    loading = {"dtype": torch.bfloat16 if precision == "bf16" else torch.float16}
    if precision == "nf4":
        loading["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    device_map = "auto" if device is None else {"": device}
    model = loader.from_pretrained(
        name, revision=revision, device_map=device_map, use_safetensors=True, **loading
    )
    if any(str(value) in ("cpu", "disk") for value in getattr(model, "hf_device_map", {}).values()):
        raise RuntimeError(
            "Model offloaded to CPU/disk; choose sufficient GPU memory before generation."
        )
    model._conversation_name = name
    model._conversation_family = config.model_type
    model._conversation_context = getattr(
        getattr(config, "text_config", config), "max_position_embeddings", None
    )
    # Some chat checkpoints ship sampling fields with do_sample=False, which triggers
    # transformers warnings. Keep model defaults neutral; sampling is passed per call.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()
    return model, tok


class _NewTextCriteria(StoppingCriteria):
    """Base class: decode only the newly generated tokens."""

    def __init__(self, tok, prompt_len: int):
        self.tok = tok
        self.prompt_len = prompt_len

    def new_text(self, input_ids) -> str:
        return self.tok.decode(input_ids[0][self.prompt_len :], skip_special_tokens=True)


class SentenceEndStoppingCriteria(_NewTextCriteria):
    """Stop once a short generated turn reaches a sentence boundary (off in the study)."""

    def __init__(self, tok, prompt_len: int, min_new_tokens: int = 8):
        super().__init__(tok, prompt_len)
        self.min_new_tokens = min_new_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids[0][self.prompt_len :].shape[-1] < self.min_new_tokens:
            return False
        text = self.new_text(input_ids).strip()
        return bool(text) and text.endswith((".", "?", "!"))


class SpeakerMarkerStoppingCriteria(_NewTextCriteria):
    """Stop as soon as the model starts writing the NEXT speaker's turn.

    Output-neutral: ``clean_single_turn`` cuts at exactly the same marker, so the kept turn
    is identical; this only skips tokens that would be thrown away. Without it, Vicuna
    often writes a whole fake dialogue up to ``max_new_tokens``.
    """

    def __init__(self, tok, prompt_len: int, labels):
        super().__init__(tok, prompt_len)
        self.labels = labels

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return next_speaker_marker(self.new_text(input_ids), self.labels) is not None


class MetaArtifactStoppingCriteria(_NewTextCriteria):
    """Stop as soon as the model starts writing non-speech residue.

    Output-neutral like :class:`SpeakerMarkerStoppingCriteria`: ``strip_meta_artifacts``
    cuts the turn at the same match.
    """

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return has_meta_artifact(self.new_text(input_ids))


@torch.inference_mode()
def chat(
    model,
    tok,
    messages,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
    stop_at_sentence=False,
    min_new_tokens=2,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    stop_labels=None,
) -> tuple[str, dict]:
    """Generate a reply to ``messages`` (a list of {role, content}). Returns (text, info).

    ``info = {"n_new_tokens": int, "hit_token_cap": bool}``. Cap hits are recorded by the
    generators because a binding cap would truncate turn length, a measured quantity.
    ``stop_labels``: speaker labels for turn-wise generation; generation stops once the
    next speaker's marker appears. None for C1, which writes the whole conversation.
    Defaults mirror ``generation/config.py``; per-run values come from the config/CLI.
    """
    template_kwargs = {}
    if getattr(model, "_conversation_family", "") in ("qwen3_5", "gemma4"):
        template_kwargs["enable_thinking"] = False
    if getattr(tok, "chat_template", None):
        encoded = tok.apply_chat_template(
            messages,
            # Processors (Qwen3.5/Gemma4) default to tokenize=False and return a string.
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **template_kwargs,
        )
    elif getattr(model, "_conversation_name", "").startswith("lmsys/vicuna-"):
        # Vicuna v1.5 may not ship a chat template: use its USER/ASSISTANT format.
        encoded = tok(_vicuna_format(messages), return_tensors="pt")
    else:
        raise ValueError(
            "Native chat template unavailable; refusing Vicuna fallback for this model."
        )

    if isinstance(encoded, torch.Tensor):
        model_inputs = {"input_ids": encoded.to(model.device)}
    else:
        model_inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v for k, v in encoded.items()
        }
    input_len = model_inputs["input_ids"].shape[-1]
    context = getattr(model, "_conversation_context", None)
    if context and input_len + max_new_tokens > context:
        raise ValueError(
            f"Input ({input_len}) plus output budget ({max_new_tokens}) exceeds context {context}."
        )
    tokenizer = getattr(tok, "tokenizer", tok)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        # Forbids the end-of-sequence token before this many tokens. The study uses 2,
        # which only blocks empty replies: a real floor would make backchannels impossible.
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "top_k": 0,  # neutralize model-specific generation defaults
        "min_p": 0.0,
    }
    if do_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    criteria = []
    if stop_at_sentence:
        criteria.append(SentenceEndStoppingCriteria(tok, input_len, min_new_tokens=min_new_tokens))
    if stop_labels:
        criteria.append(SpeakerMarkerStoppingCriteria(tok, input_len, stop_labels))
        criteria.append(MetaArtifactStoppingCriteria(tok, input_len))
    if criteria:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(criteria)

    out = model.generate(**model_inputs, **gen_kwargs)
    new_tokens = out[0][input_len:]
    raw = tok.decode(new_tokens, skip_special_tokens=False)
    text, leak = spoken_response(raw, tok.decode(new_tokens, skip_special_tokens=True))
    info = {
        "n_new_tokens": int(new_tokens.shape[-1]),
        "hit_token_cap": int(new_tokens.shape[-1]) >= max_new_tokens,
        "raw_output": raw,
        "reasoning_leak": leak,
        "input_tokens": input_len,
    }
    return text, info


def _vicuna_format(messages) -> str:
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    parts = [system] if system else []
    for m in messages:
        if m["role"] == "user":
            parts.append(f"USER: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"ASSISTANT: {m['content']}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)
