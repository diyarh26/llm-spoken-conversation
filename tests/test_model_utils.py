"""CPU mocks exercise loader selection and native chat calls without downloading weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def backend(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from generation import model_utils as module

    calls = []
    processor = SimpleNamespace(
        chat_template="native",
        pad_token_id=0,
        eos_token_id=9,
        apply_chat_template=lambda messages, **kwargs: (
            calls.append(("template", kwargs)) or {"input_ids": torch.tensor([[1, 2]])}
        ),
        decode=lambda tokens, **kwargs: "Yeah.",
    )
    model = SimpleNamespace(
        hf_device_map={"": 0},
        generation_config=SimpleNamespace(),
        device=torch.device("cpu"),
        eval=lambda: None,
        generate=lambda **kwargs: calls.append(("generate", kwargs)) or torch.tensor([[1, 2, 3]]),
    )

    def factory(label, value):
        return SimpleNamespace(from_pretrained=lambda *a, **kw: calls.append((label, kw)) or value)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "BitsAndBytesConfig", lambda **kw: kw)
    monkeypatch.setattr(module, "AutoTokenizer", factory("tokenizer", processor))
    monkeypatch.setattr(module, "AutoModelForCausalLM", factory("causal", model))
    monkeypatch.setattr(
        module,
        "transformers",
        SimpleNamespace(
            AutoProcessor=factory("processor", processor),
            AutoModelForMultimodalLM=factory("multimodal", model),
        ),
    )
    return module, calls, processor, model, factory


@pytest.mark.parametrize(
    ("family", "loader"), [("qwen3_5", "multimodal"), ("gemma4", "multimodal"), ("olmo3", "causal")]
)
@pytest.mark.parametrize("precision", ["nf4", "bf16"])
def test_selected_checkpoint_revision_precision_and_loader(
    backend, monkeypatch, family, loader, precision
):
    module, calls, _processor, _model, factory = backend
    monkeypatch.setattr(module, "AutoConfig", factory("config", SimpleNamespace(model_type=family)))
    module.load_model("org/checkpoint", revision="frozen-sha", precision=precision)
    options = next(value for label, value in calls if label == loader)
    assert options["revision"] == "frozen-sha"
    assert ("quantization_config" in options) == (precision == "nf4")
    assert ("processor" in [label for label, _ in calls]) == (loader == "multimodal")


def test_native_template_disables_thinking_and_uses_neutral_sampling(backend):
    module, calls, processor, model, _ = backend
    model._conversation_family = "gemma4"
    result, info = module.chat(model, processor, [{"role": "user", "content": "Hello"}])
    assert result == "Yeah." and not info["reasoning_leak"]
    template = next(value for label, value in calls if label == "template")
    generated = next(value for label, value in calls if label == "generate")
    assert template["enable_thinking"] is False
    assert generated["top_k"] == 0 and generated["min_p"] == 0.0
    assert info["raw_output"] == "Yeah."


def test_broken_native_template_never_falls_back_to_vicuna(backend):
    module, _, processor, model, _ = backend

    def broken(*args, **kwargs):
        raise ValueError("bad roles")

    processor.apply_chat_template = broken
    with pytest.raises(ValueError, match="bad roles"):
        module.chat(model, processor, [{"role": "user", "content": "hello"}])
    processor.chat_template = None
    with pytest.raises(ValueError, match="refusing Vicuna fallback"):
        module.chat(model, processor, [{"role": "user", "content": "hello"}])


def test_context_overflow_fails_without_truncating_history(backend):
    module, _, processor, model, _ = backend
    model._conversation_context = 4
    with pytest.raises(ValueError, match="exceeds context"):
        module.chat(model, processor, [{"role": "user", "content": "hello"}], max_new_tokens=3)
