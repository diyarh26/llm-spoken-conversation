"""
Model loading + chat-generation helpers for the VM (Vicuna / Mistral).

4-bit quantized load to fit the V100 (16 GB). Uses each model's chat template so prompt
formatting is correct, with a Vicuna-v1.5 fallback if no template is shipped.

Requires (VM only): torch, transformers, accelerate, bitsandbytes.
"""

from __future__ import annotations

import re

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria,
    StoppingCriteriaList,
)

from generation.quality import is_near_duplicate  # shared with the post-hoc scorer

VICUNA = "lmsys/vicuna-13b-v1.5-16k"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.2"


def load_model(name: str, device: str | None = None):
    """Load a 4-bit quantized causal LM + tokenizer onto the GPU.

    device=None -> device_map="auto" (default; on a single-GPU box this puts the whole
    model on that GPU, on a multi-GPU box it splits layers to balance memory — correct for
    the V100). device="cuda:N" pins the ENTIRE model to one GPU — used only as a multi-GPU
    stopgap (e.g. C4 on the 2×M60 box: Vicuna on cuda:0, Mistral on cuda:1, so two models
    don't collide). Placement is output-neutral: same weights, same decoding, same result.
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(name)
    device_map = "auto" if device is None else {"": device}
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=bnb, device_map=device_map, use_safetensors=True
    )
    # Some chat checkpoints ship sampling fields with do_sample=False, which triggers
    # transformers warnings. Keep model defaults neutral; pass sampling choices per call.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()
    return model, tok


class SentenceEndStoppingCriteria(StoppingCriteria):
    """Stop once a short generated turn reaches a sentence boundary."""

    def __init__(self, tok, prompt_len: int, min_new_tokens: int = 8):
        self.tok = tok
        self.prompt_len = prompt_len
        self.min_new_tokens = min_new_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        new_ids = input_ids[0][self.prompt_len:]
        if new_ids.shape[-1] < self.min_new_tokens:
            return False
        text = self.tok.decode(new_ids, skip_special_tokens=True).strip()
        if not text:
            return False
        return text.endswith((".", "?", "!"))


@torch.inference_mode()
def chat(model, tok, messages, max_new_tokens=512, temperature=0.8, top_p=0.95,
         do_sample=True, stop_at_sentence=False, min_new_tokens=2,
         repetition_penalty=1.0, no_repeat_ngram_size=0) -> tuple[str, dict]:
    """messages: list of {role, content}. Returns (text, info).

    info = {"n_new_tokens": int, "hit_token_cap": bool} — cap-hits are logged by the
    generators because a bound cap truncates turn length, a measured DV.
    Defaults mirror generation/config.py (DV-safe: penalties off, no real token floor);
    per-run values come from the config/CLI, not from here.
    """
    try:
        encoded = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    except Exception:
        # Vicuna v1.5 may not ship a chat template — use its USER/ASSISTANT format.
        encoded = tok(_vicuna_format(messages), return_tensors="pt")

    if isinstance(encoded, torch.Tensor):
        model_inputs = {"input_ids": encoded.to(model.device)}
    else:
        model_inputs = {
            k: v.to(model.device) if hasattr(v, "to") else v
            for k, v in encoded.items()
        }
    input_len = model_inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        # min_new_tokens forbids the end-of-sequence token before this many tokens are
        # generated — this is what actually prevents 1-word "fragment" turns. It used to be
        # passed into this function but only fed the sentence-stop criteria, never generate(),
        # so there was no real floor on turn length. Now it is enforced.
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tok.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    if do_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    if stop_at_sentence:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
            SentenceEndStoppingCriteria(tok, input_len, min_new_tokens=min_new_tokens)
        ])

    out = model.generate(**model_inputs, **gen_kwargs)
    new_tokens = out[0][input_len:]
    info = {
        "n_new_tokens": int(new_tokens.shape[-1]),
        "hit_token_cap": int(new_tokens.shape[-1]) >= max_new_tokens,
    }
    return tok.decode(new_tokens, skip_special_tokens=True).strip(), info


def generate_turn(model, tok, messages, history, labels, *,
                  max_new_tokens, temperature, top_p, min_new_tokens,
                  stop_at_sentence, repetition_penalty, no_repeat_ngram_size,
                  dup_min_words=8, dup_jaccard=0.8, resample_temp_bump=0.15,
                  counters=None) -> str:
    """One clean conversational turn, with the procedural quality guards.

    Shared by the C2/C3/C4 turn-by-turn generators (single place, per the no-copy-paste
    rule). Pipeline: generate → truncate to one turn → strip chatbot residue → then
      - empty result: ONE fresh retry (models may emit instant EOS now that the token
        floor is gone); still empty = the model left the conversation, return "".
      - near-duplicate of an earlier turn (loop): ONE resample at temperature+bump —
        the procedural replacement for logit-level repetition penalties.
    `counters` (a dict) accumulates: multi_turn_emissions, hit_token_cap, empty_retries,
    dup_resamples, dup_kept — all recorded in the output JSON for the degeneration score.
    """
    counters = counters if counters is not None else {}

    def _once(temp: float) -> str:
        raw, info = chat(
            model, tok, messages,
            max_new_tokens=max_new_tokens, temperature=temp, top_p=top_p,
            min_new_tokens=min_new_tokens, stop_at_sentence=stop_at_sentence,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        counters["hit_token_cap"] = counters.get("hit_token_cap", 0) + int(info["hit_token_cap"])
        turn, ran_past = clean_single_turn(raw, labels)
        counters["multi_turn_emissions"] = counters.get("multi_turn_emissions", 0) + int(ran_past)
        return strip_meta_artifacts(turn)

    turn = _once(temperature)
    if not turn:
        counters["empty_retries"] = counters.get("empty_retries", 0) + 1
        turn = _once(temperature)
        if not turn:
            return ""
    if is_near_duplicate(turn, history, dup_min_words, dup_jaccard):
        counters["dup_resamples"] = counters.get("dup_resamples", 0) + 1
        retry = _once(temperature + resample_temp_bump)
        if retry and not is_near_duplicate(retry, history, dup_min_words, dup_jaccard):
            turn = retry
        else:
            counters["dup_kept"] = counters.get("dup_kept", 0) + 1
            turn = retry or turn
    return turn


def clean_single_turn(text: str, labels=("ParticipantA", "ParticipantB")) -> tuple[str, bool]:
    """Return the first utterance and whether the model ran on past a single turn.

    Truncates at the first speaker/role marker. Besides the participant labels, this also
    catches chat-role residue the agent path leaks when Vicuna rambles into a whole fake
    dialogue (line-initial USER:, ASSISTANT:, and degraded 4-bit variants ASSISTATIVE: /
    ASSISTY:, plus HUMAN:/AI:/SYSTEM:/BOT:). A True flag here means the model did NOT keep
    to one turn — which we count as a multi-turn emission.
    """
    label_alt = "|".join(re.escape(label) for label in labels)
    # Tolerant participant label: catches degraded 4-bit variants like "ParticipantsA:"
    # (stray 's') and "Participant A:" (space) that the exact label misses. The C2
    # single-model path — where one model writes both speakers — leaks these often.
    fuzzy_label = r"Participants?\s*[AB]"
    marker_re = re.compile(
        rf"(?:\b(?:{label_alt}|{fuzzy_label})\s*:)"
        rf"|(?:(?:^|\n)\s*(?:USER|ASSIST\w*|HUMAN|AI|SYSTEM|BOT)\s*:)",
        re.I,
    )
    # Leading label — aggressive. The single-model C2 path (one model writes BOTH speakers off a
    # labelled transcript) emits a wide variety of degraded/misspelled speaker labels:
    # "ParticipentB:", "ParticipB:", "Participation:", "ParticipANT_A:", the vocative
    # "ParticipantB," (comma), even doubled "ParticipParticipant B:". Peel any participant/role
    # prefix ending in a colon OR comma from the very START (repeatably). A trailing colon/comma
    # is required, so legitimate words ("Part of...", "...every part: the cost") are never touched.
    lead_label = re.compile(
        r"^\s*(?:particip\w*|part(?:ner)?|user|assist\w*|human|ai|system|bot)"
        r"[\s_]*[ab]?\s*[:,]\s*",
        re.I,
    )
    t = text.strip()
    prev = None
    while prev != t:
        prev = t
        t = lead_label.sub("", t, count=1)
    nxt = marker_re.search(t)
    ran_past = nxt is not None
    if ran_past:
        t = t[:nxt.start()]
    return t.strip().strip('"'), ran_past


# Farewell / sign-off cues used to end a conversation naturally (see generate_c3.py loop).
_CLOSING_RE = re.compile(
    r"\b(?:good-?bye|bye-?bye|bye|take care|farewell|see you(?: around| soon| later| next time)?|"
    r"talk (?:to you )?(?:soon|later)|catch you later|until next time|happy chatting|"
    r"(?:nice|great|lovely|a pleasure) (?:talking|chatting|speaking)(?: (?:to|with) you)?|"
    r"enjoy (?:the rest of )?your day|"
    r"have a (?:great|good|nice|wonderful|lovely|fantastic) (?:day|one|time|evening|weekend))\b",
    re.I,
)

# Assistant / template / end-of-session residue the model emits once it drops out of the
# conversation (observed in C3 tails: "[End of Response]", "*Session closed.*", code fences,
# "Here's a summary", stray role tokens, and garbage like "** | **" / "-> |" / "V V V").
_META_RE = re.compile(
    r"(?:"
    r"\[(?:end of|turn|tur|t\b|this|do you|assist|closed|/)"
    r"|\*{1,}\s*(?:conversation|chat|session|closed|ended|assistance|connection|end of)"
    r"|here'?s (?:the |a )?(?:quick )?(?:summary|recap)"
    r"|this (?:concludes|conversation (?:covered|concludes|ends))"
    r"|```"
    r"|(?:^|\n)\s*(?:USER|ASSISTANT|ASSISTMENT|SYSTEM|BOT)\b"
    r"|\*\*\s*\||\|\s*->|->\s*\||\bV\s+V\s+V\b"
    r"|(?:^|\n)\s*---\s*(?:$|\n)"
    r")",
    re.I | re.M,
)


def strip_meta_artifacts(text: str) -> str:
    """Cut a turn at the first assistant/template/end-of-conversation artifact.

    Keeps the clean leading part so chatbot residue never enters the transcript; if the whole
    turn was such residue this returns "" (the caller then ends the conversation).
    """
    m = _META_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip().strip('"').strip()


def looks_like_closing(text: str) -> bool:
    """True if a turn contains a farewell / sign-off (used to stop the conversation)."""
    return bool(_CLOSING_RE.search(text))


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

