"""
Turn-level text cleaning for generated dialogue (torch-free).

Raw model output is rarely a single clean turn. Models run on into the partner's turn,
leak speaker labels (including misspelled 4-bit variants such as ``ParticipentB:``), or
append chatbot residue such as ``[End of Response]`` or ``(Your turn count is 14)``. The
functions here turn raw output into exactly one spoken turn:

- :func:`clean_single_turn` keeps the first utterance and reports whether the model wrote
  more than one turn (a "multi-turn emission", recorded in every output file).
- :func:`strip_meta_artifacts` cuts a turn at the first piece of non-speech residue.
- :func:`looks_like_closing` detects farewells, used to end conversations naturally.

The same patterns also drive the generation-time stopping criteria in
:mod:`generation.model_utils`. Those criteria stop at exactly the point where these
functions would cut, so stopping early never changes the kept text.
"""

from __future__ import annotations

import re

LABELS = ("ParticipantA", "ParticipantB")


def clean_single_turn(text: str, labels=LABELS) -> tuple[str, bool]:
    """Return the first utterance and whether the model ran on past a single turn.

    Truncates at the first speaker/role marker. Besides the participant labels, this also
    catches chat-role residue the agent path leaks when Vicuna rambles into a whole fake
    dialogue (line-initial USER:, ASSISTANT:, and degraded 4-bit variants ASSISTATIVE: /
    ASSISTY:, plus HUMAN:/AI:/SYSTEM:/BOT:). A True flag here means the model did NOT keep
    to one turn, which we count as a multi-turn emission.
    """
    t = strip_lead_labels(text.strip())
    cut = _marker_in(t, labels)
    ran_past = cut is not None
    if ran_past:
        t = t[:cut]
    return t.strip().strip('"'), ran_past


def next_speaker_marker(text: str, labels=LABELS) -> int | None:
    """Position of the next-speaker marker in raw model output (None = still one turn)."""
    return _marker_in(strip_lead_labels(text.strip()), labels)


def _marker_in(t: str, labels) -> int | None:
    label_alt = "|".join(re.escape(label) for label in labels)
    # Tolerant participant label: catches degraded 4-bit variants like "ParticipantsA:"
    # (stray 's') and "Participant A:" (space) that the exact label misses. The C2
    # single-model path, where one model writes both speakers, leaks these often.
    fuzzy_label = r"Participants?\s*[AB]"
    marker_re = re.compile(
        rf"(?:\b(?:{label_alt}|{fuzzy_label})\s*:)"
        rf"|(?:(?:^|\n)\s*(?:USER|ASSIST\w*|HUMAN|AI|SYSTEM|BOT)\s*:)",
        re.IGNORECASE,
    )
    m = marker_re.search(t)
    return m.start() if m else None


# Leading label, matched aggressively. The single-model C2 path (one model writes BOTH
# speakers off a labelled transcript) emits a wide variety of degraded/misspelled speaker
# labels: "ParticipentB:", "ParticipB:", "Participation:", "ParticipANT_A:", the vocative
# "ParticipantB," (comma), even doubled "ParticipParticipant B:". Any participant/role
# prefix ending in a colon OR comma is peeled from the very START (repeatably). A trailing
# colon/comma is required, so ordinary words ("Part of...", "...every part: the cost") are
# never touched.
_LEAD_LABEL_RE = re.compile(
    r"^\s*(?:particip\w*|part(?:ner)?|user|assist\w*|human|ai|system|bot)"
    r"[\s_]*[ab]?\s*[:,]\s*",
    re.IGNORECASE,
)


def strip_lead_labels(t: str) -> str:
    """Remove any (possibly misspelled, possibly repeated) speaker label at the start."""
    prev = None
    while prev != t:
        prev = t
        t = _LEAD_LABEL_RE.sub("", t, count=1)
    return t


# Farewell / sign-off cues used to end a conversation naturally (see generation.turnwise).
_CLOSING_RE = re.compile(
    r"\b(?:good-?bye|bye-?bye|bye|take care|farewell|see you(?: around| soon| later| next time)?|"
    r"talk (?:to you )?(?:soon|later)|catch you later|until next time|happy chatting|"
    r"(?:nice|great|lovely|a pleasure) (?:talking|chatting|speaking)(?: (?:to|with) you)?|"
    r"enjoy (?:the rest of )?your day|"
    r"have a (?:great|good|nice|wonderful|lovely|fantastic) (?:day|one|time|evening|weekend))\b",
    re.IGNORECASE,
)

# Assistant / template / end-of-session residue a model emits once it drops out of the
# conversation. Every alternative below was observed in real outputs: "[End of Response]",
# "*Session closed.*", code fences, "Here's a summary", stray role tokens, and garbage like
# "** | **" / "-> |" / "V V V".
META_RE = re.compile(
    r"(?:"
    r"\[(?:end of|turn|tur|t\b|this|do you|assist|closed|/)"
    r"|\*{1,}\s*(?:conversation|chat|session|closed|ended|assistance|connection|end of)"
    r"|here'?s (?:the |a )?(?:quick )?(?:summary|recap)"
    r"|this (?:concludes|conversation (?:covered|concludes|ends))"
    r"|```"
    r"|(?:^|\n)\s*(?:USER|ASSISTANT|ASSISTMENT|SYSTEM|BOT)\b"
    r"|\*\*\s*\||\|\s*->|->\s*\||\bV\s+V\s+V\b"
    r"|(?:^|\n)\s*---\s*(?:$|\n)"
    # Parenthesized stage directions / prompt echoes and fake headings: "(Your turn count
    # is 68)", "(End of conversation.)", "## Example response:". Only these meta openers
    # match; ordinary parentheses in speech ("(IoT)", "401(k)", "(writing, drawing,
    # composing music)") never do.
    r"|\((?:your turn|end of (?:the )?conversation|the conversation (?:can|has|ends|is|so far)|"
    r"note:|i'll (?:now )?(?:wait|end)|here'?s a suggested|this is the final turn|"
    r"it'?s important to remember that these conversations)"
    r"|(?:^|\n)\s*#{1,6}\s"
    r"|\bexample response\s*:"
    # Prompt echo: fake "**Prompt:** ... **Reply:** ..." pairs that copy our own
    # instructions back ("Act like a 42-year-old...", "...turns of talk; do not end it too
    # early"). A leading digit is included so a turn that STARTS mid-echo ("0 turns of
    # talk...") is cut to empty rather than to a stray "0".
    r"|\*\*\s*(?:prompt|reply|response|answer|user|assistant|system)\s*:?\s*\*\*"
    r"|\bact like an? \d+-year-old"
    r"|\d*\s*\bturns of talk\b"
    r"|\bdo not end it too early"
    r"|\bthe topic of the conversation is\s*:"
    r"|\bthe conversation will have about"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def has_meta_artifact(text: str) -> bool:
    """Does raw output contain non-speech residue (searched as the turn cleaner sees it)?"""
    return META_RE.search(strip_lead_labels(text.strip())) is not None


def strip_meta_artifacts(text: str) -> str:
    """Cut a turn at the first assistant/template/end-of-conversation artifact.

    Keeps the clean leading part so chatbot residue never enters the transcript; if the
    whole turn was such residue this returns "" (the caller then ends the conversation).
    """
    m = META_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip().strip('"').strip()


def looks_like_closing(text: str) -> bool:
    """True if a turn contains a farewell / sign-off (used to stop the conversation)."""
    return bool(_CLOSING_RE.search(text))
