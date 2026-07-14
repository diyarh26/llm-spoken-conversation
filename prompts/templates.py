"""
Prompt and message builders for the conversation-generation experiment.

Architectures
-------------
C1  all-at-once          : one model writes the entire dialogue in a single call.
C2  turn-by-turn single  : one model sees the whole transcript and writes the next turn for
                           the named speaker — replicates the paper's GPT4-1 setup.
C3  two agents, same     : two independent first-person sessions of the SAME model.
C4  two agents, different : two independent first-person sessions of DIFFERENT models.

Prompt levels
-------------
P0  basic / replication : faithfully follows the paper's BASIC prompt (GPT4-1 wording —
                          "act like a {persona}", topic = the verbatim SB instruction, ~50
                          turns). No marker forcing, no scripted openings/closings, no guard.
P1  spoken intervention : OUR prompt — spoken register + brevity, natural ending, peer/anti-
                          assistant guard, and a seeded PERSONA CARD (occupation, stance
                          toward the topic, one life anchor). Design principle (2026-07-14):
                          EVOKE behavior through persona and situation; NEVER instruct a
                          behavior we measure. No dialogue act, coordination marker, or topic
                          behavior is ever named — stories/disagreement/tangents must emerge
                          from giving the speakers something to say, not from naming the acts.
                          (Turn length is the one instructed exception, reported as such.)
P2  few-shot            : P1 + ONE real Switchboard excerpt (from a DIFFERENT topic) shown as
                          a style example. NOTE: for P2, turn-length and conceptual-alignment
                          effects are the valid story; lexical alignment and specific marker
                          rates are partly example-driven and must be read as confounded.

We deliberately do NOT replicate the paper's *enhanced* prompts (which order "use okay/oh/
uh-huh" and script openings/closings, then measure exactly those). The generators seed every
turn-by-turn dialogue with a mutual "Hello!" opening, as the paper did.
"""

import random
from dataclasses import dataclass


@dataclass
class Persona:
    label: str        # "ParticipantA" / "ParticipantB"
    gender: str       # "man" / "woman"
    age: int
    education: str    # e.g. "a college degree"
    # Seeded persona-card fields (P1/P2 only; P0 stays the paper's bare demographics).
    occupation: str | None = None
    stance: str | None = None      # index into _STANCES at assignment time; stored as text
    anchor: str | None = None

    def describe(self) -> str:
        return f"a {self.age}-year-old {self.gender} with {self.education}"

    def card_first_person(self) -> str:
        """P1/P2 agent view: 'You work as ... You have mixed feelings ...'."""
        if not self.occupation:
            return ""
        return (f"You work as {self.occupation}. {_STANCES[self.stance][0]} "
                f"{_ANCHORS[self.anchor][0]}")

    def card_third_person(self) -> str:
        """P1/P2 writer view (C1/C2): '... works as ..., has mixed feelings ...'."""
        if not self.occupation:
            return ""
        return (f"{self.label} works as {self.occupation}, "
                f"{_STANCES[self.stance][1]}, and {_ANCHORS[self.anchor][1]}.")


# Persona-card pools, (first-person, third-person) pairs. Attitude and content only —
# no interactional verbs (no "react", "agree", "ask", "interrupt"), so no dialogue act
# is ever prompted. Disagreement etc. can only EMERGE from stance combinations, which
# keeps the DA metric non-circular.
_OCCUPATIONS = [
    "a school bus driver", "a nurse", "an electrician", "a grocery store manager",
    "a high-school teacher", "an accountant", "a car mechanic", "a restaurant server",
    "a farmer", "an office receptionist", "a carpenter", "a pharmacist",
    "a real estate agent", "a factory supervisor", "a librarian", "a plumber",
    "a bank teller", "a truck driver", "a dental hygienist", "an insurance adjuster",
]
_STANCES = [
    ("You find this topic genuinely interesting and have plenty of opinions about it.",
     "finds the topic genuinely interesting with plenty of opinions about it"),
    ("You have mixed feelings about this topic — parts of it sit fine with you, parts don't.",
     "has mixed feelings about the topic — parts of it sit fine, parts don't"),
    ("Privately, you are a bit skeptical about this topic.",
     "is privately a bit skeptical about the topic"),
]
_ANCHORS = [
    ("You have a personal experience related to it that you sometimes tell people about.",
     "has a personal experience related to it that they sometimes tell people about"),
    ("Someone close to you has dealt with it, which shaped how you see it.",
     "knows someone close who has dealt with it, which shaped their view"),
    ("You made up your mind about it years ago, though you would struggle to say exactly why.",
     "made up their mind about it years ago without quite being able to say why"),
]


def assign_cards(a: "Persona", b: "Persona", conversation_no: int) -> None:
    """Fill both personas' card fields, seeded by (conversation_no, label).

    Deterministic across machines and reruns (random.Random hashes str seeds with
    sha512, independent of PYTHONHASHSEED). Stances are drawn independently, so
    ~2/3 of pairs hold different stances and disagreement can emerge unscripted.
    Stores pool indices (stance/anchor) so the draw is documented in the output JSON.
    """
    for p in (a, b):
        rng = random.Random(f"card:v3:{conversation_no}:{p.label}")
        p.occupation = rng.choice(_OCCUPATIONS)
        p.stance = rng.randrange(len(_STANCES))
        p.anchor = rng.randrange(len(_ANCHORS))


def render_transcript(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{spk}: {txt}" for spk, txt in history)


# --- shared prompt pieces ------------------------------------------------------------

def _length_clause() -> str:
    return "The conversation will have about 30 turns of talk; do not end it too early."


def _p1_style() -> str:
    return (
        "Speak casually and informally, the way people actually talk out loud on the phone. "
        "Keep each turn short — often just a sentence or two. Let the conversation end "
        "naturally when it feels finished; do not pad or stretch it out."
    )


def _turn_status(history: list, max_turns: int | None) -> str:
    """Supervisor fix (2026-07-14): tell the model where it is in the turn budget, so it
    can bring the call to a natural close before the hard cap instead of being cut off
    mid-conversation. Phrased as position only — it never says HOW to close.
    NOTE for analysis: this makes ending behavior harness-assisted in every turn-by-turn
    condition; closing-act results must be reported with that caveat."""
    if not max_turns:
        return ""
    return (f"\n(The conversation so far has {len(history)} turns of talk; "
            f"it cannot go past {max_turns} turns, so it has to be over by then.)")


def _peer_guard() -> str:
    return (
        "You are an ordinary member of the public — NOT an AI assistant, agent, operator, or a "
        "representative of any company or service. The other caller is also an ordinary person; "
        "the two of you are equals who were each asked to phone a stranger and chat. Never say "
        "things like 'how can I help you' or 'is this the ... service', and do not offer help "
        "or recommendations as if it were your job — just share your own views and ask about "
        "theirs, the way two regular people would."
    )


def _topic_clause(topic: str, sb_prompt: str | None, level: str) -> str:
    """How the Switchboard task is presented (plain for P0, reframed as peer goal otherwise)."""
    instruction = sb_prompt or topic
    if level == "P0":
        return f"The topic of the conversation is: {instruction}"
    return (
        f"You were both asked to chat about this topic: {topic}. Shared discussion goal (talk "
        f"it over together as equals, comparing your own experiences and opinions): {instruction}"
    )


_FEWSHOT_EXAMPLE: str | None = None


def _fewshot_block(level: str) -> str:
    """For P2 only: a real Switchboard excerpt (different topic) shown as a style example."""
    global _FEWSHOT_EXAMPLE
    if level != "P2":
        return ""
    if _FEWSHOT_EXAMPLE is None:
        try:
            from analysis.swda import fewshot_example
            try:
                # Source the excerpt outside the sampled target/dev ids AND their topics,
                # so P2's style example never overlaps what we generate or compare against.
                from generation.sampling import load_manifest
                m = load_manifest()
                excl = set(m["target_ids"]) | set(m["dev_ids"])
                _FEWSHOT_EXAMPLE = (
                    fewshot_example(exclude_ids=excl,
                                    exclude_topics=set(m["target_topics"]))
                    # if no long-enough conversation exists outside the target topics,
                    # relax to id-exclusion only rather than losing the example
                    or fewshot_example(exclude_ids=excl)
                )
            except Exception:
                _FEWSHOT_EXAMPLE = fewshot_example()  # legacy skip-80 fallback
        except Exception:
            _FEWSHOT_EXAMPLE = ""
    if not _FEWSHOT_EXAMPLE:
        return ""
    return (
        "\n\nHere is an example of a real, natural telephone conversation between two strangers "
        "on a DIFFERENT topic, to show the spoken style (short, casual back-and-forth):\n"
        f"{_FEWSHOT_EXAMPLE}\n(End of example.)\n"
    )


# --- C1: all at once -----------------------------------------------------------------

def build_c1(prompt_level: str, a: Persona, b: Persona, topic: str,
             sb_prompt: str | None = None) -> list[dict]:
    if prompt_level == "P0":
        prompt = (
            "Write the log of a telephone conversation between two people who do not know each "
            f"other and have equal roles in the discussion. {a.label} is {a.describe()}. "
            f"{b.label} is {b.describe()}. {_topic_clause(topic, sb_prompt, 'P0')} "
            f"{_length_clause()} The log starts with:\n{a.label}: Hello!\n{b.label}: Hello!\n"
            "Each line is one turn beginning with the speaker's label and a colon. Write the "
            "complete conversation, continuing from those greetings."
        )
    else:  # P1 / P2
        prompt = (
            "Write a realistic telephone conversation between two ordinary people who do not "
            f"know each other. {a.label} is {a.describe()}. {b.label} is {b.describe()}. "
            f"{a.card_third_person()} {b.card_third_person()} "
            f"{_topic_clause(topic, sb_prompt, 'P1')} {_p1_style()} {_peer_guard()}"
            f"{_fewshot_block(prompt_level)}\n"
            f"It opens with:\n{a.label}: Hello!\n{b.label}: Hello!\n"
            f"Write the full conversation, one turn per line as '{a.label}: ...' / '{b.label}: ...'."
        )
    return [{"role": "user", "content": prompt}]


# --- C2: turn-by-turn, single model sees the whole script (replicates GPT4-1) --------

def build_c2(prompt_level: str, a: Persona, b: Persona, topic: str,
             sb_prompt: str | None,
             history: list[tuple[str, str]], next_speaker: str,
             max_turns: int | None = None) -> list[dict]:
    transcript = render_transcript(history) if history else f"{a.label}: Hello!\n{b.label}: Hello!"
    me = {a.label: a, b.label: b}[next_speaker]
    if prompt_level == "P0":
        prompt = (
            f"Act like {me.describe()} in a phone conversation with someone you do not know. "
            f"{_topic_clause(topic, sb_prompt, 'P0')} {_length_clause()} "
            f"The conversation log so far is:\n'''{transcript}'''\n"
            f"Each line is one turn; the speaker label precedes the colon. Your label is "
            f"{next_speaker}. Your response is the next turn — respond to the last line but take "
            "the whole log into account. Do not include more than one turn, and do not write a "
            f"speaker label.{_turn_status(history, max_turns)}"
        )
    else:  # P1 / P2
        prompt = (
            "You are writing a realistic phone conversation between two ordinary people who do "
            f"not know each other. {a.label} is {a.describe()}; {b.label} is {b.describe()}. "
            f"{a.card_third_person()} {b.card_third_person()} "
            f"{_topic_clause(topic, sb_prompt, 'P1')} {_p1_style()} {_peer_guard()}"
            f"{_fewshot_block(prompt_level)}\n"
            f"Conversation so far:\n{transcript}\n{_turn_status(history, max_turns)}\n"
            f"Write ONLY {next_speaker}'s next single turn — just the utterance, no label."
        )
    return [{"role": "user", "content": prompt}]


# --- C3 / C4: independent first-person agents ----------------------------------------

def build_agent(prompt_level: str, me: Persona, partner: Persona, topic: str,
                sb_prompt: str | None,
                history: list[tuple[str, str]],
                max_turns: int | None = None) -> list[dict]:
    """Build the message list from `me`'s point of view (used for both C3 and C4).

    The partner is deliberately NOT described beyond "a stranger" (P1/P2): the agents are
    independent sessions, and real strangers don't know each other's stance — any
    convergence has to happen in the talk itself.
    """
    if prompt_level == "P0":
        system = (
            f"Act like {me.describe()} in a telephone conversation with someone you do not know. "
            f"{_topic_clause(topic, sb_prompt, 'P0')} {_length_clause()} "
            "Reply with only your next single turn of talk — no speaker label, one turn only."
            f"{_turn_status(history, max_turns)}"
        )
    else:  # P1 / P2
        system = (
            f"You are {me.describe()} on a telephone call with an ordinary stranger you just met. "
            f"{me.card_first_person()} "
            f"{_topic_clause(topic, sb_prompt, 'P1')} {_p1_style()} {_peer_guard()}"
            f"{_fewshot_block(prompt_level)} "
            "Reply with only what you say next, as a single spoken turn — no speaker label."
            f"{_turn_status(history, max_turns)}"
        )
    messages = [{"role": "system", "content": system}]
    if not history:
        messages.append({"role": "user", "content": "(The phone connects — your partner is on the line.)"})
        return messages
    if history[0][0] == me.label:
        messages.append({"role": "user", "content": "(The call is already underway.)"})
    for spk, txt in history:
        role = "assistant" if spk == me.label else "user"
        messages.append({"role": role, "content": txt})
    if messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "(Your partner is quiet — continue if you wish.)"})
    return messages
