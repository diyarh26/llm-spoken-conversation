"""
Map Task prompts for the four architectures, plus the route reader and the move tagger.

Who sees what is the whole point of this domain:
  C1 (all at once) and C2 (turn by turn, one writer): one model sees BOTH maps and the
      route, and writes both speakers. It knows where the maps differ.
  C3/C4 (two agents): the giver agent sees only the giver's map and route; the follower
      agent sees only the follower's map. Neither knows where the maps differ, exactly
      as for the human pairs.
The design rule of the main study holds here too: no prompt names or asks for a dialogue
move the analysis measures (no "ask", "check", "confirm", "acknowledge", "clarify").
The speakers are only told the task, that the maps may differ, and to talk naturally.
Coordinates are given so the model can locate things, and the speakers are told that the
other person's map has no numbers, so coordinates are not a shared vocabulary.
"""

from __future__ import annotations

from maptask.score import passes

LABELS = ("Giver", "Follower")
GIVER, FOLLOWER = LABELS

COORDS = (
    "Positions are given as (x, y): x runs from 0 at the west (left) edge of the map to 65 "
    "at the east (right) edge, and y runs from 0 at the north (top) edge to 90 at the "
    "south (bottom) edge."
)

SPOKEN = (
    "Talk the way people actually do out loud on the phone: casual and uneven, "
    "sometimes a sentence or two, sometimes just a word or two."
)

NO_NUMBERS = (
    "The numbers are only there to help you picture the map: the other person's map has "
    "no numbers on it, so describe things the way you would to someone looking at a "
    "drawing."
)


def _label(lm: dict) -> str:
    if lm["name"] == "start":
        return "START (marked with a cross)"
    if lm["name"] == "finish":
        return "FINISH (marked with a cross)"
    return lm.get("label", lm["name"])


def map_text(side: dict) -> str:
    """One map as a landmark list, west-to-east within north-to-south bands."""
    lms = sorted(side["landmarks"], key=lambda lm: (lm["y"] // 10, lm["x"]))
    return "\n".join(f"- {_label(lm)} at ({lm['x']}, {lm['y']})" for lm in lms)


def route_text(giver: dict) -> str:
    """The giver's route: the polyline, and the landmarks it passes with their side."""
    points = " -> ".join(f"({x}, {y})" for x, y in giver["route"])
    near = passes(giver["route"], giver["landmarks"])
    along = "; ".join(f"the {p['label']} on the {p['side']}" for p in near)
    return (
        f"The route is drawn on your map as a line from START to FINISH through these "
        f"points, in order: {points}.\n"
        f"Travelling along it, you pass close to: {along}."
    )


def _turn_status(history: list, max_turns: int | None) -> str:
    if not max_turns:
        return ""
    return (
        f"\n(The call so far has {len(history)} turns of talk; it cannot go past "
        f"{max_turns} turns, so it has to be over by then.)"
    )


def _task() -> str:
    return (
        "Two strangers are doing a map task over the phone. Each has a map of the same "
        "area, and they cannot see each other's map. The instruction giver's map shows a "
        "route from START to FINISH; the instruction follower's map shows START but no "
        "route, and the follower has to draw the route on it from the giver's directions. "
        "The two maps are not quite the same: some landmarks are on one map but not the "
        "other, and some may be in different places. Neither person knows where the "
        "differences are."
    )


def render_transcript(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"{spk}: {txt}" for spk, txt in history)


# --- C1: all at once --------------------------------------------------------------------


def build_c1(m: dict) -> list[dict]:
    prompt = (
        f"Write a realistic transcript of this conversation. {_task()}\n\n"
        f"{COORDS}\n\nThe giver's map:\n{map_text(m['giver'])}\n{route_text(m['giver'])}\n\n"
        f"The follower's map:\n{map_text(m['follower'])}\n\n"
        f"{SPOKEN} The speakers never say coordinates, because their maps have no "
        "numbers. The call ends when the follower has drawn the route to FINISH.\n"
        f"Write the full conversation, one turn per line as '{GIVER}: ...' / "
        f"'{FOLLOWER}: ...', starting with the giver."
    )
    return [{"role": "user", "content": prompt}]


# --- C2: turn by turn, one writer who sees both maps ----------------------------------


def build_c2(
    m: dict, history: list[tuple[str, str]], next_speaker: str, max_turns: int | None = None
) -> list[dict]:
    so_far = render_transcript(history) if history else "(nothing yet: the giver starts)"
    prompt = (
        f"You are writing a realistic transcript of this conversation. {_task()}\n\n"
        f"{COORDS}\n\nThe giver's map:\n{map_text(m['giver'])}\n{route_text(m['giver'])}\n\n"
        f"The follower's map:\n{map_text(m['follower'])}\n\n"
        f"{SPOKEN} The speakers never say coordinates, because their maps have no "
        "numbers. The call ends when the follower has drawn the route to FINISH.\n\n"
        f"Conversation so far:\n{so_far}\n{_turn_status(history, max_turns)}\n"
        f"Write ONLY the {next_speaker.lower()}'s next single turn: just the words, no label."
    )
    return [{"role": "user", "content": prompt}]


# --- C3/C4: two agents, each with only its own map ---------------------------------------


def build_agent(
    role: str, m: dict, history: list[tuple[str, str]], max_turns: int | None = None
) -> list[dict]:
    if role == GIVER:
        mine = (
            "You are the instruction giver. Your map:\n"
            f"{map_text(m['giver'])}\n{route_text(m['giver'])}\n"
            "Your partner's map shows START but no route, and they will draw the route "
            "from what you tell them."
        )
    else:
        mine = (
            "You are the instruction follower. Your map:\n"
            f"{map_text(m['follower'])}\n"
            "There is no route on your map. Your partner has the route on theirs, and "
            "you have to draw it on your map from what they tell you."
        )
    system = (
        "You are an ordinary person doing a map task over the phone with a stranger. You "
        "cannot see each other's maps, and your partner's map is not quite the same as "
        "yours: some landmarks may be on one map but not the other, or in different "
        f"places, and you do not know where.\n\n{COORDS} {NO_NUMBERS}\n\n{mine}\n\n"
        f"{SPOKEN} The call ends when the route has been drawn to FINISH.\n"
        "Reply with only what you say next, as a single spoken turn, with no speaker "
        f"label.{_turn_status(history, max_turns)}"
    )
    messages = [{"role": "system", "content": system}]
    if not history:
        messages.append(
            {"role": "user", "content": "(The call connects. You speak first.)"}
            if role == GIVER
            else {"role": "user", "content": "(The call connects.)"}
        )
        return messages
    for spk, txt in history:
        messages.append({"role": "assistant" if spk == role else "user", "content": txt})
    if messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "(Your partner is quiet.)"})
    return messages


# --- scoring helpers ----------------------------------------------------------------------


def build_reconstruction(m: dict, turns: list[tuple[str, str]]) -> list[dict]:
    """The route reader: sees ONLY the follower's map and the finished conversation, and
    draws the route the conversation describes. Applied identically to human dialogues."""
    prompt = (
        "Below is a map and the transcript of a phone call. In the call, a giver described "
        "a route to the follower, who was holding this map and had to draw the route on "
        "it. The giver's map was not quite the same as this one.\n\n"
        f"{COORDS}\n\nThe follower's map:\n{map_text(m['follower'])}\n\n"
        f"Transcript:\n{render_transcript(turns)}\n\n"
        "Draw the route the follower should have drawn from what was said, starting at "
        "START. Answer with only a JSON list of [x, y] points in order, for example "
        "[[5, 61], [10, 70], [20, 72]], with enough points to follow every turn of the route."
    )
    return [{"role": "user", "content": prompt}]


# Short definitions of the 12 HCRC moves (after Carletta et al. 1997, "The reliability of
# a dialogue structure coding scheme"). Used only to LABEL finished dialogues, never in a
# generation prompt.
MOVE_DEFINITIONS = {
    "instruct": "tells the partner to carry out an action (e.g. move along the route)",
    "explain": "states information the partner did not ask for (e.g. what is on my map)",
    "check": "asks the partner to confirm something the speaker believes but is unsure of",
    "align": "checks that the partner is following or ready to go on (e.g. 'okay?')",
    "query_yn": "any other yes/no question",
    "query_w": "any other question (where, what, which, how...)",
    "acknowledge": "shows the speaker heard and understood (e.g. 'okay', 'right', 'mmhmm')",
    "reply_y": "answers a question with yes or an equivalent",
    "reply_n": "answers a question with no or an equivalent",
    "reply_w": "any other answer to a question",
    "clarify": "repeats or rephrases what was already said, to make it clearer",
    "ready": "a signal that the speaker is about to start a new step (e.g. 'right', 'now')",
}


def build_move_tagger(turns: list[tuple[str, str]]) -> list[dict]:
    """Label each turn with its main dialogue move (one label per turn)."""
    defs = "\n".join(f"- {k}: {v}" for k, v in MOVE_DEFINITIONS.items())
    numbered = "\n".join(f"{i}. {spk}: {txt}" for i, (spk, txt) in enumerate(turns, 1))
    prompt = (
        "Label each turn of this map-task conversation with its main dialogue move, using "
        f"exactly one of these labels:\n{defs}\n\nConversation:\n{numbered}\n\n"
        f"Answer with only a JSON list of {len(turns)} labels, one per turn, in order."
    )
    return [{"role": "user", "content": prompt}]
