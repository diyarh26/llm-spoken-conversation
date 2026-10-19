"""The turn-by-turn core: per-turn quality guards and the conversation loop (stub models)."""

from __future__ import annotations

from generation.quality import is_near_duplicate, token_jaccard
from generation.turnwise import SEED_TURNS, generate_turn, run_conversation

LONG = "I really think the city should spend more money fixing the roads near my house"


def scripted(*outputs, cap_hits=()):
    """A stub model returning the given outputs in order; records the temperatures used."""
    calls = []

    def chat_fn(temperature):
        i = len(calls)
        calls.append(temperature)
        return outputs[i], {"hit_token_cap": i in cap_hits}

    chat_fn.calls = calls
    return chat_fn


class TestNearDuplicate:
    def test_jaccard(self):
        assert token_jaccard("a b c", "a b c") == 1.0
        assert token_jaccard("a b", "c d") == 0.0
        assert token_jaccard("", "") == 0.0

    def test_long_verbatim_repeat_is_a_loop(self):
        assert is_near_duplicate(LONG + " today", [("A", LONG)])

    def test_backchannels_may_repeat_freely(self):
        history = [("A", "Uh-huh."), ("B", "Yeah."), ("A", "Uh-huh.")]
        assert not is_near_duplicate("Uh-huh.", history)

    def test_different_long_turn_is_not_a_loop(self):
        assert not is_near_duplicate(
            "My sister moved to Dallas because the jobs are better", [("A", LONG)]
        )


class TestGenerateTurn:
    def test_clean_output_passes_through(self):
        counters = {}
        chat = scripted("Sure, I drive to work every day.")
        assert (
            generate_turn(chat, [], temperature=0.8, counters=counters)
            == "Sure, I drive to work every day."
        )
        assert counters == {"hit_token_cap": 0, "multi_turn_emissions": 0}

    def test_empty_output_is_retried_once(self):
        counters = {}
        chat = scripted("", "Okay, sounds good.")
        assert generate_turn(chat, [], temperature=0.8, counters=counters) == "Okay, sounds good."
        assert counters["empty_retries"] == 1

    def test_twice_empty_means_the_model_left(self):
        assert generate_turn(scripted("", "   "), [], temperature=0.8) == ""

    def test_loop_is_resampled_at_higher_temperature(self):
        counters = {}
        chat = scripted(LONG, "Honestly, my commute is only ten minutes so I hardly notice.")
        turn = generate_turn(chat, [("A", LONG)], temperature=0.8, counters=counters)
        assert turn.startswith("Honestly")
        assert chat.calls == [0.8, 0.8 + 0.15]
        assert counters["dup_resamples"] == 1
        assert "dup_kept" not in counters

    def test_loop_is_kept_and_counted_when_resample_also_loops(self):
        counters = {}
        turn = generate_turn(
            scripted(LONG, LONG), [("A", LONG)], temperature=0.8, counters=counters
        )
        assert turn == LONG
        assert counters["dup_kept"] == 1

    def test_label_leak_and_residue_are_cleaned_and_counted(self):
        counters = {}
        chat = scripted(
            "Right, exactly. (Your turn count is 7) ParticipantB: and then", cap_hits={0}
        )
        assert generate_turn(chat, [], temperature=0.8, counters=counters) == "Right, exactly."
        assert counters == {"hit_token_cap": 1, "multi_turn_emissions": 1}


class TestRunConversation:
    @staticmethod
    def speaker(*turns):
        it = iter(turns)
        return lambda speaker, history: next(it)

    def test_starts_from_the_scripted_greeting_and_alternates(self):
        history, _ = run_conversation(
            self.speaker("So, pets?", "I have two dogs.", ""), max_turns=10
        )
        assert history[:2] == list(SEED_TURNS)
        assert [s for s, _ in history[2:]] == ["ParticipantA", "ParticipantB"]

    def test_mutual_goodbye_ends_immediately(self):
        _, ended = run_conversation(self.speaker("Well, bye now!", "Bye, take care!"), max_turns=10)
        assert ended == "mutual_closing"

    def test_one_reciprocal_turn_after_a_farewell(self):
        history, ended = run_conversation(
            self.speaker("I have to go, nice talking to you.", "Sure thing, you too."), max_turns=10
        )
        assert ended == "closing"
        assert len(history) == 4

    def test_empty_turn_ends_the_call(self):
        history, ended = run_conversation(self.speaker("Hi!", ""), max_turns=10)
        assert ended == "empty_turn"
        assert history[-1] == ("ParticipantA", "Hi!")

    def test_turn_cap(self):
        history, ended = run_conversation(lambda s, h: "Tell me more about that.", max_turns=6)
        assert ended == "turn_cap"
        assert len(history) == len(SEED_TURNS) + 6
