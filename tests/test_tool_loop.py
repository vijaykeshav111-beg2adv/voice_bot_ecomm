"""run_turn with a scripted fake LLM: tool calls arrive in streamed fragments, tools hit the (SQLite) DB."""
import json
from datetime import datetime
from types import SimpleNamespace as NS

import pytest

import booking
import booking_tools
import core


def text(s):
    return NS(choices=[NS(delta=NS(content=s, tool_calls=None))])


def call(index, name=None, args="", cid=None):
    fn = NS(name=name, arguments=args)
    return NS(choices=[NS(delta=NS(content=None, tool_calls=[NS(index=index, id=cid, function=fn)]))])


class FakeLLM:
    """Each element of `rounds` is the list of chunks for one streaming request."""

    def __init__(self, rounds):
        self.rounds, self.requests = list(rounds), []

    def __call__(self, messages, tools=None):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        return iter(self.rounds.pop(0))


@pytest.fixture
def llm(monkeypatch, sqlite_db):
    def install(rounds):
        fake = FakeLLM(rounds)
        monkeypatch.setattr(core, "_open_llm_stream", fake)
        return fake
    monkeypatch.setattr(core, "_tts_clip", lambda t: {"audio": b"mp3:" + t.encode(), "first_byte": 0.0,
                                                      "seconds": 0.0, "done_at": core.time.perf_counter()})
    return install


def run(user_text, conv=None, source="text"):
    conv = conv if conv is not None else core.new_conversation()
    rec = core.new_record("t" * 36, source=source)
    events = list(core.run_turn(conv, rec, text=user_text))
    return events, conv, rec


def kinds(events):
    return [e["type"] for e in events]


def test_availability_question_runs_a_tool_then_speaks_the_answer(llm):
    fake = llm([
        [call(0, "check_availab", "", "call_1"), call(0, "ility", '{"date": "2026'), call(0, None, '-10-26"}')],
        [text("We have 9 AM to 12:30 PM on Monday. "), text("Which time suits you?")],
    ])
    events, conv, rec = run("Any slot on Monday?")
    tool = next(e for e in events if e["type"] == "tool")
    assert tool["name"] == "check_availability" and tool["args"] == {"date": "2026-10-26"}
    assert tool["result"]["ok"] and tool["result"]["free_count"] == 14
    assert kinds(events)[-1] == "done" and "clip" in kinds(events)
    spoken = " ".join(e["text"] for e in events if e["type"] == "text")
    assert "9 AM to 12:30 PM" in spoken
    # the 2nd request carries the assistant tool call and the tool result, and no tools after round 1 is NOT required
    msgs = fake.requests[1]["messages"]
    assert msgs[-2]["role"] == "assistant" and msgs[-2]["tool_calls"][0]["id"] == "call_1"
    assert msgs[-2]["tool_calls"][0]["function"]["name"] == "check_availability"
    assert msgs[-1]["role"] == "tool" and msgs[-1]["tool_call_id"] == "call_1"
    assert json.loads(msgs[-1]["content"])["free_summary"].startswith("9:00 AM")
    # only user + final assistant text are kept as history
    assert [m["role"] for m in conv] == ["system", "user", "assistant"]
    assert rec["status"] == "success" and rec["tools_called"] == "check_availability" and rec["tool_seconds"] >= 0


def test_no_tool_or_prompt_asks_for_a_phone_number(llm):
    fake = llm([[text("Hello.")]])
    run("hi")
    assert "phone" not in json.dumps(booking_tools.TOOLS).lower()
    sp = fake.requests[0]["messages"][0]["content"]
    assert "Do not ask for a phone number" in sp and "6-digit booking reference" in sp


def test_cancel_with_reference_through_the_assistant(llm):
    a = booking.book("Asha Rao", datetime(2026, 10, 26, 10, 0), purpose="Fever")
    args = json.dumps({"reference": " ".join(a["ref_code"]), "confirmed": True})
    llm([[call(0, "cancel_appointment", args)], [text("Cancelled.")]])
    events, _, _ = run("cancel it")
    assert next(e for e in events if e["type"] == "tool")["result"]["ok"]
    assert booking.list_appointments() == []


def test_reschedule_and_find_with_reference_through_the_assistant(llm):
    a = booking.book("Asha Rao", datetime(2026, 10, 26, 10, 0), purpose="Fever")
    llm([[call(0, "find_my_appointment", json.dumps({"reference": a["ref_code"]}))], [text("Found it.")],
         [call(0, "reschedule_appointment", json.dumps({"reference": a["ref_code"], "new_date": "2026-10-27",
                                                        "new_time": "15:00", "confirmed": True}))], [text("Moved.")]])
    events, _, _ = run("find")
    assert next(e for e in events if e["type"] == "tool")["result"]["appointment"]["time"] == "10:00"
    events, _, _ = run("move it to tuesday 3pm")
    assert next(e for e in events if e["type"] == "tool")["result"]["appointment"]["date"] == "2026-10-27"


def test_wrong_reference_gives_not_found(llm):
    llm([[call(0, "cancel_appointment", json.dumps({"reference": "123456", "confirmed": True}))], [text("Not found.")]])
    events, _, _ = run("cancel 123456")
    assert next(e for e in events if e["type"] == "tool")["result"]["error"] == "not_found"


def test_system_prompt_carries_todays_date_and_is_refreshed_each_turn(llm):
    fake = llm([[text("Hello.")]])
    run("hi")
    system = fake.requests[0]["messages"][0]["content"]
    assert "Friday 23 October, 2026" in system and "2026-10-26 = Monday 26 October" in system
    assert fake.requests[0]["tools"] is booking_tools.TOOLS


def test_booking_through_the_assistant_writes_to_the_database(llm):
    args = json.dumps({"date": "2026-10-26", "time": "10:00", "customer_name": "Asha Rao",
                       "purpose": "Check-up", "confirmed": True})
    llm([[call(0, "book_appointment", args, "c1")], [text("Done, your reference is shown on screen.")]])
    events, _, rec = run("yes, book it", source="voice")
    result = next(e for e in events if e["type"] == "tool")["result"]
    assert result["ok"] and result["appointment"]["time"] == "10:00"
    (row,) = booking.list_appointments()
    assert row["customer_name"] == "Asha Rao" and row["channel"] == "voice" and row["purpose"] == "Check-up"


def test_prompt_asks_for_reason_first_then_time_then_name(llm):
    fake = llm([[text("Hello.")]])
    run("hi")
    sp = fake.requests[0]["messages"][0]["content"]
    assert sp.index("STEP 1 - REASON") < sp.index("STEP 2 - TIME") < sp.index("STEP 3 - NAME") < sp.index("STEP 4")
    assert "emergency" in sp


def test_model_cannot_book_without_a_reason(llm):
    args = json.dumps({"date": "2026-10-26", "time": "10:00", "customer_name": "Asha Rao",
                       "confirmed": True})            # no purpose given
    llm([[call(0, "book_appointment", args)], [text("What is the problem?")]])
    events, _, _ = run("book monday 10")
    result = next(e for e in events if e["type"] == "tool")["result"]
    assert result["ok"] is False and result["error"] == "invalid_purpose"
    assert booking.list_appointments() == []


def test_model_cannot_book_without_confirmation(llm):
    args = json.dumps({"date": "2026-10-26", "time": "10:00", "customer_name": "Asha Rao",
                       "purpose": "Fever", "confirmed": False})
    llm([[call(0, "book_appointment", args)], [text("Shall I go ahead?")]])
    events, _, _ = run("book monday 10")
    result = next(e for e in events if e["type"] == "tool")["result"]
    assert result["ok"] is False and result["error"] == "not_confirmed"
    assert booking.list_appointments() == []


def test_tool_error_is_given_to_the_model_not_raised(llm):
    fake = llm([[call(0, "check_availability", '{"date": "2020-01-01"}')], [text("That date has passed.")]])
    events, _, rec = run("jan 2020?")
    assert next(e for e in events if e["type"] == "tool")["result"]["error"] == "past"
    assert json.loads(fake.requests[1]["messages"][-1]["content"])["ok"] is False
    assert rec["status"] == "success"


def test_unparseable_arguments_and_unknown_tool(llm):
    llm([[call(0, "check_availability", "{not json")], [call(0, "delete_everything", "{}")], [text("Sorry.")]])
    events, _, _ = run("x")
    results = [e["result"] for e in events if e["type"] == "tool"]
    assert results[0]["error"] == "bad_arguments" and results[1]["error"] == "unknown_tool"


def test_two_tool_calls_in_one_round(llm):
    llm([[call(0, "check_availability", '{"date": "2026-10-26"}', "a"),
          call(1, "check_availability", '{"date": "2026-10-27"}', "b")], [text("Both days have room.")]])
    events, _, rec = run("monday or tuesday?")
    assert [e["args"]["date"] for e in events if e["type"] == "tool"] == ["2026-10-26", "2026-10-27"]
    assert rec["tools_called"] == "check_availability,check_availability"


def test_silent_model_after_tool_falls_back_to_the_tools_own_summary(llm):
    llm([[call(0, "check_availability", '{"date": "2026-10-26"}')], []])
    events, _, _ = run("monday?")
    assert "free times" in " ".join(e["text"] for e in events if e["type"] == "text")


def test_runaway_tool_loop_is_cut_off(llm, monkeypatch):
    monkeypatch.setattr(core, "MAX_TOOL_ROUNDS", 2)
    fake = llm([[call(0, "check_availability", '{"date": "2026-10-26"}')]] * 2 + [[text("Enough searching.")]])
    events, _, rec = run("loop")
    assert len(fake.requests) == 3 and fake.requests[2]["tools"] is None      # last round: no tools offered
    assert kinds(events)[-1] == "done" and rec["status"] == "success"


def test_database_down_gives_the_model_a_polite_failure(llm, monkeypatch):
    import db

    def broken():
        raise ConnectionError("mysql is down")
    monkeypatch.setattr(db, "connection", broken)
    llm([[call(0, "check_availability", '{"date": "2026-10-26"}')], [text("Sorry, the system is down.")]])
    events, _, rec = run("monday?")
    result = next(e for e in events if e["type"] == "tool")["result"]
    assert result["error"] == "system_unavailable" and "mysql" not in json.dumps(result).lower()
    assert rec["status"] == "success"


def test_booking_disabled_means_plain_chat(llm, monkeypatch):
    monkeypatch.setattr(core, "BOOKING_ENABLED", False)
    fake = llm([[text("Hi there.")]])
    run("hello")
    assert fake.requests[0]["tools"] is None
    assert "voice receptionist" not in fake.requests[0]["messages"][0]["content"]


def test_streamed_events_serialise_for_the_server(llm):
    llm([[call(0, "check_availability", '{"date": "2026-10-26"}')], [text("Plenty of room.")]])
    events, _, _ = run("monday?")
    for e in events:
        if e["type"] != "clip":
            json.dumps(e, ensure_ascii=False)          # server._line() does exactly this
