import threading
from datetime import datetime

import pytest

import booking
import db
from booking import BookingError

MON = datetime(2026, 10, 26, 10, 0)     # Monday after the frozen "now" (Fri 23 Oct 2026, 10:00)


def book(start=MON, name="Asha Rao", purpose="Fever and cough", **kw):
    return booking.book(name, start, purpose=purpose, **kw)


def code(excinfo):
    return excinfo.value.code


# ---------------------------------------------------------------- the slot grid
def test_monday_grid_skips_lunch_break(sqlite_db):
    res = booking.availability("2026-10-26")
    assert res["free_count"] == 14                      # 9-17 in 30 min steps = 16, minus 13:00 and 13:30
    assert "13:00" not in res["free_times"] and "13:30" not in res["free_times"]
    assert res["free_summary"] == "9:00 AM to 12:30 PM, 2:00 PM to 4:30 PM"


def test_sunday_is_closed_and_suggests_next_days(sqlite_db):
    res = booking.availability("2026-10-25")
    assert res["is_working_day"] is False and res["free_times"] == []
    assert res["next_available"][0]["date"] == "2026-10-26"
    assert res["next_available"][0]["time"] == "09:00"


def test_today_hides_slots_inside_the_notice_window(sqlite_db):
    free = booking.availability("2026-10-23")["free_times"]
    assert "10:00" not in free and "10:30" in free      # now = 10:00, notice = 30 min
    assert free[0] == "10:30"


def test_booked_slot_disappears_and_shows_in_day_status(sqlite_db):
    book()
    assert "10:00" not in booking.availability("2026-10-26")["free_times"]
    states = {s["time"]: s["status"] for s in booking.day_status(booking.parse_date("2026-10-26"))}
    assert states["10:00"] == "booked" and states["10:30"] == "free"


# ---------------------------------------------------------------- booking rules
def test_book_returns_details_and_ref_code_and_stores_no_phone(sqlite_db):
    a = book(purpose="Check-up")
    assert a["time"] == "10:00" and a["end_time"] == "10:30" and a["status"] == "booked"
    assert a["spoken"] == "Monday 26 October at 10:00 AM"
    assert len(a["ref_code"]) == 6 and a["ref_code"].isdigit() and a["purpose"] == "Check-up"
    assert "phone" not in a


def test_double_booking_is_refused_with_alternatives(sqlite_db):
    book()
    with pytest.raises(BookingError) as e:
        book(name="Other Person")
    assert code(e) == "slot_taken"
    assert "10:00" not in e.value.extra["free_times"] and e.value.extra["free_times"]


def test_cancelled_slot_can_be_booked_again(sqlite_db):
    a = book()
    booking.cancel(a["ref_code"])
    b = book(name="Other Person")
    assert b["time"] == "10:00" and b["id"] != a["id"]


@pytest.mark.parametrize("start,expected", [
    (datetime(2026, 10, 22, 10, 0), "past"),            # yesterday
    (datetime(2026, 10, 23, 10, 0), "past"),            # inside the 30 min notice window
    (datetime(2026, 12, 30, 10, 0), "too_far"),         # beyond 30 days
    (datetime(2026, 10, 25, 10, 0), "closed"),          # Sunday
    (datetime(2026, 10, 26, 10, 15), "not_a_slot"),     # off the 30-minute grid
    (datetime(2026, 10, 26, 13, 0), "not_a_slot"),      # lunch break
    (datetime(2026, 10, 26, 8, 30), "not_a_slot"),      # before opening
    (datetime(2026, 10, 26, 17, 0), "not_a_slot"),      # at closing time
])
def test_invalid_slots(sqlite_db, start, expected):
    with pytest.raises(BookingError) as e:
        book(start)
    assert code(e) == expected


@pytest.mark.parametrize("name", ["A", "  ", ""])
def test_invalid_name(sqlite_db, name):
    with pytest.raises(BookingError) as e:
        book(name=name)
    assert code(e) == "invalid_name"


@pytest.mark.parametrize("purpose", ["", "  ", "ab", None])
def test_reason_or_symptoms_are_required(sqlite_db, purpose):
    with pytest.raises(BookingError) as e:
        booking.book("Asha Rao", MON, purpose=purpose)
    assert code(e) == "invalid_purpose"
    assert booking.list_appointments() == []


def test_the_same_person_may_book_several_slots(sqlite_db):
    for h in (9, 10, 11, 12):
        book(datetime(2026, 10, 26, h, 0))
    assert len(booking.list_appointments()) == 4


# ---------------------------------------------------------------- reference based find / cancel / reschedule
def test_reference_normalisation():
    assert booking.normalize_ref("482 913") == "482913"
    assert booking.normalize_ref("४८२९१३") == "482913"           # Devanagari digits from Hindi speech
    for bad in ("", "12345", "1234567", "abcdef"):
        with pytest.raises(BookingError) as e:
            booking.normalize_ref(bad)
        assert code(e) == "invalid_reference"


def test_find_by_reference(sqlite_db):
    a = book()
    assert booking.find_by_ref(a["ref_code"])["customer_name"] == "Asha Rao"
    assert booking.find_by_ref(" ".join(a["ref_code"]))["id"] == a["id"]      # "4 8 2 9 1 3" read out digit by digit


def test_cancel_with_reference(sqlite_db):
    a = book()
    res = booking.cancel(a["ref_code"])
    assert res["status"] == "cancelled"
    with pytest.raises(BookingError) as e:
        booking.find_by_ref(a["ref_code"])
    assert code(e) == "not_found"
    assert "10:00" in booking.availability("2026-10-26")["free_times"]


def test_wrong_reference_cannot_cancel_anything(sqlite_db):
    a = book()
    wrong = "000000" if a["ref_code"] != "000000" else "111111"
    with pytest.raises(BookingError) as e:
        booking.cancel(wrong)
    assert code(e) == "not_found"
    assert len(booking.list_appointments()) == 1


def test_cancel_twice_reports_not_found(sqlite_db):
    a = book()
    booking.cancel(a["ref_code"])
    with pytest.raises(BookingError) as e:
        booking.cancel(a["ref_code"])
    assert code(e) == "not_found"


def test_reschedule_moves_the_slot_and_keeps_the_reference(sqlite_db):
    a = book()
    res = booking.reschedule(a["ref_code"], datetime(2026, 10, 27, 15, 0))
    assert res["date"] == "2026-10-27" and res["time"] == "15:00" and res["ref_code"] == a["ref_code"]
    assert "10:00" in booking.availability("2026-10-26")["free_times"]       # old slot released
    assert "15:00" not in booking.availability("2026-10-27")["free_times"]


def test_reschedule_to_taken_slot_keeps_the_old_one(sqlite_db):
    a = book()
    book(datetime(2026, 10, 27, 15, 0), name="Other Person")
    with pytest.raises(BookingError) as e:
        booking.reschedule(a["ref_code"], datetime(2026, 10, 27, 15, 0))
    assert code(e) == "slot_taken"
    assert booking.find_by_ref(a["ref_code"])["time"] == "10:00"


def test_reschedule_to_same_time_is_rejected(sqlite_db):
    a = book()
    with pytest.raises(BookingError) as e:
        booking.reschedule(a["ref_code"], MON)
    assert code(e) == "same_slot"


def test_staff_cancel_by_id_and_listing(sqlite_db):
    a = book()
    assert [x["id"] for x in booking.list_appointments()] == [a["id"]]
    booking.cancel_by_id(a["id"])
    assert booking.list_appointments() == []
    with pytest.raises(BookingError):
        booking.cancel_by_id(a["id"])


def test_past_appointments_are_not_listed_as_upcoming(sqlite_db, clock):
    a = book()
    clock["now"] = datetime(2026, 10, 26, 11, 0)
    assert booking.list_appointments() == []
    assert len(booking.list_appointments(upcoming_only=False)) == 1
    with pytest.raises(BookingError):
        booking.find_by_ref(a["ref_code"])


# ---------------------------------------------------------------- concurrency
def test_simultaneous_bookings_of_one_slot_produce_exactly_one_winner(sqlite_db):
    results, barrier = [], threading.Barrier(8)

    def worker(i):
        barrier.wait()
        try:
            booking.book(f"Caller {i}", MON, purpose="Headache")
            results.append("ok")
        except BookingError as exc:
            results.append(exc.code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("ok") == 1 and results.count("slot_taken") == 7
    assert len(booking.list_appointments()) == 1


# ---------------------------------------------------------------- db helpers
def test_integrity_error_detection():
    import sqlite3
    assert db.is_integrity_error(sqlite3.IntegrityError("x"))
    assert not db.is_integrity_error(ValueError("x"))


def test_schema_has_no_phone_and_parses_into_one_statement():
    stmts = db._schema_statements()
    assert len(stmts) == 1 and stmts[0].startswith("CREATE TABLE IF NOT EXISTS appointments")
    assert "UNIQUE KEY uq_active_slot (slot_start, active)" in stmts[0]
    assert "phone" not in stmts[0].lower()


class RecordingCursor:
    def __init__(self, counts):
        self.counts, self.sql = list(counts), []

    def execute(self, sql, params=None):
        self.sql.append(sql)

    def fetchall(self):
        return [{"n": self.counts.pop(0)}]


def test_old_phone_column_and_index_are_dropped_from_existing_tables():
    cur = RecordingCursor([1, 1])                    # column exists, index exists
    db._drop_phone_column(cur)
    assert cur.sql[-2:] == ["ALTER TABLE appointments DROP INDEX idx_phone_start",
                            "ALTER TABLE appointments DROP COLUMN phone"]


def test_migration_does_nothing_on_a_fresh_database():
    cur = RecordingCursor([0])                       # no phone column
    db._drop_phone_column(cur)
    assert len(cur.sql) == 1 and cur.sql[0].startswith("SELECT")
