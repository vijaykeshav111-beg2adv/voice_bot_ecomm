"""Appointment logic: which slots exist, which are free, and atomic book / cancel / reschedule.

Slots are generated from the working-hours settings below (no rows are stored for free slots);
only booked appointments live in MySQL. All times are local wall-clock time in BOOKING_TIMEZONE.

Every function raises BookingError(code, message) for expected problems (slot taken, closed day,
bad reference ...), which the voice/LLM layer and the REST API turn into friendly messages.
"""
import os
import random
import re
from datetime import date, datetime, time, timedelta

import db

# ---------------------------------------------------------------- configuration (.env)
BUSINESS_NAME = os.getenv("BUSINESS_NAME", "our clinic").strip() or "our clinic"
TZ_NAME = os.getenv("BOOKING_TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata"
SLOT_MINUTES = max(5, int(os.getenv("SLOT_MINUTES", "30") or 30))
BOOKING_DAYS_AHEAD = max(1, int(os.getenv("BOOKING_DAYS_AHEAD", "30") or 30))
MIN_NOTICE_MINUTES = max(0, int(os.getenv("MIN_NOTICE_MINUTES", "30") or 30))

_DAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def _parse_hhmm(value, name):
    try:
        h, m = value.strip().split(":")
        return time(int(h), int(m))
    except Exception:
        raise RuntimeError(f"{name} must look like 09:30, got {value!r}") from None


OPEN_TIME = _parse_hhmm(os.getenv("OPEN_TIME", "09:00"), "OPEN_TIME")
CLOSE_TIME = _parse_hhmm(os.getenv("CLOSE_TIME", "17:00"), "CLOSE_TIME")
_break = os.getenv("BREAK_TIME", "13:00-14:00").strip()          # empty = no break
BREAK = tuple(_parse_hhmm(p, "BREAK_TIME") for p in _break.split("-")) if _break else None
WORKING_DAYS = {_DAY_INDEX[d.strip()[:3].lower()]
                for d in os.getenv("WORKING_DAYS", "mon,tue,wed,thu,fri,sat").split(",")
                if d.strip()[:3].lower() in _DAY_INDEX}
HOLIDAYS = set()
for _h in os.getenv("HOLIDAYS", "").split(","):                  # e.g. 2026-11-08,2026-12-25
    if _h.strip():
        HOLIDAYS.add(date.fromisoformat(_h.strip()))

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo(TZ_NAME)
except Exception:                                                  # missing tzdata on Windows etc.
    TZ = None


class BookingError(Exception):
    def __init__(self, code, message, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra


# ---------------------------------------------------------------- formatting helpers
def now():
    """Current local time at the business (naive datetime). Tests patch this."""
    return (datetime.now(TZ) if TZ else datetime.now()).replace(tzinfo=None, microsecond=0)


def fmt_time(dt):
    h = dt.hour % 12 or 12
    return f"{h}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def fmt_date(d):
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"


def spoken(dt):
    return f"{fmt_date(dt.date())} at {fmt_time(dt)}"


def parse_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        raise BookingError("invalid_date", "The date must look like 2026-10-27.") from None


def parse_time(value):
    """Accepts 14:30, 2:30 PM, 2pm, 9, 09:00:00."""
    s = str(value).strip().lower().replace(".", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(?::\d{2})?\s*(am|pm)?", s)
    if not m:
        raise BookingError("invalid_time", "The time must look like 14:30.")
    h, minute, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    if not (0 <= h <= 23 and 0 <= minute <= 59):
        raise BookingError("invalid_time", "That is not a valid time.")
    return time(h, minute)


def normalize_ref(value):
    """The booking reference: 6 digits (Devanagari digits and spaces are accepted)."""
    digits = "".join(str(int(c)) for c in str(value or "") if c.isdigit())
    if len(digits) != 6:
        raise BookingError("invalid_reference", "The booking reference is a 6-digit number.")
    return digits


def clean_name(value):
    name = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(name) < 2 or len(name) > 100:
        raise BookingError("invalid_name", "I need the customer's name (2 to 100 characters).")
    return name


def clean_purpose(value):
    """The reason for the visit / symptoms. Required: the receptionist asks for it first."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) < 3:
        raise BookingError("invalid_purpose", "Please tell me the reason for the visit or your symptoms first.")
    return text[:255]


# ---------------------------------------------------------------- slot grid
def check_window(d):
    today = now().date()
    if d < today:
        raise BookingError("past", "That date has already passed.")
    last = today + timedelta(days=BOOKING_DAYS_AHEAD)
    if d > last:
        raise BookingError("too_far", f"Bookings open only up to {fmt_date(last)}.", last_date=last.isoformat())


def day_slots(d):
    """Every bookable start time on date d according to the working hours (free or not)."""
    if d.weekday() not in WORKING_DAYS or d in HOLIDAYS:
        return []
    step = timedelta(minutes=SLOT_MINUTES)
    cur, end = datetime.combine(d, OPEN_TIME), datetime.combine(d, CLOSE_TIME)
    out = []
    while cur + step <= end:
        in_break = BREAK and cur.time() < BREAK[1] and (cur + step).time() > BREAK[0]
        if not in_break:
            out.append(cur)
        cur += step
    return out


def _booked_starts(cur, d):
    cur.execute("SELECT slot_start FROM appointments WHERE active = 1 AND slot_start >= %s AND slot_start < %s",
                (datetime.combine(d, time.min), datetime.combine(d + timedelta(days=1), time.min)))
    return {r["slot_start"] for r in cur.fetchall()}


def _earliest():
    return now() + timedelta(minutes=MIN_NOTICE_MINUTES)


def day_status(d):
    """For the UI and the LLM: every slot of the day with free/booked/past."""
    check_window(d)
    slots = day_slots(d)
    with db.connection() as conn, conn.cursor() as cur:
        booked = _booked_starts(cur, d) if slots else set()
    earliest = _earliest()
    out = []
    for s in slots:
        state = "booked" if s in booked else ("past" if s < earliest else "free")
        out.append({"time": s.strftime("%H:%M"), "label": fmt_time(s), "status": state})
    return out


def _ranges(slots):
    """[10:00, 10:30, 11:00, 15:00] -> '10:00 AM to 11:30 AM, 3:00 PM to 3:30 PM' (start of first to end of last)."""
    if not slots:
        return ""
    step = timedelta(minutes=SLOT_MINUTES)
    groups, run = [], [slots[0]]
    for s in slots[1:]:
        if s - run[-1] == step:
            run.append(s)
        else:
            groups.append(run)
            run = [s]
    groups.append(run)
    parts = []
    for g in groups:
        parts.append(fmt_time(g[0]) if len(g) == 1 else f"{fmt_time(g[0])} to {fmt_time(g[-1])}")
    return ", ".join(parts)


def availability(d, include_next=True):
    """Free slots for one date, plus the next free slots on later days if this day has none."""
    d = parse_date(d)
    check_window(d)
    status = day_status(d)
    free = [datetime.combine(d, parse_time(s["time"])) for s in status if s["status"] == "free"]
    res = {
        "date": d.isoformat(), "weekday": d.strftime("%A"), "is_working_day": bool(status),
        "slot_minutes": SLOT_MINUTES,
        "free_times": [s.strftime("%H:%M") for s in free],
        "free_summary": _ranges(free),
        "free_count": len(free),
    }
    if not free and include_next:
        res["next_available"] = next_available(d + timedelta(days=1), 4)
    return res


def next_available(start_date, count=4, max_days=21):
    """The first `count` free slots from start_date onward (looks at most max_days ahead)."""
    out = []
    today = now().date()
    last = today + timedelta(days=BOOKING_DAYS_AHEAD)
    d = max(start_date, today)
    for _ in range(max_days):
        if d > last or len(out) >= count:
            break
        if day_slots(d):
            try:
                for s in day_status(d):
                    if s["status"] == "free" and len(out) < count:
                        dt = datetime.combine(d, parse_time(s["time"]))
                        out.append({"date": d.isoformat(), "time": s["time"], "spoken": spoken(dt)})
            except BookingError:
                break
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------- reading appointments
def _appt(row):
    start, end = row["slot_start"], row["slot_end"]
    return {
        "id": row["id"], "ref_code": row["ref_code"], "customer_name": row["customer_name"],
        "purpose": row.get("purpose") or "", "status": row["status"],
        "channel": row.get("channel") or "", "date": start.date().isoformat(), "time": start.strftime("%H:%M"),
        "end_time": end.strftime("%H:%M"), "spoken": spoken(start),
    }


def list_appointments(upcoming_only=True, limit=100):
    sql = "SELECT * FROM appointments WHERE active = 1"
    params = []
    if upcoming_only:
        sql += " AND slot_end >= %s"
        params.append(now())
    sql += " ORDER BY slot_start ASC LIMIT %s"
    params.append(int(max(1, min(limit, 500))))
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [_appt(r) for r in cur.fetchall()]


def _upcoming_by_ref(cur, ref):
    cur.execute("SELECT * FROM appointments WHERE active = 1 AND ref_code = %s AND slot_end >= %s",
                (ref, now()))
    return cur.fetchall()


def find_by_ref(ref):
    """The caller's own upcoming appointment, found by the reference they were given at booking."""
    ref = normalize_ref(ref)
    with db.connection() as conn, conn.cursor() as cur:
        rows = _upcoming_by_ref(cur, ref)
    if not rows:
        raise BookingError("not_found", "I could not find an upcoming appointment with that reference.")
    return _appt(rows[0])


# ---------------------------------------------------------------- writing
def _validate_slot(start):
    """Raise BookingError unless `start` is a real, free-in-principle, bookable slot."""
    check_window(start.date())
    if start not in day_slots(start.date()):
        times = [fmt_time(s) for s in day_slots(start.date())]
        if not times:
            raise BookingError("closed", f"We are closed on {fmt_date(start.date())}.")
        raise BookingError("not_a_slot", f"{fmt_time(start)} is not a bookable time. Slots are {SLOT_MINUTES} "
                                         f"minutes long between {times[0]} and {times[-1]}, not during the break.")
    if start < _earliest():
        raise BookingError("past", "That time is too soon or has already passed.")


def _taken_error(start):
    alt = availability(start.date())
    return BookingError("slot_taken", f"{spoken(start)} was just taken.",
                        free_times=alt["free_times"], free_summary=alt["free_summary"],
                        next_available=alt.get("next_available", []))


def _new_ref(cur):
    for _ in range(10):
        ref = f"{random.randint(0, 999999):06d}"
        cur.execute("SELECT 1 FROM appointments WHERE ref_code = %s", (ref,))
        if not cur.fetchall():
            return ref
    raise BookingError("system_busy", "Could not create a booking reference, please try again.")


def book(customer_name, start, purpose="", channel="web"):
    """Book one slot. The unique key on (slot_start, active) makes double-booking impossible."""
    name, purpose = clean_name(customer_name), clean_purpose(purpose)
    start = start.replace(second=0, microsecond=0)          # `start` is a datetime
    _validate_slot(start)
    end = start + timedelta(minutes=SLOT_MINUTES)
    stamp = now()
    for attempt in range(3):
        try:
            with db.connection() as conn, conn.cursor() as cur:
                ref = _new_ref(cur)
                cur.execute(
                    "INSERT INTO appointments (ref_code, customer_name, purpose, slot_start, slot_end, "
                    "status, active, channel, created_at, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, 'booked', 1, %s, %s, %s)",
                    (ref, name, purpose, start, end,
                     (channel or "web")[:10], stamp, stamp))
                new_id = cur.lastrowid
                cur.execute("SELECT * FROM appointments WHERE id = %s", (new_id,))
                return _appt(cur.fetchall()[0])
        except BookingError:
            raise
        except Exception as exc:
            if not db.is_integrity_error(exc):
                raise
            # Duplicate key: either someone took the slot a moment ago, or our random ref collided.
            with db.connection() as conn, conn.cursor() as cur:
                taken = start in _booked_starts(cur, start.date())
            if taken:
                raise _taken_error(start) from None
    raise BookingError("system_busy", "Could not complete the booking, please try again.")


def _own(cur, ref):
    rows = _upcoming_by_ref(cur, normalize_ref(ref))
    if not rows:
        raise BookingError("not_found", "I could not find an upcoming appointment with that reference.")
    return rows[0]


def cancel(ref):
    """Cancel the appointment that belongs to this booking reference."""
    with db.connection() as conn, conn.cursor() as cur:
        row = _own(cur, ref)
        cur.execute("UPDATE appointments SET status = 'cancelled', active = NULL, updated_at = %s "
                    "WHERE id = %s AND active = 1", (now(), row["id"]))
        if cur.rowcount != 1:
            raise BookingError("not_found", "That appointment was already cancelled.")
        return _appt(dict(row, status="cancelled"))


def cancel_by_id(appt_id):
    """Staff cancel from the web panel."""
    with db.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM appointments WHERE id = %s AND active = 1", (int(appt_id),))
        rows = cur.fetchall()
        if not rows:
            raise BookingError("not_found", "No active appointment with that id.")
        cur.execute("UPDATE appointments SET status = 'cancelled', active = NULL, updated_at = %s "
                    "WHERE id = %s AND active = 1", (now(), int(appt_id)))
        return _appt(dict(rows[0], status="cancelled"))


def reschedule(ref, new_start):
    """Move the appointment with this reference to a new free slot, atomically (never leaves them with none)."""
    new_start = new_start.replace(second=0, microsecond=0)
    _validate_slot(new_start)
    try:
        with db.connection() as conn, conn.cursor() as cur:
            row = _own(cur, ref)
            if row["slot_start"] == new_start:
                raise BookingError("same_slot", "The appointment is already at that time.")
            cur.execute("UPDATE appointments SET slot_start = %s, slot_end = %s, updated_at = %s "
                        "WHERE id = %s AND active = 1",
                        (new_start, new_start + timedelta(minutes=SLOT_MINUTES), now(), row["id"]))
            cur.execute("SELECT * FROM appointments WHERE id = %s", (row["id"],))
            return _appt(cur.fetchall()[0])
    except BookingError:
        raise
    except Exception as exc:
        if db.is_integrity_error(exc):
            raise _taken_error(new_start) from None
        raise
