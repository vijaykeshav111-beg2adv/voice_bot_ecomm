"""Web server for the voice bot.   Run:  python server.py   then open http://localhost:8000"""
import base64
import hmac
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, File, Form, Header, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import booking
import core
import db
from knowledge_base import search_knowledge, ingest_documents

app = FastAPI(title="Voice E-commerce Assistant")

# Staff key: when ADMIN_KEY is set, the endpoints that expose patient data (appointment list with patient names,
# staff cancel, conversation history/CSV) need it, sent as the X-Admin-Key header (or ?key= for the CSV link).
# The public booking form, slot list and chat stay open. Leave it empty only for local use.
ADMIN_KEY = os.getenv("ADMIN_KEY", "").strip()
HOST = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
PORT = int(os.getenv("PORT", "8000") or 8000)


def _admin_denied(key):
    """None if allowed, else a 401 response."""
    if not ADMIN_KEY or hmac.compare_digest((key or "").encode(), ADMIN_KEY.encode()):
        return None
    return JSONResponse({"error": "staff_key_required", "message": "Enter the staff key to see this."}, 401)
STATIC = core.BASE_DIR / "static"

_sessions = {}
_sessions_lock = threading.Lock()


def conversation_for(session_id):
    with _sessions_lock:
        return _sessions.setdefault(session_id, core.new_conversation())


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    stt = (f"groq {core.GROQ_STT_MODEL}" if core.STT_BACKEND == "groq"
           else f"whisper-{core.WHISPER_MODEL} (local)")
    return {
        "llm": core.GROQ_MODEL, "tts": core.TTS_MODEL, "stt": stt,
        "max_record_seconds": core.MAX_WEB_RECORD_SECONDS,
        "missing_keys": core.missing_keys(),
        "store": {"enabled": True, "business": os.getenv("BUSINESS_NAME", "Everyday Online Store")},
    }


@app.get("/api/knowledge/search")
def knowledge_search(q: str = Query(..., min_length=1, max_length=1000), top_k: int = Query(3, ge=1, le=5)):
    """Search the local RAG knowledge base, including PDF, Markdown, and text documents."""
    return search_knowledge(q, top_k)


@app.get("/api/knowledge/status")
def knowledge_status():
    """Report local document-ingestion/index statistics."""
    return ingest_documents()


@app.post("/api/knowledge/ingest")
def knowledge_ingest():
    """Force re-ingestion after adding or editing approved knowledge documents."""
    return ingest_documents(force=True)


@app.post("/api/warm")
def warm():
    """The browser calls this when recording starts, so the HTTPS connections to Groq and
    ElevenLabs are already open when the user finishes speaking."""
    core.prewarm()
    return {"ok": True}


def _suffix_for(upload: UploadFile):
    ct = (upload.content_type or "").lower()
    for key, ext in (("webm", ".webm"), ("ogg", ".ogg"), ("mp4", ".mp4"), ("m4a", ".m4a"),
                     ("wav", ".wav"), ("mpeg", ".mp3")):
        if key in ct:
            return ext
    return ".webm"


def _line(event):
    """Encode one pipeline event as a line of JSON (NDJSON). Audio bytes become base64."""
    raw_audio = event.get("audio")
    if event.get("type") == "clip":
        event = {k: v for k, v in event.items() if k != "audio"}
        event["audio_b64"] = base64.b64encode(raw_audio).decode()
    return json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/api/turn")
def turn(
    session_id: str = Form(...),
    audio: UploadFile | None = File(None),
    text: str = Form(""),
    recording_seconds: float = Form(0.0),
):
    """One conversation turn: audio (or typed text) -> STT -> LLM -> TTS.

    The reply is STREAMED as newline-delimited JSON events (see core.run_turn): the first
    spoken sentence reaches the browser while the rest of the answer is still being generated.
    """
    miss = core.missing_keys()
    if miss:
        return JSONResponse({"error": f"{', '.join(miss)} missing in .env", "stage": "config"}, 400)

    audio_bytes = audio.file.read() if audio is not None else None
    suffix = _suffix_for(audio) if audio is not None else ".webm"
    if not audio_bytes:
        audio_bytes = None

    rec = core.new_record(str(uuid.uuid4()), source="voice" if audio_bytes else "text")
    rec["recording_seconds"] = round(max(0.0, min(recording_seconds, 600.0)), 4)
    conv = conversation_for(session_id)

    def stream():
        for event in core.run_turn(conv, rec, audio=audio_bytes, audio_suffix=suffix, text=text):
            yield _line(event)

    return StreamingResponse(
        stream(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class Playback(BaseModel):
    playback_seconds: float
    heard_after_seconds: float | None = None


@app.post("/api/turn/{turn_id}/playback")
def playback(turn_id: str, body: Playback):
    """The browser reports how long the reply played and how long after the user stopped
    speaking it first became audible. This completes the turn's totals and is logged."""
    rec = core.report_playback(turn_id, body.playback_seconds, body.heard_after_seconds)
    if rec is None:
        return JSONResponse({"error": "unknown turn"}, 404)
    return {"turn": rec}


@app.get("/api/history")
def history(limit: int = 50, x_admin_key: str | None = Header(None)):
    denied = _admin_denied(x_admin_key)
    if denied:
        return denied
    return {"turns": core.read_metadata(max(1, min(limit, 500)))}


@app.post("/api/reset")
def reset(session_id: str = Form(...)):
    with _sessions_lock:
        _sessions.pop(session_id, None)
    return {"ok": True}


# ---------------------------------------------------------------- appointments (REST)
# The same functions the voice/text assistant uses through its tools, exposed for the web panel.
# There is no login: the server binds to 127.0.0.1. Add authentication before exposing it.
_STATUS = {"slot_taken": 409, "not_found": 404, "ambiguous": 409, "limit": 409, "same_slot": 409}


def _guarded(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except booking.BookingError as exc:
        extra = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in exc.extra.items()}
        return JSONResponse({"error": exc.code, "message": exc.message, **extra}, _STATUS.get(exc.code, 400))
    except Exception as exc:
        logging.exception("Database request failed")
        return JSONResponse({"error": "database_unavailable", "message": f"{type(exc).__name__}: {exc}"}, 503)


@app.get("/api/booking/status")
def booking_status():
    """Health check for the MySQL connection (the page shows a notice if this fails)."""
    ok, message = db.status()
    return {"ok": ok, "message": message, "business": booking.BUSINESS_NAME,
            "timezone": booking.TZ_NAME, "slot_minutes": booking.SLOT_MINUTES,
            "staff_key_required": bool(ADMIN_KEY),
            "today": booking.now().date().isoformat(),
            "last_date": (booking.now().date() + timedelta(days=booking.BOOKING_DAYS_AHEAD)).isoformat()}


@app.get("/api/slots")
def slots(date: str | None = None):
    """Every slot of one day with its state: free / booked / past."""
    def go():
        d = booking.parse_date(date) if date else booking.now().date()
        status = booking.day_status(d)
        return {"date": d.isoformat(), "weekday": d.strftime("%A"), "is_working_day": bool(status),
                "slots": status, "free_count": sum(1 for x in status if x["status"] == "free")}
    return _guarded(go)


@app.get("/api/appointments")
def appointments(limit: int = 100, upcoming: bool = True, x_admin_key: str | None = Header(None)):
    denied = _admin_denied(x_admin_key)
    if denied:
        return denied
    return _guarded(lambda: {"appointments": booking.list_appointments(upcoming, limit)})


class BookingIn(BaseModel):
    customer_name: str
    date: str
    time: str
    purpose: str = ""


@app.post("/api/appointments")
def create_appointment(body: BookingIn):
    def go():
        start = datetime.combine(booking.parse_date(body.date), booking.parse_time(body.time))
        return {"appointment": booking.book(body.customer_name, start, body.purpose, "form")}
    return _guarded(go)


@app.post("/api/appointments/{appt_id}/cancel")
def cancel_appointment(appt_id: int, x_admin_key: str | None = Header(None)):
    denied = _admin_denied(x_admin_key)
    if denied:
        return denied
    return _guarded(lambda: {"appointment": booking.cancel_by_id(appt_id)})


@app.get("/api/export.csv")
def export_csv(key: str | None = Query(None), x_admin_key: str | None = Header(None)):
    denied = _admin_denied(x_admin_key or key)
    if denied:
        return denied
    if not core.METADATA_CSV.exists():
        return JSONResponse({"error": "no turns recorded yet"}, 404)
    return FileResponse(core.METADATA_CSV, media_type="text/csv", filename="conversation_metadata.csv")



# E-commerce API (demo catalog and SQLite order records)
import ecommerce
from fastapi import Body

@app.get("/api/products")
def products(q: str = "", category: str = "", limit: int = 20):
    return {"products": ecommerce.list_products(q, category, limit)}

@app.get("/api/products/{sku}")
def product_details(sku: str):
    product = ecommerce.get_product(sku)
    if not product:
        return JSONResponse({"ok": False, "error": "not_found", "message": "Product not found."}, 404)
    return {"ok": True, "product": product}

@app.post("/api/orders")
def create_store_order(body: dict = Body(...)):
    result = ecommerce.create_order(body.get("customer_name", ""), body.get("customer_email", ""), body.get("address", ""), body.get("items", []), body.get("confirmed") is True)
    return JSONResponse(result, 200 if result.get("ok") else 400)

@app.get("/api/orders/{order_ref}")
def store_order_status(order_ref: str):
    result = ecommerce.find_order(order_ref)
    return JSONResponse(result, 200 if result.get("ok") else 404)

@app.get("/api/orders")
def recent_store_orders(limit: int = 20, x_admin_key: str | None = Header(None)):
    denied = _admin_denied(x_admin_key)
    if denied:
        return denied
    return {"orders": ecommerce.list_orders(limit)}

if __name__ == "__main__":
    if core.missing_keys():
        print("Warning:", ", ".join(core.missing_keys()), "missing - the UI will show a setup notice.")
    else:
        core.prewarm(force=True)
        if core.STT_BACKEND != "groq":
            print("Loading local Whisper model (first run downloads it)...")
            core.warm_local_whisper()
        else:
            core.preload_local_fallback()   # background; so a Groq hiccup never also pays a model load
    if core.BOOKING_ENABLED:
        ok, msg = db.status()
        print(("MySQL: connected to " if ok else "MySQL NOT available - booking will fail until this is fixed: ") + msg)
    print(f"STT: {core.STT_BACKEND} | LLM: {core.GROQ_MODEL} | TTS: {core.TTS_MODEL}")
    if HOST not in ("127.0.0.1", "localhost") and not ADMIN_KEY:
        print("WARNING: listening on a public address without ADMIN_KEY - set ADMIN_KEY in .env.")
    print(f"Open http://localhost:{PORT}   (latency lines are written to logs/voicebot.log)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
