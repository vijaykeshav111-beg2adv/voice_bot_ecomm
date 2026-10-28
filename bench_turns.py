"""Benchmark speech-to-text on 24 real turns (10 English, 10 Hindi, 4 Hinglish) with ground truth.

Why this exists: bench_stt.py times 2 clips and only checks "same text as another config". This one
reads KNOWN sentences, so it can score accuracy (word / character error rate) as well as latency, for
both languages, and it compares the settings that could still change Groq STT time or accuracy.

Cost: Groq free tier only (paced to stay under its requests-per-minute limit). No ElevenLabs, no LLM.

Steps (PowerShell, from the voice_bot folder, venv active):
  python bench_turns.py --record             # you read 24 short sentences aloud (~4 min); resumable
  python bench_turns.py                      # benchmark: current .env setup vs alternatives (~5 min)
  python bench_turns.py --variants A,E       # choose variants (see below)

Variants:
  A  current .env setup, exactly as the bot runs it (the production code path)
  B  same, response_format=json (smaller reply; no language field)
  C  language forced to the clip's true language. ORACLE: shows the best case a language hint could
     give, but the bot cannot use it for mixed Hindi/English speech
  D  A + a short vocabulary prompt (--prompt "..." to try your own)
  E  whisper-large-v3 (non-turbo): slower, sometimes better Hindi
  L  local faster-whisper with your WHISPER_* settings (your previous setup; free, offline)

Latency is timed around the transcription call only, with one discarded warm-up request first. For
each clip the variants run back to back in a rotating order so network drift hits all of them equally.
"""
import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import core

BENCH_DIR = core.AUDIO_DIR / "bench"
MANIFEST = BENCH_DIR / "manifest.json"
RESULTS_JSON = core.LOG_DIR / "bench_turns_results.json"
RESULTS_CSV = core.LOG_DIR / "bench_turns_results.csv"

DEFAULT_PROMPT = "Hindi and English conversation. Python, machine learning, neural network, RAM."

# (language, sentence you read aloud, scored). Hinglish is not scored: Whisper may write the English
# words in Roman or Devanagari, both are right, so a word-match score would be meaningless.
PHRASES = [
    ("en", "What is machine learning?", True),
    ("en", "Introduce yourself.", True),
    ("en", "What is the capital of India?", True),
    ("en", "Explain how a neural network learns in simple words.", True),
    ("en", "Tell me a short joke.", True),
    ("en", "How do I reverse a string in Python?", True),
    ("en", "Why is the sky blue during the day?", True),
    ("en", "Remind me to call my mother this evening.", True),
    ("en", "Who wrote the Indian national anthem?", True),
    ("en", "What is the difference between RAM and storage?", True),
    ("hi", "मुझे पायथन के बारे में बताइए।", True),
    ("hi", "अपना परिचय दीजिए।", True),
    ("hi", "भारत की राजधानी क्या है?", True),
    ("hi", "मशीन लर्निंग क्या होती है?", True),
    ("hi", "मुझे एक छोटा सा चुटकुला सुनाइए।", True),
    ("hi", "जयपुर को गुलाबी नगरी क्यों कहते हैं?", True),
    ("hi", "स्वस्थ रहने के लिए क्या खाना चाहिए?", True),
    ("hi", "शाम को मुझे माँ को फोन करने की याद दिलाइए।", True),
    ("hi", "राष्ट्रीय गान किसने लिखा था?", True),
    ("hi", "रैम और स्टोरेज में क्या अंतर होता है?", True),
    ("mix", "Mujhe Python ke baare mein batao.", False),
    ("mix", "Machine learning kya hota hai, simple words mein samjhao.", False),
    ("mix", "आज मौसम कैसा है, what should I wear?", False),
    ("mix", "Mujhe ek chhota sa joke sunao.", False),
]


# ---------------------------------------------------------------- recording
def record_clip(path, max_seconds=10.0, silence=1.0):
    """Record until you pause (same idea as the CLI bot), save 16 kHz mono wav."""
    import queue

    import numpy as np
    import scipy.io.wavfile as wavfile
    import sounddevice as sd

    blocks, frames = queue.Queue(), []
    noise, heard, last_voice = 0.0, False, 0.0
    t0 = time.perf_counter()
    with sd.InputStream(samplerate=core.SAMPLE_RATE, channels=1, dtype="int16",
                        blocksize=int(0.05 * core.SAMPLE_RATE),
                        callback=lambda indata, n, t, s: blocks.put(indata.copy())):
        while True:
            block = blocks.get()
            frames.append(block)
            level = float(np.abs(block.astype(np.int32)).mean())
            now = time.perf_counter() - t0
            if now < 0.4:
                noise = max(noise, level)
                continue
            if level > max(250.0, noise * 3):
                heard, last_voice = True, now
            if now >= max_seconds or (heard and now > 1.0 and now - last_voice > silence):
                break
    wavfile.write(str(path), core.SAMPLE_RATE, np.concatenate(frames))
    return time.perf_counter() - t0


def record_all(only, rerecord):
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (lang, text, scored) in enumerate(PHRASES, 1):
        if only and lang != only:
            continue
        name = f"{lang}_{i:02d}.wav"
        path = BENCH_DIR / name
        manifest.append({"file": name, "lang": lang, "ref": text, "scored": scored})
        if path.exists() and not rerecord:
            print(f"[{i:02d}/{len(PHRASES)}] {name} already recorded (use --rerecord to redo)")
            continue
        label = {"en": "ENGLISH", "hi": "HINDI", "mix": "HINGLISH"}[lang]
        while True:
            input(f"\n[{i:02d}/{len(PHRASES)}] {label}: read this aloud\n    {text}\n  Press ENTER, speak, then pause... ")
            secs = record_clip(path)
            if input(f"  recorded {secs:.1f}s. ENTER = keep, r = redo: ").strip().lower() != "r":
                break
    # keep entries for clips from an earlier partial session
    existing = {m["file"]: m for m in load_manifest(optional=True)}
    for m in manifest:
        existing[m["file"]] = m
    MANIFEST.write_text(json.dumps(list(existing.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSaved {len(existing)} clips + manifest in {BENCH_DIR}")


def load_manifest(optional=False):
    if not MANIFEST.exists():
        if optional:
            return []
        sys.exit("No recordings yet. Run:  python bench_turns.py --record")
    return [m for m in json.loads(MANIFEST.read_text(encoding="utf-8")) if (BENCH_DIR / m["file"]).exists()]


# ---------------------------------------------------------------- scoring
def norm_text(text):
    """Lower-case, drop punctuation (incl. the Hindi danda) and ignore spelling variants that are the
    same word in Hindi: nukta (फ़ vs फ) and chandrabindu vs anusvara (माँ vs मां)."""
    text = unicodedata.normalize("NFC", text.lower())
    text = text.replace("\u093c", "").replace("\u0901", "\u0902")
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    r, h = norm_text(ref).split(), norm_text(hyp).split()
    return edit_distance(r, h) / max(1, len(r))


def cer(ref, hyp):
    r, h = norm_text(ref).replace(" ", ""), norm_text(hyp).replace(" ", "")
    return edit_distance(r, h) / max(1, len(r))


def script_flag(lang, text):
    """Arabic-script output for Hindi speech (Whisper sometimes labels Hindustani as Urdu), or
    Devanagari for English speech."""
    if lang == "hi" and any("\u0600" <= c <= "\u06ff" for c in text):
        return "urdu-script"
    if lang == "en" and any("\u0900" <= c <= "\u097f" for c in text):
        return "devanagari"
    return ""


# ---------------------------------------------------------------- transcription variants
class Pacer:
    """Keeps Groq calls under the free tier's requests-per-minute limit."""

    def __init__(self, rpm):
        self.gap = 60.0 / rpm if rpm else 0.0
        self.last = 0.0

    def wait(self):
        left = self.last + self.gap - time.perf_counter()
        if left > 0:
            time.sleep(left)
        self.last = time.perf_counter()


def groq_call(data, suffix, *, model=None, fmt="verbose_json", language=None, prompt=None):
    """One Groq transcription with explicit options. Returns (text, language)."""
    client = core.groq_client()
    try:
        client = client.with_options(max_retries=0, timeout=max(core.GROQ_STT_TIMEOUT, 15))
    except AttributeError:
        pass
    kw = dict(file=(f"speech{suffix}", data), model=model or core.GROQ_STT_MODEL,
              response_format=fmt, temperature=0.0)
    if language:
        kw["language"] = language
    if prompt:
        kw["prompt"] = prompt
    resp = client.audio.transcriptions.create(**kw)
    get = (lambda k: resp.get(k)) if isinstance(resp, dict) else (lambda k: getattr(resp, k, ""))
    return (get("text") or "").strip(), core._norm_lang(get("language"))


def make_variants(args):
    return {
        "A": ("current .env", lambda d, s, c: core._transcribe_groq(d, s), True),
        "B": ("json format", lambda d, s, c: groq_call(d, s, fmt="json"), True),
        "C": ("language forced (ORACLE)",
              lambda d, s, c: groq_call(d, s, language=c["lang"] if c["lang"] in ("en", "hi") else None), True),
        "D": ("with vocabulary prompt", lambda d, s, c: groq_call(d, s, prompt=args.prompt), True),
        "E": ("whisper-large-v3", lambda d, s, c: groq_call(d, s, model="whisper-large-v3"), True),
        "L": (f"local {core.WHISPER_MODEL} (previous)", lambda d, s, c: core._transcribe_local(d, s), False),
    }


def is_rate_limit(exc):
    return getattr(exc, "status_code", None) == 429 or type(exc).__name__ == "RateLimitError"


def timed_call(fn, data, suffix, clip, pacer, uses_network):
    """Returns (seconds, text, lang, error, retries). 429s are waited out and not counted as time."""
    retries = 0
    while True:
        if uses_network:
            pacer.wait()
        t0 = time.perf_counter()
        try:
            text, lang = fn(data, suffix, clip)
            return time.perf_counter() - t0, text, lang, "", retries
        except Exception as exc:
            if is_rate_limit(exc) and retries < 3:
                retries += 1
                wait = 20 * retries
                print(f"    rate limited, waiting {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            return time.perf_counter() - t0, "", "", f"{type(exc).__name__}: {str(exc)[:80]}", retries


def prepare_payload(path, fmt, kbps):
    """Bytes + suffix to upload. The browser sends webm/opus (~100 kbps), so by default we do the same
    when ffmpeg exists; otherwise the wav is sent as it is (about 5x bigger, slightly pessimistic)."""
    if fmt == "auto":
        fmt = "webm" if shutil.which("ffmpeg") else "wav"
    if fmt == "wav":
        return path.read_bytes(), ".wav"
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-c:a", "libopus", "-b:a", f"{kbps}k",
         "-f", "webm", "pipe:1"], capture_output=True, check=True).stdout
    return out, ".webm"


# ---------------------------------------------------------------- reporting
def pctl(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))] if vals else 0.0


def summarize(results, clips, keys):
    by_clip = {c["file"]: c for c in clips}
    base = results.get("A", {})
    rows = {}
    for k in keys:
        rs = [r for r in results[k].values() if not r["error"]]
        row = {"n": len(rs), "errors": len(results[k]) - len(rs)}
        lat = [r["seconds"] for r in rs]
        row.update(mean=statistics.mean(lat) if lat else 0, median=statistics.median(lat) if lat else 0,
                   p90=pctl(lat, 0.9), max=max(lat) if lat else 0)
        for lang in ("en", "hi"):
            sc = [r for f, r in results[k].items()
                  if not r["error"] and by_clip[f]["lang"] == lang and by_clip[f]["scored"]]
            row[f"{lang}_wer"] = statistics.mean(wer(by_clip[r["file"]]["ref"], r["text"]) for r in sc) if sc else None
            row[f"{lang}_cer"] = statistics.mean(cer(by_clip[r["file"]]["ref"], r["text"]) for r in sc) if sc else None
        det = [(by_clip[f]["lang"], r["lang"]) for f, r in results[k].items()
               if not r["error"] and r["lang"] and by_clip[f]["lang"] in ("en", "hi")]
        row["lang_ok"] = f"{sum(a == b for a, b in det)}/{len(det)}" if det else "-"
        row["script_flags"] = sum(1 for f, r in results[k].items()
                                  if not r["error"] and script_flag(by_clip[f]["lang"], r["text"]))
        paired = [r["seconds"] - base[f]["seconds"] for f, r in results[k].items()
                  if f in base and not r["error"] and not base[f]["error"]]
        row["paired_ms"] = statistics.median(paired) * 1000 if paired and k != "A" else None
        rows[k] = row
    return rows


def pct(x):
    return "   -  " if x is None else f"{x * 100:5.1f}%"


def print_report(rows, names, clips, results, keys):
    print("\n" + "=" * 118)
    print(f"{'variant':<34}{'n':>3}{'mean':>7}{'median':>8}{'p90':>7}{'max':>7}{'vs A':>9} |"
          f"{'EN WER':>8}{'EN CER':>8}{'HI WER':>8}{'HI CER':>8} | lang  script")
    for k in keys:
        r = rows[k]
        vs = "" if r["paired_ms"] is None else f"{r['paired_ms']:+.0f} ms"
        print(f"{k + '  ' + names[k]:<34}{r['n']:>3}{r['mean']:7.2f}{r['median']:8.2f}{r['p90']:7.2f}{r['max']:7.2f}{vs:>9} |"
              f"{pct(r['en_wer']):>8}{pct(r['en_cer']):>8}{pct(r['hi_wer']):>8}{pct(r['hi_cer']):>8} |"
              f" {r['lang_ok']:>5} {r['script_flags']:>4}" + (f"   ({r['errors']} errors)" if r["errors"] else ""))
    print("Times in seconds; 'vs A' = median per-clip difference to the current setup (negative = faster).")
    print("WER/CER = word/character error rate against what you read aloud (lower is better).")

    print("\nClips where any variant has WER > 20%  (ref -> hypothesis):")
    shown = 0
    for c in clips:
        if not c["scored"]:
            continue
        bad = [(k, results[k][c["file"]]) for k in keys if c["file"] in results[k]
               and not results[k][c["file"]]["error"] and wer(c["ref"], results[k][c["file"]]["text"]) > 0.2]
        if bad:
            shown += 1
            print(f"  {c['file']}  ref: {c['ref']}")
            for k, r in bad:
                print(f"      {k}: {r['text']}  [{r['lang'] or '-'}] WER {wer(c['ref'], r['text']) * 100:.0f}%")
    if not shown:
        print("  none")

    print("\nHinglish clips (not scored, read them yourself):")
    for c in clips:
        if not c["scored"]:
            print(f"  said: {c['ref']}")
            for k in keys:
                r = results[k].get(c["file"])
                if r and not r["error"]:
                    print(f"      {k}: {r['text']}  [{r['lang'] or '-'}]")


def recommend(rows, names, keys):
    """Suggest a change only when it is faster AND not less accurate than the current setup."""
    a = rows.get("A")
    print("\n" + "=" * 118)
    if not a:
        return
    best = None
    for k in keys:
        if k in ("A", "C", "L") or rows[k]["paired_ms"] is None:
            continue
        r = rows[k]
        same_acc = all((r[m] is None or a[m] is None or r[m] <= a[m] + 0.01) for m in ("en_wer", "hi_wer"))
        if same_acc and r["paired_ms"] <= -30 and (best is None or r["paired_ms"] < rows[best]["paired_ms"]):
            best = k
    if best:
        print(f"Variant {best} ({names[best]}) is faster by {-rows[best]['paired_ms']:.0f} ms (median per clip) with no accuracy loss.")
        print({"B": "  .env:  GROQ_STT_RESPONSE_FORMAT=json",
               "D": "  .env:  GROQ_STT_PROMPT=<the prompt used by --prompt>",
               "E": "  .env:  GROQ_STT_MODEL=whisper-large-v3"}.get(best, ""))
    else:
        print("No variant is both faster (by 30 ms or more) and at least as accurate as the current setup.")
        print("Keep the current settings. Differences under ~30-50 ms are within network noise at this sample size.")
    acc = [k for k in keys if k not in ("A", "C", "L")
           and all(rows[k][m] is not None and a[m] is not None and rows[k][m] < a[m] - 0.02 for m in ("en_wer", "hi_wer"))]
    for k in acc:
        print(f"Note: {k} ({names[k]}) is clearly MORE accurate than A (WER lower by > 2 points in both languages).")
    if "L" in rows and rows["L"]["n"]:
        print(f"Previous local setup for comparison: median {rows['L']['median']:.2f} s vs {a['median']:.2f} s on Groq.")
    if "C" in rows and rows["C"]["n"]:
        c = rows["C"]
        print(f"Oracle language hint: HI WER {pct(c['hi_wer']).strip()} vs {pct(a['hi_wer']).strip()}; "
              f"EN WER {pct(c['en_wer']).strip()} vs {pct(a['en_wer']).strip()} (upper bound of what a language hint could add).")


def save(results, clips, keys, names):
    by_clip = {c["file"]: c for c in clips}
    RESULTS_JSON.write_text(json.dumps({"variants": {k: names[k] for k in keys}, "results": results},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
    with RESULTS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["variant", "file", "lang_expected", "lang_detected", "seconds", "wer", "cer", "reference", "transcript", "error"])
        for k in keys:
            for fn, r in results[k].items():
                c = by_clip[fn]
                sc = c["scored"] and not r["error"]
                w.writerow([k, fn, c["lang"], r["lang"], f"{r['seconds']:.3f}",
                            f"{wer(c['ref'], r['text']):.3f}" if sc else "", f"{cer(c['ref'], r['text']):.3f}" if sc else "",
                            c["ref"], r["text"], r["error"]])
    print(f"\nPer-clip results: {RESULTS_CSV}")


# ---------------------------------------------------------------- main
def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="record the 24 sentences (resumable)")
    ap.add_argument("--rerecord", action="store_true", help="with --record: redo clips that already exist")
    ap.add_argument("--only", choices=["en", "hi", "mix"], help="restrict recording/benchmark to one language")
    ap.add_argument("--variants", default="A,B,D,L", help="comma list from A,B,C,D,E,L (default A,B,D,L)")
    ap.add_argument("--runs", type=int, default=1, help="timed runs per clip and variant, median kept (default 1)")
    ap.add_argument("--rpm", type=int, default=18, help="max Groq requests per minute (free tier allows 20); 0 = no limit")
    ap.add_argument("--format", choices=["auto", "webm", "wav"], default="auto", help="upload format (auto = webm if ffmpeg is installed)")
    ap.add_argument("--opus-kbps", type=int, default=96, help="bitrate when converting to webm (browser default is about 100)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="text for variant D")
    args = ap.parse_args(argv)

    if args.record:
        record_all(args.only, args.rerecord)
        return
    clips = [c for c in load_manifest() if not args.only or c["lang"] == args.only]
    if not clips:
        sys.exit("No clips for that selection.")
    variants = make_variants(args)
    keys = [k.strip().upper() for k in args.variants.split(",") if k.strip()]
    if "A" not in keys:
        keys.insert(0, "A")      # every comparison is relative to the current setup
    unknown = [k for k in keys if k not in variants]
    if unknown:
        sys.exit(f"Unknown variant(s): {', '.join(unknown)}")
    if any(variants[k][2] for k in keys):
        core.require_keys()
    names = {k: variants[k][0] for k in keys}

    payloads = {c["file"]: prepare_payload(BENCH_DIR / c["file"], args.format, args.opus_kbps) for c in clips}
    sample = next(iter(payloads.values()))
    print(f"{len(clips)} clips, variants {','.join(keys)}, upload format {sample[1]} "
          f"(~{statistics.mean(len(p[0]) for p in payloads.values()) / 1024:.0f} KB per clip), "
          f"Groq model {core.GROQ_STT_MODEL}")

    pacer = Pacer(args.rpm)
    first = clips[0]
    if any(variants[k][2] for k in keys):          # warm-up: DNS + TLS + model, not counted
        print("warm-up ...", flush=True)
        timed_call(variants["A"][1], *payloads[first["file"]], first, pacer, True)
    if "L" in keys:
        print("loading local model (not counted) ...", flush=True)
        timed_call(variants["L"][1], *payloads[first["file"]], first, pacer, False)

    results = {k: {} for k in keys}
    for i, clip in enumerate(clips):
        order = keys[i % len(keys):] + keys[:i % len(keys)]      # rotate so no variant is always first
        for k in order:
            fn, uses_net = variants[k][1], variants[k][2]
            data, suffix = payloads[clip["file"]]
            runs = [timed_call(fn, data, suffix, clip, pacer, uses_net) for _ in range(max(1, args.runs))]
            good = [r for r in runs if not r[3]]
            if good:
                secs = statistics.median(r[0] for r in good)
                text, lang = good[-1][1], good[-1][2]
                err, retries = "", sum(r[4] for r in runs)
            else:
                secs, text, lang, err, retries = runs[-1][0], "", "", runs[-1][3], runs[-1][4]
            results[k][clip["file"]] = {"file": clip["file"], "seconds": secs, "text": text, "lang": lang,
                                        "error": err, "retries": retries}
        a = results["A"][clip["file"]]
        print(f"[{i + 1:02d}/{len(clips)}] {clip['file']:<10} A {a['seconds']:.2f}s [{a['lang'] or '-'}] {a['text'][:60]}"
              + (f"   ERROR {a['error']}" if a["error"] else ""), flush=True)

    rows = summarize(results, clips, keys)
    print_report(rows, names, clips, results, keys)
    recommend(rows, names, keys)
    save(results, clips, keys, names)


if __name__ == "__main__":
    main()
