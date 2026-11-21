"""Hype commentary voice + captions + 1080x1920 render.

The voice is NOT a narration: a few short, fun reaction lines placed at the exact
moments they react to. Between lines the real game audio plays at full volume.

Fixes vs. the old version:
  * captions were plain SRT sentences rendered in a 384x288 coordinate space
    (huge text, MarginV pushed them off-screen). Now: a proper 1080x1920 ASS file
    with big 2-3 word "TikTok style" captions and the spoken word highlighted.
  * banner text was drawn with drawtext and could run off the screen; it now wraps.
  * amix halved the voice volume; now the game audio is ducked under the voice
    (sidechain compression) and the final mix is loudness-normalised to -14 LUFS.
  * voiceover longer than the max length is sped up instead of being cut mid-sentence.
  * works with vertical sources and sources without audio.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import textwrap
from pathlib import Path

from common import READY_DIR, VO_DIR, env, load_config, probe, run

log = logging.getLogger("autopilot")

# --------------------------------------------------------------------------- TTS

async def _edge_tts(text: str, voice: str, rate: str, out: Path, boundary: str = "WordBoundary",
                    pitch: str = "+0Hz", volume: str = "+0%") -> list[dict]:
    import edge_tts
    words: list[dict] = []
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume, boundary=boundary)
    with open(out, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                start = chunk["offset"] / 1e7
                end = start + chunk["duration"] / 1e7
                if chunk["type"] == "WordBoundary":
                    words.append({"text": chunk["text"], "start": start, "end": end})
                else:  # spread the sentence's words across its time span
                    words.extend((w | {"start": w["start"] + start, "end": w["end"] + start})
                                 for w in _even_word_timings(chunk["text"], end - start, lead=0.0))
    return words


def _even_word_timings(text: str, duration: float, lead: float = 0.15) -> list[dict]:
    tokens = text.split()
    weights = [max(2, len(t)) for t in tokens]
    total = sum(weights) or 1
    t, out = lead, []
    span = max(0.2, duration - 2 * lead)
    for tok, w in zip(tokens, weights):
        d = span * w / total
        out.append({"text": tok, "start": t, "end": t + d})
        t += d
    return out


def _edge_line(text: str, voice: str, rate: str, out: Path) -> list[dict]:
    """One short line with edge-tts. Word timings relative to the line start.
    Tries the energetic settings (pitch/volume lift) first, then plain settings."""
    vcfg = load_config().get("voice", {})
    styles = [(vcfg.get("pitch", "+2Hz"), vcfg.get("volume", "+10%")), ("+0Hz", "+0%")]
    for pitch, volume in styles:
        for boundary in ("WordBoundary", "SentenceBoundary"):
            try:
                words = asyncio.run(_edge_tts(text, voice, rate, out, boundary, pitch, volume))
                if out.exists() and out.stat().st_size > 2000:
                    return words
            except Exception as exc:  # noqa: BLE001
                log.debug("[voice] edge %s/%s/%s: %s", voice, boundary, pitch, exc)
                time.sleep(1.5)
    raise RuntimeError(f"edge-tts {voice} gave no audio")


_gemini_tts_cache: list[str] | None = None


def _gemini_tts_models() -> list[str]:
    """Gemini text-to-speech models this key can use (names change, so ask the API)."""
    global _gemini_tts_cache
    if _gemini_tts_cache is None:
        _gemini_tts_cache = list(load_config().get("voice", {}).get("gemini_tts_models", []))
        try:
            import gemini_writer
            for m in gemini_writer._client_beta().models.list():
                name = (m.name or "").removeprefix("models/")
                if "tts" in name and name not in _gemini_tts_cache:
                    _gemini_tts_cache.append(name)
            # newest flash TTS first (best quality, bigger free quota), lite/pro last
            _gemini_tts_cache.sort(key=lambda n: ("flash" not in n, "lite" in n,
                                                  -float((re.findall(r"(\d+(?:\.\d+)?)", n) or ["0"])[0])))
            log.info("[voice] Gemini TTS models: %s", _gemini_tts_cache)
        except Exception as exc:  # noqa: BLE001
            log.warning("[voice] could not list Gemini TTS models: %.150s", exc)
    return _gemini_tts_cache


class HumanVoiceUnavailable(RuntimeError):
    """Gemini's human voice failed and config says never to fall back to a robotic voice."""


class _SplitError(RuntimeError):
    """The one-request audio can't be cut into the script's lines (or contains extra speech)."""


def _tts_prompt(lines: list[str]) -> str:
    """What Gemini TTS receives: ONLY the script, nothing else. (Style instructions in the request get read
    out loud by the TTS models - that put "read this like a 25-year-old..." into a published video.)
    The casual YouTuber style comes from how the script itself is written; Fenrir performs it."""
    return "\n\n".join(t.strip() for t in lines if t.strip())


def _expected_sec(text: str) -> float:
    return len(text.split()) / 2.6


def _gemini_line(text: str, out: Path, model: str, voice: str) -> None:
    """One line, one Gemini TTS request (used when the all-in-one audio can't be split)."""
    wav = _gemini_tts_wav(_tts_prompt([text]), model, voice, out.with_suffix(".wav"))
    # trim silence at both ends so the line lands exactly on its moment
    trim = "silenceremove=start_periods=1:start_threshold=-45dB"
    res = run(["ffmpeg", "-y", "-i", str(wav), "-af", f"{trim},areverse,{trim},areverse", "-ar", "48000",
               str(out)], timeout=60)
    wav.unlink(missing_ok=True)
    if res.returncode != 0 or probe(out)["duration"] < 0.3:
        raise RuntimeError("Gemini TTS returned no usable audio")
    if probe(out)["duration"] > _expected_sec(text) * 1.8 + 2.5:
        raise RuntimeError(f"line audio far too long for '{text[:40]}' - extra speech")


def _rate_limited(exc: Exception) -> bool:
    msg = str(exc)
    return ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and "PerDay" not in msg and "per day" not in msg.lower()


def _gemini_voice(lines: list[dict], paths: list[Path], voice: str) -> str:
    """Voice all lines with Gemini's human voice. Tries every TTS model; on a per-minute rate limit waits
    once and retries; if the one-request audio can't be split, records the lines one by one.
    Returns the model used, raises RuntimeError if nothing worked."""
    errors = []
    models = _gemini_tts_models()
    if not models:
        raise RuntimeError("no Gemini TTS model available for this API key")
    for model in models:
        for attempt in (1, 2):
            try:
                try:
                    _gemini_all_lines(lines, paths, model, voice)
                except _SplitError as exc:
                    log.info("[voice] %s - recording the %d lines one by one", exc, len(lines))
                    for ln, path in zip(lines, paths):
                        _gemini_line(ln["text"], path, model, voice)
                return model
            except Exception as exc:  # noqa: BLE001
                if attempt == 1 and _rate_limited(exc):
                    log.info("[voice] %s rate-limited - waiting 25s and retrying once", model)
                    time.sleep(25)
                    continue
                log.warning("[voice] Gemini TTS %s failed (%.150s)", model, exc)
                errors.append(f"{model}: {str(exc)[:80]}")
                break
    raise RuntimeError("; ".join(errors) or "Gemini TTS failed")


def _speech_segments(path: Path, min_silence: float) -> list[tuple[float, float]]:
    """[(start, end)] of speech parts, using ffmpeg silencedetect."""
    res = run(["ffmpeg", "-hide_banner", "-i", str(path), "-af",
               f"silencedetect=noise=-38dB:d={min_silence}", "-f", "null", "-"], timeout=60)
    dur = probe(path)["duration"]
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", res.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", res.stderr)]
    segs, cursor = [], 0.0
    for a, b in zip(starts, ends + [dur] * (len(starts) - len(ends))):
        if a - cursor > 0.15:
            segs.append((cursor, a))
        cursor = b
    if dur - cursor > 0.15:
        segs.append((cursor, dur))
    return segs


def _gemini_tts_wav(prompt: str, model: str, voice: str, out_wav: Path) -> Path:
    """One Gemini TTS request -> 48 kHz wav."""
    import base64
    import gemini_writer
    from google.genai import types
    resp = gemini_writer._client_beta().models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)))),
    )
    part = resp.candidates[0].content.parts[0]
    data = part.inline_data.data
    if isinstance(data, str):
        data = base64.b64decode(data)
    m = re.search(r"rate=(\d+)", part.inline_data.mime_type or "")
    raw = out_wav.with_suffix(".pcm")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(data)
    run(["ffmpeg", "-y", "-f", "s16le", "-ar", m.group(1) if m else "24000", "-ac", "1", "-i", str(raw),
         "-ar", "48000", str(out_wav)], timeout=60, check=True)
    raw.unlink(missing_ok=True)
    return out_wav


SAMPLE_LINES = [
    "GTA 6 is finally out and it's insane. Let me show you what it's really like.",
    "One of the craziest things: cars now need gas. See this? My tank isn't even full.",
    "Nah bro, he's not seriously trying this... wait wait wait - OH NO!",
    "This is Jason's house. It's pretty small, but hey, it's better than nothing.",
]
SAMPLE_VOICES = ["Fenrir", "Zubenelgenubi", "Achird", "Umbriel", "Algenib", "Orus", "Sadachbia", "Charon"]


def voice_samples() -> int:
    """'voice_samples' slot: the same lines in several voices -> mp3 files in the run's Artifacts."""
    from common import write_summary
    READY_DIR.mkdir(parents=True, exist_ok=True)
    vcfg = load_config().get("voice", {})
    prompt = _tts_prompt(SAMPLE_LINES)
    rows = ["## 🎙️ Voice samples", "", "Download **voice-samples** from Artifacts below, listen, then tell Claude "
            "which one to use.", "", "| File | Voice | Result |", "|---|---|---|"]
    models = _gemini_tts_models()
    for i, voice in enumerate(SAMPLE_VOICES):
        ok, err = False, ""
        for model in (models[i % len(models):] + models[:i % len(models)]) if models else []:
            try:
                wav = _gemini_tts_wav(prompt, model, voice, READY_DIR / f"gemini_{voice}.wav")
                run(["ffmpeg", "-y", "-i", str(wav), "-b:a", "128k", str(wav.with_suffix(".mp3"))], timeout=60)
                wav.unlink(missing_ok=True)
                ok = True
                break
            except Exception as exc:  # noqa: BLE001
                err = str(exc)[:80]
        rows.append(f"| gemini_{voice}.mp3 | Gemini {voice} | {'✅' if ok else '❌ ' + err} |")
        log.info("[voice] sample %s: %s", voice, "ok" if ok else err)
        time.sleep(4)
    for edge_voice in ("en-US-AndrewNeural", "en-US-AndrewMultilingualNeural"):
        paths = [READY_DIR / f"edge_{edge_voice}_{k}.mp3" for k in range(len(SAMPLE_LINES))]
        try:
            for text, path in zip(SAMPLE_LINES, paths):
                _edge_line(text, edge_voice, vcfg.get("rate", "+23%"), path)
            lst = READY_DIR / "concat.txt"
            lst.write_text("".join(f"file '{p.name}'\n" for p in paths))
            run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                 str(READY_DIR / f"edge_{edge_voice}.mp3")], timeout=60)
            ok = True
        except Exception:  # noqa: BLE001
            ok = False
        for p in paths:
            p.unlink(missing_ok=True)
        (READY_DIR / "concat.txt").unlink(missing_ok=True)
        rows.append(f"| edge_{edge_voice}.mp3 | Microsoft {edge_voice} (current) | {'✅' if ok else '❌'} |")
    write_summary("\n".join(rows))
    return 0


def _align_cuts(segs: list[tuple[float, float]], words: list[int]) -> list[tuple[float, float]] | None:
    """Split speech into len(words) pieces: choose the pauses whose positions best match where each line
    should end (by word count), preferring longer pauses. Returns [(start, end)] per line or None."""
    n = len(words)
    if n == 1:
        return [(segs[0][0], segs[-1][1])]
    gaps = [((segs[i][1] + segs[i + 1][0]) / 2, segs[i + 1][0] - segs[i][1], i) for i in range(len(segs) - 1)]
    if len(gaps) < n - 1:
        return None
    t0, t1, total = segs[0][0], segs[-1][1], sum(words)
    cum, run_w = [], 0
    for w in words[:-1]:
        run_w += w
        cum.append(t0 + (t1 - t0) * run_w / total)
    inf = float("inf")
    m = len(gaps)
    # best[k][j] = cost when boundary k uses gap j (gaps strictly increasing)
    best = [[inf] * m for _ in range(n - 1)]
    prev = [[-1] * m for _ in range(n - 1)]
    for k in range(n - 1):
        for j in range(m):
            pos, length, _ = gaps[j]
            c = abs(pos - cum[k]) - 1.2 * min(length, 1.5)
            if k == 0:
                best[k][j] = c
                continue
            for i in range(j):
                if best[k - 1][i] + c < best[k][j]:
                    best[k][j], prev[k][j] = best[k - 1][i] + c, i
    j = min(range(m), key=lambda x: best[n - 2][x])
    if best[n - 2][j] == inf:
        return None
    chosen = []
    for k in range(n - 2, -1, -1):
        chosen.append(gaps[j][2])
        j = prev[k][j]
    chosen.reverse()
    pieces, start = [], segs[0][0]
    for gi in chosen:
        pieces.append((start, segs[gi][1]))
        start = segs[gi + 1][0]
    pieces.append((start, segs[-1][1]))
    return pieces


_NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def _norm_tokens(text: str) -> list[str]:
    """Words for matching script vs Whisper: lower case, no punctuation, small numbers as words ("15" = "fifteen")."""
    out = []
    text = re.sub(r"#\s*(\d)", r"number \1", text)  # Whisper writes "Number four" as "#4" or "No. 4"
    text = re.sub(r"\bno\.?\s*(\d)", r"number \1", text, flags=re.I)
    for tok in re.findall(r"[a-z0-9']+", text.lower().replace(".", "")):
        tok = tok.strip("'")
        if tok.isdigit() and int(tok) <= 20:
            tok = _NUM[int(tok)]
        if tok:
            out.append(tok)
    return out


def _whisper_cuts(full: Path, texts: list[str]) -> list[tuple[float, float]] | None:
    """Exact cut points: Whisper listens to the one-take narration and finds where every script line is
    spoken. None = Whisper not available (caller uses pauses + word counts). Raises _SplitError if a line
    can't be found (the caller then records the lines one by one)."""
    import difflib
    import transcribe
    if transcribe._get_model() is None:
        return None
    heard, _ = transcribe.words(full)
    if not heard:
        raise _SplitError("Whisper heard no speech in the narration")
    h_tok, h_idx = [], []
    for i, w in enumerate(heard):
        for t in _norm_tokens(w["text"]):
            h_tok.append(t)
            h_idx.append(i)
    s_tok, s_line = [], []
    for li, t in enumerate(texts):
        for tok in _norm_tokens(t):
            s_tok.append(tok)
            s_line.append(li)
    match = {}
    for a, b, size in difflib.SequenceMatcher(None, s_tok, h_tok, autojunk=False).get_matching_blocks():
        for k in range(size):
            match[a + k] = b + k
    spans = []  # (first heard word index, last heard word index) per line
    for li, text in enumerate(texts):
        idx = [k for k in range(len(s_tok)) if s_line[k] == li]
        got = [h_idx[match[k]] for k in idx if k in match]
        if not idx or len(got) < 0.6 * len(idx):
            raise _SplitError(f"Whisper could not find the line '{text[:40]}' in the narration")
        spans.append((min(got), max(got)))
    for i in range(1, len(spans)):
        if spans[i][0] <= spans[i - 1][1]:
            raise _SplitError("narration lines overlap / out of order")
    dur = probe(full)["duration"]
    # boundary between line i-1 and i: the longest pause between their matched words. Each piece runs from
    # its first spoken word to its last (tight - a piece that starts with the pause makes the voice come
    # late against the captions; QA measured +0.8-1.3 s)
    starts, ends = [max(0.0, heard[spans[0][0]]["start"] - 0.08)], []
    for i in range(1, len(spans)):
        lo, hi = spans[i - 1][1], spans[i][0]
        j = max(range(lo, hi), key=lambda x: heard[x + 1]["start"] - heard[x]["end"])
        gap_mid = (heard[j]["end"] + heard[j + 1]["start"]) / 2
        ends.append(min(gap_mid, heard[j]["end"] + 0.35))  # Whisper marks word ends early - never clip the tail
        starts.append(max(gap_mid, heard[j + 1]["start"] - 0.08))
    ends.append(min(dur, heard[spans[-1][1]]["end"] + 0.6))  # generous tail (QA heard the final sentence "cut off")
    return list(zip(starts, ends))


def _gemini_all_lines(lines: list[dict], paths: list[Path], model: str, voice: str | None = None) -> None:
    """ONE Gemini TTS request with the whole script (natural flow, saves quota), then cut into lines at the
    pauses that match the script. Every piece is checked against its line; any doubt -> _SplitError and the
    caller records line by line, so captions always match what the voice says."""
    voice = voice or load_config().get("voice", {}).get("gemini_tts_voice", "Fenrir")
    texts = [ln["text"] for ln in lines]
    words = [max(1, len(t.split())) for t in texts]
    full = _gemini_tts_wav(_tts_prompt(texts), model, voice, paths[0].with_name(paths[0].stem + "_all.wav"))
    try:
        segs = _speech_segments(full, 0.2)
        if not segs:
            raise _SplitError("no speech in Gemini audio")
        speech = segs[-1][1] - segs[0][0]
        expected = sum(words) / 2.6
        if not 0.55 * expected <= speech <= 1.6 * expected + 2:
            raise _SplitError(f"{speech:.0f}s of speech for {sum(words)} words - doesn't match the script")
        # Whisper hears where each line is actually spoken (cutting by word counts drifted on long scripts:
        # half a sentence ended up on the next clip - found by QA)
        pieces = _whisper_cuts(full, texts) if len(lines) > 1 else None
        if pieces:
            log.info("[voice] Whisper split the narration into %d lines", len(pieces))
        else:  # Whisper not available: pauses + word counts
            pieces = _align_cuts(segs, words)
            if not pieces:
                raise _SplitError(f"could not split Gemini audio into {len(lines)} lines (found {len(segs)} parts)")
        rate = speech / sum(words)  # this take's real seconds per word
        for (a, b), text, w in zip(pieces, texts, words):
            if not (0.5 * w * rate - 0.4 <= b - a <= 1.7 * w * rate + 1.0):
                raise _SplitError(f"piece {b - a:.1f}s doesn't match line '{text[:40]}'")
        for (a, b), path in zip(pieces, paths):
            # leading silence trimmed: Whisper stretches the first word after a pause back into the silence, so a
            # piece could start with ~1 s of quiet and the voice came in late (QA: "spoken +1.2 s")
            run(["ffmpeg", "-y", "-ss", f"{max(0, a - 0.05):.2f}", "-to", f"{b + 0.12:.2f}", "-i", str(full),
                 "-af", "silenceremove=start_periods=1:start_threshold=-42dB:start_silence=0.06",
                 "-ar", "48000", str(path)], timeout=60, check=True)
    finally:
        full.unlink(missing_ok=True)


def synthesize_lines(lines: list[dict], clip_id: str) -> list[dict]:
    """Voice every commentary line with ONE consistent voice.
    Gemini TTS (human voice, config voice.gemini_tts_voice). With engine "gemini" and robot_fallback off it
    raises HumanVoiceUnavailable instead of using edge-tts / gTTS. Returns lines + path/dur/words."""
    vcfg = load_config().get("voice", {})
    engine = vcfg.get("engine", "auto")
    VO_DIR.mkdir(parents=True, exist_ok=True)  # tours voice the lines before render() creates it
    paths = [VO_DIR / f"{clip_id}_l{i}.mp3" for i in range(len(lines))]

    def finish(words_per_line, label):
        out = []
        for ln, path, words in zip(lines, paths, words_per_line):
            dur = probe(path)["duration"]
            exact = bool(words) and len(words) >= len(ln["text"].split()) * 0.6  # the TTS gave word times
            if not exact:
                words = _even_word_timings(ln["text"], dur, lead=0.05)
            out.append({**ln, "path": path, "dur": dur, "words": words, "words_exact": exact})
        log.info("[voice] %d lines voiced with %s", len(out), label)
        return out

    human_only = engine == "gemini" and not vcfg.get("robot_fallback", False)
    if engine in ("auto", "gemini"):
        voice = vcfg.get("gemini_tts_voice", "Fenrir")
        if not env("GEMINI_API_KEY"):
            err = "GEMINI_API_KEY is not set"
        else:
            try:
                model = _gemini_voice(lines, paths, voice)
                return finish([[] for _ in lines], f"Gemini TTS {model}/{voice}")
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
        if human_only:
            raise HumanVoiceUnavailable(f"Gemini voice {voice} failed ({err[:200]}) - robot_fallback is off")
        log.warning("[voice] Gemini voice failed (%.150s) - falling back to edge-tts", err)

    voices = [env("TTS_VOICE")] + vcfg.get("voices", ["en-US-AndrewMultilingualNeural", "en-US-AndrewNeural"])
    rate = vcfg.get("rate", "+12%")
    for voice in [v for v in dict.fromkeys(voices) if v]:
        try:
            words = [_edge_line(ln["text"], voice, rate, path) for ln, path in zip(lines, paths)]
            return finish(words, f"edge-tts {voice}")
        except Exception as exc:  # noqa: BLE001
            log.warning("[voice] %s failed: %s", voice, exc)

    from gtts import gTTS
    for ln, path in zip(lines, paths):
        raw = path.with_suffix(".gtts.mp3")
        gTTS(text=ln["text"], lang="en", tld="com").save(str(raw))
        run(["ffmpeg", "-y", "-i", str(raw), "-filter:a", "atempo=1.15", str(path)], check=True)
        raw.unlink(missing_ok=True)
    return finish([[] for _ in lines], "gTTS")


def _overlap(a: float, b: float, spans: list) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in spans)


def place_lines(voiced: list[dict], total: float, avoid: list | None = None) -> list[dict]:
    """Put each line at its moment without overlaps. If the script is longer than the clip, middle lines are
    dropped first so the hook and the ending question always stay. `avoid` = [(start, end)] where the footage
    itself talks (in-game dialogue, found by Whisper): a line slides a little (max 0.4 s earlier / 1.5 s later;
    the first line only up to 0.4 s later) into a gap instead of talking over the characters."""
    voiced = sorted(voiced, key=lambda x: x["t"])
    while len(voiced) > 2 and sum(v["dur"] + 0.35 for v in voiced) > total - 0.6:
        drop = max(range(1, len(voiced) - 1), key=lambda i: voiced[i]["dur"])
        log.info("[voice] script too long for the clip - dropping: %s", voiced[drop]["text"])
        voiced = voiced[:drop] + voiced[drop + 1:]
    placed, cursor = [], 0.0
    for i, ln in enumerate(voiced):
        last = i == len(voiced) - 1
        while True:
            start = max(ln["t"], cursor + 0.35, 0.25)
            if start + ln["dur"] > total - 0.3:
                start = max(cursor + 0.35, total - 0.3 - ln["dur"])
            fits = start + ln["dur"] <= total and start >= cursor
            if fits or not last or len(placed) < 2:
                break
            # the closing question always stays: drop the line before it instead
            gone = placed.pop()
            log.info("[voice] making room for the ending question - dropping: %s", gone["text"])
            cursor = placed[-1]["start"] + placed[-1]["dur"] if placed else 0.0
        if not fits:
            log.info("[voice] dropping line that doesn't fit: %s", ln["text"])
            continue
        if avoid and _overlap(start, start + ln["dur"], avoid) > 0.05:
            best, cost = start, _overlap(start, start + ln["dur"], avoid)
            # the hook never moves earlier (it would talk over the cold open) and waits at most 0.4 s
            s = start if not placed else max(cursor + 0.35, 0.25, start - 0.4)
            while s <= min(start + (0.4 if not placed else 1.5), total - 0.3 - ln["dur"]) + 1e-6:
                c = _overlap(s, s + ln["dur"], avoid) + 0.25 * abs(s - start)
                if c < cost - 1e-6:
                    best, cost = s, c
                s += 0.1
            if best != start:
                log.info("[voice] moved '%s' %+.1fs so it doesn't talk over the game's dialogue", ln["text"][:40],
                         best - start)
            start = best
        placed.append({**ln, "start": start})
        cursor = start + ln["dur"]
    return placed


# --------------------------------------------------------------------------- word timings (captions)
# Captions show the script's own words (= exactly what Fenrir says); only their TIMES are measured:
#   1. Whisper on each voice piece (on the runner) -> exact word times, mapped onto the script words
#   2. no Whisper: the pauses Fenrir makes at commas / full stops are found in the audio and every phrase is
#      pinned to its pause, words inside a phrase are spread by syllables (much closer than an even spread)

_ACRONYMS = {"gta", "npc", "npcs", "rp", "fps", "xp", "ufo", "atm", "ceo", "fbi", "irl", "op", "dlc", "ai", "hud",
             "vip", "rpg", "suv", "mph", "kmh", "ok"}


def _syllables(tok: str) -> float:
    """Rough spoken length of one script word in syllables (numbers and acronyms are spoken long)."""
    core = tok.strip(".,!?;:\"'()-–—…").lower()
    if not core:
        return 0.3
    if re.search(r"\d", core):
        digits = len(re.sub(r"\D", "", core))
        return 1.0 + 0.9 * digits + (1.5 if "$" in core else 0) + (1.0 if "%" in core else 0)
    if core.replace("'", "") in _ACRONYMS:
        return float(len(re.sub(r"[^a-z0-9]", "", core)))  # spelled out: G-T-A
    w = re.sub(r"[^a-z]", "", core)
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return float(max(1, n))


def _spread(toks: list[str], a: float, b: float) -> list[dict]:
    """Spread words over [a, b] by their spoken length."""
    w = [_syllables(t) + 0.35 for t in toks]
    tot = sum(w) or 1.0
    out, t = [], a
    for tok, wk in zip(toks, w):
        d = (b - a) * wk / tot
        out.append({"text": tok, "start": round(t, 3), "end": round(t + d, 3)})
        t += d
    return out


_PUNCT_END = re.compile(r"[,.!?;:…—–-]$")


def _estimate_word_times(text: str, path: Path | None, dur: float) -> list[dict]:
    """Word times without Whisper: phrases pinned to the pauses in the voice piece, syllables inside a phrase."""
    toks = text.split()
    if not toks:
        return []
    segs: list[tuple[float, float]] = []
    try:
        if path and Path(path).exists():
            segs = _speech_segments(Path(path), 0.12)
    except Exception:  # noqa: BLE001
        segs = []
    t0, t1 = (segs[0][0], segs[-1][1]) if segs else (0.05, max(0.3, dur - 0.1))
    syl = [_syllables(t) + 0.35 for t in toks]
    gaps = [(segs[i][1], segs[i + 1][0]) for i in range(len(segs) - 1)]
    spans, lo_i, lo_t, used = [], 0, t0, -1
    for i in (k for k, t in enumerate(toks[:-1]) if _PUNCT_END.search(t)):
        expect = lo_t + (t1 - lo_t) * sum(syl[lo_i:i + 1]) / (sum(syl[lo_i:]) or 1.0)
        tol = 0.3 + 0.15 * (expect - lo_t)
        cands = [(k, g) for k, g in enumerate(gaps) if k > used and abs((g[0] + g[1]) / 2 - expect) <= tol]
        if not cands:
            continue
        k, g = min(cands, key=lambda kg: abs((kg[1][0] + kg[1][1]) / 2 - expect) - 0.5 * min(kg[1][1] - kg[1][0], 0.6))
        spans.append((lo_i, i, lo_t, g[0]))
        lo_i, lo_t, used = i + 1, g[1], k
    spans.append((lo_i, len(toks) - 1, lo_t, t1))
    out: list[dict] = []
    for fi, li, a, b in spans:
        out += _spread(toks[fi:li + 1], a, max(b, a + 0.1 * (li - fi + 1)))
    return out


def _whisper_word_times(path: Path, text: str) -> list[dict] | None:
    """Exact word times from Whisper, mapped onto the SCRIPT's words (the caption text never comes from
    Whisper). None if Whisper isn't available or can't find most of the line."""
    import difflib
    import transcribe
    toks = text.split()
    if not toks or transcribe._get_model() is None:
        return None
    heard, _ = transcribe.words(Path(path))
    if not heard:
        return None
    h_tok, h_idx = [], []
    for i, w in enumerate(heard):
        for t in _norm_tokens(w["text"]):
            h_tok.append(t)
            h_idx.append(i)
    s_tok, s_word = [], []
    for wi, t in enumerate(toks):
        for n in _norm_tokens(t):
            s_tok.append(n)
            s_word.append(wi)
    times: list[list[float] | None] = [None] * len(toks)
    for a, b, size in difflib.SequenceMatcher(None, s_tok, h_tok, autojunk=False).get_matching_blocks():
        for k in range(size):
            h, wi = heard[h_idx[b + k]], s_word[a + k]
            if times[wi] is None:
                times[wi] = [h["start"], h["end"]]
            else:
                times[wi][1] = max(times[wi][1], h["end"])
    if sum(1 for t in times if t) < 0.6 * len(toks):
        return None
    out: list[dict] = []
    i = 0
    while i < len(toks):
        if times[i]:
            out.append({"text": toks[i], "start": times[i][0], "end": times[i][1]})
            i += 1
            continue
        j = i
        while j < len(toks) and not times[j]:
            j += 1
        a = out[-1]["end"] if out else max(0.0, times[j][0] - 0.3 * (j - i))
        b = times[j][0] if j < len(toks) else a + 0.3 * (j - i)
        out += _spread(toks[i:j], a, max(b, a + 0.08 * (j - i)))
        i = j
    for k in range(1, len(out)):
        out[k]["start"] = max(out[k]["start"], out[k - 1]["start"])
        out[k]["end"] = max(out[k]["end"], out[k]["start"] + 0.05)
    return out


def _timed_words(ln: dict) -> list[dict]:
    """Word times (relative to the line's start) for one placed voice line."""
    if ln.get("words_exact") and ln.get("words"):
        return ln["words"]
    words = None
    if load_config().get("video", {}).get("fx", {}).get("caption_whisper", True):
        try:
            words = _whisper_word_times(Path(ln["path"]), ln["text"])
        except Exception as exc:  # noqa: BLE001
            log.debug("[captions] Whisper word times failed: %s", exc)
    return words or _estimate_word_times(ln["text"], Path(ln["path"]), float(ln.get("dur") or 0))


# --------------------------------------------------------------------------- captions (ASS)

def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _ass_text(s: str) -> str:
    return re.sub(r"[{}\\]", "", s)


def _ass_color(value: str, default: str) -> str:
    """'&H0000F0FF' / '&H00F0FF&' / '00F0FF' -> inline '&H00F0FF&' (BGR)."""
    hexpart = re.sub(r"[^0-9A-Fa-f]", "", str(value or "").upper().replace("&H", ""))
    return f"&H{(hexpart or default)[-6:]:0>6}&"


# Montserrat Black advance widths (1/1000 em, measured from the font the workflow installs). libass draws
# Fontsize 100 with a ~73.5 px em, so a caption's layout is known well enough to put the highlighted word
# exactly over its own slot (and to keep every line inside the safe zone).
_ADV = {"A": 806, "B": 773, "C": 744, "D": 826, "E": 673, "F": 644, "G": 769, "H": 804, "I": 350, "J": 574,
        "K": 765, "L": 617, "M": 954, "N": 804, "O": 848, "P": 743, "Q": 848, "R": 745, "S": 657, "T": 654,
        "U": 784, "V": 786, "W": 1206, "X": 762, "Y": 710, "Z": 687, "0": 692, "1": 418, "2": 608, "3": 614,
        "4": 712, "5": 619, "6": 661, "7": 645, "8": 678, "9": 661, " ": 300, ".": 304, ",": 304, "!": 314,
        "?": 606, "'": 254, "’": 254, '"': 489, "-": 390, "$": 657, "%": 917, "&": 777, ":": 304, ";": 304,
        "(": 381, ")": 381, "/": 440, "#": 740, "+": 620, "*": 473, "@": 1037, "…": 912, "—": 800, "–": 600}


def _text_w(text: str, size: float) -> float:
    return sum(_ADV.get(ch, 720) for ch in text.upper()) * size * 0.000735


def _wrap(text: str, size: float, max_w: float, max_lines: int = 3, min_size: float = 40) -> tuple[list[str], float]:
    """Greedy word wrap inside max_w; shrinks the font until it fits in max_lines."""
    words = text.split()
    while True:
        lines, cur = [], ""
        for w in words:
            cand = f"{cur} {w}".strip()
            if cur and _text_w(cand, size) > max_w:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            lines.append(cur)
        widest = max((_text_w(ln, size) for ln in lines), default=0)
        if (len(lines) <= max_lines and widest <= max_w) or size <= min_size:
            return _balance(words, len(lines), size) if 1 < len(lines) <= 3 and len(words) <= 14 else lines, size
        size = max(min_size, size * 0.92)


def _balance(words: list[str], n: int, size: float) -> list[str]:
    """Same number of lines, but as even as possible (no lonely last word: 'HE TOOK THE JUMP / AT 200 MPH')."""
    import itertools
    best, best_w = None, float("inf")
    for cuts in itertools.combinations(range(1, len(words)), n - 1):
        bounds = (0, *cuts, len(words))
        lines = [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:])]
        widest = max(_text_w(ln, size) for ln in lines)
        if widest < best_w - 1:
            best, best_w = lines, widest
    return best or [" ".join(words)]


_HYPE = {"insane", "crazy", "wild", "huge", "massive", "never", "nobody", "impossible", "unreal", "secret", "hidden",
         "free", "money", "cash", "rich", "broke", "million", "millions", "billion", "fastest", "best", "worst",
         "biggest", "illegal", "brutal", "boom", "destroyed", "exploded", "explodes", "legendary", "perfect", "craziest",
         "dead", "rare", "easy", "fast", "instantly", "everything", "nothing", "wrong", "cops", "police", "wanted",
         "sniper", "jackpot", "heist", "robbery", "chaos", "gone"}
_NUM_WORDS = {"two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "fifteen",
              "twenty", "fifty", "hundred", "thousand", "million", "billion", "one"}
_UNITS = {"dollars", "dollar", "bucks", "grand", "k", "million", "thousand", "hundred", "stars", "star", "seconds",
          "minutes", "hours", "mph", "cars", "percent", "times", "cops", "kills"}
_WEAK_END = {"a", "an", "the", "to", "of", "and", "or", "but", "in", "on", "at", "for", "with", "my", "your", "his",
             "her", "its", "our", "their", "so", "if", "as", "from", "by", "into", "than", "gonna"}
_SOFT_END = {"is", "are", "was", "i", "you", "he", "she", "we", "they", "it's", "i'm", "that", "this", "just",
             "really", "can", "will"}


def _bare(tok: str) -> str:
    return re.sub(r"[^\w$%']", "", tok)


def _emphasis_flags(texts: list[str]) -> list[bool]:
    """Key words that get the emphasis colour (+ a punch-in): numbers / money, countdown numbers, hype words,
    words the script shouts in CAPS, words with '!'. 'GTA 6' is the game's name, never emphasised."""
    shouty = sum(1 for t in texts if _bare(t).isupper() and len(_bare(t)) > 1) > 0.5 * max(1, len(texts))
    flags = []
    for i, raw in enumerate(texts):
        core = _bare(raw)
        low = core.lower().strip("'")
        prev = _bare(texts[i - 1]).lower() if i else ""
        nxt = _bare(texts[i + 1]).lower() if i + 1 < len(texts) else ""
        if prev == "gta" or low in ("gta", ""):
            e = False
        elif re.search(r"\d", core):
            e = True
        elif low in _NUM_WORDS:
            e = prev == "number" or nxt in _UNITS
        elif low in _HYPE:
            e = True
        elif not shouty and core.isupper() and len(core) >= 2 and low not in _ACRONYMS:
            e = True
        else:
            e = raw.endswith("!") and len(core) >= 3
        flags.append(e)
    return flags


def _split_run(run: list[dict], size: float, max_w: float, max_words: int) -> list[list[dict]]:
    """Best 1-4 word caption chunks for a stretch of speech without pauses: 2-3 words, breaks at punctuation,
    never ends on 'the' / 'a' / 'to'..., never wider than the safe zone, no chunk on screen much over 2 s."""
    n = len(run)
    texts = [w["text"] for w in run]
    widths = [_text_w(t, size) for t in texts]
    space = _text_w(" ", size)
    inf = float("inf")
    best, prev = [0.0] + [inf] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(max(0, j - max_words), j):
            k = j - i
            if k > 1 and sum(widths[i:j]) + space * (k - 1) > max_w:
                continue
            c = 0.6 * (k - 2.6) ** 2
            last = _bare(texts[j - 1]).lower()
            if j < n and last in _WEAK_END:
                c += 2.5
            elif j < n and last in _SOFT_END:
                c += 1.2
            if j < n and re.search(r"\d", last) and _bare(texts[j]).lower() in _UNITS:
                c += 2.0  # keep "500 DOLLARS" together
            if j < n and texts[j - 1][:1].isupper() and texts[j][:1].isupper() and not \
                    _PUNCT_END.search(texts[j - 1]) and not texts[j].isupper():
                c += 2.0  # keep names together ("VICE CITY", "KAI CENAT")
            if _PUNCT_END.search(texts[j - 1]):
                c -= 1.2
            c += 3.0 * sum(1 for m in range(i, j - 1) if _PUNCT_END.search(texts[m]))
            dur = run[j - 1]["end"] - run[i]["start"]
            if dur > 2.0:
                c += (dur - 2.0) * 2.0
            if k == 1 and n > 1:  # a lonely word looks like a glitch (unless it is long or ends a phrase)
                c += 0.8 + (0.7 if widths[i] < max_w * 0.6 and not _PUNCT_END.search(texts[j - 1]) else 0.0)
            if best[i] + c < best[j]:
                best[j], prev[j] = best[i] + c, i
    cuts, j = [], n
    while j > 0:
        cuts.append((prev[j], j))
        j = prev[j]
    return [run[a:b] for a, b in reversed(cuts)]


def _caption_groups(words: list[dict], size: float, max_w: float, max_words: int = 4) -> list[list[dict]]:
    groups, run = [], []
    for w in words:
        if not w["text"].strip():
            continue
        if run and w["start"] - run[-1]["end"] > 0.45:  # a pause starts a new caption (no stale words)
            groups += _split_run(run, size, max_w, max_words)
            run = []
        run.append(w)
    if run:
        groups += _split_run(run, size, max_w, max_words)
    return groups


def build_ass(path: Path, banner: str, words: list[dict], total: float, cta: str, banner_v: int = 200,
              flashes: list | None = None, question: dict | None = None) -> dict:
    """1080x1920 ASS: hook banner (pops in), word-by-word captions synced to the voice (current word pops in
    the highlight colour, key words in the emphasis colour), white flash frames, end card with the closing
    question + call to action. Everything stays inside the TikTok / Reels / Shorts safe zone (no text in the
    top ~10 %, the bottom ~20 % or under the right-hand button column).
    Returns {"emphasis": [(start, end)], "pops": [t]} for the punch-in zooms / pop sounds."""
    vcfg = load_config().get("video", {})
    fxc = vcfg.get("fx", {})
    font = vcfg.get("caption_font", "Montserrat Black")
    hl = _ass_color(vcfg.get("highlight_color_ass", "&H0000F0FF"), "00F0FF")
    em = _ass_color(vcfg.get("emphasis_color_ass", "&H0055FF3C"), "55FF3C")
    cs = float(vcfg.get("caption_size", 104))
    cx, cy = int(vcfg.get("caption_x", 500)), int(vcfg.get("caption_y", 1400))
    cap_w = float(vcfg.get("caption_max_w", 800))
    style = str(fxc.get("caption_style", "karaoke"))  # karaoke = whole chunk shown, current word pops
    words = sorted((w for w in words if str(w.get("text", "")).strip()), key=lambda w: w["start"])

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Banner,{font},84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,8,4,8,40,40,0,1
Style: Caption,{font},{cs:.0f},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,9,5,5,0,0,0,1
Style: Card,{font},68,&H00141414,&H00141414,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,3,16,0,8,40,40,0,1
Style: CTA,{font},46,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,6,3,8,40,40,0,1
Style: Flash,{font},20,&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev: list[str] = []
    pops: list[float] = []

    def dlg(layer: int, a: float, b: float, style_name: str, text: str) -> None:
        if _ass_time(b) > _ass_time(a) or b - a >= 0.01:
            ev.append(f"Dialogue: {layer},{_ass_time(a)},{_ass_time(b)},{style_name},,0,0,0,,{text}")

    # --- hook banner: pops in with a bounce on frame 1, key words in the highlight colour
    b_lines, b_size = _wrap(_ass_text(banner.upper()), 84, float(vcfg.get("banner_max_w", 920)), 3, 52)
    b_words = " ".join(b_lines).split()
    b_flags = _emphasis_flags(b_words)
    if not any(b_flags) and len(b_lines) >= 2:  # nothing to stress: the last line gets the colour
        b_flags = [False] * (len(b_words) - len(b_lines[-1].split())) + [True] * len(b_lines[-1].split())
    k, rows = 0, []
    for line in b_lines:
        parts = []
        for w in line.split():
            parts.append(f"{{\\1c{hl}}}{w}{{\\1c&HFFFFFF&}}" if b_flags[k] else w)
            k += 1
        rows.append(" ".join(parts))
    banner_h = len(b_lines) * b_size * 1.18
    q_text = _ass_text(str((question or {}).get("text", "")).upper()).strip()
    q_at = float((question or {}).get("start", total)) if q_text else total
    q_at = min(max(0.5, q_at), total)
    if b_lines:
        # already readable on frame 1 (TikTok's default cover), then bounces: 85 % -> 112 % -> 100 %
        dlg(3, 0, q_at, "Banner", f"{{\\an8\\pos(540,{banner_v})\\fs{b_size:.0f}\\fad(0,{150 if q_at < total else 0})"
                                  f"\\fscx85\\fscy85\\t(0,130,\\fscx112\\fscy112)\\t(130,260,\\fscx100\\fscy100)}}"
                                  + "\\N".join(rows))
        pops.append(0.0)

    # --- word-by-word captions
    groups = _caption_groups(words, cs, cap_w, int(fxc.get("caption_max_words", 4)))
    flags = _emphasis_flags([w["text"] for w in words])
    flag_of = {id(w): f for w, f in zip(words, flags)}
    emphasis = [(w["start"], w["end"]) for w, f in zip(words, flags) if f]
    shadow_a = "64"
    gap = int(fxc.get("caption_word_gap", 150))  # word gap in % of a normal space (room for the popped word)
    hidden, shown = "{\\1a&HFF&\\3a&HFF&\\4a&HFF&}", f"{{\\1a&H00&\\3a&H00&\\4a&H{shadow_a}&}}"
    shown_until = 0.0
    for gi, g in enumerate(groups):
        texts = [_ass_text(w["text"].upper()) for w in g]
        size = cs
        space = _text_w(" ", size) * gap / 100
        width = sum(_text_w(t, size) for t in texts) + space * (len(texts) - 1)
        if width > cap_w:  # one very long word: shrink it into the safe zone
            size = cs * cap_w / width
            space, width = space * cap_w / width, cap_w
        extra = space - _text_w(" ", size)  # letter-spacing on the space only (keeps the pop scale working)
        sep = f"{{\\fsp{extra:.1f}}} {{\\fsp0}}" if extra > 0.5 else " "
        xs, x = [], cx - width / 2
        for t in texts:
            tw = _text_w(t, size)
            xs.append(x + tw / 2)
            x += tw + space
        g_start = max(g[0]["start"] - 0.04, shown_until, 0.0)
        g_end = g[-1]["end"] + 0.3
        if gi + 1 < len(groups):
            nxt = groups[gi + 1][0]["start"] - 0.04
            g_end = nxt if nxt - g[-1]["end"] < 0.4 else min(g_end, nxt)
        g_end = shown_until = max(min(g_end, total), g_start)
        for k, w in enumerate(g):
            a = g_start if k == 0 else max(w["start"], g_start)
            b = max(g[k + 1]["start"], a) if k + 1 < len(g) else g_end
            if b - a < 0.02:
                continue
            base = []
            for j, t in enumerate(texts):
                if j == k or (style == "reveal" and j > k):
                    base.append(f"{hidden}{t}{shown}")
                elif flag_of.get(id(g[j])):
                    base.append(f"{{\\1c{em}}}{t}{{\\1c&HFFFFFF&}}")
                else:
                    base.append(t)
            hot = flag_of.get(id(w))
            col = em if hot else hl
            peak, hold = (126, 116) if hot else (118, 110)
            if k == 0:  # the whole chunk pops in, the first word grows into its slot with it
                pop = 72
                gpop = f"\\fscx{pop}\\fscy{pop}\\t(0,110,\\fscx100\\fscy100)"
                x0 = cx + (xs[0] - cx) * pop / 100
                over = (f"{{\\an5\\move({x0:.0f},{cy},{xs[0]:.0f},{cy},0,110)\\fs{size:.0f}\\1c{col}"
                        f"\\fscx{pop}\\fscy{pop}\\t(0,110,\\fscx{peak}\\fscy{peak})\\t(110,190,\\fscx{hold}\\fscy{hold})}}")
            else:
                gpop = ""
                over = (f"{{\\an5\\pos({xs[k]:.0f},{cy})\\fs{size:.0f}\\1c{col}"
                        f"\\t(0,70,\\fscx{peak}\\fscy{peak})\\t(70,150,\\fscx{hold}\\fscy{hold})}}")
            dlg(1, a, b, "Caption", f"{{\\an5\\pos({cx},{cy})\\fs{size:.0f}{gpop}}}" + sep.join(base))
            dlg(2, a, b, "Caption", over + texts[k])

    # --- white flash frames (cut out of the cold open, impacts)
    for f in flashes or []:
        t, strength = float(f[0]), float(f[1]) if len(f) > 1 else 0.8
        dur = float(f[2]) if len(f) > 2 else 0.2
        if not 0 <= t < total:
            continue
        alpha = max(0, min(255, int(255 * (1 - strength))))
        dlg(0, t, min(total, t + dur), "Flash",
            f"{{\\an7\\pos(0,0)\\bord0\\shad0\\alpha&H{alpha:02X}&\\t(0,{int(dur * 1000)},\\alpha&HFF&)\\p1}}"
            "m 0 0 l 1080 0 1080 1920 0 1920{\\p0}")

    # --- end card: the closing question replaces the banner (comment bait), then the call to action
    end_y = banner_v + banner_h
    if q_text and q_at < total - 0.3 and fxc.get("question_card", True):
        c_lines, c_size = _wrap(q_text, 68, float(vcfg.get("card_max_w", 860)), 3, 46)
        dlg(3, q_at, total, "Card", f"{{\\an8\\pos(540,{banner_v + 10})\\fs{c_size:.0f}\\fad(80,0)"
                                    f"\\fscx40\\fscy40\\t(0,150,\\fscx106\\fscy106)\\t(150,260,\\fscx100\\fscy100)}}"
                                    + "\\N".join(c_lines))
        pops.append(q_at)
        end_y = banner_v + 10 + len(c_lines) * c_size * 1.28 + 16
    if cta:
        c_at = max(0.0, total - 3.2, q_at + 0.6 if q_at < total else 0.0)
        if c_at < total - 0.8:
            dlg(3, c_at, total, "CTA", f"{{\\an8\\pos(540,{end_y + 28:.0f})\\fad(100,0)\\fscx60\\fscy60"
                                       f"\\t(0,140,\\fscx108\\fscy108)\\t(140,240,\\fscx100\\fscy100)}}"
                                       f"{_ass_text(cta.upper())}")
    path.write_text(header + "\n".join(ev) + "\n", encoding="utf-8")
    return {"emphasis": emphasis, "pops": pops}


# --------------------------------------------------------------------------- effects

SFX_RECIPES = {  # generated with FFmpeg - no samples, no copyright: (inputs, filter_complex -> [o])
    "whoosh": (["-f", "lavfi", "-i", "anoisesrc=color=pink:duration=0.55:amplitude=0.9"],
               "[0:a]highpass=f=350,lowpass=f=5500,afade=t=in:st=0:d=0.3:curve=exp,"
               "afade=t=out:st=0.3:d=0.25:curve=exp,volume=1.4[o]"),
    "boom": (["-f", "lavfi", "-i", "aevalsrc='0.95*sin(2*PI*(42+55*exp(-9*t))*t)*exp(-3.2*t)':s=48000:d=1.2",
              "-f", "lavfi", "-i", "anoisesrc=color=brown:duration=0.14:amplitude=0.7"],
             "[1:a]lowpass=f=400,afade=t=out:st=0.04:d=0.1[n];[0:a][n]amix=inputs=2:normalize=0,"
             "lowpass=f=220,volume=1.8[o]"),
    # soft "bloop" when a text pops in (pitch falls from ~1.4 kHz)
    "pop": (["-f", "lavfi", "-i", "aevalsrc='0.8*sin(2*PI*1400*(1-exp(-30*t))/30)*exp(-34*t)':s=48000:d=0.16"],
            "[0:a]highpass=f=180,afade=t=out:st=0.1:d=0.06[o]"),
    # 1 s build-up into the payoff: rising tone + rising air
    "riser": (["-f", "lavfi", "-i", "aevalsrc='0.3*sin(2*PI*(160*t+420*t*t))*t*t':s=48000:d=1.0",
               "-f", "lavfi", "-i", "anoisesrc=color=pink:duration=1.0:amplitude=0.6"],
              "[1:a]highpass=f=900,afade=t=in:st=0:d=1.0:curve=exp,volume=0.8[n];"
              "[0:a][n]amix=inputs=2:normalize=0,afade=t=out:st=0.94:d=0.06[o]"),
    # punchy thump on a key action (crash, shot, landing): falling low tone + a short click for the attack
    "hit": (["-f", "lavfi", "-i", "aevalsrc='0.95*sin(2*PI*(70*t+6.4*(1-exp(-25*t))))*exp(-11*t)':s=48000:d=0.4",
             "-f", "lavfi", "-i", "anoisesrc=color=white:duration=0.04:amplitude=0.6"],
            "[1:a]highpass=f=1500,afade=t=out:st=0.005:d=0.03[c];[0:a][c]amix=inputs=2:normalize=0,volume=1.3[o]"),
}
SFX_GAIN = {"pop": 0.55, "riser": 0.7, "hit": 0.8}


def _sfx(kind: str) -> Path | None:
    out = VO_DIR.parent / "sfx" / f"{kind}.wav"
    if out.exists():
        return out
    if kind not in SFX_RECIPES:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    inputs, fc = SFX_RECIPES[kind]
    res = run(["ffmpeg", "-y", *inputs, "-filter_complex", fc, "-map", "[o]", "-ar", "48000", "-ac", "2", str(out)],
              timeout=60)
    return out if res.returncode == 0 and out.exists() else None


def _music_track() -> Path | None:
    """A random background track: your own music/ folder, else CC0 beats from Freesound (music.py)."""
    import music
    return music.pick()


def _zoom_filter(events: list[dict], shakes: list | None = None, shake_px: float = 14.0,
                 size: str = "1080x1920") -> str:
    """Punch-in zooms with eased curves (fast ease-out in, smooth settle back - no hard snap) and an optional
    camera shake on impacts. events: [{"a", "b", "amt", "in"?, "out"?}] = zoomed from a to b (fully zoomed
    after `in` seconds, back to normal `out` seconds after b). Output keeps its size / 30 fps."""
    if not events:
        return ""
    terms = []
    for e in events:
        a, b, amt = float(e["a"]), float(e["b"]), float(e["amt"])
        ti, to = max(0.01, float(e.get("in", 0.12))), max(0.01, float(e.get("out", 0.22)))
        pin = f"clip((it-{a:.3f})/{ti:.3f},0,1)"
        pout = f"clip(({b + to:.3f}-it)/{to:.3f},0,1)"
        terms.append(f"{amt:.4f}*(1-pow(1-{pin},3))*{pout}*{pout}*(3-2*{pout})")
    x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    for t, d in shakes or []:  # decaying shake (only visible while zoomed in - zoompan keeps it in frame)
        decay = f"clip(({t + d:.3f}-it)/{d:.3f},0,1)*gte(it,{t:.3f})"
        x += f"+{shake_px:.1f}*sin({2 * 3.14159 * 17:.2f}*(it-{t:.3f}))*{decay}"
        y += f"+{shake_px * 0.7:.1f}*cos({2 * 3.14159 * 13:.2f}*(it-{t:.3f}))*{decay}"
    return f",zoompan=z='1+{'+'.join(terms)}':x='{x}':y='{y}':d=1:s={size}:fps=30"


def _plan_zooms(fx: dict, emphasis: list[tuple[float, float]], total: float, amount: float, cap: float,
                fxcfg: dict) -> list[dict]:
    """Zoom events: the editor's windows (cold open, payoff, cuts) + small punch-ins on emphasised words
    (at most one every couple of seconds, never on top of a bigger zoom) + a zoom-out reveal on frame 1."""
    events = []
    for w in fx.get("zoom", []):
        e = dict(w) if isinstance(w, dict) else {"a": w[0], "b": w[1]}
        e["amt"] = min(cap, float(e.get("amt") or amount))
        if e["a"] < total and e["b"] > -1:
            events.append(e)
    if fxcfg.get("emphasis_zoom", True):
        amt = min(cap, float(fxcfg.get("emphasis_zoom_amount", 0.08)))
        gap, last = float(fxcfg.get("emphasis_zoom_gap", 2.2)), -99.0
        busy = [(e["a"] - 0.5, e["b"] + float(e.get("out", 0.22)) + 0.3) for e in events]
        for a, b in emphasis:
            if a - last < gap or a > total - 0.6 or any(x <= a <= y or x <= b <= y for x, y in busy):
                continue
            events.append({"a": a - 0.03, "b": min(b + 0.35, a + 1.0), "amt": amt, "in": 0.08, "out": 0.25})
            last = a
    intro = min(cap, float(fxcfg.get("intro_zoom", 0.1)))
    if intro > 0 and not any(e["a"] <= 0.4 for e in events):  # starts punched in, eases out: motion on frame 1
        events.append({"a": -1.0, "b": 0.0, "amt": intro, "in": 0.01, "out": 0.6})
    return sorted(events, key=lambda e: e["a"])


# --------------------------------------------------------------------------- render

def render(clip: Path, clip_id: str, banner: str, commentary: list[dict],
           voiced: list[dict] | None = None, fx: dict | None = None, caption_words: list[dict] | None = None,
           game_volume: float | None = None, fg_zoom: float | None = None) -> dict | None:
    """commentary = [{"t": seconds, "text": "line"}, ...]. `voiced` = lines already synthesised
    (with path/dur/words) - used by the feature-tour format where the edit follows the voice.
    fx (all optional): zoom [(a, b) | {"a","b","amt","in","out"}], sfx [{"t","kind"}], flash [(t, strength)],
    shake [(t, dur)], facecam (x, y, w, h)."""
    cfg = load_config()
    vcfg = cfg.get("video", {})
    VO_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)
    fx = fx or {}
    fxcfg = vcfg.get("fx", {})

    info = probe(clip)
    # never loop footage: short stream clips simply make a shorter video
    total = round(min(float(vcfg.get("max_duration_sec", 90)), info["duration"] or 60), 2)
    lines = place_lines(voiced if voiced is not None else synthesize_lines(commentary, clip_id), total,
                        avoid=fx.get("dialogue"))
    words = []
    for ln in lines:  # exact word times (Whisper / the voice's own pauses) -> captions in sync with Fenrir
        words += [{"text": w["text"], "start": w["start"] + ln["start"], "end": w["end"] + ln["start"]}
                  for w in _timed_words(ln)]
    if caption_words is not None:  # stream clips: subtitles of what the streamer says, no voice-over
        words = caption_words
    question = None
    if lines and lines[-1]["text"].rstrip().endswith("?"):
        question = {"text": lines[-1]["text"], "start": lines[-1]["start"]}

    vertical_src = info["height"] > info["width"]
    cam = fx.get("facecam")
    ass = VO_DIR / f"{clip_id}.ass"
    marks = build_ass(ass, banner, words, total, vcfg.get("cta_text", "FOLLOW FOR DAILY GTA 6 MOMENTS"),
                      banner_v=640 if cam and not vertical_src else int(vcfg.get("banner_y", 200)),
                      flashes=fx.get("flash") if fxcfg.get("flash", True) else None, question=question)
    out = READY_DIR / f"{clip_id}.mp4"

    # sharp picture: one high-quality (lanczos) scale of the footage, never blown up much past its real size
    src_w, src_h = max(1, info["width"]), max(1, info["height"])
    max_up = float(vcfg.get("max_upscale", 1.35))
    if cam and not vertical_src:
        # pro stream-clip layout: streamer's facecam on top, gameplay below - both big
        cx, cy, cw, ch = cam
        video_chain = (
            f"[0:v]split=2[a][b];[a]crop=iw*{cw}:ih*{ch}:iw*{cx}:ih*{cy},"
            "scale=1080:720:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:720,setsar=1[cam];"
            "[b]scale=-2:1200:flags=lanczos,crop=1080:1200,setsar=1[game];[cam][game]vstack=inputs=2,setsar=1[base]")
        scale_up = 0.0  # split layout: no whole-frame zooms (they would break the layout)
    elif vertical_src:
        video_chain = ("[0:v]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920,"
                       "setsar=1[base]")
        scale_up = max(1080 / src_w, 1920 / src_h)
    else:
        zoom = float(fg_zoom or vcfg.get("foreground_zoom", 1.3))
        # a low-res source is not cropped in as far (that only makes it blurry)
        zoom = max(1.0, min(zoom, float(vcfg.get("max_fg_upscale", 1.2)) * src_w / 1080))
        fg_w = int(1080 * zoom) // 2 * 2
        scale_up = fg_w / src_w
        video_chain = (
            "[0:v]split=2[a][b];"
            "[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=12:3,"
            "scale=1080:1920,eq=brightness=-0.12:saturation=1.2,setsar=1[bg];"
            f"[b]scale={fg_w}:-2:flags=lanczos,crop=1080:ih,setsar=1[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2-40[base]"
        )
    ass_escaped = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    game_vol = float(game_volume if game_volume is not None else
                     vcfg.get("game_audio_volume_tour" if voiced else "game_audio_volume", 0.9))
    vo_gain = float(vcfg.get("voice_volume", 1.25))

    # each voice line delayed to its moment, mixed into one sparse voice track
    delayed = []
    for i, ln in enumerate(lines, start=1):
        ms = int(ln["start"] * 1000)
        delayed.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,volume={vo_gain},adelay={ms}|{ms}[v{i}]")
    if lines:
        vo_mix = "".join(f"[v{i}]" for i in range(1, len(lines) + 1))
        voice = ";".join(delayed) + f";{vo_mix}amix=inputs={len(lines)}:normalize=0:duration=longest,apad[vo];"
    else:
        voice = "anullsrc=r=48000:cl=stereo[vo];"
    # sound effects (whoosh on cuts, boom on the payoff, riser into it, pop when text pops in) + music
    extra_inputs, extra_chains, extra_labels = [], [], []
    idx = len(lines) + 1
    if fxcfg.get("sfx", True):
        sfx_vol = float(fxcfg.get("sfx_volume", 0.55))
        events = list(fx.get("sfx", []))
        if fxcfg.get("pop_sfx", True):
            events += [{"t": t + 0.02, "kind": "pop"} for t in marks["pops"]]
        for ev in events:
            path = _sfx(ev["kind"]) if 0 <= ev["t"] < total - 0.2 else None
            if not path:
                continue
            ms = int(max(0.0, ev["t"]) * 1000)
            extra_inputs += ["-i", str(path)]
            extra_chains.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,"
                                f"volume={sfx_vol * SFX_GAIN.get(ev['kind'], 1.0):.3f},adelay={ms}|{ms}[x{idx}]")
            extra_labels.append(f"[x{idx}]")
            idx += 1
    music = _music_track()
    music_label = ""
    if music:
        extra_inputs += ["-stream_loop", "-1", "-i", str(music)]
        mvol = float(fxcfg.get("music_volume", 0.09))
        beats = [(a, b) for a, b in fx.get("action", []) if b > 0 and a < total]
        if beats and float(fxcfg.get("music_action_boost", 0.5)) > 0:  # a bit louder on the action beats
            on = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in beats)
            mvol_f = f"volume='{mvol}*(1+{float(fxcfg.get('music_action_boost', 0.5))}*min(1,{on}))':eval=frame"
        else:
            mvol_f = f"volume={mvol}"
        extra_chains.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,{mvol_f},afade=t=in:d=0.6[m{idx}]")
        music_label = f"[m{idx}]"
        log.info("[render] background music: %s", music.name)
    extras = (";".join(extra_chains) + ";") if extra_chains else ""

    # --- pass 1: build the audio mix (fast, audio only)
    mix_wav = VO_DIR / f"{clip_id}_mix.wav"
    if music_label:  # music dips under the voice too (pro mix), then joins the other extras
        afc_music = f"[vo]asplit=3[vo1][vo2][vo3];{music_label}[vo3]sidechaincompress=threshold=0.02:ratio=8:" \
                    "attack=20:release=600[mduck];"
        extra_labels = extra_labels + ["[mduck]"]
    else:
        afc_music = "[vo]asplit=2[vo1][vo2];"
    afc = (
        f"{voice}{extras}{afc_music}"
        f"[0:a]aresample=48000,aformat=channel_layouts=stereo,volume={game_vol}[game];"
        # game audio stays loud between lines and dips only while the voice talks
        f"[game][vo1]sidechaincompress=threshold={float(fxcfg.get('duck_threshold', 0.015))}:"
        f"ratio={float(fxcfg.get('duck_ratio', 10))}:attack=15:release=450[ducked];"
        f"[ducked][vo2]{''.join(extra_labels)}amix=inputs={2 + len(extra_labels)}:duration=first:normalize=0[aout]"
    )
    inputs = ["-stream_loop", "-1", "-i", str(clip)] if info["duration"] < total else ["-i", str(clip)]
    for ln in lines:
        inputs += ["-i", str(ln["path"])]
    inputs += extra_inputs
    res = run(["ffmpeg", "-y", "-hide_banner", *inputs, "-filter_complex", afc, "-map", "[aout]",
               "-t", str(total), "-ar", "48000", str(mix_wav)], timeout=300)
    if res.returncode != 0:
        log.error("[render] audio mix failed: %s", res.stderr[-800:])
        return None

    # --- pass 2: measure loudness once, then level the WHOLE track evenly (no quiet start)
    norm = "loudnorm=I=-14:TP=-1.5:LRA=11"
    meas = run(["ffmpeg", "-hide_banner", "-i", str(mix_wav), "-af", norm + ":print_format=json",
                "-f", "null", "-"], timeout=120)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", meas.stderr)
    if m:
        import json as _json
        j = _json.loads(m.group(0))
        norm += (f":measured_I={j['input_i']}:measured_TP={j['input_tp']}:measured_LRA={j['input_lra']}"
                 f":measured_thresh={j['input_thresh']}:offset={j['target_offset']}:linear=true")

    # --- pass 3: video + levelled audio
    vin = ["-stream_loop", "-1", "-i", str(clip)] if info["duration"] < total else ["-i", str(clip)]
    zoom_f = ""
    if fxcfg.get("zoom", True) and scale_up > 0:
        amount = float(fxcfg.get("zoom_amount", 0.22))
        cap = max(0.04, min(amount, max_up / scale_up - 1))  # zoom only as far as the source stays sharp
        plan = _plan_zooms(fx, marks["emphasis"], total, amount, cap, fxcfg)
        shakes = [s for s in fx.get("shake", []) if 0 <= s[0] < total] if fxcfg.get("shake", True) else []
        zoom_f = _zoom_filter(plan, shakes, float(fxcfg.get("shake_px", 14)))
        log.info("[render] %d zoom moves (max %.0f%%), %d shakes, source %dx%d", len(plan), cap * 100,
                 len(shakes), src_w, src_h)
    grade = "eq=contrast=1.05:saturation=1.1,unsharp=5:5:0.4:5:5:0.0,"
    fc = (f"{video_chain};[base]fps=30{zoom_f},{grade}ass='{ass_escaped}',format=yuv420p[vout];"
          f"[1:a]{norm},aresample=48000,alimiter=limit=0.79:level=0,apad[aout]")  # limiter last + headroom (AAC overshoots)
    cmd = ["ffmpeg", "-y", "-hide_banner", *vin, "-i", str(mix_wav),
           "-filter_complex", fc, "-map", "[vout]", "-map", "[aout]",
           "-c:v", "libx264", "-preset", vcfg.get("preset", "veryfast"), "-crf", str(vcfg.get("crf", 20)),
           "-profile:v", "high", "-maxrate", "12M", "-bufsize", "24M",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-t", str(total), "-movflags", "+faststart", str(out)]
    res = run(cmd, timeout=900)
    mix_wav.unlink(missing_ok=True)
    if res.returncode != 0 or not out.exists():
        log.error("[render] FFmpeg failed: %s", res.stderr[-1200:])
        return None
    log.info("[render] %s (%.1fs, %.1f MB, %d voice lines, %d caption words)", out.name, total,
             out.stat().st_size / 1e6, len(lines), len(words))
    return {"path": str(out), "duration": total,
            "lines": [{"t": round(ln["start"], 1), "text": ln["text"]} for ln in lines],
            "placed": [{"text": ln["text"], "start": ln["start"], "dur": ln["dur"], "path": str(ln["path"])}
                       for ln in lines],
            "captions": [{"text": w["text"], "start": round(w["start"], 2), "end": round(w["end"], 2)}
                         for w in words]}
