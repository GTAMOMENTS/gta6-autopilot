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
        pieces = _align_cuts(segs, words)
        if not pieces:
            raise _SplitError(f"could not split Gemini audio into {len(lines)} lines (found {len(segs)} parts)")
        rate = speech / sum(words)  # this take's real seconds per word
        for (a, b), text, w in zip(pieces, texts, words):
            if not (0.5 * w * rate - 0.4 <= b - a <= 1.7 * w * rate + 1.0):
                raise _SplitError(f"piece {b - a:.1f}s doesn't match line '{text[:40]}'")
        for (a, b), path in zip(pieces, paths):
            run(["ffmpeg", "-y", "-ss", f"{max(0, a - 0.05):.2f}", "-to", f"{b + 0.12:.2f}", "-i", str(full),
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
            if not words or len(words) < len(ln["text"].split()) * 0.6:
                words = _even_word_timings(ln["text"], dur, lead=0.05)
            out.append({**ln, "path": path, "dur": dur, "words": words})
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


def place_lines(voiced: list[dict], total: float) -> list[dict]:
    """Put each line at its moment without overlaps. If the script is longer than the clip, middle lines are
    dropped first so the hook and the ending question always stay."""
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
        placed.append({**ln, "start": start})
        cursor = start + ln["dur"]
    return placed


# --------------------------------------------------------------------------- captions (ASS)

def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _ass_text(s: str) -> str:
    return re.sub(r"[{}\\]", "", s)


def _group_words(words: list[dict], max_words: int = 3, max_chars: int = 16) -> list[list[dict]]:
    groups, cur = [], []
    for w in words:
        txt = w["text"].strip()
        if not txt:
            continue
        candidate = " ".join(x["text"] for x in cur + [w])
        pause = cur and w["start"] - cur[-1]["end"] > 0.6  # a pause starts a new subtitle (no stale words)
        if cur and (pause or len(cur) >= max_words or len(candidate) > max_chars):
            groups.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?,;:]$", txt):  # break on punctuation for natural phrasing
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(path: Path, banner: str, words: list[dict], total: float, cta: str) -> None:
    vcfg = load_config().get("video", {})
    font = vcfg.get("font", "Montserrat")
    hl = vcfg.get("highlight_color_ass", "&H0000F0FF")  # BGR -> yellow
    wrapped = "\\N".join(textwrap.wrap(_ass_text(banner.upper()), width=17)[:3])
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Banner,{font},76,&H00FFFFFF,&H00FFFFFF,&H00000000,&HC0000000,-1,0,0,0,100,100,1,0,3,18,0,8,60,60,190,1
Style: Caption,{font},92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,7,3,5,40,40,0,1
Style: CTA,{font},46,&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,1,0,3,14,0,2,60,60,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [f"Dialogue: 2,{_ass_time(0)},{_ass_time(total)},Banner,,0,0,0,,{{\\fad(250,0)}}{wrapped}"]
    y = int(vcfg.get("caption_y", 1450))
    for group in _group_words(words):
        g_start = group[0]["start"]
        g_end_final = group[-1]["end"]
        for i, w in enumerate(group):
            start = w["start"] if i else g_start
            end = group[i + 1]["start"] if i + 1 < len(group) else g_end_final + 0.08
            parts = []
            for j, x in enumerate(group):
                t = _ass_text(x["text"].upper())
                parts.append(f"{{\\c{hl}\\fscx108\\fscy108}}{t}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}" if j == i else t)
            pop = "{\\fscx80\\fscy80\\t(0,90,\\fscx100\\fscy100)}" if i == 0 else ""
            lines.append(f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
                         f"{{\\an5\\pos(540,{y})}}{pop}{' '.join(parts)}")
    if cta:
        lines.append(f"Dialogue: 3,{_ass_time(max(0, total - 3.2))},{_ass_time(total)},CTA,,0,0,0,,"
                     f"{{\\fad(200,0)}}{_ass_text(cta.upper())}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- effects

SFX_RECIPES = {  # generated with FFmpeg - no samples, no copyright
    "whoosh": (["-f", "lavfi", "-i", "anoisesrc=color=pink:duration=0.55:amplitude=0.9"],
               "highpass=f=350,lowpass=f=5500,afade=t=in:st=0:d=0.3:curve=exp,"
               "afade=t=out:st=0.3:d=0.25:curve=exp,volume=1.4"),
    "boom": (["-f", "lavfi", "-i", "aevalsrc='0.95*sin(2*PI*(42+55*exp(-9*t))*t)*exp(-3.2*t)':s=48000:d=1.2",
              "-f", "lavfi", "-i", "anoisesrc=color=brown:duration=0.14:amplitude=0.7"],
             None),
}


def _sfx(kind: str) -> Path | None:
    out = VO_DIR.parent / "sfx" / f"{kind}.wav"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    inputs, af = SFX_RECIPES[kind]
    if kind == "boom":
        cmd = ["ffmpeg", "-y", *inputs, "-filter_complex",
               "[1:a]lowpass=f=400,afade=t=out:st=0.04:d=0.1[n];[0:a][n]amix=inputs=2:normalize=0,"
               "lowpass=f=220,volume=1.8", "-ar", "48000", "-ac", "2", str(out)]
    else:
        cmd = ["ffmpeg", "-y", *inputs, "-af", af, "-ar", "48000", "-ac", "2", str(out)]
    res = run(cmd, timeout=60)
    return out if res.returncode == 0 and out.exists() else None


def _music_track() -> Path | None:
    """A random track from the music/ folder (your own royalty-free music), if there is one."""
    import random
    from common import BASE_DIR
    folder = BASE_DIR / load_config().get("video", {}).get("fx", {}).get("music_dir", "music")
    tracks = [p for p in folder.glob("*") if p.suffix.lower() in (".mp3", ".m4a", ".wav", ".ogg")] if folder.exists() else []
    return random.choice(tracks) if tracks else None


def _zoom_filter(windows: list[tuple[float, float]], amount: float) -> str:
    """Punch-in zoom during the given [start, end] windows (output keeps 1080x1920 / 30 fps)."""
    if not windows:
        return ""
    terms = "+".join(f"between(it,{a:.2f},{b:.2f})*min(1,(it-{a:.2f})/0.12)" for a, b in windows)
    return (f",zoompan=z='1+{amount}*({terms})':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s=1080x1920:fps=30")


# --------------------------------------------------------------------------- render

def render(clip: Path, clip_id: str, banner: str, commentary: list[dict],
           voiced: list[dict] | None = None, fx: dict | None = None, caption_words: list[dict] | None = None,
           game_volume: float | None = None, fg_zoom: float | None = None) -> dict | None:
    """commentary = [{"t": seconds, "text": "line"}, ...]. `voiced` = lines already synthesised
    (with path/dur/words) - used by the feature-tour format where the edit follows the voice."""
    cfg = load_config()
    vcfg = cfg.get("video", {})
    VO_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)

    info = probe(clip)
    # never loop footage: short stream clips simply make a shorter video
    total = round(min(float(vcfg.get("max_duration_sec", 90)), info["duration"] or 60), 2)
    lines = place_lines(voiced if voiced is not None else synthesize_lines(commentary, clip_id), total)
    words = [{"text": w["text"], "start": w["start"] + ln["start"], "end": w["end"] + ln["start"]}
             for ln in lines for w in ln["words"]]
    if caption_words is not None:  # stream clips: subtitles of what the streamer says, no voice-over
        words = caption_words

    ass = VO_DIR / f"{clip_id}.ass"
    build_ass(ass, banner, words, total, vcfg.get("cta_text", "FOLLOW FOR DAILY GTA 6 MOMENTS"))
    out = READY_DIR / f"{clip_id}.mp4"

    vertical_src = info["height"] > info["width"]
    if vertical_src:
        video_chain = "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[base]"
    else:
        zoom = float(fg_zoom or vcfg.get("foreground_zoom", 1.3))
        fg_w = int(1080 * zoom) // 2 * 2
        video_chain = (
            "[0:v]split=2[a][b];"
            "[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=12:3,"
            "scale=1080:1920,eq=brightness=-0.12:saturation=1.2,setsar=1[bg];"
            f"[b]scale={fg_w}:-2,crop=1080:ih,setsar=1[fg];"
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
    # sound effects (whoosh on cuts, boom on the payoff) and optional background music
    fx = fx or {}
    fxcfg = vcfg.get("fx", {})
    extra_inputs, extra_chains, extra_labels = [], [], []
    idx = len(lines) + 1
    if fxcfg.get("sfx", True):
        sfx_vol = float(fxcfg.get("sfx_volume", 0.55))
        for ev in fx.get("sfx", []):
            path = _sfx(ev["kind"]) if 0 <= ev["t"] < total - 0.2 else None
            if not path:
                continue
            ms = int(max(0.0, ev["t"]) * 1000)
            extra_inputs += ["-i", str(path)]
            extra_chains.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,volume={sfx_vol},"
                                f"adelay={ms}|{ms}[x{idx}]")
            extra_labels.append(f"[x{idx}]")
            idx += 1
    music = _music_track()
    if music:
        extra_inputs += ["-stream_loop", "-1", "-i", str(music)]
        extra_chains.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,"
                            f"volume={float(fxcfg.get('music_volume', 0.09))}[x{idx}]")
        extra_labels.append(f"[x{idx}]")
        log.info("[render] background music: %s", music.name)
    extras = (";".join(extra_chains) + ";") if extra_chains else ""

    # --- pass 1: build the audio mix (fast, audio only)
    mix_wav = VO_DIR / f"{clip_id}_mix.wav"
    afc = (
        f"{voice}{extras}[vo]asplit=2[vo1][vo2];"
        f"[0:a]aresample=48000,aformat=channel_layouts=stereo,volume={game_vol}[game];"
        # game audio stays loud between lines and dips only while the voice talks
        "[game][vo1]sidechaincompress=threshold=0.02:ratio=6:attack=15:release=450[ducked];"
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
    zoom = _zoom_filter([w for w in fx.get("zoom", []) if w[0] < total], float(fxcfg.get("zoom_amount", 0.22))) \
        if fxcfg.get("zoom", True) else ""
    fc = (f"{video_chain};[base]fps=30{zoom},ass='{ass_escaped}',format=yuv420p[vout];"
          f"[1:a]{norm},aresample=48000[aout]")
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
    log.info("[render] %s (%.1fs, %.1f MB, %d voice lines)", out.name, total, out.stat().st_size / 1e6, len(lines))
    return {"path": str(out), "duration": total, "lines": [{"t": round(ln["start"], 1), "text": ln["text"]} for ln in lines]}
