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


def _gemini_line(text: str, out: Path, model: str, voice: str) -> None:
    """One line, one Gemini TTS request (used when the all-in-one audio can't be split)."""
    style = load_config().get("voice", {}).get("gemini_tts_style", "")
    wav = _gemini_tts_wav(f"{style}\nSay only this line:\n\n{text}", model, voice, out.with_suffix(".wav"))
    # trim silence at both ends so the line lands exactly on its moment
    trim = "silenceremove=start_periods=1:start_threshold=-45dB"
    res = run(["ffmpeg", "-y", "-i", str(wav), "-af", f"{trim},areverse,{trim},areverse", "-ar", "48000",
               str(out)], timeout=60)
    wav.unlink(missing_ok=True)
    if res.returncode != 0 or probe(out)["duration"] < 0.3:
        raise RuntimeError("Gemini TTS returned no usable audio")


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
                except RuntimeError as exc:
                    if "could not split" not in str(exc):
                        raise
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


def _merge_to(segs: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Merge the closest neighbours until there are n parts (a line can contain short pauses)."""
    segs = list(segs)
    while len(segs) > n:
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        i = gaps.index(min(gaps))
        segs[i:i + 2] = [(segs[i][0], segs[i + 1][1])]
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
    style = vcfg.get("gemini_tts_style", "")
    prompt = (f"{style}\nSay these lines one after another with a short natural pause between them:\n\n"
              + "\n\n".join(SAMPLE_LINES))
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


def _gemini_all_lines(lines: list[dict], paths: list[Path], model: str, voice: str | None = None) -> None:
    """ONE Gemini TTS request for every line (saves free quota), then split at the long pauses."""
    vcfg = load_config().get("voice", {})
    voice = voice or vcfg.get("gemini_tts_voice", "Fenrir")
    style = vcfg.get("gemini_tts_style", "Talk like a real guy reacting to a crazy clip with friends.")
    script = "\n\n".join(ln["text"] for ln in lines)
    prompt = (f"{style}\nSay the {len(lines)} lines below one after another, as separate parts. "
              "After each line, stay completely silent for about two seconds. Say only the lines.\n\n" + script)
    full = _gemini_tts_wav(prompt, model, voice, paths[0].with_name(paths[0].stem + "_all.wav"))
    segs = []
    for min_sil in (0.8, 0.5, 0.3):
        segs = _speech_segments(full, min_sil)
        if len(segs) >= len(lines):
            break
    if len(segs) < len(lines):
        raise RuntimeError(f"could not split Gemini audio into {len(lines)} lines (found {len(segs)})")
    for (a, b), path in zip(_merge_to(segs, len(lines)), paths):
        run(["ffmpeg", "-y", "-ss", f"{max(0, a - 0.05):.2f}", "-to", f"{b + 0.12:.2f}", "-i", str(full),
             "-ar", "48000", str(path)], timeout=60, check=True)
    full.unlink(missing_ok=True)


def synthesize_lines(lines: list[dict], clip_id: str) -> list[dict]:
    """Voice every commentary line with ONE consistent voice.
    Gemini TTS (human voice, config voice.gemini_tts_voice). With engine "gemini" and robot_fallback off it
    raises HumanVoiceUnavailable instead of using edge-tts / gTTS. Returns lines + path/dur/words."""
    vcfg = load_config().get("voice", {})
    engine = vcfg.get("engine", "auto")
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
    """Put each line at its moment without overlaps; drop lines that don't fit."""
    placed, cursor = [], 0.0
    for ln in sorted(voiced, key=lambda x: x["t"]):
        start = max(ln["t"], cursor + 0.35, 0.25)
        if start + ln["dur"] > total - 0.3:
            start = max(cursor + 0.35, total - 0.3 - ln["dur"])
        if start + ln["dur"] > total or start < cursor:
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
        if cur and (len(cur) >= max_words or len(candidate) > max_chars):
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


# --------------------------------------------------------------------------- render

def render(clip: Path, clip_id: str, banner: str, commentary: list[dict],
           voiced: list[dict] | None = None) -> dict | None:
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

    ass = VO_DIR / f"{clip_id}.ass"
    build_ass(ass, banner, words, total, vcfg.get("cta_text", "FOLLOW FOR DAILY GTA 6 MOMENTS"))
    out = READY_DIR / f"{clip_id}.mp4"

    vertical_src = info["height"] > info["width"]
    if vertical_src:
        video_chain = "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[base]"
    else:
        zoom = float(vcfg.get("foreground_zoom", 1.3))
        fg_w = int(1080 * zoom) // 2 * 2
        video_chain = (
            "[0:v]split=2[a][b];"
            "[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=12:3,"
            "scale=1080:1920,eq=brightness=-0.12:saturation=1.2,setsar=1[bg];"
            f"[b]scale={fg_w}:-2,crop=1080:ih,setsar=1[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2-40[base]"
        )
    ass_escaped = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    game_vol = float(vcfg.get("game_audio_volume_tour" if voiced is not None else "game_audio_volume", 0.9))
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
    # --- pass 1: build the audio mix (fast, audio only)
    mix_wav = VO_DIR / f"{clip_id}_mix.wav"
    afc = (
        f"{voice}[vo]asplit=2[vo1][vo2];"
        f"[0:a]aresample=48000,aformat=channel_layouts=stereo,volume={game_vol}[game];"
        # game audio stays loud between lines and dips only while the voice talks
        "[game][vo1]sidechaincompress=threshold=0.02:ratio=6:attack=15:release=450[ducked];"
        "[ducked][vo2]amix=inputs=2:duration=first:normalize=0[aout]"
    )
    inputs = ["-stream_loop", "-1", "-i", str(clip)] if info["duration"] < total else ["-i", str(clip)]
    for ln in lines:
        inputs += ["-i", str(ln["path"])]
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
    fc = (f"{video_chain};[base]ass='{ass_escaped}',fps=30,format=yuv420p[vout];"
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
