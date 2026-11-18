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

async def _edge_tts(text: str, voice: str, rate: str, out: Path, boundary: str = "WordBoundary") -> list[dict]:
    import edge_tts
    words: list[dict] = []
    communicate = edge_tts.Communicate(text, voice, rate=rate, boundary=boundary)
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
    """One short line with edge-tts. Word timings relative to the line start."""
    for boundary in ("WordBoundary", "SentenceBoundary"):
        try:
            words = asyncio.run(_edge_tts(text, voice, rate, out, boundary))
            if out.exists() and out.stat().st_size > 2000:
                return words
        except Exception as exc:  # noqa: BLE001
            log.debug("[voice] edge %s/%s: %s", voice, boundary, exc)
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
            for m in gemini_writer._client().models.list():
                name = (m.name or "").removeprefix("models/")
                if "tts" in name and name not in _gemini_tts_cache:
                    _gemini_tts_cache.append(name)
            _gemini_tts_cache.sort(key=lambda n: ("flash" not in n, "lite" in n))  # flash first: bigger free quota
            log.info("[voice] Gemini TTS models: %s", _gemini_tts_cache)
        except Exception as exc:  # noqa: BLE001
            log.warning("[voice] could not list Gemini TTS models: %.150s", exc)
    return _gemini_tts_cache


def _gemini_line(text: str, out: Path, model: str) -> None:
    """One line with Gemini TTS - natural, steerable 'hyped YouTuber' delivery."""
    import base64
    import gemini_writer
    from google.genai import types
    vcfg = load_config().get("voice", {})
    style = vcfg.get("gemini_tts_style", "Say this like a hyped, fun gaming YouTuber reacting live to an insane clip")
    resp = gemini_writer._client().models.generate_content(
        model=model,
        contents=f"{style}: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=vcfg.get("gemini_tts_voice", "Puck")))),
        ),
    )
    part = resp.candidates[0].content.parts[0]
    data = part.inline_data.data
    if isinstance(data, str):
        data = base64.b64decode(data)
    m = re.search(r"rate=(\d+)", part.inline_data.mime_type or "")
    raw = out.with_suffix(".pcm")
    raw.write_bytes(data)
    # PCM -> mp3, trimming silence at both ends so the line lands exactly on its moment
    trim = "silenceremove=start_periods=1:start_threshold=-45dB"
    res = run(["ffmpeg", "-y", "-f", "s16le", "-ar", m.group(1) if m else "24000", "-ac", "1", "-i", str(raw),
               "-af", f"{trim},areverse,{trim},areverse", "-ar", "48000", str(out)], timeout=60)
    raw.unlink(missing_ok=True)
    if res.returncode != 0 or probe(out)["duration"] < 0.3:
        raise RuntimeError("Gemini TTS returned no usable audio")


def synthesize_lines(lines: list[dict], clip_id: str) -> list[dict]:
    """Voice every commentary line with ONE consistent voice.
    Order: Gemini TTS (most human) -> edge-tts voices -> gTTS. Returns lines + path/dur/words."""
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

    if engine in ("auto", "gemini") and env("GEMINI_API_KEY"):
        for model in _gemini_tts_models()[:2]:
            try:
                for ln, path in zip(lines, paths):
                    _gemini_line(ln["text"], path, model)
                return finish([[] for _ in lines], f"Gemini TTS {model}/{vcfg.get('gemini_tts_voice', 'Puck')}")
            except Exception as exc:  # noqa: BLE001
                log.warning("[voice] Gemini TTS %s failed (%.120s) - trying next option", model, exc)

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

def render(clip: Path, clip_id: str, banner: str, commentary: list[dict]) -> dict | None:
    """commentary = [{"t": seconds, "text": "short hype line"}, ...]"""
    cfg = load_config()
    vcfg = cfg.get("video", {})
    VO_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)

    info = probe(clip)
    total = round(min(float(vcfg.get("duration_sec", 48)), float(vcfg.get("max_duration_sec", 58)),
                      info["duration"] if info["duration"] >= 20 else 999), 2)
    lines = place_lines(synthesize_lines(commentary, clip_id), total)
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
    game_vol = float(vcfg.get("game_audio_volume", 0.9))
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
