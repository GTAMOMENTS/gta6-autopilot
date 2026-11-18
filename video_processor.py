"""Voiceover + captions + 1080x1920 render.

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
import textwrap
from pathlib import Path

from common import READY_DIR, VO_DIR, env, load_config, probe, run

log = logging.getLogger("autopilot")

# --------------------------------------------------------------------------- TTS

async def _edge_tts(text: str, voice: str, rate: str, out: Path) -> list[dict]:
    import edge_tts
    words: list[dict] = []
    communicate = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    with open(out, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7
                words.append({"text": chunk["text"], "start": start, "end": start + chunk["duration"] / 1e7})
    return words


def _even_word_timings(text: str, duration: float) -> list[dict]:
    tokens = text.split()
    weights = [max(2, len(t)) for t in tokens]
    total = sum(weights) or 1
    t, out = 0.15, []
    span = max(1.0, duration - 0.3)
    for tok, w in zip(tokens, weights):
        d = span * w / total
        out.append({"text": tok, "start": t, "end": t + d})
        t += d
    return out


def synthesize(text: str, out: Path) -> tuple[float, list[dict]]:
    """Return (duration, word timings). Tries several edge-tts voices, then gTTS."""
    vcfg = load_config().get("voice", {})
    voices = [env("TTS_VOICE")] + vcfg.get("voices", ["en-US-AndrewNeural", "en-US-GuyNeural"])
    rate = vcfg.get("rate", "+10%")
    words: list[dict] = []
    for voice in [v for v in dict.fromkeys(voices) if v]:
        try:
            words = asyncio.run(_edge_tts(text, voice, rate, out))
            if out.exists() and out.stat().st_size > 5000:
                log.info("[voice] edge-tts voice %s (%d word timings)", voice, len(words))
                break
        except Exception as exc:  # noqa: BLE001
            log.warning("[voice] %s failed: %s", voice, exc)
        words = []
    else:
        log.info("[voice] falling back to gTTS")
        from gtts import gTTS
        raw = out.with_suffix(".gtts.mp3")
        gTTS(text=text, lang="en", tld="com").save(str(raw))
        run(["ffmpeg", "-y", "-i", str(raw), "-filter:a", "atempo=1.1", str(out)], check=True)
        raw.unlink(missing_ok=True)

    duration = probe(out)["duration"] or 45.0
    if len(words) < len(text.split()) * 0.6:
        words = _even_word_timings(text, duration)
    return duration, words


def fit_voice(audio: Path, words: list[dict], duration: float, max_len: float) -> tuple[Path, list[dict], float]:
    """Speed the voiceover up (max 1.25x) if it is longer than the allowed video length."""
    if duration <= max_len:
        return audio, words, duration
    factor = min(1.25, duration / max_len)
    fitted = audio.with_name(audio.stem + "_fit.mp3")
    run(["ffmpeg", "-y", "-i", str(audio), "-filter:a", f"atempo={factor:.4f}", str(fitted)], check=True)
    log.info("[voice] voiceover %.1fs > %.1fs, sped up x%.2f", duration, max_len, factor)
    words = [{"text": w["text"], "start": w["start"] / factor, "end": w["end"] / factor} for w in words]
    return fitted, words, probe(fitted)["duration"]


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

def render(clip: Path, clip_id: str, banner: str, script: str) -> dict | None:
    cfg = load_config()
    vcfg = cfg.get("video", {})
    VO_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)

    max_len = float(vcfg.get("max_duration_sec", 58))
    vo_path = VO_DIR / f"{clip_id}.mp3"
    vo_dur, words = synthesize(script, vo_path)
    vo_path, words, vo_dur = fit_voice(vo_path, words, vo_dur, max_len - 1.5)
    total = round(min(max_len, vo_dur + 1.5), 2)

    ass = VO_DIR / f"{clip_id}.ass"
    build_ass(ass, banner, words, total, vcfg.get("cta_text", "FOLLOW FOR DAILY GTA 6 MOMENTS"))
    out = READY_DIR / f"{clip_id}.mp4"

    info = probe(clip)
    vertical_src = info["height"] > info["width"]
    if vertical_src:
        video_chain = "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1[base]"
    else:
        zoom = float(vcfg.get("foreground_zoom", 1.18))
        fg_w = int(1080 * zoom) // 2 * 2
        video_chain = (
            "[0:v]split=2[a][b];"
            "[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=12:3,"
            "scale=1080:1920,eq=brightness=-0.12:saturation=1.2,setsar=1[bg];"
            f"[b]scale={fg_w}:-2,crop=1080:ih,setsar=1[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2-40[base]"
        )
    ass_escaped = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    game_vol = float(vcfg.get("game_audio_volume", 0.5))
    fc = (
        f"{video_chain};[base]ass='{ass_escaped}',fps=30,format=yuv420p[vout];"
        f"[0:a]volume={game_vol},aresample=48000[game];"
        "[1:a]aresample=48000,volume=1.0,asplit=2[vo1][vo2];"
        "[game][vo1]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=350[ducked];"
        "[ducked][vo2]amix=inputs=2:duration=longest:normalize=0,"
        "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]"
    )
    inputs = ["-stream_loop", "-1", "-i", str(clip)] if info["duration"] < total else ["-i", str(clip)]
    cmd = ["ffmpeg", "-y", "-hide_banner", *inputs, "-i", str(vo_path),
           "-filter_complex", fc, "-map", "[vout]", "-map", "[aout]",
           "-c:v", "libx264", "-preset", vcfg.get("preset", "veryfast"), "-crf", str(vcfg.get("crf", 20)),
           "-profile:v", "high", "-maxrate", "12M", "-bufsize", "24M",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
           "-t", str(total), "-movflags", "+faststart", str(out)]
    res = run(cmd, timeout=900)
    if res.returncode != 0 or not out.exists():
        log.error("[render] FFmpeg failed: %s", res.stderr[-1200:])
        return None
    log.info("[render] %s (%.1fs, %.1f MB)", out.name, total, out.stat().st_size / 1e6)
    return {"path": str(out), "duration": total}
