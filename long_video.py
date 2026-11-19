"""Daily long YouTube video: "Top N craziest GTA 6 stream moments today" (16:9, 5-10 min).

  1. gather the day's most-viewed GTA 6 clips from the top streams (Twitch, Kick, X clip accounts)
  2. AI watches them in batches (saves quota): real GTA 6? spoiler? unsafe? hype score, best part, a short
     label + one narration line per moment
  3. countdown edit: cold open of #1, title card + Fenrir line per moment (worst -> best), whoosh between clips,
     subscribe outro; even loudness
  4. YouTube: title with streamer names, chapters + credits in the description, auto thumbnail, first comment
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from common import RAW_DIR, READY_DIR, STATE_DIR, WORK_DIR, env, load_config, load_json, now_local, probe, run, \
    save_json

log = logging.getLogger("autopilot")
STATE_FILE = STATE_DIR / "long_videos.json"
LONG_DIR = WORK_DIR / "long"
W, H = 1920, 1080


def _cfg() -> dict:
    return load_config().get("long_video", {})


# --------------------------------------------------------------------------- 1. gather

def gather(c: dict, used: set) -> list[dict]:
    import stream_clips as sc
    from source_fetcher import blocked_channel
    scfg = {**load_config().get("stream_clips", {}), "window_hours": float(c.get("window_hours", 24)),
            "top_streams": int(c.get("top_streams", 15))}
    clips = []
    for fn in (sc.twitch_clips, sc.kick_clips, sc.x_account_clips):
        try:
            clips += fn(scfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("[long] %s failed: %.150s", fn.__name__, exc)
    langs = set(scfg.get("languages", ["en"]))
    bars = scfg.get("viral_bar", {})
    pool = []
    for x in clips:
        if x["key"] in used or blocked_channel(x["streamer"]):
            continue
        if not sc.is_famous(x["streamer"], scfg):  # famous English-speaking streamers only
            continue
        if x["platform"] == "twitch" and langs and x["language"] and x["language"] not in langs:
            continue
        if not 10 <= x["duration"] <= 180:  # chapters need 10 s+
            continue
        # views relative to what's normal on that platform (X counts millions, Kick hundreds)
        base = float((bars.get(x["platform"]) or {}).get("min_views", 1000)) or 1000
        x["score"] = math.log10(x["views"] + 1) - math.log10(base)
        pool.append(x)
    pool.sort(key=lambda x: x["score"], reverse=True)
    picked, per_streamer = [], {}
    for x in pool:
        s = x["streamer"].lower()
        if per_streamer.get(s, 0) >= int(c.get("max_per_streamer", 2)):
            continue
        if any(p["streamer"].lower() == s and abs(p["created"] - x["created"]) < 600 for p in picked):
            continue  # same moment clipped twice
        picked.append(x)
        per_streamer[s] = per_streamer.get(s, 0) + 1
        if len(picked) >= int(c.get("candidates", 24)):
            break
    log.info("[long] %d clips found, %d candidates: %s", len(clips), len(picked),
             [f"{x['platform']}:{x['streamer']} {x['views']}v" for x in picked[:12]])
    return picked


def download(cands: list[dict]) -> list[dict]:
    import stream_clips as sc
    from source_fetcher import Candidate, _download
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out, why = [], {}
    for x in cands:
        path = RAW_DIR / f"long_{re.sub(r'[^a-zA-Z0-9]', '_', x['key'])[:50]}.mp4"
        cand = Candidate(key=x["key"], url=x["url"], title=x["title"], uploader=x["streamer"],
                         provider=f"{x['platform']}_clip", duration=x["duration"], kind="stream",
                         media_url=x.get("media_url") or None)
        try:
            ok = _download(cand, 0, int(x["duration"]) + 1, path)
        except Exception as exc:  # noqa: BLE001
            log.info("[long] download failed %s: %.100s", x["key"], exc)
            ok = False
        if not ok:
            why[f"download failed ({x['platform']})"] = why.get(f"download failed ({x['platform']})", 0) + 1
            continue
        info = probe(path)
        if min(info["width"], info["height"]) < 480 or info["duration"] < 6:
            r = f"low res ({x['platform']} {info['width']}x{info['height']})" if info["duration"] >= 6 else "too short"
            why[r] = why.get(r, 0) + 1
            path.unlink(missing_ok=True)
            continue
        if not _has_audio(path):
            why["no sound"] = why.get("no sound", 0) + 1
            path.unlink(missing_ok=True)
            continue
        rep = sc.smoothness(path)
        if rep > float(_cfg().get("max_repeated_frames", 0.2)):
            why["choppy"] = why.get("choppy", 0) + 1
            log.info("[long] ✋ %s (%s, %s) is choppy: %.0f%% repeated frames", x["key"], x["streamer"], x["platform"],
                     rep * 100)
            path.unlink(missing_ok=True)
            continue
        out.append({**x, "path": path, "dur": info["duration"], "repeat": rep})
    log.info("[long] downloaded %d/%d clips (skipped: %s)", len(out), len(cands), why)
    return out


# --------------------------------------------------------------------------- 2. AI review (batches)

REVIEW_PROMPT = """You are the editor of "{brand}", a GTA 6 channel. The attached video is {k} short GTA 6 livestream
clips joined one after another. Clip time ranges in this video:
{ranges}
For EACH clip judge honestly and write:
- is_gta6_footage: true only if the game shown is real Grand Theft Auto VI.
- is_spoiler: true for story spoilers (story cutscenes, plot reveals, endings, character deaths).
- is_english: true if the streamer speaks English (or nobody talks); false for any other language.
- unsafe: true ONLY for nudity, slurs / hate speech, real-world violence, doxxing - things YouTube removes.
  In-game violence, crime, police chases and the streamer swearing are normal GTA content and are NOT unsafe.
- hype_score 1-10: how funny / wild / shareable the moment and the streamer's reaction are (strict).
- label: max 6 words, Title Case, what happens ("Kai Cenat Vs The Yacht"). No spoilers of the punchline.
- line: what our narrator says right before it plays: 8-20 words, casual YouTuber voice, names the streamer,
  builds curiosity without giving away the punchline ("Number seven, Kai thought a jet ski could outrun a police
  helicopter. It could not.").  Say "GTA 6", never "GTA VI".
- trim_start / trim_end: seconds WITHIN that clip (0 = start of the clip) of the best part, 10-45 seconds.
- best_t: second within the clip with the most eye-catching frame (for the thumbnail).
Return ONLY JSON: {{"clips": [{{"clip": number, "is_gta6_footage": bool, "is_spoiler": bool, "unsafe": bool,
 "is_english": bool, "hype_score": int, "label": str, "line": str, "trim_start": number, "trim_end": number,
 "best_t": number}}]}}"""


def _proxy(path: Path, out: Path) -> bool:
    res = run(["ffmpeg", "-y", "-i", str(path), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
               "-filter_complex", "[0:v]scale=-2:480,fps=12,setsar=1,format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2[v]",
               "-map", "[v]", "-map", "0:a?", "-map", "1:a", "-shortest", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "32", "-c:a", "aac", "-ar", "48000", "-ac", "1", "-b:a", "48k", str(out)], timeout=180)
    return res.returncode == 0 and out.exists()


def review(moments: list[dict], brand: str) -> list[dict]:
    import gemini_writer
    if not env("GEMINI_API_KEY"):
        log.warning("[long] GEMINI_API_KEY not set - can't review clips")
        return []
    LONG_DIR.mkdir(parents=True, exist_ok=True)
    size = int(_cfg().get("review_batch", 6))
    reviewed = []
    for b in range(0, len(moments), size):
        batch = moments[b:b + size]
        parts, ranges, t = [], [], 0.0
        for i, m in enumerate(batch, 1):
            p = LONG_DIR / f"proxy_{b + i}.mp4"
            if not _proxy(m["path"], p):
                continue
            d = probe(p)["duration"]
            parts.append((i, m, p, t, t + d))
            ranges.append(f"{i}: {t:.0f}-{t + d:.0f} s ({m['streamer']} on {m['platform'].title()}: \"{m['title'][:60]}\")")
            t += d
        if not parts:
            continue
        lst = LONG_DIR / "proxy_list.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for _, _, p, _, _ in parts))
        reel = LONG_DIR / f"review_{b}.mp4"
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(reel)], timeout=120)
        verdicts = {}
        for attempt in (1, 2):  # the AI sometimes answers nothing (busy) - one retry before giving up
            data = None
            try:
                gemini_writer._unavailable_until = 0.0  # a busy moment in one batch shouldn't skip the rest
                up = gemini_writer._upload_video(reel)
                if up is not None:
                    data = gemini_writer._generate_json([up, REVIEW_PROMPT.format(
                        brand=brand, k=len(parts), ranges="\n".join(ranges))], temperature=0.6, prefer_pro=True)
                    try:
                        gemini_writer._client().files.delete(name=up.name)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as exc:  # noqa: BLE001
                log.warning("[long] review batch failed: %.150s", exc)
            verdicts = {}
            for v in (data or {}).get("clips", []) if isinstance(data, dict) else []:
                try:
                    verdicts[int(v.get("clip", 0))] = v
                except (TypeError, ValueError, AttributeError):
                    continue
            if verdicts:
                break
            log.warning("[long] AI gave no verdicts for batch %d (attempt %d)", b // size + 1, attempt)
            time.sleep(20)
        for i, m, p, _, _ in parts:
            p.unlink(missing_ok=True)
            v = verdicts.get(i)
            if not v:
                continue
            reason = ("not GTA 6" if not v.get("is_gta6_footage") else "spoiler" if v.get("is_spoiler")
                      else "not English" if v.get("is_english") is False else "unsafe" if v.get("unsafe") else None)
            if reason:
                log.info("[long] ✋ %s (%s): %s", m["key"], m["streamer"], reason)
                continue
            try:
                a = max(0.0, float(v.get("trim_start") or 0))
                e = min(m["dur"], float(v.get("trim_end") or m["dur"]))
            except (TypeError, ValueError):
                a, e = 0.0, m["dur"]
            if e - a < 10:  # chapters need 10 s+; keep the whole clip if the AI cut too tight
                a, e = 0.0, min(m["dur"], 45.0)
            if e - a < 10:
                continue
            reviewed.append({**m, "hype": int(v.get("hype_score") or 0), "label": str(v.get("label") or m["title"])[:60],
                             "line": str(v.get("line") or "").strip(), "a": a, "b": min(e, a + 45),
                             "best_t": float(v.get("best_t") or (a + e) / 2)})
        reel.unlink(missing_ok=True)
        log.info("[long] batch %d: %d/%d clips passed", b // size + 1,
                 sum(1 for r in reviewed if r["key"] in {m["key"] for m in batch}), len(batch))
    return reviewed


# --------------------------------------------------------------------------- 3. package (title etc.)

PACKAGE_PROMPT = """You write for "{brand}", a GTA 6 YouTube channel. Today's video is a countdown of the {n} craziest
GTA 6 livestream moments of the day. Moments (from #{n} up to #1):
{moments}
Write:
- title: max 70 characters, the style of the biggest gaming compilation channels, name 1-3 of the biggest
  streamers, e.g. "Top 12 CRAZIEST GTA 6 Moments Today (IShowSpeed, Kai Cenat, caseoh)". No hashtags.
- intro_line: what the narrator says over the cold open of #1, 10-20 words, makes people stay to the end.
- outro_line: 8-15 words, asks which moment was the best and to subscribe for daily GTA 6 moments.
- thumbnail_text: 2-4 words, ALL CAPS, huge thumbnail text that makes people click (e.g. "HE JUMPED?!").
- summary: 1-2 sentences for the description.
- tags: 10 YouTube search tags (strings, no #).
Say "GTA 6", never "GTA VI". Return ONLY JSON with exactly these keys."""


def package(moments: list[dict], brand: str) -> dict:
    import ai_fallback
    import gemini_writer
    n = len(moments)
    listing = "\n".join(f"#{n - i}: {m['label']} ({m['streamer']})" for i, m in enumerate(moments))
    prompt = PACKAGE_PROMPT.format(brand=brand, n=n, moments=listing)
    data = gemini_writer._generate_json(prompt, temperature=0.8, prefer_pro=True)
    if not data and ai_fallback.available():
        data = ai_fallback.chat_json(prompt)
    data = data or {}
    if data:  # critic AI: second opinion on title / thumbnail text / intro
        import critic
        fields = {k: str(data[k]) for k in ("title", "thumbnail_text", "intro_line") if data.get(k)}
        data.update(critic.polish(fields, {"title": 90, "thumbnail_text": 28, "intro_line": 160}, listing))
    names = list(dict.fromkeys(m["streamer"] for m in reversed(moments)))[:3]
    title = str(data.get("title") or f"Top {n} CRAZIEST GTA 6 Stream Moments Today ({', '.join(names)})")
    return {
        "title": re.sub(r"[<>]", "", title)[:95],
        "intro_line": str(data.get("intro_line") or f"These are the {n} craziest GTA 6 moments from today's streams, "
                                                     "and number one is unreal."),
        "outro_line": str(data.get("outro_line") or "Which one was the best? Subscribe for daily GTA 6 moments."),
        "thumbnail_text": re.sub(r"[^A-Z0-9 !?'.]", "", str(data.get("thumbnail_text") or "GTA 6 CHAOS").upper())[:28],
        "summary": str(data.get("summary") or f"The {n} craziest GTA 6 livestream moments from today."),
        "tags": [str(t)[:30] for t in (data.get("tags") or [])][:12] or ["GTA 6", "GTA 6 funny moments"],
    }


# --------------------------------------------------------------------------- 4. render (16:9)

FIT = ("split=2[a][b];[a]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=20:2,"
       "eq=brightness=-0.15[bg];[b]scale=1920:1080:force_original_aspect_ratio=decrease[fg];"
       "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30,format=yuv420p")
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def _has_audio(path: Path) -> bool:
    res = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
               str(path)], timeout=30)
    return bool(res.stdout.strip())


def _segment(src: Path, a: float, b: float, out: Path, silent_ok: bool = False) -> bool:
    """One piece of the video: 1920x1080/30fps, even loudness. Audio and video are cut to EXACTLY the same
    length (whole frames) - otherwise tiny differences add up over 15 clips and the sound drifts out of sync."""
    d = round((b - a) * 30) / 30
    vchain = f"[0:v]{FIT},trim=duration={d:.4f},setpts=PTS-STARTPTS[v]"
    if _has_audio(src):
        achain = (f"[0:a]aresample=48000,aformat=channel_layouts=stereo,loudnorm=I=-16:TP=-1.5:LRA=11,"
                  f"aresample=48000,apad,atrim=duration={d:.4f},asetpts=PTS-STARTPTS[a]")
    elif silent_ok:
        achain = f"anullsrc=r=48000:cl=stereo,atrim=duration={d:.4f}[a]"
    else:
        log.info("[long] %s has no sound - not used", src.name)
        return False
    res = run(["ffmpeg", "-y", "-ss", f"{a:.3f}", "-t", f"{d + 0.5:.3f}", "-i", str(src), "-filter_complex",
               f"{vchain};{achain}", "-map", "[v]", "-map", "[a]", *ENC, str(out)], timeout=600)
    if res.returncode != 0 or not out.exists() or probe(out)["duration"] < 1:
        log.warning("[long] could not prepare %s: %.300s", src.name, res.stderr[-300:])
        return False
    return True


def _join(segs: list[Path], out: Path) -> bool:
    """Join with the concat FILTER (decodes + re-encodes): timestamps line up exactly, no drift between clips."""
    inputs = []
    for sg in segs:
        inputs += ["-i", str(sg)]
    pads = "".join(f"[{i}:v][{i}:a]" for i in range(len(segs)))
    res = run(["ffmpeg", "-y", *inputs, "-filter_complex", f"{pads}concat=n={len(segs)}:v=1:a=1[v][a]",
               "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-c:a", "aac", "-b:a", "256k", "-ar", "48000", str(out)], timeout=2400)
    if res.returncode != 0:
        log.error("[long] join failed: %s", res.stderr[-500:])
    return res.returncode == 0 and out.exists()


def _vdur(path: Path) -> float:
    """Duration of the VIDEO stream (the container duration can include a longer audio tail)."""
    res = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of",
               "csv=p=0", str(path)], timeout=30)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return probe(path)["duration"]


def _ass_text(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _t(sec: float) -> str:
    h, rem = divmod(max(0.0, sec), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _build_ass(path: Path, title: str, cards: list[dict], outro_at: float, total: float, handle: str) -> None:
    font = load_config().get("video", {}).get("font", "Montserrat")
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,{font},84,&H00FFFFFF,&H00FFFFFF,&H00000000,&HB0000000,-1,0,0,0,100,100,1,0,3,20,0,8,80,80,70,1
Style: Num,{font},150,&H0000F0FF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,9,4,7,70,70,50,1
Style: Label,{font},64,&H00FFFFFF,&H00FFFFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,1,0,3,16,0,7,70,70,220,1
Style: Who,{font},40,&H0000F0FF,&H00FFFFFF,&H00000000,&HA0000000,-1,0,0,0,100,100,1,0,3,12,0,7,70,70,320,1
Style: Outro,{font},90,&H00FFFFFF,&H00FFFFFF,&H00000000,&HB0000000,-1,0,0,0,100,100,1,0,3,22,0,5,80,80,0,1
Style: Handle,{font},34,&H90FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,1,0,1,3,0,3,40,40,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    import textwrap
    short = re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip().upper()  # drop "(IShowSpeed, Kai...)"
    wrapped = "\\N".join(textwrap.wrap(_ass_text(short), 26)[:3])
    ev = [f"Dialogue: 2,{_t(0)},{_t(cards[0]['start'] if cards else 4)},Title,,0,0,0,,{{\\fad(200,200)}}"
          f"{wrapped}"]
    for cd in cards:
        a, b = cd["start"] + 0.2, cd["start"] + 4.5
        pop = "{\\fad(150,250)\\fscx70\\fscy70\\t(0,180,\\fscx100\\fscy100)}"
        ev.append(f"Dialogue: 3,{_t(a)},{_t(b)},Num,,0,0,0,,{pop}#{cd['n']}")
        ev.append(f"Dialogue: 2,{_t(a + 0.15)},{_t(b)},Label,,0,0,0,,{{\\fad(150,250)}}{_ass_text(cd['label'].upper())}")
        ev.append(f"Dialogue: 2,{_t(a + 0.3)},{_t(b)},Who,,0,0,0,,{{\\fad(150,250)}}{_ass_text(cd['who'])}")
    ev.append(f"Dialogue: 2,{_t(outro_at + 0.3)},{_t(total)},Outro,,0,0,0,,{{\\fad(250,0)}}"
              "WHICH ONE WAS THE BEST?\\N\\NSUBSCRIBE FOR DAILY GTA 6 MOMENTS")
    if handle:
        ev.append(f"Dialogue: 1,{_t(0)},{_t(total)},Handle,,0,0,0,,{_ass_text(handle)}")
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def _font_file() -> str:
    for q in ("Montserrat:style=Black", "Montserrat:style=ExtraBold", "Montserrat:bold", "DejaVu Sans:bold"):
        res = run(["fc-match", "-f", "%{file}", q], timeout=10)
        if res.returncode == 0 and res.stdout.strip() and ("ontserrat" in res.stdout or "DejaVu" in q):
            return res.stdout.strip()
    return "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def thumbnail(src: Path, t: float, text: str, out: Path) -> Path | None:
    """1280x720 thumbnail: the #1 moment's best frame, punchier colours, huge text with a thick outline."""
    words = text.split()
    lines = [text]
    if len(text) > 9 and len(words) >= 2:  # two lines, split where the halves are most even
        cut = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
        lines = [" ".join(words[:cut]), " ".join(words[cut:])]
    # font size that fits the widest line in ~1160 px (bold caps are ~0.72 em wide)
    size = int(min(200, 1160 / (0.72 * max(len(ln) for ln in lines))))
    font = _font_file()
    LONG_DIR.mkdir(parents=True, exist_ok=True)
    draw = []
    for i, ln in enumerate(lines):
        tf = LONG_DIR / f"thumb_{i}.txt"
        tf.write_text(ln, encoding="utf-8")
        y = f"h-{(len(lines) - i) * int(size * 1.08) + 40}"
        draw.append(f"drawtext=fontfile='{font}':textfile='{tf}':fontsize={size}:fontcolor={'yellow' if i == 0 else 'white'}"
                    f":borderw={max(8, size // 12)}:bordercolor=black:x=(w-text_w)/2:y={y}")
    vf = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,eq=saturation=1.4:contrast=1.12," \
         "drawbox=x=0:y=ih*0.45:w=iw:h=ih*0.55:color=black@0.35:t=fill," + ",".join(draw)
    dur = probe(src)["duration"] or 0
    t = min(max(0.0, t), max(0.0, dur - 0.5))  # the AI's "best frame" can point past the end of the clip
    plain = "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,eq=saturation=1.4:contrast=1.12"
    for filt in (vf, plain):  # with the big text; else at least the punchy frame
        out.unlink(missing_ok=True)
        res = run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1", "-vf", filt,
                   "-q:v", "3", str(out)], timeout=60)
        if res.returncode == 0 and out.exists() and out.stat().st_size > 1000:
            return out
        log.warning("[long] thumbnail attempt failed (t=%.1f of %.1fs): %.200s", t, dur, res.stderr[-200:])
    return None


def render_long(moments: list[dict], pkg: dict, clip_id: str) -> dict | None:
    from video_processor import HumanVoiceUnavailable, _sfx, synthesize_lines
    LONG_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)
    c = _cfg()
    top = moments[-1]  # #1 (moments are in countdown order: worst first)
    # narration first (Fenrir): its lengths decide the cold-open length; the video still works without it
    texts = [pkg["intro_line"]] + [m.get("line") or "" for m in moments] + [pkg["outro_line"]]
    voice_by_text: dict[str, dict] = {}
    try:
        spoken = [t for t in texts if t.strip()]
        for v in synthesize_lines([{"t": 0.0, "text": t} for t in spoken], clip_id):
            voice_by_text[v["text"]] = v
    except HumanVoiceUnavailable as exc:
        log.warning("[long] narration skipped: %s", exc)
    intro_v = voice_by_text.get(pkg["intro_line"])
    segs, cards, voiced = [], [], []
    # cold open: #1 around its best frame, long enough for the intro line
    teaser = LONG_DIR / "seg_teaser.mp4"
    t_len = min(9.0, max(float(c.get("teaser_sec", 4.0)), (intro_v["dur"] + 0.8) if intro_v else 0))
    ta = max(0.0, min(top["best_t"] - t_len / 2, top["dur"] - t_len))
    if _segment(top["path"], ta, ta + t_len, teaser):
        segs.append(teaser)
    cursor = sum(_vdur(s) for s in segs)
    if intro_v and segs:
        voiced.append({**intro_v, "t": 0.3})
    n = len(moments)
    for i, m in enumerate(moments):
        out = LONG_DIR / f"seg_{i:02d}.mp4"
        if not _segment(m["path"], m["a"], m["b"], out):
            log.warning("[long] could not prepare %s - dropped", m["key"])
            continue
        d = _vdur(out)
        num = n - i
        cards.append({"start": cursor, "n": num, "label": m["label"],
                      "who": f"{m['streamer']} on {m['platform'].title()}", "m": m})
        v = voice_by_text.get(m.get("line") or "")
        if v and v["dur"] < d - 0.6:  # a line longer than its moment would talk over the next one
            voiced.append({**v, "t": cursor + 0.35})
        segs.append(out)
        cursor += d
    if len(cards) < 3:
        return None
    # renumber if some segments were dropped
    for k, cd in enumerate(cards):
        cd["n"] = len(cards) - k
    outro_at = cursor
    frame = LONG_DIR / "outro.jpg"
    run(["ffmpeg", "-y", "-ss", f"{top['best_t']:.2f}", "-i", str(top["path"]), "-frames:v", "1", str(frame)], timeout=60)
    outro = LONG_DIR / "seg_outro.mp4"
    outro_len = float(c.get("outro_sec", 7.0))
    if frame.exists():
        run(["ffmpeg", "-y", "-loop", "1", "-i", str(frame), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
             "-filter_complex", "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
             f"boxblur=25:2,eq=brightness=-0.3,fps=30,format=yuv420p,trim=duration={outro_len}[v];"
             f"[1:a]atrim=duration={outro_len}[a]", "-map", "[v]", "-map", "[a]", *ENC, str(outro)], timeout=120)
        if outro.exists():
            segs.append(outro)
            ov = voice_by_text.get(pkg["outro_line"])
            if ov and ov["dur"] < outro_len - 0.5:
                voiced.append({**ov, "t": outro_at + 0.3})
    total = outro_at + (outro_len if outro.exists() else 0)
    body = LONG_DIR / "body.mp4"
    if not _join(segs, body):
        return None
    total = _vdur(body) or total

    ass = LONG_DIR / f"{clip_id}.ass"
    handle = load_config().get("brand", {}).get("handles", {}).get("youtube", "")
    _build_ass(ass, pkg["title"], cards, outro_at, total, handle)

    inputs, chains, labels = ["-i", str(body)], [], []
    k = 1
    for v in voiced:
        ms = int(v["t"] * 1000)
        inputs += ["-i", str(v["path"])]
        chains.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,volume=1.35,adelay={ms}|{ms}[v{k}]")
        labels.append(f"[v{k}]")
        k += 1
    sfx = _sfx("whoosh")
    sfx_labels = []
    if sfx:
        for cd in cards[1:] + ([{"start": outro_at}] if outro.exists() else []):
            ms = int(max(0.0, cd["start"] - 0.25) * 1000)
            inputs += ["-i", str(sfx)]
            chains.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.5,adelay={ms}|{ms}[x{k}]")
            sfx_labels.append(f"[x{k}]")
            k += 1
    if labels:
        voice = "".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=longest,apad[vo];[vo]asplit=2[vo1][vo2];"
        duck = "[0:a][vo1]sidechaincompress=threshold=0.03:ratio=5:attack=20:release=500[g];"
        mix_in, n_in = "[g][vo2]", 2
    else:
        voice, duck, mix_in, n_in = "", "[0:a]anull[g];", "[g]", 1
    afc = (";".join(chains) + ";" if chains else "") + voice + duck + \
        f"{mix_in}{''.join(sfx_labels)}amix=inputs={n_in + len(sfx_labels)}:normalize=0:duration=first," \
        "loudnorm=I=-14:TP=-1.5:LRA=11[aout]"
    ass_esc = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    out = READY_DIR / f"{clip_id}.mp4"
    res = run(["ffmpeg", "-y", "-hide_banner", *inputs, "-filter_complex", f"[0:v]ass='{ass_esc}'[vout];{afc}",
               "-map", "[vout]", "-map", "[aout]", "-t", f"{total:.2f}", "-c:v", "libx264", "-preset",
               str(c.get("preset", "veryfast")), "-crf", str(c.get("crf", 21)), "-profile:v", "high",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out)], timeout=3000)
    if res.returncode != 0 or not out.exists():
        log.error("[long] final render failed: %s", res.stderr[-1200:])
        return None
    thumb = thumbnail(top["path"], top["best_t"], pkg["thumbnail_text"], READY_DIR / f"{clip_id}_thumb.jpg")
    chapters = ["0:00 Intro"] + [f"{int(cd['start'] // 60)}:{int(cd['start'] % 60):02d} #{cd['n']} {cd['label']} "
                                 f"({cd['m']['streamer']})" for cd in cards]
    credits = [f"#{cd['n']} {cd['m']['streamer']} ({cd['m']['platform'].title()}): {cd['m']['url']}" for cd in cards]
    log.info("[long] %s: %.0fs, %d moments, narration %s, thumbnail %s", out.name, total, len(cards),
             "yes" if voiced else "no", "yes" if thumb else "no")
    return {"path": str(out), "duration": total, "thumbnail": str(thumb) if thumb else "", "chapters": chapters,
            "credits": credits, "moments": [{"n": cd["n"], "key": cd["m"]["key"], "label": cd["label"],
                                             "streamer": cd["m"]["streamer"], "hype": cd["m"]["hype"]} for cd in cards]}


# --------------------------------------------------------------------------- pending upload (retry)

PENDING_DIR = STATE_DIR / "pending_long"  # saved with the Actions state cache until YouTube accepts it


def _keep_for_retry(rendered: dict, caps: dict, clip_id: str) -> None:
    """YouTube refused the upload (e.g. uploadLimitExceeded): keep the finished video and retry on the next runs."""
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    for f in PENDING_DIR.glob("*"):
        f.unlink(missing_ok=True)  # only the newest long video is kept
    shutil.copy(rendered["path"], PENDING_DIR / "video.mp4")
    if rendered.get("thumbnail"):
        shutil.copy(rendered["thumbnail"], PENDING_DIR / "thumb.jpg")
    save_json(PENDING_DIR / "meta.json", {"clip_id": clip_id, "caps": caps, "at": time.time(),
                                          "moments": rendered.get("moments", []), "duration": rendered["duration"]})
    log.info("[long] kept the finished video - the upload is retried on the next runs")


def retry_pending(dry_run: bool) -> dict | None:
    """Upload a long video YouTube refused earlier. Returns a history-style result, or None if nothing pending."""
    meta_f = PENDING_DIR / "meta.json"
    if dry_run or not meta_f.exists() or not (PENDING_DIR / "video.mp4").exists():
        return None
    from publisher import _available, youtube_api_upload, youtube_set_thumbnail
    if not _available("youtube_api", "youtube"):
        return None
    meta = load_json(meta_f, {})
    if time.time() - meta.get("at", 0) > 6 * 86400:  # too old to post as "today's" moments
        shutil.rmtree(PENDING_DIR, ignore_errors=True)
        return None
    log.info("[long] retrying the upload of %s", meta.get("clip_id"))
    try:
        yt = youtube_api_upload(PENDING_DIR / "video.mp4", meta["caps"])
        if (PENDING_DIR / "thumb.jpg").exists() and yt.get("id"):
            yt["thumbnail"] = youtube_set_thumbnail(yt["id"], PENDING_DIR / "thumb.jpg")
    except Exception as exc:  # noqa: BLE001
        log.warning("[long] retry still refused: %.300s", exc)
        return {"ok": False, "error": str(exc)[:300]}
    from source_fetcher import load_manifest, save_manifest
    mf = load_manifest()
    mf["items"].append({"id": meta["clip_id"], "slot": "long_video", "format_id": "long_daily", "status": "published",
                        "created_at": datetime.now(timezone.utc).isoformat(), "duration": meta.get("duration"),
                        "moments": meta.get("moments", []), "copy": {"title": meta["caps"]["youtube_title"]},
                        "results": {"youtube": {**yt, "provider": "youtube_api"}}})
    save_manifest(mf)
    shutil.rmtree(PENDING_DIR, ignore_errors=True)
    log.info("[long] ✅ pending long video published: %s", yt.get("url"))
    return {"ok": True, "id": meta["clip_id"], "results": {"youtube": {**yt, "provider": "youtube_api"}}}


# --------------------------------------------------------------------------- 5. run

def run_long(dry_run: bool) -> tuple[bool, dict]:
    from publisher import _available, youtube_api_upload, youtube_set_thumbnail
    c = _cfg()
    if not c.get("enabled", True):
        return True, {"skipped": "long videos disabled"}
    if not dry_run and not _available("youtube_api", "youtube"):
        return True, {"skipped": "YouTube uploads not configured"}
    brand = load_config().get("brand", {}).get("display_name", "GTA VI Top Moments")
    state = load_json(STATE_FILE, {"used": [], "videos": []})
    used = set(state.get("used", []))
    cands = gather(c, used)
    if len(cands) < 4:
        return False, {"error": f"only {len(cands)} usable clips today - no long video"}
    moments = review(download(cands), brand)
    min_hype = int(c.get("min_hype", 6))
    good = sorted([m for m in moments if m["hype"] >= min_hype], key=lambda m: m["hype"], reverse=True)
    target, cap = float(c.get("target_sec", 480)), int(c.get("max_moments", 15))
    chosen, total = [], 0.0
    for m in good:
        if len(chosen) >= cap or total >= target:
            break
        chosen.append(m)
        total += m["b"] - m["a"]
    if total < float(c.get("min_sec", 240)) or len(chosen) < 4:
        return False, {"error": f"not enough good moments today ({len(chosen)} moments, {total:.0f}s; "
                                f"{len(moments)} passed the AI check)"}
    # Whisper + scene changes: every moment starts at a sentence / scene change and ends after the punchline
    import transcribe
    for m in chosen:
        lo, hi = max(0.0, m["a"] - 3.0), min(m["dur"], m["b"] + 4.0)
        heard, meta = transcribe.words(m["path"], lo, hi)
        if meta.get("language") and meta["language"] != "en" and meta.get("language_probability", 0) >= 0.7 \
                and len(heard) > 5:
            log.info("[long] ✋ %s (%s): Whisper hears %s, not English - dropped", m["key"], m["streamer"], meta["language"])
            m["drop"] = True
            continue
        a, b = transcribe.smart_cut(m["path"], m["a"], m["b"], heard, offset=lo, max_len=50.0)
        if b - a >= 10:
            m["a"], m["b"] = a, b
    chosen = [m for m in chosen if not m.get("drop")]
    if len(chosen) < 4:
        return False, {"error": f"only {len(chosen)} English moments left after the language check"}
    chosen.sort(key=lambda m: (m["hype"], m["views"]))  # countdown: weakest first, #1 last
    pkg = package(chosen, brand)
    clip_id = "long_" + now_local().strftime("%Y%m%d_%H%M")
    rendered = render_long(chosen, pkg, clip_id)
    for m in moments:
        Path(m["path"]).unlink(missing_ok=True)
    if not rendered:
        return False, {"error": "long video render failed"}
    desc = "\n".join([pkg["summary"], "", "⏱️ Chapters", *rendered["chapters"], "", "🎥 Clips (all credit to the streamers):",
                      *rendered["credits"], "", "#GTA6 #GTA6Funny #GTA6Stream"])
    caps = {"youtube_title": pkg["title"], "youtube_description": desc[:4900],
            "tags": pkg["tags"] + ["GTA 6", "GTA 6 stream moments", "GTA 6 funny moments"],
            "first_comment": "Which moment was the best? Drop the number 👇"}
    # everything needed to upload it by hand (title, description with chapters + credits, tags) next to the video
    (READY_DIR / f"{clip_id}_YOUTUBE_UPLOAD.txt").write_text(
        "TITLE:\n" + caps["youtube_title"] + "\n\nDESCRIPTION:\n" + caps["youtube_description"] + "\n\nTAGS:\n"
        + ", ".join(dict.fromkeys(caps["tags"])) + "\n\nPINNED COMMENT:\n" + caps["first_comment"]
        + "\n\nTHUMBNAIL: " + Path(rendered.get("thumbnail") or "none").name + "\n", encoding="utf-8")
    results = {}
    if not dry_run:
        try:
            yt = youtube_api_upload(Path(rendered["path"]), caps)
            if rendered["thumbnail"] and yt.get("id"):
                yt["thumbnail"] = youtube_set_thumbnail(yt["id"], Path(rendered["thumbnail"]))
            results["youtube"] = {**yt, "provider": "youtube_api"}
        except Exception as exc:  # noqa: BLE001
            results["youtube"] = {"ok": False, "error": str(exc)[:300], "provider": "youtube_api"}
            log.error("[long] YouTube upload failed: %.400s", exc)
            _keep_for_retry(rendered, caps, clip_id)
    ok = dry_run or results.get("youtube", {}).get("ok", False)
    record = {"id": clip_id, "slot": "long_video", "format_id": "long_daily",
              "created_at": datetime.now(timezone.utc).isoformat(), "copy": {**pkg, "banner": pkg["title"]},
              "rendered": rendered["path"], "duration": rendered["duration"], "moments": rendered["moments"],
              "thumbnail": rendered["thumbnail"], "results": results, "dry_run": dry_run,
              "status": "published" if ok and not dry_run else ("dry_run" if dry_run else "failed"),
              "source": {"title": pkg["title"], "url": "", "credit": ", ".join(m["streamer"] for m in chosen),
                         "provider": "stream_clips", "section": [], "segments": []}}
    if ok and not dry_run:
        from source_fetcher import load_manifest, save_manifest
        mf = load_manifest()
        mf["items"].append(record)
        save_manifest(mf)
        state["used"] = (state.get("used", []) + [m["key"] for m in chosen])[-2000:]
        state["videos"] = (state.get("videos", []) + [{"id": clip_id, "at": time.time(),
                                                       "url": results.get("youtube", {}).get("url", "")}])[-200:]
        save_json(STATE_FILE, state)
    return ok, record
