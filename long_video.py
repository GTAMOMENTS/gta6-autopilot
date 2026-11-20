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
import subprocess
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


def _fetch_one(x: dict) -> tuple[dict | None, str | None]:
    """Download + check one candidate. Returns (clip, None) or (None, reason it was skipped)."""
    import stream_clips as sc
    from source_fetcher import Candidate, _download
    path = RAW_DIR / f"long_{re.sub(r'[^a-zA-Z0-9]', '_', x['key'])[:50]}.mp4"
    whole = x["platform"] in ("youtube", "dailymotion")  # guide footage: a normal video, not a stream clip
    cand = Candidate(key=x["key"], url=x["url"], title=x["title"], uploader=x["streamer"],
                     provider=x["platform"] if whole else f"{x['platform']}_clip", duration=x["duration"],
                     kind="gameplay" if whole else "stream", media_url=x.get("media_url") or None)
    a0, b0 = x.get("section") or (0, int(x["duration"]) + 1)
    if x["platform"] == "youtube":  # "GTA 6" videos uploaded before launch are fakes (GTA V / mods)
        from source_fetcher import _fill_metadata
        try:
            if not _fill_metadata(cand, load_config()):
                return None, "old / no info (before launch)"
        except Exception:  # noqa: BLE001
            return None, "download failed (youtube)"
    try:
        ok = _download(cand, a0, b0, path)
    except Exception as exc:  # noqa: BLE001
        log.info("[long] download failed %s: %.100s", x["key"], exc)
        ok = False
    if not ok:
        return None, f"download failed ({x['platform']})"
    info = probe(path)
    if min(info["width"], info["height"]) < 480 or info["duration"] < 6:
        path.unlink(missing_ok=True)
        return None, (f"low res ({x['platform']} {info['width']}x{info['height']})" if info["duration"] >= 6
                      else "too short")
    if not _has_audio(path):
        path.unlink(missing_ok=True)
        return None, "no sound"
    errs = _decode_errors(path)
    if errs > int(_cfg().get("max_decode_errors", 30)):
        log.info("[long] ✋ %s (%s, %s) is corrupted: %d decoding errors", x["key"], x["streamer"], x["platform"], errs)
        path.unlink(missing_ok=True)
        return None, "corrupted"
    rep = sc.smoothness(path)
    if rep > float(_cfg().get("max_repeated_frames", 0.2)):
        log.info("[long] ✋ %s (%s, %s) is choppy: %.0f%% repeated frames", x["key"], x["streamer"], x["platform"],
                 rep * 100)
        path.unlink(missing_ok=True)
        return None, "choppy"
    return {**x, "path": path, "dur": info["duration"], "repeat": rep}, None


def _decode_errors(path: Path) -> int:
    """Broken video data (pixel garbage / green-purple smears come from it): count the decoder's error lines."""
    res = run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"], timeout=300)
    return len([ln for ln in res.stderr.splitlines() if ln.strip()])


def _region_ok(src: Path, t: float, box: list) -> bool:
    """Is there a real picture (a face cam) in this part of the frame? A blank / one-colour area = no.
    (QA found a REACTION close-up that showed only an empty black/red area: wrong webcam coordinates.)"""
    x, y, w, h = box
    res = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(src), "-frames:v", "1",
                          "-vf", f"crop=iw*{w:.3f}:ih*{h:.3f}:iw*{x:.3f}:ih*{y:.3f},scale=48:48,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, timeout=60)
    px = res.stdout
    if res.returncode != 0 or len(px) < 48 * 48:
        return False
    mean = sum(px) / len(px)
    std = (sum((p - mean) ** 2 for p in px) / len(px)) ** 0.5
    return std >= float(_cfg().get("facecam_min_contrast", 18)) and 15 <= mean <= 240


def download(cands: list[dict]) -> list[dict]:
    """Downloads + checks the candidates a few at a time (one by one took ~18 min for 40 clips).
    The result keeps the candidates' order (most viewed first)."""
    from concurrent.futures import ThreadPoolExecutor
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(_cfg().get("download_workers", 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_fetch_one, cands))
    out, why = [], {}
    for clip, reason in results:
        if clip:
            out.append(clip)
        elif reason:
            why[reason] = why.get(reason, 0) + 1
    log.info("[long] downloaded %d/%d clips (skipped: %s)", len(out), len(cands), why)
    return out


# --------------------------------------------------------------------------- 2. AI review (batches)

REVIEW_PROMPT = """You are the editor of "{brand}", a GTA 6 channel. GAME FACTS (newer than your training data - trust
them): {game}
The attached video is {k} short GTA 6 livestream clips joined one after another. Clip time ranges in this video:
{ranges}
For EACH clip judge honestly and write:
- is_gta6_footage: true only if real Grand Theft Auto VI GAMEPLAY is on screen for most of the clip. A streamer
  only talking / eating / reacting at the desk with no game visible = false.
- broken: true if the picture is corrupted (pixelated blocks, green / purple smears, glitching image), mostly
  frozen or black.
- is_spoiler: true for story spoilers (story cutscenes, plot reveals, endings, character deaths).
- is_english: true if the streamer speaks English (or nobody talks); false for any other language.
- unsafe: true ONLY for nudity, slurs / hate speech, real-world violence, doxxing - things YouTube removes.
  In-game violence, crime, police chases and the streamer swearing are normal GTA content and are NOT unsafe.
- hype_score 1-10: how funny / wild / shareable the moment and the streamer's reaction are (strict).
- label: max 6 words, Title Case, what happens ("Kai Cenat Vs The Yacht"). No spoilers of the punchline.
- line: what our narrator says right before it plays: 8-20 words, casual YouTuber voice, names the streamer,
  builds curiosity without giving away the punchline ("Kai thought a jet ski could outrun a police helicopter.
  It could not."). NO numbers or rankings in the line (the countdown number is added later). Say "GTA 6".
- trim_start / trim_end: seconds WITHIN that clip (0 = start of the clip) of the best part, 10-45 seconds.
- best_t: second within the clip with the most eye-catching frame / the punchline (for the replay + thumbnail).
- facecam: [x, y, w, h] as fractions of the frame (0-1) where the streamer's webcam is, or null if there is none.
Return ONLY JSON: {{"clips": [{{"clip": number, "is_gta6_footage": bool, "broken": bool, "is_spoiler": bool,
 "unsafe": bool, "is_english": bool, "hype_score": int, "label": str, "line": str, "trim_start": number, "trim_end": number,
 "best_t": number, "facecam": [number, number, number, number] | null}}]}}"""


def _bbox(v) -> list | None:
    """AI facecam box [x, y, w, h] (fractions) -> validated, or None."""
    try:
        x, y, w, h = (float(t) for t in v)
    except (TypeError, ValueError):
        return None
    if max(x, y, w, h) > 1.5:  # pixels or percent -> assume percent
        x, y, w, h = (t / 100 for t in (x, y, w, h))
    if not (0 <= x < 1 and 0 <= y < 1 and 0.08 <= w <= 0.6 and 0.08 <= h <= 0.6 and x + w <= 1.02 and y + h <= 1.02):
        return None
    return [round(x, 3), round(y, 3), round(min(w, 1 - x), 3), round(min(h, 1 - y), 3)]


_NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
              "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def _strip_rank(line: str) -> str:
    """Remove any 'Number seven,' / '#7:' / 'At number 3' the AI put in - the real rank is added at render time."""
    pat = r"^\s*(at\s+)?(number|no\.?|#)\s*(\d+|" + "|".join(_NUM_WORDS) + r")\b[\s,.:;!-]*"
    return re.sub(pat, "", line, flags=re.I).strip()


def _ranked(n: int, line: str) -> str:
    """The narrator says the SAME number the title card shows."""
    word = _NUM_WORDS[n] if n < len(_NUM_WORDS) else str(n)
    body = _strip_rank(line)
    body = body[:1].upper() + body[1:]
    return f"Number {word}. {body}" if body else ""


def _proxy(path: Path, out: Path) -> bool:
    res = run(["ffmpeg", "-y", "-i", str(path), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
               "-filter_complex", "[0:v]scale=-2:480,fps=12,setsar=1,format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2[v]",
               "-map", "[v]", "-map", "0:a?", "-map", "1:a", "-shortest", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "32", "-c:a", "aac", "-ar", "48000", "-ac", "1", "-b:a", "48k", str(out)], timeout=180)
    return res.returncode == 0 and out.exists()


def _game_facts() -> str:
    import gemini_writer
    return gemini_writer._game_context()


def _batch_verdicts(moments: list[dict], prompt_for):
    """The AI watches the clips in batches (one upload per batch saves quota). prompt_for(k, ranges) -> prompt.
    Yields (batch, [(moment, verdict or None)])."""
    import gemini_writer
    LONG_DIR.mkdir(parents=True, exist_ok=True)
    size = int(_cfg().get("review_batch", 6))
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
                    data = gemini_writer._generate_json([up, prompt_for(len(parts), "\n".join(ranges))],
                                                        temperature=0.6, prefer_pro=True)
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
        for _, _, p, _, _ in parts:
            p.unlink(missing_ok=True)
        reel.unlink(missing_ok=True)
        yield batch, [(m, verdicts.get(i)) for i, m, _, _, _ in parts]


def review(moments: list[dict], brand: str) -> list[dict]:
    if not env("GEMINI_API_KEY"):
        log.warning("[long] GEMINI_API_KEY not set - can't review clips")
        return []
    reviewed = []
    for bi, (batch, pairs) in enumerate(_batch_verdicts(
            moments, lambda k, ranges: REVIEW_PROMPT.format(brand=brand, k=k, ranges=ranges,
                                                            game=_game_facts())), 1):
        for m, v in pairs:
            if not v:
                continue
            reason = ("not GTA 6" if not v.get("is_gta6_footage") else "broken picture" if v.get("broken")
                      else "spoiler" if v.get("is_spoiler")
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
            cam = _bbox(v.get("facecam"))
            if cam and not _region_ok(m["path"], float(v.get("best_t") or (a + e) / 2), cam):
                log.info("[long] %s: the AI's webcam position shows no picture - no facecam for this clip", m["key"])
                cam = None
            reviewed.append({**m, "hype": int(v.get("hype_score") or 0), "label": str(v.get("label") or m["title"])[:60],
                             "line": str(v.get("line") or "").strip(), "a": a, "b": min(e, a + 45),
                             "best_t": float(v.get("best_t") or (a + e) / 2), "facecam": cam})
        log.info("[long] batch %d: %d/%d clips passed", bi,
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
    title = re.sub(r"\b(Top|TOP|top) \d+\b", lambda mm: f"{mm.group(1)} {n}", title)  # the number must be true
    return {
        "title": re.sub(r"[<>]", "", title)[:95],
        "intro_line": str(data.get("intro_line") or f"These are the {n} craziest GTA 6 moments from today's streams, "
                                                     "and number one is unreal."),
        "outro_line": str(data.get("outro_line") or "Which one was the best? Subscribe for daily GTA 6 moments."),
        "thumbnail_text": re.sub(r"[^A-Z0-9 !?'.]", "", str(data.get("thumbnail_text") or "GTA 6 CHAOS").upper())[:28],
        "summary": str(data.get("summary") or f"The {n} craziest GTA 6 livestream moments from today."),
        "tags": [str(t)[:30] for t in (data.get("tags") or [])][:12] or ["GTA 6", "GTA 6 funny moments"],
    }


# --------------------------------------------------------------------------- 3b. guide videos (money / trend)

GUIDE_BAD = re.compile(r"glitch|exploit|unlimited|infinite|mod menu|modded|\bmods?\b|cheat|leak|hack|trainer|"
                       r"duplicat|rumou?r|fake|gta 5|gta v\b|gta online", re.I)
_STOP = {"the", "a", "an", "in", "gta", "6", "to", "for", "of", "and", "your", "best", "way", "ways", "money", "make"}


def guide_spec(kind: str) -> dict | None:
    """What a guide video is about. money_guide: the best ways to make money; trend_guide: today's hottest
    trend topic (trends.py)."""
    c = _cfg()
    if kind == "money_guide":
        return {"kind": kind, "topic": "", "subject": "the best ways to make money in GTA 6 (no story spoilers)",
                "noun": "way to make money", "short": "money tips",
                "example": "Top 6 BEST Ways To Make Money (No Spoilers)",
                "queries": c.get("money_queries", ["GTA 6 how to make money", "GTA 6 money making guide",
                                                   "GTA 6 money making no commentary", "GTA 6 fast money",
                                                   "GTA 6 best money method", "GTA 6 make money early"]),
                "proof": ("the money actually being EARNED on screen: the HUD cash counter going up, a payout / "
                          "sale / reward screen, or picking up cash (stealing or wrecking things without getting "
                          "paid on screen does not count)"),
                "hashtags": "#GTA6 #GTA6Money #GTA6Tips",
                "outro": "WHICH ONE WILL YOU TRY?\\N\\NSUBSCRIBE FOR MORE GTA 6 MONEY TIPS",
                "comment": "Which money method are you trying first? 👇"}
    if kind == "trend_guide":
        import trends
        t = trends.pick(money=None, min_demand=int(c.get("trend_min_demand", 5)))
        if not t:
            return None
        return {"kind": kind, "topic": t["topic"], "subject": f"GTA 6 - {t['topic']}: {t['angle']}",
                "noun": f"example of '{t['topic']}'", "short": "videos",
                "example": f"Top 7 {t['topic']} (No Spoilers)",
                "queries": list(dict.fromkeys(t["queries"] + [f"GTA 6 {t['topic']}"])),
                "hashtags": "#GTA6 #GTA6Tips #GTA6Gameplay",
                "outro": "WHICH ONE IS YOUR FAVOURITE?\\N\\NSUBSCRIBE FOR DAILY GTA 6 VIDEOS",
                "comment": "Which one surprised you the most? 👇"}
    return None


def gather_guide(spec: dict, used: set) -> list[dict]:
    """GTA 6 gameplay videos that show the subject (YouTube + Dailymotion), one ~110 s section each."""
    from source_fetcher import TitleFilter, blocked_channel, discover_dailymotion, discover_youtube
    cfg, c = load_config(), _cfg()
    tf = TitleFilter(cfg)
    found = []
    for name, fn in (("youtube", lambda: discover_youtube(cfg, spec["queries"], tf, "gameplay")),
                     ("dailymotion", lambda: discover_dailymotion(cfg, spec["queries"][:3], tf))):
        try:
            found += fn()
        except Exception as exc:  # noqa: BLE001
            log.warning("[guide] %s search failed: %.150s", name, exc)
    pool, per = _guide_stream_clips(c, used), {}
    # footage without someone talking over it keeps its game sound (muted voice-overs feel empty) -> first
    for x in sorted(found, key=lambda x: ("no commentary" in (x.title or "").lower(), x.views), reverse=True):
        dur = float(x.duration or 0)
        ch = (x.uploader or "unknown").strip()
        if x.key in used or GUIDE_BAD.search(x.title or "") or blocked_channel(ch, cfg) or not 30 <= dur <= 5400:
            continue
        if per.get(ch.lower(), 0) >= int(c.get("guide_max_per_channel", 2)):
            continue
        per[ch.lower()] = per.get(ch.lower(), 0) + 1
        if dur <= 150:
            sec = (0, int(dur) + 1)
        else:  # skip the intro, take the part where the method is shown
            st = int(min(max(20.0, dur * 0.12), dur - 120))
            sec = (st, st + int(c.get("guide_section_sec", 110)))
        pool.append({"key": x.key, "url": x.url, "title": x.title, "streamer": ch, "platform": x.provider,
                     "duration": dur, "section": sec, "views": int(x.views or 0), "created": 0})
        if len(pool) >= int(c.get("guide_candidates", 24)) + int(c.get("guide_stream_clips", 18)):
            break
    log.info("[guide] %d videos found, %d candidates: %s", len(found), len(pool),
             [f"{x['streamer']}: {x['title'][:40]} ({x['views']}v)" for x in pool[:8]])
    return pool


_MONEYISH = re.compile(r"money|cash|\$|rich|million|bank|rob|heist|sell|sold|stash|safe|loot|paid|pay|job|hustle|"
                       r"business|buy|bought|price|dollar|grand\b|\d+k\b", re.I)


def _guide_stream_clips(c: dict, used: set) -> list[dict]:
    """GTA VI-category clips from famous English streamers (Twitch / Kick): guaranteed real GTA 6 gameplay
    (most "GTA 6" YouTube uploads are still GTA V / mods). Clips whose title sounds like money come first."""
    import stream_clips as sc
    n = int(c.get("guide_stream_clips", 18))
    if n <= 0:
        return []
    scfg = {**load_config().get("stream_clips", {}), "window_hours": float(c.get("window_hours", 24)) * 2,
            "top_streams": int(c.get("top_streams", 15))}
    clips = []
    for fn in (sc.twitch_clips, sc.kick_clips):
        try:
            clips += fn(scfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("[guide] %s failed: %.150s", fn.__name__, exc)
    langs = set(scfg.get("languages", ["en"]))
    pool = [x for x in clips if x["key"] not in used and sc.is_famous(x["streamer"], scfg)
            and 15 <= x["duration"] <= 180 and not (x["platform"] == "twitch" and langs and x["language"]
                                                     and x["language"] not in langs)]
    pool.sort(key=lambda x: (bool(_MONEYISH.search(x["title"] or "")), x["views"]), reverse=True)
    out, per = [], {}
    for x in pool:
        s = x["streamer"].lower()
        if per.get(s, 0) >= 3:
            continue
        per[s] = per.get(s, 0) + 1
        out.append({**x, "trusted": True})
        if len(out) >= n:
            break
    log.info("[guide] %d GTA 6 stream clips (trusted, %d sound like money): %s", len(out),
             sum(1 for x in out if _MONEYISH.search(x["title"] or "")), [f"{x['streamer']}: {x['title'][:30]}" for x in out[:6]])
    return out


GUIDE_REVIEW_PROMPT = """You are the researcher and editor of "{brand}", a GTA 6 channel. GAME FACTS (newer than your
training data - trust them): {game}
We are making a long YouTube countdown about {subject}. The attached video is {k} GTA 6 gameplay clips from other creators joined one after another:
{ranges}
For EACH clip judge honestly:
- is_gta6_footage: true only if real Grand Theft Auto VI GAMEPLAY is on screen for most of the clip (not GTA 5 /
  Online, not a mod, not a trailer, not just a person talking to the camera)
- broken: true if the picture is corrupted (pixelated blocks, green / purple smears), mostly frozen or black
- is_spoiler: true if it shows or talks about story missions, story heists, plot, endings or character deaths
- unsafe: true ONLY for nudity, slurs / hate speech, real-world violence
- is_glitch: true for glitches, exploits, cheats or mods (they get patched - we never show them)
- shows_item: true only if the clip clearly SHOWS one {noun} that you can explain from what is seen and heard
- item: that {noun} in 2-6 words, Title Case ("Selling Stolen Cars")
- facts: what the clip shows or its creator SAYS about it (where, how, what you get). Amounts ONLY if they are
  visible on screen (HUD, menus, prices) or said out loud - otherwise no numbers at all.
- lines: 3-5 sentences for OUR narrator explaining it to the viewer in "you" form, 8-22 words each, casual
  YouTuber voice, ONLY facts from this clip, no numbers that are not in facts, no rankings ("number three") - the
  countdown number is added later. Say "GTA 6".
- usefulness: 1-10, how good this {noun} is for viewers (pays more / faster / easier = higher). Be strict.
- trim_start / trim_end: seconds WITHIN that clip of the part that shows it best (20-45 seconds)
- best_t: second within the clip with the most eye-catching frame (for the thumbnail)
- has_narrator: true if someone talks over the clip (we mute it and use our own narrator)
{proof_rule}
Return ONLY JSON: {{"clips": [{{"clip": number, "is_gta6_footage": bool, "broken": bool, "is_spoiler": bool,
 "unsafe": bool, "is_glitch": bool, "shows_item": bool, "item": str, "facts": str, "lines": [str], "usefulness": int,
 "trim_start": number, "trim_end": number, "best_t": number, "has_narrator": bool, "proof_t": number | null}}]}}"""


def _nums(text: str) -> set:
    return set(re.findall(r"\d[\d,.]*\d|\d", text or ""))


def _item_key(label: str) -> str:
    return " ".join(w for w in re.sub(r"[^a-z0-9 ]", "", label.lower()).split() if w not in _STOP)


def review_guide(clips: list[dict], spec: dict, brand: str) -> list[dict]:
    """AI research per clip -> items (deduplicated: the same method from several creators counts as confirmed)."""
    if not env("GEMINI_API_KEY"):
        log.warning("[guide] GEMINI_API_KEY not set - can't review clips")
        return []
    import difflib
    out = []
    for bi, (batch, pairs) in enumerate(_batch_verdicts(clips, lambda k, ranges: GUIDE_REVIEW_PROMPT.format(
            brand=brand, subject=spec["subject"], noun=spec["noun"], k=k, ranges=ranges, game=_game_facts(),
            proof_rule=(f"- proof_t: the second within the clip where you SEE {spec['proof']}; null if the clip "
                        "never shows it (then shows_item must be false)" if spec.get("proof")
                        else "- proof_t: null"))), 1):
        passed = 0
        for m, v in pairs:
            if not v:
                continue
            facts = str(v.get("facts") or "")[:400]
            lines = []
            for ln in v.get("lines") or []:
                ln = _strip_rank(re.sub(r"\s+", " ", str(ln)).strip())
                if ln and not (_nums(ln) - _nums(facts) - {"6"}):  # a number the clip never showed = made up
                    lines.append(ln)
            item = re.sub(r"\s+", " ", str(v.get("item") or "")).strip(" .")[:50]
            proof_t = v.get("proof_t") if isinstance(v.get("proof_t"), (int, float)) else None
            if spec.get("proof") and proof_t is None:
                v["shows_item"] = False  # no money visibly earned = not a money method
            reason = ("not GTA 6" if not v.get("is_gta6_footage") else "broken picture" if v.get("broken")
                      else "spoiler" if v.get("is_spoiler")
                      else "unsafe" if v.get("unsafe") else "glitch / exploit" if v.get("is_glitch")
                      else ("no money earned on screen" if spec.get("proof") and proof_t is None else "doesn't show one")
                      if not v.get("shows_item") else "no explanation"
                      if not item or len(lines) < 2 else None)
            if reason:
                log.info("[guide] ✋ %s (%s): %s", m["key"], m["streamer"], reason)
                continue
            try:
                a = max(0.0, float(v.get("trim_start") or 0))
                e = min(m["dur"], float(v.get("trim_end") or m["dur"]))
            except (TypeError, ValueError):
                a, e = 0.0, m["dur"]
            if e - a < 15:  # the explanation needs room: take more footage around the AI's pick
                a, e = max(0.0, min(a, m["dur"] - 40)), min(m["dur"], max(a, 0) + 40)
            if e - a < 12:
                continue
            if proof_t is not None and 0 <= proof_t <= m["dur"] and not a <= proof_t <= e:
                length = e - a  # the proof must be in the video: move the part over it
                a = max(0.0, min(proof_t - length * 0.6, m["dur"] - length))
                e = min(m["dur"], a + length)
            try:
                useful = max(1, min(10, int(v.get("usefulness") or 0)))
            except (TypeError, ValueError):
                useful = 1
            out.append({**m, "label": item.title() if item.islower() else item, "lines": lines[:5], "facts": facts,
                        "hype": useful, "a": a, "b": min(e, a + 45), "best_t": float(v.get("best_t") or (a + e) / 2),
                        "has_narrator": bool(v.get("has_narrator", True)), "facecam": None, "sources": 1})
            passed += 1
        log.info("[guide] batch %d: %d/%d clips usable", bi, passed, len(batch))
    items: list[dict] = []
    for r in sorted(out, key=lambda r: r["hype"], reverse=True):
        k = _item_key(r["label"])
        twin = next((x for x in items if difflib.SequenceMatcher(None, _item_key(x["label"]), k).ratio() > 0.72), None)
        if twin:
            twin["sources"] += 1  # several creators show the same method -> more trustworthy
        else:
            items.append(r)
    log.info("[guide] %d different items: %s", len(items),
             [f"{x['label']} ({x['hype']}, {x['sources']} src)" for x in items])
    return items


GUIDE_PACKAGE_PROMPT = """You write for "{brand}", a GTA 6 YouTube channel. Today's video is a countdown guide:
{subject}.{part}
Items (from #{n} up to #1, the best one last):
{items}
Write:
- title: max 70 characters, search-friendly, "GTA 6" near the start, promises exactly what the video shows, e.g.
  "GTA 6: {example}". No hashtags.
- intro_line: what the narrator says over the cold open, 10-20 words, makes people stay for number one.
- outro_line: 8-15 words, asks which one they will try and to subscribe for more GTA 6 {short}.
- thumbnail_text: 2-4 words, ALL CAPS, huge thumbnail text that makes people click (e.g. "EASY MONEY").
- summary: 1-2 sentences for the description.
- tags: 10 YouTube search tags (strings, no #).
Say "GTA 6", never "GTA VI". No story spoilers. Never invent numbers or amounts. Return ONLY JSON with exactly these keys."""


def package_guide(items: list[dict], spec: dict, brand: str, part: int = 1) -> dict:
    import ai_fallback
    import gemini_writer
    n = len(items)
    listing = "\n".join(f"#{n - i}: {m['label']} - {m['facts'][:120]}" for i, m in enumerate(items))
    prompt = GUIDE_PACKAGE_PROMPT.format(brand=brand, subject=spec["subject"], n=n, items=listing,
                                         example=spec["example"].replace("6", str(n), 1) if spec["example"].startswith("Top 6") else spec["example"],
                                         short=spec["short"], part=f" This is part {part} of the series." if part > 1 else "")
    data = gemini_writer._generate_json(prompt, temperature=0.7, prefer_pro=True)
    if not data and ai_fallback.available():
        data = ai_fallback.chat_json(prompt)
    data = data or {}
    if data:
        import critic
        fields = {k: str(data[k]) for k in ("title", "thumbnail_text", "intro_line") if data.get(k)}
        data.update(critic.polish(fields, {"title": 90, "thumbnail_text": 28, "intro_line": 160}, listing))
    if spec["kind"] == "money_guide":
        fallback_title = f"GTA 6: Top {n} BEST Ways To Make Money (No Spoilers)"
        fallback_thumb = "EASY MONEY"
    else:
        fallback_title = f"GTA 6: Top {n} {spec['topic']} (No Spoilers)"
        fallback_thumb = " ".join(spec["topic"].upper().split()[:3])
    title = re.sub(r"\b(Top|TOP|top) \d+\b", lambda mm: f"{mm.group(1)} {n}", str(data.get("title") or fallback_title))
    if "gta 6" not in title.lower():
        title = f"GTA 6: {title}"
    return {
        "title": re.sub(r"[<>]", "", title)[:95],
        "intro_line": str(data.get("intro_line") or f"These are the {n} best ones in GTA 6 right now, and number one "
                                                     "is the one you need."),
        "outro_line": str(data.get("outro_line") or f"Which one are you trying first? Subscribe for more GTA 6 {spec['short']}."),
        "thumbnail_text": re.sub(r"[^A-Z0-9 !?'.$]", "", str(data.get("thumbnail_text") or fallback_thumb).upper())[:28],
        "summary": str(data.get("summary") or f"{spec['subject'][0].upper()}{spec['subject'][1:]} - {n} picks, best last."),
        "tags": [str(t)[:30] for t in (data.get("tags") or [])][:12] or ["GTA 6", "GTA 6 tips"],
    }


def make_guide(kind: str, brand: str, used: set, state: dict) -> dict:
    """Guide long video: {chosen, pkg, rendered, spec} or {error}."""
    c = _cfg()
    spec = guide_spec(kind)
    if not spec:
        return {"error": f"no topic for {kind} (no trend data yet)"}
    log.info("[guide] long video: %s", spec["subject"])
    clips = download(gather_guide(spec, used))
    import authentic
    clips = authentic.keep_real(clips, "[guide]")  # GTA V / mod footage uploaded as "GTA 6" is dropped here
    if len(clips) < 4:
        for m in clips:
            Path(m["path"]).unlink(missing_ok=True)
        return {"error": f"only {len(clips)} usable gameplay videos for {kind}"}
    items = review_guide(clips, spec, brand)
    prev = {_item_key(x) for x in state.get("guide_items", {}).get(kind, [])}
    for it in items:  # confirmed by several creators = better; already covered in an earlier guide = later
        it["rank"] = (it["hype"] + 0.5 * min(it["sources"] - 1, 2) - (2.5 if _item_key(it["label"]) in prev else 0)
                      - (0.3 if it.get("has_narrator", True) else 0))
    good = sorted([it for it in items if it["hype"] >= int(c.get("guide_min_usefulness", 5))],
                  key=lambda it: it["rank"], reverse=True)[: int(c.get("guide_max_items", 7))]
    if len(good) < int(c.get("guide_min_items", 4)):
        for m in clips:
            Path(m["path"]).unlink(missing_ok=True)
        return {"error": f"only {len(good)} good items for {kind} ({len(items)} found)"}
    good.sort(key=lambda it: it["rank"])  # countdown: weakest first, #1 last
    part = int(state.get("guide_count", {}).get(kind, 0)) + 1
    pkg = package_guide(good, spec, brand, part)
    clip_id = f"long_{kind}_" + now_local().strftime("%Y%m%d_%H%M")
    rendered = render_long(good, pkg, clip_id, mode="guide", outro_text=spec["outro"])
    for m in clips:
        Path(m["path"]).unlink(missing_ok=True)
    if not rendered:
        return {"error": f"{kind} render failed"}
    return {"chosen": good, "pkg": pkg, "rendered": rendered, "spec": spec, "clip_id": clip_id}


# --------------------------------------------------------------------------- 4. render (16:9)

FIT = ("split=2[a][b];[a]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,boxblur=20:2,"
       "eq=brightness=-0.15[bg];[b]scale=1920:1080:force_original_aspect_ratio=decrease[fg];"
       "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30,eq=contrast=1.06:saturation=1.12,"
       "unsharp=5:5:0.45:5:5:0.0,format=yuv420p")
# (5) audio polish: rumble cut, light noise reduction, compression, even loudness, limiter (no distortion)
AUDIO_POLISH = ("highpass=f=70,afftdn=nr=8:nf=-42,acompressor=threshold=-20dB:ratio=3:attack=5:release=120,"
                "loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,alimiter=limit=0.95")
ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def _has_audio(path: Path) -> bool:
    res = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
               str(path)], timeout=30)
    return bool(res.stdout.strip())


def _segment(src: Path, a: float, b: float, out: Path, silent_ok: bool = False, keep: list | None = None,
             speed: float = 1.0, crop: list | None = None, mute: bool = False) -> bool:
    """One piece of the video: 1920x1080/30fps, polished audio. Audio and video are cut to EXACTLY the same
    length (whole frames) - otherwise tiny differences add up over 15 clips and the sound drifts out of sync.
    keep: [(x, y)] source ranges to keep (jump cuts); speed < 1 = slow motion; crop = [x, y, w, h] fractions."""
    pre_v, pre_a = "", ""
    if keep:  # jump cuts: drop the quiet / dead parts, timestamps rebuilt so sound stays in sync
        expr = "+".join(f"between(t,{x - a:.3f},{y - a:.3f})" for x, y in keep)
        pre_v = f"select='{expr}',setpts=N/(30*TB),"
        pre_a = f"aselect='{expr}',asetpts=N/SR/TB,"
        d = round(sum(y - x for x, y in keep) * 30) / 30
    else:
        d = round((b - a) * 30) / 30
    slow_v, slow_a = "", ""
    if speed != 1.0:
        slow_v, slow_a = f",setpts=PTS/{speed}", f"atempo={speed},"
        d = round(d / speed * 30) / 30
    crop_v = (f"crop=iw*{crop[2]:.3f}:ih*{crop[3]:.3f}:iw*{crop[0]:.3f}:ih*{crop[1]:.3f},"
              "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,") if crop else ""
    vchain = f"[0:v]fps=30,{pre_v}{crop_v}{FIT}{slow_v},trim=duration={d:.4f},setpts=PTS-STARTPTS[v]"
    if mute:  # guide footage: the other creator's voice-over is removed, our narrator explains instead
        achain = f"anullsrc=r=48000:cl=stereo,atrim=duration={d:.4f}[a]"
    elif _has_audio(src):
        achain = (f"[0:a]aresample=48000,aformat=channel_layouts=stereo,{pre_a}{slow_a}{AUDIO_POLISH},"
                  f"apad,atrim=duration={d:.4f},asetpts=PTS-STARTPTS[a]")
    elif silent_ok:
        achain = f"anullsrc=r=48000:cl=stereo,atrim=duration={d:.4f}[a]"
    else:
        log.info("[long] %s has no sound - not used", src.name)
        return False
    res = run(["ffmpeg", "-y", "-ss", f"{a:.3f}", "-t", f"{(b - a) + 0.5:.3f}", "-i", str(src), "-filter_complex",
               f"{vchain};{achain}", "-map", "[v]", "-map", "[a]", *ENC, str(out)], timeout=600)
    if res.returncode != 0 or not out.exists() or probe(out)["duration"] < 1:
        log.warning("[long] could not prepare %s: %.300s", src.name, res.stderr[-300:])
        return False
    return True


def _jump_keep(src: Path, a: float, b: float, protect: float, freeze_only: bool = False) -> list | None:
    """Ranges to keep inside [a, b]: quiet stretches (no talking, no action sound) over 1 s are cut out,
    except around the punchline; frozen picture always. freeze_only: only the frozen parts (guides).
    None = nothing worth cutting."""
    if not _cfg().get("jump_cuts", True) and not freeze_only:
        return None
    quiet = []
    res = run(["ffmpeg", "-hide_banner", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(src), "-vn", "-af",
               "silencedetect=noise=-35dB:d=1.0", "-f", "null", "-"], timeout=120) if not freeze_only else None
    for st, en in zip(re.findall(r"silence_start: ([\d.]+)", res.stderr), re.findall(r"silence_end: ([\d.]+)", res.stderr)) \
            if res else []:
        x, y = a + float(st) + 0.3, a + float(en) - 0.3  # keep a little air around each cut
        if y - x >= 0.7 and not (x - 1.5 <= protect <= y + 1.5):
            quiet.append((x, y))
    # frozen picture in the stream itself (lag) - always cut out
    fr = run(["ffmpeg", "-hide_banner", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i", str(src), "-an", "-vf",
              "scale=320:-2,freezedetect=n=0.003:d=0.8", "-f", "null", "-"], timeout=180)
    fs = re.findall(r"freeze_start: ([\d.]+)", fr.stderr)
    fe = re.findall(r"freeze_end: ([\d.]+)", fr.stderr) + [str(b - a)] * 3
    for st, en in zip(fs, fe):
        quiet.append((a + float(st) - 0.05, a + float(en) + 0.05))
    quiet = sorted(quiet)
    merged = []
    for x, y in quiet:  # overlapping cuts -> one
        if merged and x <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], y))
        else:
            merged.append((max(a, x), min(b, y)))
    quiet = merged
    if not quiet:
        return None
    keep, cur = [], a
    for x, y in quiet:
        if x > cur:
            keep.append((cur, x))
        cur = y
    if cur < b:
        keep.append((cur, b))
    kept = sum(y - x for x, y in keep)
    if kept < 10 or len(keep) > 8:
        return None
    log.info("[cut] %s: %.1fs -> %.1fs (%d part(s) removed)", "frozen parts" if freeze_only else "jump cuts",
             b - a, kept, len(quiet))
    return keep


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


def _trim(seg: Path, d: float) -> bool:
    """Shorten a prepared piece to exactly d seconds (audio + video, whole frames)."""
    d = round(d * 30) / 30
    tmp = seg.with_name(seg.stem + "_t.mp4")
    res = run(["ffmpeg", "-y", "-i", str(seg), "-filter_complex",
               f"[0:v]trim=duration={d:.4f},setpts=PTS-STARTPTS[v];[0:a]atrim=duration={d:.4f},asetpts=PTS-STARTPTS[a]",
               "-map", "[v]", "-map", "[a]", *ENC, str(tmp)], timeout=300)
    if res.returncode == 0 and tmp.exists():
        tmp.replace(seg)
        return True
    tmp.unlink(missing_ok=True)
    return False


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


def _build_ass(path: Path, title: str, cards: list[dict], outro_at: float, total: float, handle: str,
               inserts: list | None = None, brand: str = "",
               outro_text: str = "WHICH ONE WAS THE BEST?\\N\\NSUBSCRIBE FOR DAILY GTA 6 MOMENTS") -> None:
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
Style: Brand,{font},110,&H0000F0FF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,4,0,1,10,5,5,80,80,0,1
Style: Tag,{font},70,&H00FFFFFF,&H00FFFFFF,&H000000C0,&H000000C0,-1,0,0,0,100,100,3,0,3,18,0,9,70,70,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    import textwrap
    short = re.sub(r"\s*[\(\[].*?[\)\]]", "", title).strip().upper()  # drop "(IShowSpeed, Kai...)"
    wrapped = "\\N".join(textwrap.wrap(_ass_text(short), 26)[:3])
    ev = []
    if brand:  # 1.4 s branded sting before the title
        ev.append(f"Dialogue: 4,{_t(0)},{_t(1.4)},Brand,,0,0,0,,{{\\fad(80,250)\\fscx60\\fscy60"
                  f"\\t(0,220,\\fscx105\\fscy105)\\t(220,380,\\fscx100\\fscy100)}}{_ass_text(brand.upper())}")
    ev.append(f"Dialogue: 2,{_t(1.2 if brand else 0)},{_t(cards[0]['start'] if cards else 4)},Title,,0,0,0,,"
              f"{{\\fad(200,200)}}{wrapped}")
    for ins in inserts or []:
        ev.append(f"Dialogue: 4,{_t(ins['start'] + 0.1)},{_t(ins['end'] - 0.1)},Tag,,0,0,0,,{{\\fad(100,150)}}"
                  f"{_ass_text(ins['label'])}")
    for cd in cards:
        a, b = cd["start"] + 0.2, cd["start"] + 4.5
        pop = "{\\fad(150,250)\\fscx70\\fscy70\\t(0,180,\\fscx100\\fscy100)}"
        ev.append(f"Dialogue: 3,{_t(a)},{_t(b)},Num,,0,0,0,,{pop}#{cd['n']}")
        ev.append(f"Dialogue: 2,{_t(a + 0.15)},{_t(b)},Label,,0,0,0,,{{\\fad(150,250)}}{_ass_text(cd['label'].upper())}")
        ev.append(f"Dialogue: 2,{_t(a + 0.3)},{_t(b)},Who,,0,0,0,,{{\\fad(150,250)}}{_ass_text(cd['who'])}")
    ev.append(f"Dialogue: 2,{_t(outro_at + 0.3)},{_t(total)},Outro,,0,0,0,,{{\\fad(250,0)}}{outro_text}")
    if handle:
        ev.append(f"Dialogue: 1,{_t(0)},{_t(total)},Handle,,0,0,0,,{_ass_text(handle)}")
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def _font_file() -> str:
    for q in ("Montserrat:style=Black", "Montserrat:style=ExtraBold", "Montserrat:bold", "DejaVu Sans:bold"):
        res = run(["fc-match", "-f", "%{file}", q], timeout=10)
        if res.returncode == 0 and res.stdout.strip() and ("ontserrat" in res.stdout or "DejaVu" in q):
            return res.stdout.strip()
    return "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def thumbnail(src: Path, t: float, text: str, out: Path, cam: list | None = None) -> Path | None:
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
    if cam:  # the streamer's reaction, big, top-right with a white frame (classic pro thumbnail)
        cx, cy, cw, ch = cam
        vf = (f"split=2[m][c];[c]crop=iw*{cw}:ih*{ch}:iw*{cx}:ih*{cy},scale=-2:380,pad=iw+16:ih+16:8:8:white[cam];"
              f"[m]{vf}[base];[base][cam]overlay=W-w-30:30")
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


def render_long(moments: list[dict], pkg: dict, clip_id: str, mode: str = "moments",
                outro_text: str = "") -> dict | None:
    """mode "moments": stream-moment countdown (streamer audio, one line per moment, replays).
    mode "guide": countdown guide (money methods / a trend topic): the narrator explains each item over its
    footage (several lines), other creators' voice-overs muted, no jump cuts / replays."""
    from video_processor import HumanVoiceUnavailable, _sfx, synthesize_lines
    LONG_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)
    c = _cfg()
    guide = mode == "guide"
    top = moments[-1]  # #1 (moments are in countdown order: worst first)
    # 1) prepare every moment (jump cuts) + the punchline inserts first, so the final countdown numbers are known
    prepared = []
    n_all = len(moments)
    for i, m in enumerate(moments):
        out = LONG_DIR / f"seg_{i:02d}.mp4"
        keep = _jump_keep(m["path"], m["a"], m["b"], m["best_t"], freeze_only=guide)
        if not _segment(m["path"], m["a"], m["b"], out, keep=keep, silent_ok=guide,
                        mute=guide and m.get("has_narrator", True)):
            log.warning("[long] could not prepare %s - dropped", m["key"])
            continue
        ins, label = None, ""
        if not guide and i >= n_all - int(c.get("replays", 3)) and c.get("punchline_inserts", True):
            ins = LONG_DIR / f"ins_{i:02d}.mp4"
            bt = min(max(m["best_t"], 1.2), m["dur"] - 1.6)
            if m.get("facecam") and _region_ok(m["path"], bt, m["facecam"]) and \
                    _segment(m["path"], bt - 0.6, bt + 1.8, ins, crop=m["facecam"]):
                label = "REACTION"
            elif _segment(m["path"], bt - 1.0, bt + 0.9, ins, speed=0.55):
                label = "REPLAY"
            else:
                ins = None
        prepared.append({"m": m, "seg": out, "ins": ins, "label": label})
    if len(prepared) < 3:
        return None
    n = len(prepared)
    for k, pr in enumerate(prepared):  # the narrator says exactly the number the card shows
        pr["n"] = n - k
        if guide:  # "Number five. Selling Stolen Cars." + the explanation lines
            pr["line"] = _ranked(pr["n"], pr["m"]["label"].rstrip(".!? ") + ".")
            pr["lines"] = [pr["line"]] + [_strip_rank(x) for x in pr["m"].get("lines", []) if _strip_rank(x)]
        else:
            pr["line"] = _ranked(pr["n"], pr["m"].get("line") or "")
            pr["lines"] = [pr["line"]]
    # 2) narration (Fenrir) with the final numbers; its lengths decide the cold-open length
    texts = [pkg["intro_line"]] + [ln for pr in prepared for ln in pr["lines"]] + [pkg["outro_line"]]
    voice_by_text: dict[str, dict] = {}
    try:
        spoken = [t for t in texts if t.strip()]
        for v in synthesize_lines([{"t": 0.0, "text": t} for t in spoken], clip_id):
            voice_by_text[v["text"]] = v
    except HumanVoiceUnavailable as exc:
        log.warning("[long] narration skipped: %s", exc)
    intro_v = voice_by_text.get(pkg["intro_line"])
    if guide:  # each item lasts as long as its explanation (+ a beat); lines that don't fit the footage are dropped
        for pr in prepared:
            seg_d, need, kept = _vdur(pr["seg"]), 0.35, []
            for ln in pr["lines"]:
                v = voice_by_text.get(ln)
                if not v:
                    continue
                if need + v["dur"] + 0.9 > seg_d:
                    break
                kept.append(v)
                need += v["dur"] + 0.3
            pr["voices"] = kept
            # footage with its own game sound may run longer than the narration; muted footage may not (silence)
            floor = 10.0 if pr["m"].get("has_narrator", True) else float(c.get("guide_min_item_sec", 25))
            target = min(seg_d, max(need + 0.9, floor))
            if target < seg_d - 0.4:
                _trim(pr["seg"], target)
    segs, cards, voiced, inserts = [], [], [], []
    # 3) cold open: #1 around its best frame, long enough for the intro line
    teaser = LONG_DIR / "seg_teaser.mp4"
    t_len = min(9.0, max(float(c.get("teaser_sec", 4.0)), (intro_v["dur"] + 0.8) if intro_v else 0))
    ta = max(0.0, min(top["best_t"] - t_len / 2, top["dur"] - t_len))
    if _segment(top["path"], ta, ta + t_len, teaser, silent_ok=guide, mute=guide and top.get("has_narrator", True)):
        segs.append(teaser)
    cursor = sum(_vdur(s) for s in segs)
    if intro_v and segs:
        voiced.append({**intro_v, "t": 0.3})
    # 4) timeline: card + voice line + moment (+ REPLAY / REACTION insert)
    for pr in prepared:
        m, d = pr["m"], _vdur(pr["seg"])
        who = f"Gameplay: {m['streamer']}" if guide else f"{m['streamer']} on {m['platform'].title()}"
        cards.append({"start": cursor, "n": pr["n"], "label": m["label"], "who": who, "m": m})
        if guide:
            t = cursor + 0.35
            for v in pr.get("voices", []):
                voiced.append({**v, "t": t})
                t += v["dur"] + 0.3
        else:
            v = voice_by_text.get(pr["line"])
            if v and v["dur"] < d - 0.6:  # a line longer than its moment would talk over the next one
                voiced.append({**v, "t": cursor + 0.35})
        segs.append(pr["seg"])
        cursor += d
        if pr["ins"] is not None:
            di = _vdur(pr["ins"])
            inserts.append({"start": cursor, "end": cursor + di, "label": pr["label"]})
            segs.append(pr["ins"])
            cursor += di
    if len(cards) < 3:
        return None
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
    _build_ass(ass, pkg["title"], cards, outro_at, total, handle, inserts,
               brand=load_config().get("brand", {}).get("display_name", "") if c.get("intro_sting", True) else "",
               **({"outro_text": outro_text} if outro_text else {}))

    inputs, chains, labels = ["-i", str(body)], [], []
    k = 1
    for v in voiced:
        ms = int(v["t"] * 1000)
        inputs += ["-i", str(v["path"])]
        chains.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,volume=1.35,adelay={ms}|{ms}[v{k}]")
        labels.append(f"[v{k}]")
        k += 1
    sfx_labels = []
    events = [("whoosh", cd["start"] - 0.25, 0.5) for cd in cards[1:]] + \
        ([("whoosh", outro_at - 0.25, 0.5)] if outro.exists() else []) + \
        [("boom", ins["start"], 0.6) for ins in inserts] + ([("boom", 0.05, 0.55)] if c.get("intro_sting", True) else [])
    for kind, at, vol in events:
        f = _sfx(kind)
        if not f:
            continue
        ms = int(max(0.0, at) * 1000)
        inputs += ["-i", str(f)]
        chains.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,volume={vol},adelay={ms}|{ms}[x{k}]")
        sfx_labels.append(f"[x{k}]")
        k += 1
    from video_processor import _music_track
    music = _music_track()
    if music:  # optional royalty-free bed from music/ (owner's tracks), ducked under everything
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        chains.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,volume={c.get('music_volume', 0.07)}[x{k}]")
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
        "loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.95,apad[aout]"
    ass_esc = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    out = READY_DIR / f"{clip_id}.mp4"
    res = run(["ffmpeg", "-y", "-hide_banner", *inputs, "-filter_complex", f"[0:v]ass='{ass_esc}'[vout];{afc}",
               "-map", "[vout]", "-map", "[aout]", "-t", f"{total:.2f}", "-c:v", "libx264", "-preset",
               str(c.get("preset", "veryfast")), "-crf", str(c.get("crf", 21)), "-profile:v", "high",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out)], timeout=3000)
    if res.returncode != 0 or not out.exists():
        log.error("[long] final render failed: %s", res.stderr[-1200:])
        return None
    thumb = thumbnail(top["path"], top["best_t"], pkg["thumbnail_text"], READY_DIR / f"{clip_id}_thumb.jpg",
                      cam=top.get("facecam"))
    # two more thumbnails for YouTube Studio's "Test & Compare" (A/B test - pick all 3 there)
    second = moments[-2] if len(moments) > 1 else top
    thumbnail(second["path"], second["best_t"], pkg["thumbnail_text"], READY_DIR / f"{clip_id}_thumb2.jpg",
              cam=second.get("facecam"))
    alt = " ".join(top["label"].upper().split()[:3])
    thumbnail(top["path"], max(0.0, top["best_t"] - 0.7), alt, READY_DIR / f"{clip_id}_thumb3.jpg")
    if guide:
        chapters = ["0:00 Intro"] + [f"{int(cd['start'] // 60)}:{int(cd['start'] % 60):02d} #{cd['n']} {cd['label']}"
                                     for cd in cards]
        credits = [f"#{cd['n']} {cd['label']} - gameplay by {cd['m']['streamer']}: {cd['m']['url']}" for cd in cards]
    else:
        chapters = ["0:00 Intro"] + [f"{int(cd['start'] // 60)}:{int(cd['start'] % 60):02d} #{cd['n']} {cd['label']} "
                                     f"({cd['m']['streamer']})" for cd in cards]
        credits = [f"#{cd['n']} {cd['m']['streamer']} ({cd['m']['platform'].title()}): {cd['m']['url']}" for cd in cards]
    log.info("[long] %s: %.0fs, %d moments, narration %s, thumbnail %s", out.name, total, len(cards),
             "yes" if voiced else "no", "yes" if thumb else "no")
    return {"path": str(out), "duration": total, "thumbnail": str(thumb) if thumb else "", "chapters": chapters,
            "placed": [{"text": v["text"], "start": v["t"], "dur": v["dur"], "path": str(v["path"])} for v in voiced],
            "still_ok": [(outro_at, total)] + [(x["start"], x["end"]) for x in inserts if x["label"] == "REACTION"],
            "inserts": inserts,
            "credits": credits, "moments": [{"n": cd["n"], "key": cd["m"]["key"], "label": cd["label"],
                                             "streamer": cd["m"]["streamer"], "hype": cd["m"]["hype"]} for cd in cards]}


# --------------------------------------------------------------------------- pending upload (retry)

PENDING_DIR = STATE_DIR / "pending_long"  # saved with the Actions state cache until YouTube accepts it


def _keep_for_retry(rendered: dict, caps: dict, clip_id: str, awaiting_approval: bool = False) -> None:
    """YouTube refused the upload (e.g. uploadLimitExceeded): keep the finished video and retry on the next runs."""
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    for f in PENDING_DIR.glob("*"):
        f.unlink(missing_ok=True)  # only the newest long video is kept
    shutil.copy(rendered["path"], PENDING_DIR / "video.mp4")
    if rendered.get("thumbnail"):
        shutil.copy(rendered["thumbnail"], PENDING_DIR / "thumb.jpg")
    save_json(PENDING_DIR / "meta.json", {"clip_id": clip_id, "caps": caps, "at": time.time(),
                                          "moments": rendered.get("moments", []), "duration": rendered["duration"],
                                          "awaiting_approval": awaiting_approval})
    log.info("[long] kept the finished video - the upload is retried on the next runs")


def retry_pending(dry_run: bool, approve: bool = False) -> dict | None:
    """Upload a long video YouTube refused earlier, or (approve=True) the one waiting for the owner's approval.
    Returns a history-style result, or None if nothing to do."""
    meta_f = PENDING_DIR / "meta.json"
    if dry_run or not meta_f.exists() or not (PENDING_DIR / "video.mp4").exists():
        return None
    if load_json(meta_f, {}).get("awaiting_approval") and not approve:
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

LONG_KINDS = ("moments", "money_guide", "trend_guide")


def _long_kinds(state: dict) -> list[str]:
    """Which long video to try, in order. FORMAT (manual run) = only that one. Otherwise the rotation
    (long_video.rotation, default moments <-> money_guide) with the others as fallbacks; when GTA 6 streams are
    quiet, a trend guide replaces the stream-moments countdown."""
    wanted = env("FORMAT").strip().lower()
    if wanted in LONG_KINDS:
        return [wanted]
    rot = [k for k in _cfg().get("rotation", ["moments", "money_guide"]) if k in LONG_KINDS] or ["moments"]
    last = state.get("last_kind")
    nxt = rot[(rot.index(last) + 1) % len(rot)] if last in rot else rot[0]
    order = list(dict.fromkeys([nxt] + rot + ["trend_guide", "money_guide"]))
    try:
        import trends
        if trends.stream_health()["weak"]:  # streams died down: follow trends, stream moments only as last resort
            if nxt == "moments":
                log.info("[long] GTA 6 streams are quiet - a trend guide instead of the stream-moments countdown")
                order.insert(0, "trend_guide")
            order = [k for k in dict.fromkeys(order) if k != "moments"] + ["moments"]
    except Exception:  # noqa: BLE001
        pass
    return order


def make_moments(brand: str, used: set) -> dict:
    """Stream-moments countdown: {chosen, pkg, rendered, clip_id} or {error}."""
    c = _cfg()
    cands = gather(c, used)
    if len(cands) < 4:
        return {"error": f"only {len(cands)} usable clips today"}
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
        for m in moments:
            Path(m["path"]).unlink(missing_ok=True)
        return {"error": f"not enough good moments today ({len(chosen)} moments, {total:.0f}s; "
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
        for m in moments:
            Path(m["path"]).unlink(missing_ok=True)
        return {"error": f"only {len(chosen)} English moments left after the language check"}
    chosen.sort(key=lambda m: (m["hype"], m["views"]))  # countdown: weakest first, #1 last
    pkg = package(chosen, brand)
    clip_id = "long_" + now_local().strftime("%Y%m%d_%H%M")
    rendered = render_long(chosen, pkg, clip_id)
    for m in moments:
        Path(m["path"]).unlink(missing_ok=True)
    if not rendered:
        return {"error": "long video render failed"}
    return {"chosen": chosen, "pkg": pkg, "rendered": rendered, "clip_id": clip_id}


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
    made, errors, kind = None, [], ""
    for kind in _long_kinds(state):
        try:
            made = make_moments(brand, used) if kind == "moments" else make_guide(kind, brand, used, state)
        except Exception as exc:  # noqa: BLE001 - one kind failing must not stop the fallback
            log.exception("[long] %s failed", kind)
            made = {"error": f"crashed: {exc}"}
        if not made.get("error"):
            break
        errors.append(f"{kind}: {made['error']}")
        log.warning("[long] %s not possible today: %s", kind, made["error"])
        made = None
    if not made:
        return False, {"error": "no long video: " + " | ".join(errors)}
    chosen, pkg, rendered, clip_id = made["chosen"], made["pkg"], made["rendered"], made["clip_id"]
    spec = made.get("spec")
    if spec:  # guide
        desc = "\n".join([pkg["summary"], "", "⏱️ Chapters", *rendered["chapters"], "",
                          "🎥 Gameplay footage (all credit to the creators):", *rendered["credits"], "",
                          "No story spoilers. " + spec["hashtags"]])
        caps = {"youtube_title": pkg["title"], "youtube_description": desc[:4900],
                "tags": pkg["tags"] + ["GTA 6", "GTA 6 guide", "GTA 6 tips"] +
                        (["GTA 6 money", "GTA 6 how to make money"] if kind == "money_guide" else []),
                "first_comment": spec["comment"]}
    else:
        desc = "\n".join([pkg["summary"], "", "⏱️ Chapters", *rendered["chapters"], "",
                          "🎥 Clips (all credit to the streamers):", *rendered["credits"], "", "#GTA6 #GTA6Funny #GTA6Stream"])
        caps = {"youtube_title": pkg["title"], "youtube_description": desc[:4900],
                "tags": pkg["tags"] + ["GTA 6", "GTA 6 stream moments", "GTA 6 funny moments"],
                "first_comment": "Which moment was the best? Drop the number 👇"}
    # everything needed to upload it by hand (title, description with chapters + credits, tags) next to the video
    (READY_DIR / f"{clip_id}_YOUTUBE_UPLOAD.txt").write_text(
        "TITLE:\n" + caps["youtube_title"] + "\n\nDESCRIPTION:\n" + caps["youtube_description"] + "\n\nTAGS:\n"
        + ", ".join(dict.fromkeys(caps["tags"])) + "\n\nPINNED COMMENT:\n" + caps["first_comment"]
        + "\n\nTHUMBNAIL: " + Path(rendered.get("thumbnail") or "none").name + "\n", encoding="utf-8")
    # strict QA: technical checks, Whisper voice-vs-captions check, AI editor watching the final video
    import qa
    if spec:
        ctx = (f"16:9 countdown GUIDE: {spec['subject']}. Title cards show '#N  item  Gameplay: creator'; our narrator "
               "explains each item over its footage (the original creators' voice-overs are muted on purpose, so "
               "some parts only have narration); the last seconds are a static 'subscribe' card. Check that each "
               "explanation matches what the footage shows. Narration lines: ")
    else:
        ctx = ("16:9 countdown of GTA 6 livestream moments. Title cards show '#N  label  streamer'; after the top "
               "moments a REPLAY (slow motion) or REACTION (facecam close-up) insert is intentional; the last seconds "
               "are a static 'subscribe' card. Narration lines: ")
    verdict = qa.check(Path(rendered["path"]), "long", lines=rendered.get("placed"), still_ok=rendered.get("still_ok"),
                       context=ctx + " | ".join(p_["text"] for p_ in rendered.get("placed", [])))
    results = {}
    awaiting = False
    if not verdict["pass"]:
        log.error("[long] ✋ the long video FAILED QA - not publishing it (report + video are in the run's Artifacts)")
        results["youtube"] = {"ok": False, "error": "failed QA: " + " / ".join(verdict["report"].splitlines()[1:4]),
                              "provider": "qa"}
    elif not dry_run and c.get("require_approval", True):
        # the owner watches it first: video, thumbnails, upload text + QA report are in the Artifacts;
        # publish by hand, or run the workflow with slot "approve_long" to let the bot upload it
        _keep_for_retry(rendered, caps, clip_id, awaiting_approval=True)
        awaiting = True
        log.info("[long] ✅ passed QA - waiting for the owner's approval (slot approve_long, or upload by hand)")
    elif not dry_run:
        try:
            yt = youtube_api_upload(Path(rendered["path"]), caps)
            if rendered["thumbnail"] and yt.get("id"):
                yt["thumbnail"] = youtube_set_thumbnail(yt["id"], Path(rendered["thumbnail"]))
            results["youtube"] = {**yt, "provider": "youtube_api"}
        except Exception as exc:  # noqa: BLE001
            results["youtube"] = {"ok": False, "error": str(exc)[:300], "provider": "youtube_api"}
            log.error("[long] YouTube upload failed: %.400s", exc)
            _keep_for_retry(rendered, caps, clip_id)
    ok = (dry_run and verdict["pass"]) or awaiting or results.get("youtube", {}).get("ok", False)
    record = {"id": clip_id, "slot": "long_video", "format_id": f"long_{kind}", "qa": verdict["report"],
              "created_at": datetime.now(timezone.utc).isoformat(), "copy": {**pkg, "banner": pkg["title"]},
              "rendered": rendered["path"], "duration": rendered["duration"], "moments": rendered["moments"],
              "thumbnail": rendered["thumbnail"], "results": results, "dry_run": dry_run,
              "status": ("awaiting_approval" if awaiting else "published" if ok and not dry_run
                         else ("dry_run" if dry_run else "failed")),
              "source": {"title": pkg["title"], "url": "", "credit": ", ".join(dict.fromkeys(m["streamer"] for m in chosen)),
                         "provider": "guide" if spec else "stream_clips", "section": [], "segments": []},
              "long_kind": kind}
    if ok and not dry_run:
        from source_fetcher import load_manifest, save_manifest
        mf = load_manifest()
        if not awaiting:
            mf["items"].append(record)
        save_manifest(mf)
        state["used"] = (state.get("used", []) + [m["key"] for m in chosen])[-2000:]
        state["last_kind"] = kind
        if spec:
            state.setdefault("guide_items", {})[kind] = (state.get("guide_items", {}).get(kind, [])
                                                         + [m["label"] for m in chosen])[-60:]
            state.setdefault("guide_count", {})[kind] = int(state.get("guide_count", {}).get(kind, 0)) + 1
            if spec.get("topic"):
                import trends
                trends.mark_used(spec["topic"])
        state["videos"] = (state.get("videos", []) + [{"id": clip_id, "at": time.time(),
                                                       "url": results.get("youtube", {}).get("url", "")}])[-200:]
        save_json(STATE_FILE, state)
    return ok, record
