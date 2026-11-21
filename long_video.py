"""Daily long YouTube video: "Top N craziest GTA 6 stream moments today" (16:9, 5-10 min).

  1. gather the day's most-viewed GTA 6 clips from the top streams (Twitch, Kick, X clip accounts)
  2. AI watches them in batches (saves quota): real GTA 6? spoiler? unsafe? hype score, best part, a short
     label + one narration line per moment
  3. countdown edit: cold open of #1, title card + Fenrir line per moment (worst -> best), whoosh between clips,
     subscribe outro; even loudness
  4. YouTube: title with streamer names, chapters + credits in the description, auto thumbnail, first comment
"""
from __future__ import annotations

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
    import ai_cache
    from source_fetcher import _launch_ts, unsafe_title_penalty
    langs = set(scfg.get("languages", ["en"]))
    bars = scfg.get("viral_bar", {})
    cutoff = _launch_ts(load_config())
    pool, skipped = [], {}
    for x in clips:
        if x["key"] in used or blocked_channel(x["streamer"]):
            continue
        if not sc.is_famous(x["streamer"], scfg):  # famous English-speaking streamers only
            continue
        if x["platform"] == "twitch" and langs and x["language"] and x["language"] not in langs:
            continue
        if not 10 <= x["duration"] <= 180:  # chapters need 10 s+
            continue
        if cutoff and x.get("created") and x["created"] < cutoff:  # only footage from after the launch
            skipped["before launch"] = skipped.get("before launch", 0) + 1
            continue
        bad = ai_cache.bad_reason(x["key"])  # an earlier review found it broken / not GTA 6 / unsafe...
        if bad:
            skipped[bad] = skipped.get(bad, 0) + 1
            continue
        # views relative to what's normal on that platform (X counts millions, Kick hundreds); strip-club
        # titles last (the AI rejects them as unsafe - a wasted download + review)
        base = float((bars.get(x["platform"]) or {}).get("min_views", 1000)) or 1000
        x["score"] = math.log10(x["views"] + 1) - math.log10(base) - unsafe_title_penalty(x["title"]) / 3
        pool.append(x)
    if skipped:
        log.info("[long] skipped (remembered from earlier runs / launch date): %s", skipped)
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
    import ai_cache
    import authentic
    from source_fetcher import Candidate, FormatsLimited, _download
    fake = authentic.known_fake(x["key"])
    if fake:
        return None, f"not GTA 6 (checked before: {fake.split()[0]})"
    bad = ai_cache.bad_reason(x["key"])
    if bad:
        return None, f"{bad} (reviewed before)"
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
    except FormatsLimited:
        return None, f"no HD stream ({x['platform']})"
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
    if not info["has_audio"]:
        path.unlink(missing_ok=True)
        return None, "no sound"
    errs, rep = _decode_errors(path, with_smoothness=True)  # one decode for both checks
    if errs > int(_cfg().get("max_decode_errors", 30)):
        log.info("[long] ✋ %s (%s, %s) is corrupted: %d decoding errors", x["key"], x["streamer"], x["platform"], errs)
        path.unlink(missing_ok=True)
        return None, "corrupted"
    if rep > float(_cfg().get("max_repeated_frames", 0.2)):
        log.info("[long] ✋ %s (%s, %s) is choppy: %.0f%% repeated frames", x["key"], x["streamer"], x["platform"],
                 rep * 100)
        path.unlink(missing_ok=True)
        return None, "choppy"
    authentic.remember(path, x["key"], x["streamer"], x["platform"], x.get("section"))
    return {**x, "path": path, "dur": info["duration"], "repeat": rep}, None


def _decode_errors(path: Path, with_smoothness: bool = False):
    """Broken video data (pixel garbage / green-purple smears come from it): count the decoder's error lines.
    with_smoothness: in the same decode, also the share of repeated frames in the first 60 s (exactly like
    stream_clips.smoothness) -> (errors, repeated share) - one full decode per clip instead of two."""
    if not with_smoothness:
        res = run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"], timeout=300)
        return len([ln for ln in res.stderr.splitlines() if ln.strip()])
    res = run(["ffmpeg", "-hide_banner", "-nostats", "-v", "error", "-progress", "pipe:1", "-i", str(path),
               "-filter_complex", "[0:v:0]split=2[full][s];[s]trim=duration=60,fps=30,scale=320:-2,"
               "mpdecimate=hi=200:lo=64:frac=0.33[m]", "-map", "[m]", "-f", "null", "-",
               "-map", "[full]", "-f", "null", "-"], timeout=300)
    errs = len([ln for ln in res.stderr.splitlines() if ln.strip()])
    frames = re.findall(r"^frame=(\d+)", res.stdout or "", re.M)
    total30 = min(60.0, probe(path)["duration"] or 0) * 30
    if res.returncode != 0 or not frames or total30 < 30:
        import stream_clips as sc  # fall back to the separate check
        return errs, sc.smoothness(path)
    return errs, round(max(0.0, 1 - int(frames[-1]) / total30), 3)


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
- unsafe: true for nudity or sexual content - IN-GAME TOO (strip clubs, lap dances, topless / naked characters,
  sex scenes, shower scenes) - and for slurs / hate speech, real-world violence, doxxing: things YouTube
  age-restricts or removes. In-game violence, crime, police chases and the streamer swearing are normal GTA
  content and are NOT unsafe.
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
    import ai_cache
    import gemini_writer
    LONG_DIR.mkdir(parents=True, exist_ok=True)
    size = int(_cfg().get("review_batch", 6))
    # verdicts from earlier runs (same prompt, same clip + section) cost no quota; errors are never remembered
    ns = ai_cache.review_ns(prompt_for(0, ""), _game_facts())
    cached = [(m, v) for m in moments if (v := ai_cache.review_get(ns, m["key"], m.get("section")))]
    if cached:
        log.info("[long] %d clip reviews reused from earlier runs (no AI call)", len(cached))
        hit = {id(m) for m, _ in cached}
        moments = [m for m in moments if id(m) not in hit]
        yield [m for m, _ in cached], cached
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
        if not verdicts:  # Gemini can't watch: a backup AI that sees images judges 2 frames per clip
            try:
                import qa
                import vision
                groups = [(f"Clip {j} ({m['streamer']})",
                           [f for f in (qa._frame(m["path"], (m.get("dur") or 20) * r) for r in (0.3, 0.7)) if f])
                          for j, (i, m, p, _, _) in enumerate(parts, 1)]
                note = ("You cannot see the joined video: you get 2 still frames per clip instead (in order). Judge "
                        "what frames can show; trim / best_t: give your best guess in seconds within each clip.\n\n")
                data = vision._backup(groups, note + prompt_for(len(parts), "\n".join(ranges)), "clips", "clip", 0.4)
                for v in (data or {}).get("clips", []):
                    try:
                        verdicts[parts[int(v.get("clip")) - 1][0]] = v
                    except (TypeError, ValueError, IndexError, AttributeError):
                        continue
                if verdicts:
                    log.info("[long] batch %d judged by a backup AI on frames", b // size + 1)
            except Exception as exc:  # noqa: BLE001
                log.warning("[long] backup review failed: %.150s", exc)
        for i, m, p, _, _ in parts:
            p.unlink(missing_ok=True)
            if verdicts.get(i):
                ai_cache.review_put(ns, m["key"], m.get("section"), verdicts[i])
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
People find these videos by SEARCHING ("gta 6 funny moments", "gta 6 stream moments", "<streamer> gta 6"), then
click the title + thumbnail that promise the most fun. Promise only what the moments above deliver.{searches}
Write:
- title: max 70 characters, "GTA 6" in the first 30 characters, the style of the biggest gaming compilation
  channels, the exact number {n}, name 1-3 of the biggest streamers, e.g.
  "Top {n} CRAZIEST GTA 6 Moments Today (IShowSpeed, Kai Cenat, caseoh)". Max 2 CAPITALISED words, no hashtags.
- intro_line: what the narrator says over the cold open of #1, 10-20 words, makes people stay to the end.
- outro_line: 8-15 words, asks which moment was the best and to subscribe for daily GTA 6 moments.
- thumbnail_text: 2-4 words, ALL CAPS, huge thumbnail text that makes people click and adds to the title instead
  of repeating it (e.g. "HE JUMPED?!").
- summary: the first 2 lines of the description (search reads them): 2 sentences, the first starts with "GTA 6" +
  the main search phrase (e.g. "GTA 6 funny moments") and names the biggest streamers.
- tags: 10 YouTube search tags people really type (e.g. "gta 6 funny moments", "kai cenat gta 6"), no #.
Say "GTA 6", never "GTA VI". No story spoilers. Never invent events or numbers. Return ONLY JSON with exactly these keys."""


def package(moments: list[dict], brand: str) -> dict:
    import ai_fallback
    import critic
    import gemini_writer
    n = len(moments)
    listing = "\n".join(f"#{n - i}: {m['label']} ({m['streamer']})" for i, m in enumerate(moments))
    searches = critic.relevant_searches("funny moments stream streamer clips " + listing, 6, 0.5)
    prompt = PACKAGE_PROMPT.format(brand=brand, n=n, moments=listing, searches=(
        "\nWhat GTA 6 players search right now (use one if it honestly fits): " + "; ".join(searches)) if searches else "")
    data = gemini_writer._generate_json(prompt, temperature=0.8, prefer_pro=True)
    if not data and ai_fallback.available():
        data = ai_fallback.chat_json(prompt)
    data = data or {}
    names = list(dict.fromkeys(m["streamer"] for m in reversed(moments)))[:3]
    if data:  # critic AI: second opinion on title / thumbnail text / intro (+ the packaging rules)
        fields = {k: str(data[k]) for k in ("title", "thumbnail_text", "intro_line") if data.get(k)}
        data.update(critic.polish(fields, {"title": 90, "thumbnail_text": 28, "intro_line": 160}, listing,
                                  money=False, keep=["stream moment", "funny moment"], count=n))
    # "GTA 6" first and the countdown number must be true (the critic once turned "Top 8" into "10 ...")
    title = critic.fix_title(str(data.get("title") or f"Top {n} CRAZIEST GTA 6 Stream Moments Today ({', '.join(names)})"),
                             n, max_len=95)
    return {
        "title": re.sub(r"[<>]", "", title)[:95],
        "intro_line": str(data.get("intro_line") or f"These are the {n} craziest GTA 6 moments from today's streams, "
                                                     "and number one is unreal."),
        "outro_line": str(data.get("outro_line") or "Which one was the best? Subscribe for daily GTA 6 moments."),
        "thumbnail_text": re.sub(r"[^A-Z0-9 !?'.]", "", critic.fix_count(critic.norm_gta(
            str(data.get("thumbnail_text") or "GTA 6 CHAOS")), n).upper())[:28],
        "summary": critic.norm_gta(str(data.get("summary") or f"GTA 6 funny moments: the {n} craziest GTA 6 livestream "
                                                               f"moments from today ({', '.join(names)}).")),
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
        # general searches + one per money topic (stash locations, safes, robberies, selling cars...) + money trends:
        # specific searches find videos that really show a method
        queries = list(c.get("money_queries", []))
        mt = next((f for f in load_config().get("formats", {}).get("rotation", []) if f.get("id") == "money_tips"), {})
        for t in mt.get("topics", []):
            queries += list(t.get("queries", []))[:1]
        try:
            import trends
            for t in load_json(trends.TRENDS_FILE, {}).get("topics", []):
                if t.get("money"):
                    queries += list(t.get("queries", []))[:2]
        except Exception:  # noqa: BLE001
            pass
        return {"kind": kind, "topic": "", "subject": "the best ways to make money in GTA 6 (no story spoilers)",
                "noun": "way to make money", "short": "money tips",
                "example": "Top 6 BEST Ways To Make Money (No Spoilers)",
                "queries": list(dict.fromkeys(queries)) or ["GTA 6 how to make money", "GTA 6 money making guide"],
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
    import ai_cache
    import authentic
    from source_fetcher import (TitleFilter, blocked_channel, discover_dailymotion, discover_youtube, load_manifest,
                                unsafe_title_penalty)
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
    srcs = load_manifest()["sources"]
    trust = {x.key: authentic.channel_verdict(x.uploader) for x in found}
    skipped: dict[str, int] = {}

    def skip(why: str) -> None:
        skipped[why] = skipped.get(why, 0) + 1
    # footage without someone talking over it keeps its game sound (muted voice-overs feel empty) -> first; then
    # channels whose videos passed the GTA 6 check before; strip-club titles last
    for x in sorted(found, key=lambda x: ("no commentary" in (x.title or "").lower(), trust[x.key][1]
                                          - unsafe_title_penalty(x.title), x.views), reverse=True):
        dur = float(x.duration or 0)
        ch = (x.uploader or "unknown").strip()
        if x.key in used or GUIDE_BAD.search(x.title or "") or blocked_channel(ch, cfg) or not 30 <= dur <= 5400:
            continue
        st = srcs.get(x.key, {})
        if st.get("rejected") or st.get("fail_count", 0) >= 2:  # e.g. found to be from before the launch
            skip("rejected before")
            continue
        if authentic.known_fake(x.key):
            skip("not GTA 6 (checked before)")
            continue
        if trust[x.key][0] == "fake":
            skip("GTA V / mod channel")
            continue
        if ai_cache.bad_reason(x.key):
            skip("bad review before")
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
    log.info("[guide] %d videos found, %d candidates%s: %s", len(found), len(pool),
             f" (skipped {skipped})" if skipped else "",
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
    import ai_cache
    from source_fetcher import _launch_ts, unsafe_title_penalty
    langs = set(scfg.get("languages", ["en"]))
    cutoff = _launch_ts(load_config())
    pool = [x for x in clips if x["key"] not in used and sc.is_famous(x["streamer"], scfg)
            and 15 <= x["duration"] <= 180 and not (x["platform"] == "twitch" and langs and x["language"]
                                                     and x["language"] not in langs)
            and not (cutoff and x.get("created") and x["created"] < cutoff) and not ai_cache.bad_reason(x["key"])]
    # strip-club clips last (always rejected as unsafe), then money-sounding titles, then views
    pool.sort(key=lambda x: (not unsafe_title_penalty(x["title"]), bool(_MONEYISH.search(x["title"] or "")),
                             x["views"]), reverse=True)
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
            twin.setdefault("extra", []).append(r)  # ...and their footage becomes extra angles of it
        else:
            items.append(r)
    log.info("[guide] %d different items: %s", len(items),
             [f"{x['label']} ({x['hype']}, {x['sources']} src)" for x in items])
    return items


GUIDE_PACKAGE_PROMPT = """You write for "{brand}", a GTA 6 YouTube channel. Today's video is a countdown guide:
{subject}.{part}
Items (from #{n} up to #1, the best one last):
{items}
People find guides by SEARCHING, so the title starts with what they type{search_hint}.{searches}
Promise exactly what the video shows - every item is legit gameplay (no glitches, no mods).
Write:
- title: max 70 characters, "GTA 6" first + the search phrase, then the hook, the exact number {n}, e.g.
  "GTA 6: {example}". Max 2 CAPITALISED words, no hashtags.{honesty}
- intro_line: what the narrator says over the cold open, 10-20 words, makes people stay for number one.
- outro_line: 8-15 words, asks which one they will try and to subscribe for more GTA 6 {short}.
- thumbnail_text: 2-4 words, ALL CAPS, huge thumbnail text that makes people click and adds to the title instead
  of repeating it (e.g. "EASY MONEY").
- summary: the first 2 lines of the description (search reads them): 2 sentences, the first starts with "GTA 6" +
  the main search phrase{search_hint}, the second says what the countdown covers.
- tags: 10 YouTube search tags people really type (e.g. "gta 6 money guide"), no #.
Say "GTA 6", never "GTA VI". No story spoilers. Never invent numbers or amounts. Return ONLY JSON with exactly these keys."""


def package_guide(items: list[dict], spec: dict, brand: str, part: int = 1) -> dict:
    import ai_fallback
    import critic
    import gemini_writer
    n = len(items)
    money = spec["kind"] == "money_guide"
    topic = critic.norm_gta(spec.get("topic") or "")
    listing = "\n".join(f"#{n - i}: {m['label']} - {m['facts'][:120]}" for i, m in enumerate(items))
    searches = critic.relevant_searches(f"{spec['subject']} {topic}" + (" how to make money fast best ways" if money else ""),
                                        6, 0.5)
    prompt = GUIDE_PACKAGE_PROMPT.format(
        brand=brand, subject=spec["subject"], n=n, items=listing, short=spec["short"],
        example=spec["example"].replace("6", str(n), 1) if spec["example"].startswith("Top 6") else spec["example"],
        part=f" This is part {part} of the series." if part > 1 else "",
        search_hint=(' (e.g. "GTA 6 how to make money", "GTA 6 best ways to make money", "GTA 6 make money fast")'
                     if money else f' (e.g. "GTA 6 {topic}")' if topic else ""),
        searches=("\nWhat GTA 6 players search right now (use one if it honestly fits): " + "; ".join(searches))
        if searches else "",
        honesty=("\n  MONEY: keep \"make money\" in the title; never words that sound like cheats or glitches (hack, glitch,"
                 " exploit, cheat, mod, trick, unlimited / infinite money) - say ways / methods / tips." if money else ""))
    data = gemini_writer._generate_json(prompt, temperature=0.7, prefer_pro=True)
    if not data and ai_fallback.available():
        data = ai_fallback.chat_json(prompt)
    data = data or {}
    if money:
        fallback_title = f"GTA 6: Top {n} BEST Ways To Make Money (No Spoilers)"
        fallback_thumb = "EASY MONEY"
    else:
        fallback_title = f"GTA 6: Top {n} {topic} (No Spoilers)"
        fallback_thumb = " ".join(topic.upper().split()[:3])
    fallbacks = {"title": fallback_title, "thumbnail_text": fallback_thumb}
    if data:  # critic AI: second opinion (+ the packaging rules: no cheat words, keep the search words, true count)
        fields = {k: str(data[k]) for k in ("title", "thumbnail_text", "intro_line") if data.get(k)}
        data.update(critic.polish(fields, {"title": 90, "thumbnail_text": 28, "intro_line": 160}, listing, money=money,
                                  keep=(["make money", "money"] if money else critic.keyword_phrases(topic)),
                                  count=n, fallbacks=fallbacks))
    title = critic.fix_title(str(data.get("title") or fallback_title), n, money, 95, fallback=fallback_title)
    if money and not critic.has_term(title, "money"):  # "money" is what people search: a money guide says it
        title = critic.fix_title(fallback_title, n, money, 95)
    rest = critic.apply_rules({k: str(data.get(k) or "") for k in ("intro_line", "outro_line", "summary", "thumbnail_text")},
                              money, n, fallbacks=fallbacks)
    return {
        "title": re.sub(r"[<>]", "", title)[:95],
        "intro_line": rest["intro_line"] or f"These are the {n} best ones in GTA 6 right now, and number one is the one you need.",
        "outro_line": rest["outro_line"] or f"Which one are you trying first? Subscribe for more GTA 6 {spec['short']}.",
        "thumbnail_text": re.sub(r"[^A-Z0-9 !?'.$]", "", (rest["thumbnail_text"] or fallback_thumb).upper())[:28],
        "summary": rest["summary"] or (f"GTA 6 how to make money: the {n} best ways to make money in GTA 6 right now, "
                                       "best last." if money else
                                       f"{spec['subject'][0].upper()}{spec['subject'][1:]} - {n} picks, best last."),
        "tags": [str(t)[:30] for t in (data.get("tags") or [])][:12] or ["GTA 6", "GTA 6 tips"],
    }


def _final_real_check(items: list[dict]) -> list[dict]:
    """Authenticity check on the exact window of every part that will be shown (main clip + extra angles),
    trusted stream clips too: QA found GTA V footage (Michael, Los Santos) at the start of a money guide."""
    import authentic
    parts = [(it, p_) for it in items for p_ in [it] + list(it.get("extra") or [])]
    wins = []
    for k, (_, p_) in enumerate(parts):
        w = LONG_DIR / f"realcheck_{k}.mp4"
        LONG_DIR.mkdir(parents=True, exist_ok=True)
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{p_['a']:.2f}", "-to", f"{p_['b']:.2f}", "-i", str(p_["path"]),
             "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", str(w)], timeout=180)
        wins.append(w)
    bad = set()
    for b0 in range(0, len(parts), 6):
        res = authentic.check(wins[b0:b0 + 6])
        for k, r in enumerate(res or [None] * len(wins[b0:b0 + 6]), b0):
            if r is None or not r["ok"]:
                bad.add(k)
                p_ = parts[k][1]
                log.info("[guide] ✋ final check: %s (%s) %s", p_.get("label") or p_["key"], p_["streamer"],
                         f"not GTA 6 - {r['game']} ({r['evidence']})" if r else "could not be checked")
    for w in wins:
        w.unlink(missing_ok=True)
    out = []
    for it in items:
        idx = [k for k, (owner, _) in enumerate(parts) if owner is it]
        main_k, extra_ks = idx[0], idx[1:]
        good_extra = [parts[k][1] for k in extra_ks if k not in bad]
        if main_k in bad:
            if not good_extra:
                continue
            new_main = {**good_extra[0], "label": it["label"], "rank": it["rank"], "sources": it["sources"]}
            new_main["extra"] = good_extra[1:]
            out.append(new_main)
        else:
            it["extra"] = good_extra
            out.append(it)
    return out


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
    good = _final_real_check(good)  # the exact parts that will be shown, streamer clips included
    if len(good) < int(c.get("guide_min_items", 4)):
        for m in clips:
            Path(m["path"]).unlink(missing_ok=True)
        return {"error": f"only {len(good)} items left after the final GTA 6 check"}
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
             speed: float = 1.0, crop: list | None = None, mute: bool = False, push: float = 0.0) -> bool:
    """One piece of the video: 1920x1080/30fps, polished audio. Audio and video are cut to EXACTLY the same
    length (whole frames) - otherwise tiny differences add up over 15 clips and the sound drifts out of sync.
    keep: [(x, y)] source ranges to keep (jump cuts); speed < 1 = slow motion; crop = [x, y, w, h] fractions;
    push > 0 = slow push-in zoom over the piece (cold open), e.g. 0.12 = ends 12% closer."""
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
    # push-in: zoompan on a 2x picture (whole-pixel steps at 1080p would wobble); one output frame per input frame
    zoom_v = (f",scale=3840:2160,zoompan=z='1+{push:.3f}*on/{max(1, round(d * 30))}':x='iw/2-(iw/zoom/2)'"
              f":y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=30,setsar=1") if push > 0 else ""
    # tpad: if the source ends early the last frame is held, so the video is never shorter than the audio
    vchain = (f"[0:v]fps=30,{pre_v}{crop_v}{FIT}{slow_v}{zoom_v},tpad=stop_mode=clone:stop_duration=1,"
              f"trim=duration={d:.4f},setpts=PTS-STARTPTS[v]")
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


PAD_CHORD = (220.0, 261.63, 329.63, 392.0, 493.88)  # A minor 9 - a calm synth pad (generated, no copyright)


def _end_screen(src: Path, t: float, length: float, out: Path, pad: bool = False) -> bool:
    """End-screen background (room for YouTube's end-screen elements): the #1 moment at half speed, heavily
    blurred and darkened - it keeps moving (not a frozen still) but stays calm. Audio: silence, or a soft
    generated pad when the video has no music bed. Exactly `length` seconds (whole frames), audio = video."""
    d = round(length * 30) / 30
    dur = probe(src)["duration"] or 0
    a = max(0.0, min(t - d / 4, dur - d / 2 - 0.2))
    vf = (f"[0:v]fps=30,scale=960:540:force_original_aspect_ratio=increase,crop=960:540,setpts=2.0*PTS,fps=30,"
          f"boxblur=14:2,eq=brightness=-0.26:saturation=0.8,scale=1920:1080,vignette=angle=PI/4,setsar=1,"
          f"format=yuv420p,trim=duration={d:.4f},setpts=PTS-STARTPTS[v]")
    if pad:
        tones = "+".join(f"{0.9 - 0.12 * i:.2f}*sin(2*PI*{f:.2f}*t)+{0.5 - 0.06 * i:.2f}*sin(2*PI*{f + 0.7:.2f}*t)"
                         for i, f in enumerate(PAD_CHORD))
        af = (f"aevalsrc='0.035*({tones})*(0.8+0.2*sin(2*PI*0.2*t))':s=48000:d={d:.4f},lowpass=f=1800,"
              f"aecho=0.8:0.6:180|360:0.3|0.2,afade=t=in:d=1.5,afade=t=out:st={max(0.0, d - 2.0):.3f}:d=2.0,"
              f"aformat=channel_layouts=stereo,atrim=duration={d:.4f}[a]")
    else:
        af = f"anullsrc=r=48000:cl=stereo,atrim=duration={d:.4f}[a]"
    res = run(["ffmpeg", "-y", "-stream_loop", "-1", "-ss", f"{a:.2f}", "-t", f"{d / 2 + 2:.2f}", "-i", str(src),
               "-filter_complex", f"{vf};{af}", "-map", "[v]", "-map", "[a]", *ENC, str(out)], timeout=300)
    if res.returncode != 0 or not out.exists() or abs(_vdur(out) - d) > 0.05:
        log.warning("[long] moving end screen failed: %.300s", res.stderr[-300:])
        out.unlink(missing_ok=True)
        return False
    return True


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


# --------------------------------------------------------------------------- brand look (cards, thumbnails)
# One consistent style everywhere (title card, number cards, lower thirds, end screen, thumbnails):
# heavy Montserrat, yellow + hot pink ("Vice City" sunset) on near-black, thick black outlines.
YELLOW, PINK, INK, WHITE = (255, 214, 0), (255, 46, 136), (12, 8, 20), (255, 255, 255)
PLATFORM_LOOK = {"twitch": ("TWITCH", (145, 70, 255), WHITE), "kick": ("KICK", (83, 252, 24), (0, 0, 0)),
                 "x": ("X", (0, 0, 0), WHITE), "youtube": ("YOUTUBE", (230, 0, 0), WHITE),
                 "dailymotion": ("DAILYMOTION", (0, 102, 220), WHITE)}
# advance widths of Montserrat Black (1/100 em) - to size boxes / fit text without measuring at run time
_GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 #!?.,:'-&$%/()+@_\""
_WIDTHS = [int(x) for x in (
    "81 77 74 83 67 64 77 80 35 57 76 62 95 80 85 74 85 74 66 65 78 79 121 76 71 69 64 70 62 70 65 43 71 70 33 33 71 "
    "33 104 70 68 70 70 46 56 46 70 64 98 64 64 57 69 42 61 61 71 62 66 64 68 66 30 74 31 61 30 30 30 25 39 78 66 92 "
    "44 38 38 62 104 50 49").split()]
_GLYPH_W = dict(zip(_GLYPHS, _WIDTHS))
ASS_EM = 0.727  # libass scales Montserrat so that 1 em = 0.727 x the ASS font size


def _em(s: str) -> float:
    """Width of a text in em (Montserrat Black; other characters count as an average capital)."""
    return sum(_GLYPH_W.get(ch, 75) for ch in s) / 100


def _ac(rgb: tuple) -> str:
    """ASS override colour (&HBBGGRR&)."""
    r, g, b = rgb
    return f"&H{b:02X}{g:02X}{r:02X}&"


def _ffc(rgb: tuple, alpha: float = 1.0) -> str:
    """FFmpeg colour (0xRRGGBB@a)."""
    r, g, b = rgb
    return f"0x{r:02X}{g:02X}{b:02X}@{alpha:.2f}"


def _rect(w: float, h: float, skew: float = 0) -> str:
    """ASS drawing: a w x h box (a parallelogram leaning right when skew > 0)."""
    return f"m {skew:.0f} 0 l {w + skew:.0f} 0 l {w:.0f} {h:.0f} l 0 {h:.0f}"


def _circle(r: float) -> str:
    """ASS drawing: a circle of radius r (centre at r, r). With \\1a&HFF& + \\bord it is a ring."""
    k, c = 0.5523 * r, r
    return (f"m {c:.0f} 0 b {c + k:.0f} 0 {2 * c:.0f} {c - k:.0f} {2 * c:.0f} {c:.0f} "
            f"b {2 * c:.0f} {c + k:.0f} {c + k:.0f} {2 * c:.0f} {c:.0f} {2 * c:.0f} "
            f"b {c - k:.0f} {2 * c:.0f} 0 {c + k:.0f} 0 {c:.0f} b 0 {c - k:.0f} {c - k:.0f} 0 {c:.0f} 0")


def _wrap(text: str, width: int, lines: int) -> list[str]:
    import textwrap
    return (textwrap.wrap(text, width) or [""])[:lines]


def _font_families() -> tuple[str, str]:
    """(heavy, bold) ASS font names: 'Montserrat Black' / 'Montserrat ExtraBold' when installed, else the
    configured font (libass then uses its bold face)."""
    font = load_config().get("video", {}).get("font", "Montserrat")
    res = run(["fc-list", ":", "family"], timeout=10)
    fams = {f.strip().lower() for ln in (res.stdout or "").splitlines() for f in ln.split(",")}
    heavy = f"{font} Black" if f"{font} black".lower() in fams else font
    bold = f"{font} ExtraBold" if f"{font} extrabold".lower() in fams else font
    return heavy, bold


def _build_ass(path: Path, tl: dict) -> None:
    """All on-screen graphics of the long video, in one brand style (ASS, burned in at the final render).
    tl (timeline, seconds): title, brand, handle, total, outro_at, outro_text, progress (bool) and
      hook (a, b, text)      cold open on the #1 moment: flash + "TODAY'S #1" tag
      coming (a, b)          'COMING UP' preview montage; flashes: [t] quick white flash cuts inside it
      intro (a, b)           title card: brand kicker + big title on a dark band
      cards [{start, end, n, label}]   number slam + label, then a corner badge "#N" for the rest of the moment
      credits [{start, end, name, platform}]   lower third: platform tag + creator name
      inserts [{start, end, label}]    REPLAY / REACTION tag
      cuts [t]               full-screen swipe transitions (synced with the whoosh)
    Text never sits in the last seconds' end-screen boxes (room for YouTube's end-screen elements)."""
    heavy, bold = _font_families()
    fb = "0" if heavy.endswith("Black") else "-1"  # a real Black / ExtraBold face needs no fake bold
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: H,{heavy},100,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,{fb},0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: B,{bold},100,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,{fb},0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: HB,{heavy},100,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,{fb},0,0,0,100,100,0,0,3,14,0,7,0,0,0,1
Style: BB,{bold},100,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,{fb},0,0,0,100,100,0,0,3,14,0,7,0,0,0,1
Style: D,{bold},100,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev: list[str] = []

    def add(layer: int, a: float, b: float, style: str, text: str) -> None:
        if b - a >= 0.05:
            ev.append(f"Dialogue: {layer},{_t(a)},{_t(b)},{style},,0,0,0,,{text}")

    y_, p_, k_, w_ = _ac(YELLOW), _ac(PINK), _ac(INK), _ac(WHITE)
    total, outro_at = float(tl["total"]), float(tl["outro_at"])
    full = _rect(1920, 1080)

    def flash(t: float, strength: str = "40", dur: float = 0.18) -> None:
        add(9, t, t + dur, "D", f"{{\\an7\\pos(0,0)\\p1\\c{w_}\\alpha&H{strength}&\\fad(0,{int(dur * 900)})}}{full}")

    def tag(a: float, b: float, text: str, box: str, fg: str, x: int = 90, y: int = 78) -> None:
        add(7, a, b, "HB", f"{{\\an7\\move(-600,{y},{x},{y},0,240)\\fs50\\fax-0.18\\bord14\\3c{box}\\c{fg}"
                           f"\\shad6\\4c&H000000&\\4a&H80&\\fad(0,200)}}{_ass_text(text)}")

    # --- cold open hook (the #1 moment) + preview montage + title card
    if tl.get("hook"):
        a, b, text = tl["hook"]
        flash(a + 0.0, "20", 0.3)
        tag(a + 0.15, b - 0.1, text, p_, w_)
    if tl.get("coming"):
        a, b = tl["coming"]
        tag(a + 0.1, b - 0.1, "COMING UP", y_, k_)
    for t in tl.get("flashes", []):
        flash(t, "60", 0.14)
    if tl.get("intro"):
        a, b = tl["intro"]
        short = re.sub(r"\s*[\(\[].*?[\)\]]", "", tl.get("title", "")).strip().upper()  # drop "(IShowSpeed, ...)"
        lines = _wrap(_ass_text(short), 24, 3)
        size = 96 if len(lines) < 3 else 84
        brand = (tl.get("brand") or "").upper()
        band_h = 70 + (62 if brand else 0) + len(lines) * size * 1.02 + 60
        top = 540 - band_h / 2
        add(2, a, b, "D", f"{{\\an5\\pos(960,540)\\p1\\c{k_}\\alpha&H40&\\fscy0\\t(0,180,\\fscy100)\\fad(0,200)}}"
                          f"{_rect(1920, band_h)}")
        for yy in (top, top + band_h):
            add(3, a, b, "D", f"{{\\an5\\pos(960,{yy:.0f})\\p1\\c{p_}\\fscx0\\t(60,320,\\fscx100)\\fad(0,200)}}"
                              f"{_rect(1920, 10)}")
        y0 = top + 60
        if brand:
            add(4, a + 0.1, b, "B", f"{{\\an8\\pos(960,{y0:.0f})\\fs46\\fsp12\\c{y_}\\fad(150,200)}}{_ass_text(brand)}")
            y0 += 62
        add(4, a + 0.12, b, "H", f"{{\\an8\\pos(960,{y0:.0f})\\fs{size}\\c{w_}\\bord5\\3c&H000000&\\shad8\\4c{p_}"
                                 f"\\fscx135\\fscy135\\t(0,220,\\fscx100\\fscy100)\\fad(100,200)}}" + "\\N".join(lines))

    # --- countdown: progress bar (one segment per item), number cards, corner badge
    cards = tl.get("cards") or []
    if cards and tl.get("progress", True):
        n_all, x0, x1, gap, by, bh = len(cards), 80, 1840, 10, 26, 9
        seg_w = (x1 - x0 - gap * (n_all - 1)) / n_all
        c0 = cards[0]["start"]
        for i, cd in enumerate(cards):
            sx = x0 + i * (seg_w + gap)
            bar = _rect(seg_w, bh)
            add(5, c0, outro_at, "D", f"{{\\an7\\pos({sx:.0f},{by})\\p1\\c{w_}\\alpha&H98&\\bord2\\3c&H000000&"
                                      f"\\3a&HB0&\\fad(300,200)}}{bar}")
            ms = int((cd["end"] - cd["start"]) * 1000)
            add(6, cd["start"], cd["end"], "D", f"{{\\an7\\pos({sx:.0f},{by})\\p1\\c{y_}\\clip({sx:.0f},0,{sx:.0f},80)"
                                                f"\\t(0,{ms},\\clip({sx:.0f},0,{sx + seg_w:.0f},80))}}{bar}")
            add(6, cd["end"], outro_at, "D", f"{{\\an7\\pos({sx:.0f},{by})\\p1\\c{y_}\\fad(0,200)}}{bar}")
    for cd in cards:
        s, e = cd["start"], cd["end"]
        out = min(s + 4.4, e - 0.2)
        num = f"#{cd['n']}"
        add(8, s + 0.08, out, "H", f"{{\\an1\\pos(112,560)\\fs300\\c{y_}\\bord14\\3c&H000000&\\shad12\\4c{p_}\\frz6"
                                   f"\\fscx260\\fscy260\\t(0,160,\\fscx92\\fscy92)\\t(160,260,\\fscx100\\fscy100)"
                                   f"\\fad(0,250)}}{num}")
        label = _wrap(_ass_text(cd.get("label", "").upper()), 22, 2)
        add(7, s + 0.3, out, "HB", f"{{\\an7\\move(-1100,592,124,592,0,260)\\fs72\\fax-0.15\\bord16\\3c{k_}"
                                   f"\\3a&H18&\\c{w_}\\fad(0,250)}}" + "\\N".join(label))
        add(7, out - 0.2, e - 0.12, "HB", f"{{\\an7\\pos(80,62)\\fs54\\fax-0.15\\bord12\\3c{p_}\\c{w_}"
                                          f"\\fad(250,150)}}{num}")

    # --- lower thirds: who made the clip / gameplay
    for cr in tl.get("credits") or []:
        a, b = cr["start"] + 0.7, min(cr["start"] + 6.5, cr["end"] - 0.3)
        if b - a < 2.0:
            continue
        tag_txt, bg, fg = PLATFORM_LOOK.get(str(cr.get("platform", "")).lower(),
                                            ("GAMEPLAY", INK, YELLOW))
        if cr.get("kind") == "gameplay":
            tag_txt, bg, fg = "GAMEPLAY", INK, YELLOW
        add(7, a, b, "BB", f"{{\\an1\\move(-900,1000,92,1000,0,280)\\fs44\\fax-0.15\\bord14\\3c{_ac(bg)}\\c{_ac(fg)}"
                           f"\\shad6\\4c&H000000&\\4a&H90&\\fad(0,250)}}{_ass_text(tag_txt)}\\h\\h\\h"
                           f"{{\\3c{w_}\\c{k_}}}{_ass_text(str(cr.get('name', '')))}")

    # --- REPLAY / REACTION inserts
    red = _ac((230, 20, 40))
    for ins in tl.get("inserts") or []:
        add(7, ins["start"] + 0.05, ins["end"] - 0.05, "HB",
            f"{{\\an9\\pos(1840,62)\\fs56\\fax-0.15\\bord14\\3c{red}\\c{w_}\\fad(80,150)}}"
            f"{{\\p1}}{_circle(16)}{{\\p0}}\\h\\h{_ass_text(ins['label'])}")
        flash(ins["start"], "70", 0.12)

    # --- swipe transitions: a pink band with a yellow edge crosses the screen; it covers the whole frame at the
    # cut itself (the cuts are on whole frames), so one clip never pops into the next
    for t in tl.get("cuts", []):
        a, b = t - 0.2, t + 0.2
        add(10, a, b, "D", f"{{\\an7\\move(-3500,0,2200,0)\\p1\\c{p_}\\blur3}}{_rect(3000, 1080, 300)}")
        add(10, a, b, "D", f"{{\\an7\\move(-500,0,5200,0)\\p1\\c{y_}\\blur3}}{_rect(150, 1080, 300)}")

    # --- handle watermark (until the end screen)
    if tl.get("handle"):
        add(1, 0, outro_at, "B", f"{{\\an3\\pos(1880,1050)\\fs30\\c{w_}\\alpha&H70&\\bord2\\3c&H000000&}}"
                                 f"{_ass_text(tl['handle'])}")

    # --- end screen: headline, two video boxes + a subscribe circle for YouTube's end-screen elements
    if total - outro_at >= 3:
        a = outro_at
        parts = [p.strip() for p in str(tl.get("outro_text") or "").split("\\N") if p.strip()]
        headline = _ass_text(parts[0] if parts else "THANKS FOR WATCHING")
        sub = _ass_text(" ".join(parts[1:]) if len(parts) > 1 else "")
        size = 92 if _em(headline) * 92 * ASS_EM < 1700 else int(1700 / (_em(headline) * ASS_EM))
        add(4, a + 0.25, total, "H", f"{{\\an8\\pos(960,92)\\fs{size}\\c{w_}\\bord6\\3c&H000000&\\shad8\\4c{p_}"
                                     f"\\fscx125\\fscy125\\t(0,220,\\fscx100\\fscy100)\\fad(200,0)}}{headline}")
        for i, (bx, cap) in enumerate(((150, "WATCH NEXT"), (1130, "MORE GTA 6"))):
            d = 0.45 + 0.15 * i
            add(3, a + d, total, "D", f"{{\\an7\\pos({bx},300)\\p1\\c{k_}\\alpha&H50&\\bord6\\3c{w_}\\fad(250,0)}}"
                                      f"{_rect(640, 360)}")
            add(4, a + d, total, "B", f"{{\\an1\\pos({bx},288)\\fs38\\fsp4\\c{y_}\\bord3\\3c&H000000&\\fad(250,0)}}{cap}")
        add(3, a + 0.6, total, "D", f"{{\\an7\\pos(830,350)\\p1\\c{k_}\\alpha&H40&\\bord8\\3c{y_}"
                                    f"\\fscx60\\fscy60\\t(0,250,\\fscx100\\fscy100)\\fad(200,0)}}{_circle(130)}")
        add(4, a + 0.7, total, "HB", f"{{\\an8\\pos(960,632)\\fs40\\fax-0.15\\bord12\\3c{p_}\\c{w_}\\fad(200,0)}}"
                                     "SUBSCRIBE")
        if sub:
            ss = 54 if _em(sub) * 54 * ASS_EM < 1600 else int(1600 / (_em(sub) * ASS_EM))
            add(4, a + 0.8, total, "B", f"{{\\an8\\pos(960,740)\\fs{ss}\\c{y_}\\bord4\\3c&H000000&\\fad(250,0)}}{sub}")
        if tl.get("handle"):
            add(4, a + 0.9, total, "B", f"{{\\an8\\pos(960,{825 if sub else 740})\\fs40\\c{w_}\\bord3\\3c&H000000&"
                                        f"\\fad(250,0)}}{_ass_text(tl['handle'])}")
    path.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def _font_file() -> str:
    for q in ("Montserrat:style=Black", "Montserrat:style=ExtraBold", "Montserrat:bold", "DejaVu Sans:bold"):
        res = run(["fc-match", "-f", "%{file}", q], timeout=10)
        if res.returncode == 0 and res.stdout.strip() and ("ontserrat" in res.stdout or "DejaVu" in q):
            return res.stdout.strip()
    return "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _gray(src: Path, t: float, w: int = 160, h: int = 90) -> bytes:
    res = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(src), "-frames:v", "1",
                          "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"], capture_output=True, timeout=60)
    return res.stdout if res.returncode == 0 and len(res.stdout) >= w * h else b""


def _best_frame(src: Path, t: float, dur: float) -> float:
    """Of a few frames around t, the sharpest / most detailed one (a motion-blurred frame makes a weak thumbnail)."""
    best, best_s = t, -1.0
    for dt in (0.0, -0.35, 0.35, -0.7, 0.7):
        tt = min(max(0.0, t + dt), max(0.0, dur - 0.4))
        px = _gray(src, tt)
        if not px:
            continue
        edges = sum(abs(px[i] - px[i + 1]) for i in range(0, len(px) - 1) if (i + 1) % 160) / len(px)
        mean = sum(px) / len(px)
        score = edges * (1.0 if 25 <= mean <= 225 else 0.4) * (1.08 if dt == 0 else 1.0)  # the AI's pick by default
        if score > best_s:
            best, best_s = tt, score
    return best


def _split_lines(text: str, max_lines: int) -> list[str]:
    """Up to max_lines lines with the widest line as narrow as possible (big, stacked thumbnail text)."""
    words = text.split()
    if len(words) <= 1 or _em(text) <= 4.2:
        return [text]
    best = [text]
    best_w = _em(text) * 1.6  # one long line is only best if it is short
    for k in range(2, min(max_lines, len(words)) + 1):
        from itertools import combinations
        for cuts in combinations(range(1, len(words)), k - 1):
            idx = (0, *cuts, len(words))
            lines = [" ".join(words[idx[i]:idx[i + 1]]) for i in range(k)]
            w = max(_em(ln) for ln in lines) * (1 + 0.12 * (k - 2))  # 3 lines are smaller overall
            if w < best_w:
                best, best_w = lines, w
    return best


def thumbnail(src: Path, t: float, text: str, out: Path, cam: list | None = None, variant: int = 0,
              badge: str = "") -> Path | None:
    """1280x720 thumbnail in the brand style: the sharpest frame near the moment, punchy grade + vignette (the
    subject pops), huge stacked text with a black + pink double outline, tilted.
    variant 0: text column left, facecam cut-out big on the right (white frame, shadow, tilt)
    variant 1: mirrored (facecam left, text right) - for YouTube's Test & Compare
    variant 2: no facecam, text across the bottom, a "#1" badge
    Falls back to a plainer version (and then the bare frame) if a filter fails."""
    LONG_DIR.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    dur = info["duration"] or 0
    t = min(max(0.0, t), max(0.0, dur - 0.5))  # the AI's "best frame" can point past the end of the clip
    t = _best_frame(src, t, dur)
    font = _font_file().replace("'", "")
    text = re.sub(r"\s+", " ", text).strip() or "GTA 6"
    W0, H0 = info["width"] or 1920, info["height"] or 1080
    side = "right" if variant == 0 else "left"
    use_cam = bool(cam) and variant in (0, 1)
    cam_w = cam_h = 0
    if use_cam:
        cx, cy, cw, ch = cam
        aspect = (cw * W0) / max(1.0, ch * H0)
        cam_h = 400
        cam_w = int(cam_h * aspect)
        if cam_w > 560:
            cam_w, cam_h = 560, int(560 / aspect)
        cam_w, cam_h = cam_w // 2 * 2, cam_h // 2 * 2
    # --- text layout
    if variant == 2:
        lines = _split_lines(text, 2)
        col_x, col_w = 40, 1200
    else:
        lines = _split_lines(text, 3)
        col_w = (1280 - cam_w - 130) if use_cam else 740
        col_x = 50 if side == "right" else 1280 - 50 - col_w
    sizes = [min(210.0, col_w / max(0.5, _em(ln))) for ln in lines]
    gaps = [max(16.0, s * 0.16) for s in sizes]
    block_h = sum(s * 0.72 for s in sizes) + sum(gaps[1:])
    max_h = 330 if variant == 2 else 560
    if block_h > max_h:
        k = max_h / block_h
        sizes, gaps = [s * k for s in sizes], [g * k for g in gaps]
        block_h = max_h
    top = (720 - 46 - block_h) if variant == 2 else (360 - block_h / 2 + 10)
    colours = [YELLOW, WHITE] if variant == 2 else [WHITE, YELLOW, WHITE] if variant == 0 else [YELLOW, WHITE, YELLOW]
    draw, y = [], top
    for i, (ln, s) in enumerate(zip(lines, sizes)):
        y += (gaps[i] if i else 0) + s * 0.72  # baseline (caps are ~0.72 em tall)
        tf = LONG_DIR / f"thumb_{out.stem}_{i}.txt"
        tf.write_text(ln, encoding="utf-8")
        bw = max(6, int(s / 13))
        if variant == 2:
            x = "(w-text_w)/2"
        elif side == "right":
            x = f"{col_x}"
        else:
            x = f"{col_x + col_w}-text_w"
        base = f"drawtext=fontfile='{font}':textfile='{tf}':fontsize={s:.0f}:x={x}:y={y:.0f}-ascent"
        draw.append(f"{base}:fontcolor={_ffc(PINK)}:borderw={bw + max(6, int(s / 16))}:bordercolor={_ffc(PINK)}")
        draw.append(f"{base}:fontcolor={_ffc(colours[i % len(colours)])}:borderw={bw}:bordercolor=black")
    tilt = -0.045 if variant != 1 else 0.04
    txt = (f"color=c=black@0.0:s=1280x720:d=1,format=rgba,{','.join(draw)},"
           f"rotate={tilt}:c=none[txt]")
    grade = ("scale=1434:806:force_original_aspect_ratio=increase,crop=1280:720,"
             "eq=contrast=1.16:saturation=1.5:brightness=0.02,unsharp=5:5:0.8:5:5:0.0,vignette=angle=PI/4.4")
    if variant == 2:  # dark at the bottom, behind the text
        alpha = "clip(235*(Y-H*0.38)/(H*0.62),0,235)"
    elif side == "right":  # text on the left
        alpha = f"clip(225*(1-X/(W*{(col_x + col_w + 60) / 1280:.2f})),0,225)"
    else:
        alpha = f"clip(225*(X-W*{(col_x - 60) / 1280:.2f})/(W*{(1280 - col_x + 60) / 1280:.2f}),0,225)"
    r, g, b = INK
    grad = f"color=c=black:s=1280x720:d=1,format=rgba,geq=r={r}:g={g}:b={b}:a='{alpha}'[grad]"
    parts = ["[0:v]split=2[m][c]" if use_cam else "[0:v]null[m]", f"[m]{grade}[bg0]", grad,
             "[bg0][grad]overlay=0:0[bg1]"]
    last = "bg1"
    if use_cam:  # the streamer's reaction: cut out of the webcam area, white frame, drop shadow, tilted
        cx, cy, cw, ch = cam
        rot = 0.05 if side == "right" else -0.05
        px = 1280 - cam_w - 60 if side == "right" else 50
        py = (720 - cam_h) // 2 - 10
        parts += [f"[c]crop=iw*{cw:.3f}:ih*{ch:.3f}:iw*{cx:.3f}:ih*{cy:.3f},scale={cam_w}:{cam_h},"
                  f"eq=contrast=1.1:saturation=1.3,pad=iw+20:ih+20:10:10:color=white,format=rgba,"
                  f"rotate={rot}:c=none:ow=rotw({rot}):oh=roth({rot}),split=2[cam1][cam2]",
                  "[cam2]colorchannelmixer=rr=0:gg=0:bb=0:aa=0.6,boxblur=10:2[sh]",
                  f"[bg1][sh]overlay={px + 14}:{py + 16}[bg2]", f"[bg2][cam1]overlay={px}:{py}[bg3]"]
        last = "bg3"
    parts.append(txt)
    parts.append(f"[{last}][txt]overlay=0:0[bg4]")
    last = "bg4"
    if badge:  # "#1" corner badge
        bf = LONG_DIR / f"thumb_{out.stem}_badge.txt"
        bf.write_text(badge, encoding="utf-8")
        parts.append(f"[{last}]drawtext=fontfile='{font}':textfile='{bf}':fontsize=96:fontcolor=white:"
                     f"box=1:boxcolor={_ffc(PINK)}:boxborderw=18:x=44:y=40:shadowx=6:shadowy=6:"
                     f"shadowcolor=black@0.6[bg5]")
        last = "bg5"
    parts.append(f"[{last}]format=yuvj420p[out]")
    full = ";".join(parts)
    # plainer fallbacks: no tilt / cam / gradient trouble -> simple text; then the bare graded frame
    simple = (f"[0:v]{grade}," + ",".join(draw[1::2]) + ",format=yuvj420p[out]")
    plain = f"[0:v]{grade},format=yuvj420p[out]"
    for fc in (full, simple, plain):
        out.unlink(missing_ok=True)
        res = run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(src), "-filter_complex", fc, "-map", "[out]",
                   "-frames:v", "1", "-q:v", "2", str(out)], timeout=90)
        if res.returncode == 0 and out.exists() and out.stat().st_size > 1000:
            break
        log.warning("[long] thumbnail attempt failed (t=%.1f of %.1fs): %.300s", t, dur, res.stderr[-300:])
    for f in LONG_DIR.glob(f"thumb_{out.stem}_*.txt"):
        f.unlink(missing_ok=True)
    return out if out.exists() and out.stat().st_size > 1000 else None


def _chapters(cards: list[dict], guide: bool) -> list[str]:
    """Description chapters. YouTube shows chapters only if the first is at 0:00 and each one is 10 s+:
    a short cold open joins the first chapter instead of being an 'Intro' of its own."""
    marks = [(cd["start"], f"#{cd['n']} {cd['label']}" + ("" if guide else f" ({cd['m']['streamer']})"))
             for cd in cards]
    if not marks:
        return []
    if marks[0][0] >= 10:
        marks.insert(0, (0.0, "Intro"))
    else:
        marks[0] = (0.0, marks[0][1])
    return [f"{int(s // 60)}:{int(s % 60):02d} {name}" for s, name in marks]


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
        # guide: the same method from other creators follows as extra "angles" (each with its own lines)
        parts = [m] + (m.get("extra") or [])[: int(c.get("guide_max_angles", 2))] if guide else [m]
        first_ok = False
        out = LONG_DIR / f"seg_{i:02d}.mp4"
        keep = _jump_keep(m["path"], m["a"], m["b"], m["best_t"], freeze_only=guide)
        if not _segment(m["path"], m["a"], m["b"], out, keep=keep, silent_ok=guide,
                        mute=guide and m.get("has_narrator", True)):
            log.warning("[long] could not prepare %s - dropped", m["key"])
            continue
        first_ok = True
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
        prepared.append({"m": m, "part": m, "seg": out, "ins": ins, "label": label, "first": True})
        for j, part in enumerate(parts[1:], 1):
            if not first_ok:
                break
            o2 = LONG_DIR / f"seg_{i:02d}_{j}.mp4"
            k2 = _jump_keep(part["path"], part["a"], part["b"], part["best_t"], freeze_only=True)
            if _segment(part["path"], part["a"], part["b"], o2, keep=k2, silent_ok=True,
                        mute=part.get("has_narrator", True)):
                prepared.append({"m": m, "part": part, "seg": o2, "ins": None, "label": "", "first": False})
    firsts = [pr for pr in prepared if pr["first"]]
    if len(firsts) < 3:
        return None
    n, k = len(firsts), 0
    for pr in prepared:  # the narrator says exactly the number the card shows
        if pr["first"]:
            k += 1
        pr["n"] = n - k + 1
        if guide:  # "Number five. Selling Stolen Cars." + the explanation lines (of THIS part's footage)
            body = [_strip_rank(x) for x in pr["part"].get("lines", []) if _strip_rank(x)]
            head = [_ranked(pr["n"], pr["m"]["label"].rstrip(".!? ") + ".")] if pr["first"] else []
            pr["lines"] = head + body
            pr["line"] = pr["lines"][0] if pr["lines"] else ""
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
            muted = pr["part"].get("has_narrator", True)
            floor = (10.0 if pr["first"] else 6.0) if muted else float(c.get("guide_min_item_sec", 25))
            target = min(seg_d, max(need + 0.9, floor))
            if target < seg_d - 0.4:
                _trim(pr["seg"], target)
        # an extra angle without narration over muted footage would be dead air -> leave it out
        prepared = [pr for pr in prepared if pr["first"] or pr["voices"] or not pr["part"].get("has_narrator", True)]
    segs, cards, voiced, inserts, credits = [], [], [], [], []
    tl: dict = {"flashes": [], "cuts": []}
    from video_processor import _music_track
    music = _music_track()
    # 3) cold open. moments: a 3-5 s hook on the #1 moment's punchline (the streamer's own reaction, slow push-in,
    #    no narrator), then a fast "COMING UP" preview of the next best moments under the narrator's intro line
    #    and the title card. guides: the preview of the best items under the intro line (muted footage needs the
    #    voice, so no silent hook).
    cursor = 0.0
    if not guide and c.get("cold_open_hook", True):
        hook = LONG_DIR / "seg_hook.mp4"
        h_len = min(5.0, max(3.0, float(c.get("teaser_sec", 4.0))))
        ta = max(0.0, min(top["best_t"] - h_len * 0.65, top["dur"] - h_len))  # cut right after the peak
        if _segment(top["path"], ta, ta + h_len, hook, push=float(c.get("hook_zoom", 0.12))):
            segs.append(hook)
            cursor = _vdur(hook)
            tl["hook"] = (0.0, cursor, "TODAY'S #1 MOMENT")
            tl["cuts"].append(cursor)
    intro_len = min(10.0, max(3.0, (intro_v["dur"] + 0.8) if intro_v else 3.0))
    pool = [pr["m"] for pr in prepared if pr["first"]][::-1]  # best first
    if tl.get("hook"):
        pool = pool[1:]  # #1 was just shown
    n_cut = max(1, min(len(pool), int(c.get("preview_cuts", 3)), round(intro_len / 1.5)))
    frames = round(intro_len * 30)
    pick = pool[:n_cut][::-1] if guide else pool[:n_cut]  # guides build up to #1
    pieces = []
    for i, m in enumerate(pick):
        ln = (frames // n_cut + (1 if i < frames % n_cut else 0)) / 30
        p = LONG_DIR / f"seg_prev_{i}.mp4"
        # the build-up just before the punchline: the preview must not give the punchline away
        bt = min(max(float(m["best_t"]), m["a"] + ln), m["b"])
        st = max(0.0, min(bt - ln - 0.3, m["dur"] - ln - 0.05))
        if not _segment(m["path"], st, st + ln, p, silent_ok=guide, mute=guide and m.get("has_narrator", True)):
            pieces = []
            break
        pieces.append(p)
    if not pieces:  # fallback: one teaser of #1 long enough for the intro line
        p = LONG_DIR / "seg_teaser.mp4"
        ta = max(0.0, min(top["best_t"] - intro_len / 2, top["dur"] - intro_len))
        if _segment(top["path"], ta, ta + intro_len, p, silent_ok=guide, mute=guide and top.get("has_narrator", True)):
            pieces = [p]
    if pieces:
        start = cursor
        for i, p in enumerate(pieces):
            if i:
                tl["flashes"].append(cursor)
            segs.append(p)
            cursor += _vdur(p)
        tl["coming"] = (start, cursor)
        tl["intro"] = (start + 0.3, cursor)
        if intro_v and intro_v["dur"] + 0.25 <= cursor - start:
            voiced.append({**intro_v, "t": start + 0.25})
    # 4) timeline: card + voice line + moment (+ REPLAY / REACTION insert)
    for pr in prepared:
        m, d = pr["m"], _vdur(pr["seg"])
        if pr["first"]:
            cards.append({"start": cursor, "n": pr["n"], "label": m["label"], "m": m})
            tl["cuts"].append(cursor)
        elif cards:
            tl["flashes"].append(cursor)  # guide: the same method from another creator
        credits.append({"start": cursor, "end": cursor + d, "name": pr["part"]["streamer"],
                        "platform": pr["part"].get("platform", "")})
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
    for i, cd in enumerate(cards):
        cd["end"] = cards[i + 1]["start"] if i + 1 < len(cards) else cursor
    outro_at = cursor
    # 5) end screen: room for YouTube's end-screen elements (two videos + subscribe) - moving, blurred #1
    outro = LONG_DIR / "seg_outro.mp4"
    outro.unlink(missing_ok=True)
    outro_len = float(c.get("outro_sec", 12.0))
    top_dur = probe(Path(top["path"]))["duration"] or 0
    bt = min(max(0.0, float(top["best_t"])), max(0.0, top_dur - 0.5))  # the AI's best_t can be past the end
    if not _end_screen(Path(top["path"]), bt, outro_len, outro, pad=bool(c.get("end_screen_pad", True)) and not music):
        frame = LONG_DIR / "outro.jpg"  # fallback: a blurred still of #1
        frame.unlink(missing_ok=True)
        for ft in (bt, top_dur / 2, 0.5):
            run(["ffmpeg", "-y", "-ss", f"{ft:.2f}", "-i", str(top["path"]), "-frames:v", "1", str(frame)], timeout=60)
            if frame.exists():
                break
        if frame.exists():
            run(["ffmpeg", "-y", "-loop", "1", "-i", str(frame), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                 "-filter_complex", "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
                 f"boxblur=25:2,eq=brightness=-0.3,fps=30,format=yuv420p,trim=duration={outro_len}[v];"
                 f"[1:a]atrim=duration={outro_len}[a]", "-map", "[v]", "-map", "[a]", *ENC, str(outro)], timeout=120)
        else:
            log.warning("[long] could not grab a frame for the end card")
    if outro.exists():
        segs.append(outro)
        tl["cuts"].append(outro_at)
        ov = voice_by_text.get(pkg["outro_line"])
        if ov and ov["dur"] < outro_len - 0.5:
            voiced.append({**ov, "t": outro_at + 0.3})
    total = outro_at + (outro_len if outro.exists() else 0)
    body = LONG_DIR / "body.mp4"
    if not _join(segs, body):
        return None
    total = _vdur(body) or total

    ass = LONG_DIR / f"{clip_id}.ass"
    brand_cfg = load_config().get("brand", {})
    tl.update({"title": pkg["title"], "total": total, "outro_at": outro_at, "cards": cards, "credits": credits,
               "inserts": inserts, "handle": brand_cfg.get("handles", {}).get("youtube", ""),
               "brand": brand_cfg.get("display_name", "") if c.get("intro_sting", True) else "",
               "outro_text": outro_text or "WHICH ONE WAS THE BEST?\\N\\NSUBSCRIBE FOR DAILY GTA 6 MOMENTS",
               "progress": bool(c.get("progress_bar", True))})
    _build_ass(ass, tl)

    inputs, chains, labels = ["-i", str(body)], [], []
    k = 1
    for v in voiced:
        ms = int(v["t"] * 1000)
        inputs += ["-i", str(v["path"])]
        chains.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,volume=1.35,adelay={ms}|{ms}[v{k}]")
        labels.append(f"[v{k}]")
        k += 1
    sfx_labels = []
    # sound design synced with the graphics: whoosh into every swipe, a soft whoosh on flash cuts, a low hit on
    # every number slam, a boom on the cold open and on REPLAY / REACTION
    events = [("whoosh", t - 0.25, 0.5) for t in tl["cuts"]] + [("whoosh", t - 0.2, 0.22) for t in tl["flashes"]] + \
        [("boom", cd["start"] + 0.1, 0.3) for cd in cards] + [("boom", ins["start"], 0.6) for ins in inserts] + \
        ([("boom", 0.05, 0.55)] if c.get("intro_sting", True) else [])
    for kind, at, vol in events:
        f = _sfx(kind)
        if not f:
            continue
        ms = int(max(0.0, at) * 1000)
        inputs += ["-i", str(f)]
        chains.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,volume={vol},adelay={ms}|{ms}[x{k}]")
        sfx_labels.append(f"[x{k}]")
        k += 1
    music_in = ""
    if music:  # background beat (own music/ or CC0 from Freesound), dips under the narration
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        chains.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo,volume={c.get('music_volume', 0.07)},"
                      f"afade=t=in:d=1.0[m{k}]")
        music_in = f"[m{k}]"
        k += 1
    if labels:
        split = "asplit=3[vo1][vo2][vo3];" if music_in else "asplit=2[vo1][vo2];"
        voice = "".join(labels) + f"amix=inputs={len(labels)}:normalize=0:duration=longest,apad[vo];[vo]{split}"
        # the streamer / game sound dips well under the narrator (QA: laughter overlapping the voice)
        duck = "[0:a][vo1]sidechaincompress=threshold=0.02:ratio=8:attack=15:release=450[g];"
        if music_in:
            duck += f"{music_in}[vo3]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=700[mduck];"
            sfx_labels.append("[mduck]")
        mix_in, n_in = "[g][vo2]", 2
    else:
        voice, duck, mix_in, n_in = "", "[0:a]anull[g];", "[g]", 1
        if music_in:
            sfx_labels.append(music_in)
    afc = (";".join(chains) + ";" if chains else "") + voice + duck + \
        f"{mix_in}{''.join(sfx_labels)}amix=inputs={n_in + len(sfx_labels)}:normalize=0:duration=first," \
        "loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.89:level=0,apad[aout]"
    ass_esc = str(ass).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    out = READY_DIR / f"{clip_id}.mp4"
    res = run(["ffmpeg", "-y", "-hide_banner", *inputs, "-filter_complex", f"[0:v]ass='{ass_esc}'[vout];{afc}",
               "-map", "[vout]", "-map", "[aout]", "-t", f"{total:.2f}", "-c:v", "libx264", "-preset",
               str(c.get("preset", "veryfast")), "-crf", str(c.get("crf", 21)), "-profile:v", "high",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", str(out)], timeout=3000)
    if res.returncode != 0 or not out.exists():
        log.error("[long] final render failed: %s", res.stderr[-1200:])
        return None
    # 3 thumbnails for YouTube Studio's "Test & Compare": #1 + facecam, #2 mirrored, #1 with the label + badge
    thumb = thumbnail(top["path"], top["best_t"], pkg["thumbnail_text"], READY_DIR / f"{clip_id}_thumb.jpg",
                      cam=top.get("facecam"), variant=0)
    second = moments[-2] if len(moments) > 1 else top
    thumbnail(second["path"], second["best_t"], pkg["thumbnail_text"], READY_DIR / f"{clip_id}_thumb2.jpg",
              cam=second.get("facecam"), variant=1)
    alt = " ".join(re.sub(r"[^A-Za-z0-9 !?'$]", "", top["label"]).upper().split()[:3]) or pkg["thumbnail_text"]
    thumbnail(top["path"], max(0.0, top["best_t"] - 0.7), alt, READY_DIR / f"{clip_id}_thumb3.jpg", variant=2,
              badge="#1")
    chapters = _chapters(cards, guide)
    if guide:
        credits = [f"#{pr['n']} {pr['m']['label']} - gameplay by {pr['part']['streamer']}: {pr['part']['url']}"
                   for pr in prepared]
    else:
        credits = [f"#{cd['n']} {cd['m']['streamer']} ({cd['m']['platform'].title()}): {cd['m']['url']}" for cd in cards]
    last_voice = max((v["t"] + v["dur"] for v in voiced), default=0.0)
    final = probe(out)["duration"] or 0
    log.info("[long] timeline: last narration ends %.1fs, end screen %.1f-%.1fs, file %.1fs%s", last_voice, outro_at,
             total, final, " - WARNING narration runs past the end" if last_voice > final - 0.2 else "")
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
    rot = [k for k in _cfg().get("rotation", ["moments"]) if k in LONG_KINDS] or ["moments"]
    last = state.get("last_kind")
    nxt = rot[(rot.index(last) + 1) % len(rot)] if last in rot else rot[0]
    fallbacks = [k for k in _cfg().get("fallbacks", ["trend_guide"]) if k in LONG_KINDS]
    order = list(dict.fromkeys([nxt] + rot + fallbacks))
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
    # title (GTA 6 first, the real count), description (2 keyword-rich lines + chapters + credits), tags, comment
    from publisher import build_long_captions
    caps = build_long_captions(pkg, rendered, spec, kind,
                               names=list(dict.fromkeys(m["streamer"] for m in reversed(chosen))) if not spec else None)
    pkg["title"] = caps["youtube_title"]
    # everything needed to upload it by hand (title, description with chapters + credits, tags) next to the video
    (READY_DIR / f"{clip_id}_YOUTUBE_UPLOAD.txt").write_text(
        "TITLE:\n" + caps["youtube_title"] + "\n\nDESCRIPTION:\n" + caps["youtube_description"] + "\n\nTAGS:\n"
        + ", ".join(dict.fromkeys(caps["tags"])) + "\n\nPINNED COMMENT:\n" + caps["first_comment"]
        + "\n\nTHUMBNAIL: " + Path(rendered.get("thumbnail") or "none").name + "\n", encoding="utf-8")
    # strict QA: technical checks, Whisper voice-vs-captions check, AI editor watching the final video
    import qa
    if spec:
        ctx = ("FOOTAGE SOURCE: every clip passed a separate GTA VI authenticity check or comes from streamers "
               "playing GTA VI live (Twitch / Kick GTA VI category) - flag 'not GTA 6' only if you clearly see GTA V "
               "places or characters (Los Santos landmarks, Franklin / Michael / Trevor). "
               f"16:9 countdown GUIDE: {spec['subject']}. Title cards show '#N  item  Gameplay: creator'; our narrator "
               "explains each item over its footage (the original creators' voice-overs are muted on purpose, so "
               "some parts only have narration). INTENTIONAL EDIT (not errors): a fast 'COMING UP' preview of the "
               "best items under the intro line, swipe transitions on cuts, a '#N' card + corner badge + lower-third "
               "credit per item, a progress bar on top, and a ~12 s end screen at the end (the #1 clip slowed and "
               "blurred with a headline and empty boxes reserved for YouTube end-screen elements). Check that each "
               "explanation matches what the footage shows. Narration lines: ")
    else:
        ctx = ("FOOTAGE SOURCE VERIFIED: every clip comes from streamers playing GTA VI live on Twitch / Kick (the "
               "GTA VI category) - it IS GTA VI, do not judge which game it is. "
               "16:9 countdown of GTA 6 livestream moments. Title cards show '#N  label  streamer'; after the top "
               "moments a REPLAY (slow motion) or REACTION (facecam close-up) insert is intentional. INTENTIONAL EDIT "
               "(not errors): it opens with a 3-5 s hook of the #1 moment (tagged TODAY'S #1 MOMENT) and a fast "
               "'COMING UP' preview of upcoming moments, so those clips appear twice on purpose; swipe transitions on "
               "cuts, a corner '#N' badge, lower-third credits and a progress bar; the last ~12 s are an end screen "
               "(the #1 clip slowed and blurred with a headline and empty boxes for YouTube end-screen elements). "
               "Narration lines: ")
    verdict = qa.check(Path(rendered["path"]), "long", lines=rendered.get("placed"), still_ok=rendered.get("still_ok"),
                       context=ctx + " | ".join(f"[{p_['start']:.1f}s] {p_['text']}" for p_ in rendered.get("placed", [])))
    try:
        import lessons
        lessons.record("long", kind, verdict["pass"], verdict["issues"],
                       lines=[p_["text"] for p_ in rendered.get("placed", [])])
    except Exception:  # noqa: BLE001
        pass
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
