"""Viral GTA 6 stream clips from Twitch + Kick, posted with the streamer's REAL voice - no AI voice-over.

Every ~30 min (workflow streams.yml):
  1. find GTA 6 clips created in the last few hours that are gaining views fast (views per hour)
  2. AI watches the best one: real GTA 6? no spoilers? viral enough? -> transcript, hook banner, captions
  3. render: bigger gameplay, big word-by-word subtitles of what the streamer says, hook banner
  4. post to TikTok / Instagram / X (YouTube off by default: its free quota is used by the main schedule)
Caps: max posts per day, min gap between posts, never the same moment twice.
State lives in queue_streams/ (its own Actions cache, separate from the main schedule).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import BASE_DIR, RAW_DIR, READY_DIR, env, load_config, load_json, probe, run, save_json, setup_logging, \
    write_summary

log = logging.getLogger("autopilot")
STATE_FILE = BASE_DIR / "queue_streams" / "stream_clips.json"
FAMOUS_LIVE: set[str] = set()  # big English streams live right now (filled by the Twitch / Kick top lists)


def is_famous(streamer: str, c: dict | None = None) -> bool:
    """Only famous English-speaking streamers: the configured list, X clip accounts, or a top English live
    stream with enough viewers right now."""
    c = c if c is not None else _cfg()
    name = (streamer or "").lower().replace(" ", "")
    known = {x.lower().replace(" ", "") for x in c.get("famous_streamers", []) + c.get("x_accounts", [])}
    return bool(name) and (name in known or name in FAMOUS_LIVE)


def smoothness(path: Path) -> float:
    """Share of repeated frames when played at 30 fps - laggy streams / clips that skip frames look choppy."""
    tot = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
               "stream=nb_read_frames", "-of", "csv=p=0", "-read_intervals", "%+60", str(path)], timeout=120)
    res = run(["ffmpeg", "-hide_banner", "-t", "60", "-i", str(path), "-vf",
               "fps=30,scale=320:-2,mpdecimate=hi=200:lo=64:frac=0.33", "-f", "null", "-"], timeout=180)
    seen = re.findall(r"frame=\s*(\d+)", res.stderr)
    dur = min(60.0, probe(path)["duration"] or 0)
    total30 = dur * 30
    if not seen or total30 < 30:
        return 0.0
    return round(max(0.0, 1 - int(seen[-1]) / total30), 3)


GTA6_RE = re.compile(r"grand theft auto v?i\b|grand theft auto 6|gta ?(vi|6)\b", re.I)


def _cfg() -> dict:
    return load_config().get("stream_clips", {})


def _ts(s: str) -> float:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _clip(platform: str, cid: str, url: str, title: str, streamer: str, views: int, created: float,
          duration: float, language: str = "", media_url: str = "") -> dict:
    age_h = max(0.25, (time.time() - created) / 3600) if created else 24.0
    return {"key": f"{platform}:{cid}", "platform": platform, "id": cid, "url": url, "title": title or "",
            "streamer": streamer, "views": views, "created": created, "duration": duration,
            "language": language, "media_url": media_url, "age_h": round(age_h, 2),
            "velocity": round(views / age_h, 1)}


# --------------------------------------------------------------------------- discovery

def twitch_clips(c: dict) -> list[dict]:
    """Twitch (official API, free TWITCH_CLIENT_ID / SECRET): the most-watched GTA 6 streams LIVE right now ->
    each of those streamers' top clips of the last hours, plus the game's top recent clips overall."""
    import requests
    from source_fetcher import _twitch_headers
    try:
        headers = _twitch_headers()
    except Exception as exc:  # noqa: BLE001
        log.warning("[streams] Twitch login failed: %s", exc)
        return []
    if not headers:
        log.info("[streams] Twitch: TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET not set - skipping Twitch")
        return []
    api = "https://api.twitch.tv/helix"
    start = datetime.fromtimestamp(time.time() - float(c.get("window_hours", 6)) * 3600, timezone.utc)
    since = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    out, seen = [], set()

    gid_box = {}

    def add(items):
        for x in items:
            if x.get("id") in seen:
                continue
            if gid_box.get("id") and x.get("game_id") and str(x["game_id"]) != str(gid_box["id"]):
                continue  # clipped while the streamer was playing something else
            seen.add(x.get("id"))
            out.append(_clip("twitch", x["id"], x["url"], x.get("title", ""), x.get("broadcaster_name", ""),
                             int(x.get("view_count") or 0), _ts(x.get("created_at", "")),
                             float(x.get("duration") or 0), x.get("language", "")))
    try:
        games = requests.get(f"{api}/games", headers=headers, timeout=20,
                             params={"name": c.get("twitch_game", "Grand Theft Auto VI")}).json().get("data", [])
        if not games:
            log.warning("[streams] Twitch: game not found")
            return []
        gid = games[0]["id"]
        gid_box["id"] = gid
        streams = requests.get(f"{api}/streams", headers=headers, timeout=20,
                               params={"game_id": gid, "first": 100, "type": "live"}).json().get("data", [])
        langs = set(c.get("languages", ["en"]))
        top = [st for st in streams if not langs or st.get("language") in langs][: int(c.get("top_streams", 10))]
        FAMOUS_LIVE.update(st["user_name"].lower() for st in top
                           if int(st.get("viewer_count") or 0) >= int(c.get("famous_min_viewers", 5000)))
        FAMOUS_LIVE.update(st.get("user_login", "").lower() for st in top
                           if int(st.get("viewer_count") or 0) >= int(c.get("famous_min_viewers", 5000)))
        log.info("[streams] Twitch top live GTA 6 streams: %s",
                 [f"{st['user_name']} ({st['viewer_count']})" for st in top])
        for st in top:
            add(requests.get(f"{api}/clips", headers=headers, timeout=20, params={
                "broadcaster_id": st["user_id"], "first": 20, "started_at": since}).json().get("data", []))
        add(requests.get(f"{api}/clips", headers=headers, timeout=20, params={
            "game_id": gid, "first": 100, "started_at": since}).json().get("data", []))
    except Exception as exc:  # noqa: BLE001
        log.warning("[streams] Twitch API error: %.200s", exc)
    log.info("[streams] Twitch: %d recent clips from the top GTA 6 streams", len(out))
    return out


def _kick_get(url: str):
    """kick.com sits behind Cloudflare - curl_cffi (installed with yt-dlp[curl-cffi]) looks like Chrome."""
    from curl_cffi import requests as creq
    r = creq.get(url, impersonate="chrome", timeout=25, headers={"Accept": "application/json"})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json()


def _items(data) -> list:
    """Kick answers come as a list, {"data": [...]}, {"clips": [...]} or {"clips": {"data": [...]}}."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("clips", "data", "livestreams"):
            v = data.get(k)
            if isinstance(v, list):
                return v
            if isinstance(v, dict) and isinstance(v.get("data"), list):
                return v["data"]
    return []


def kick_top_channels(c: dict) -> list[str]:
    """Channels streaming GTA 6 on Kick right now, most viewers first (+ the channels listed in config)."""
    slugs, tried = [], []
    for slug in c.get("kick_category_slugs", ["grand-theft-auto-vi"]):
        for url in (f"https://kick.com/stream/livestreams/en?page=1&limit=24&subcategory={slug}&sort=desc",
                    f"https://kick.com/api/v1/subcategories/{slug}/livestreams?limit=24&page=1"):
            try:
                items = _items(_kick_get(url))
            except Exception as exc:  # noqa: BLE001
                tried.append(f"{url.split('kick.com')[1][:55]}: {str(exc)[:40]}")
                continue
            live, other_lang = [], 0
            langs = {x.lower() for x in c.get("kick_languages", ["english", "en"])}
            for it in items:
                ch = it.get("channel") or {}
                slug_ch = ch.get("slug") or (ch.get("user") or {}).get("username") or it.get("slug") or ""
                lang = str(it.get("language") or (it.get("metadata") or {}).get("language") or "").lower()
                if langs and lang and lang not in langs:
                    other_lang += 1
                    continue
                live.append((int(it.get("viewer_count") or it.get("viewers") or 0), slug_ch))
            if other_lang:
                log.info("[streams] Kick: skipped %d non-English live streams", other_lang)
            live = [x for x in sorted(live, reverse=True) if x[1]]
            FAMOUS_LIVE.update(s_.lower() for v, s_ in live if v >= int(c.get("famous_min_viewers", 5000)))
            if live:
                log.info("[streams] Kick top live GTA 6 streams: %s", [f"{s_} ({v})" for v, s_ in live[:10]])
                slugs += [s_ for _, s_ in live[: int(c.get("top_streams", 10))]]
                break
        if slugs:
            break
    if not slugs and tried:
        log.info("[streams] Kick live list unavailable (%s)", "; ".join(tried[:3]))
    return list(dict.fromkeys(slugs + [x.lower() for x in c.get("kick_channels", [])]))


def kick_clips(c: dict) -> list[dict]:
    """Top clips of the last day from the most-watched GTA 6 channels on Kick (public web API, no key)."""
    out, seen, notes = [], set(), []
    max_age = float(c.get("window_hours", 6)) * 3600
    urls = [f"https://kick.com/api/v2/channels/{ch}/clips?sort=view&time=day" for ch in kick_top_channels(c)]
    for url in urls:
        try:
            items = _items(_kick_get(url))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{url.split('kick.com')[1][:50]}: {str(exc)[:30]}")
            continue
        kept = 0
        for x in items:
            if not isinstance(x, dict) or x.get("id") in seen:
                continue
            seen.add(x.get("id"))
            cat = x.get("category") or {}
            cat_name = f"{cat.get('name', '')} {cat.get('slug', '')}" if isinstance(cat, dict) else str(cat)
            if not GTA6_RE.search(cat_name.replace("-", " ")):
                continue
            created = _ts(x.get("created_at", ""))
            if created and time.time() - created > max_age:
                continue
            ch = x.get("channel") or {}
            slug = ch.get("slug") or ch.get("username") or ""
            out.append(_clip("kick", x["id"], f"https://kick.com/{slug}/clips/{x['id']}", x.get("title", ""),
                             ch.get("username") or slug, int(x.get("views") or x.get("view_count") or 0), created,
                             float(x.get("duration") or 0), "", x.get("video_url") or x.get("clip_url") or ""))
            kept += 1
            if kept == 1 and not out[-1]["media_url"]:
                log.info("[streams] Kick clip has no video URL - fields: %s", sorted(x.keys())[:25])
        notes.append(f"{url.split('kick.com')[1][:50]}: {len(items)} clips, {kept} GTA 6 + recent")
    log.info("[streams] Kick: %d recent GTA 6 clips (%s)", len(out), "; ".join(notes[:6]))
    return out


def youtube_vod_peaks(c: dict) -> list[dict]:
    """Recent GTA 6 stream recordings on YouTube - from the big streamers' channels (e.g. IShowSpeed streams on
    YouTube) and from search. YouTube's own "most replayed" graph marks the moment everyone rewound to - the
    viral moment - so we cut ~40 s around that peak."""
    from source_fetcher import _ytdlp_base, blocked_channel
    lists = [f"https://www.youtube.com/{ch.strip('/')}/streams" for ch in c.get("youtube_channels", [])]
    lists += ["https://www.youtube.com/results?search_query=" + q.replace(" ", "+") + "&sp=CAI%253D"
              for q in c.get("youtube_queries", ["GTA 6 stream"])]
    cands, why = [], {}
    for url in lists:
        res = run(_ytdlp_base() + ["--flat-playlist", "-J", "--playlist-end", "15", url], timeout=90)
        try:
            entries = json.loads(res.stdout).get("entries", []) or []
        except (json.JSONDecodeError, AttributeError):
            why["list failed"] = why.get("list failed", 0) + 1
            continue
        from_channel = "/streams" in url
        for e in entries:
            if not e:
                continue
            title = e.get("title", "")
            ch = e.get("channel") or e.get("uploader") or ""
            dur = float(e.get("duration") or 0)
            if e.get("live_status") in ("is_live", "is_upcoming"):
                r = "still live"
            elif not GTA6_RE.search(title):
                r = "no GTA 6 in title"
            elif blocked_channel(ch):
                r = "blocked channel"
            elif dur and not 600 <= dur <= 43200:
                r = "length"
            elif not from_channel and int(e.get("view_count") or 0) < int(c.get("youtube_min_views", 10000)):
                r = "few views"
            else:
                cands.append(e)
                continue
            why[r] = why.get(r, 0) + 1
    cands.sort(key=lambda e: int(e.get("view_count") or 0), reverse=True)
    clips, seen = [], set()
    for e in cands:
        if e["id"] in seen or len(clips) >= int(c.get("youtube_max_videos", 3)):
            continue
        seen.add(e["id"])
        res = run(_ytdlp_base() + ["-J", "--skip-download", f"https://www.youtube.com/watch?v={e['id']}"], timeout=90)
        try:
            meta = json.loads(res.stdout)
        except (json.JSONDecodeError, TypeError):
            why["metadata failed"] = why.get("metadata failed", 0) + 1
            continue
        created = float(meta.get("timestamp") or meta.get("release_timestamp") or 0)
        if created and time.time() - created > float(c.get("youtube_max_age_hours", 72)) * 3600:
            why["too old"] = why.get("too old", 0) + 1
            continue
        peaks = sorted((p for p in (meta.get("heatmap") or []) if p.get("start_time", 0) > 60),
                       key=lambda p: p.get("value", 0), reverse=True)
        if not peaks:
            why["no most-replayed data"] = why.get("no most-replayed data", 0) + 1
            continue
        mid = (float(peaks[0]["start_time"]) + float(peaks[0].get("end_time", peaks[0]["start_time"]))) / 2
        length = float(c.get("youtube_clip_sec", 40))
        start = max(0.0, mid - length * 0.6)
        x = _clip("youtube", f"{e['id']}@{int(start)}", meta.get("webpage_url") or f"https://www.youtube.com/watch?v={e['id']}",
                  meta.get("title", ""), meta.get("channel") or meta.get("uploader") or "",
                  int(meta.get("view_count") or 0), created, length, meta.get("language") or "")
        x["section"] = (int(start), int(start + length))
        x["velocity"] = round(x["velocity"] * float(c.get("youtube_velocity_factor", 0.2)), 1)
        clips.append(x)
    log.info("[streams] YouTube stream recordings: %d with a most-replayed peak (skipped: %s)", len(clips), why)
    return clips


def x_account_clips(c: dict) -> list[dict]:
    """Video posts from clip accounts on X (e.g. @IShowSpeedHQ) - read with the X_COOKIES login,
    cached for a couple of hours so X isn't opened every run."""
    import x_source
    accounts = c.get("x_accounts", [])
    if not accounts:
        return []
    out = []
    for acc, x in x_source.account_clips(load_config(), accounts=accounts, cache_file=STATE_FILE.with_name("x_cache.json"),
                                         cache_hours=float(c.get("x_cache_hours", 2))):
        text = re.sub(r"https?://\S+", "", x.get("text", "")).strip()
        out.append(_clip("x", x["id"], x.get("page_url") or f"https://x.com/{acc}/status/{x['id']}", text, acc,
                         int(x.get("views") or 0), x.get("created") or 0, float(x.get("duration") or 0),
                         media_url=x.get("media_url", "")))
    log.info("[streams] X accounts: %d recent video posts", len(out))
    return out


def _bar(c: dict, clip: dict, key: str, default: float) -> float:
    """Per-platform viral bar: X counts millions of views, Kick clips hundreds."""
    return float((c.get("viral_bar", {}).get(clip["platform"]) or {}).get(key, c.get(key, default)))


def find_viral(c: dict, state: dict) -> list[dict]:
    clips = []
    try:
        clips += x_account_clips(c)
    except Exception as exc:  # noqa: BLE001 - a broken X page must never stop the run
        log.warning("[streams] X accounts failed: %.200s", exc)
    if c.get("twitch", True):
        clips += twitch_clips(c)
    if c.get("kick", True):
        clips += kick_clips(c)
    if c.get("youtube_vods", True):
        clips += youtube_vod_peaks(c)
    posted = state.get("posted", [])
    posted_keys = {p["key"] for p in posted}
    langs = set(c.get("languages", ["en"]))
    good, reasons = [], {}
    for x in clips:
        why = None
        if x["key"] in posted_keys:
            why = "already posted"
        elif not is_famous(x["streamer"], c):
            why = "not a famous English streamer"
        elif x["platform"] == "twitch" and langs and x["language"] and x["language"] not in langs:
            why = "language"
        elif not float(c.get("min_sec", 8)) <= x["duration"] <= float(c.get("max_sec", 90)):
            why = "length"
        elif x["views"] < _bar(c, x, "min_views", 1000) or x["velocity"] < _bar(c, x, "min_views_per_hour", 500):
            why = f"not viral (yet) on {x['platform']}"
        else:
            # same streamer, same moment (clips made within 10 min of a posted one) or posted too recently
            from source_fetcher import blocked_channel
            if blocked_channel(x["streamer"]):
                why = "blocked channel"
            for p in posted:
                if p.get("streamer", "").lower() == x["streamer"].lower() and (
                        abs(p.get("created", 0) - x["created"]) < 600
                        or time.time() - p.get("at", 0) < float(c.get("same_streamer_gap_hours", 3)) * 3600):
                    why = "same streamer / same moment"
                    break
        if why is None:
            good.append(x)
        else:
            reasons[why] = reasons.get(why, 0) + 1
    if reasons:
        top = sorted(clips, key=lambda x: x["velocity"], reverse=True)[:5]
        log.info("[streams] not picked: %s | fastest clips: %s", reasons,
                 [f"{x['platform']}:{x['streamer']} {x['views']}v/{x['age_h']}h={x['velocity']:.0f}/h {x['duration']:.0f}s"
                  for x in top])
    good.sort(key=lambda x: x["velocity"], reverse=True)
    log.info("[streams] %d clips found, %d viral candidates: %s", len(clips), len(good),
             [f"{x['streamer']} {x['views']}v {x['velocity']:.0f}/h" for x in good[:5]])
    return good


# --------------------------------------------------------------------------- AI

PROMPT = """You prepare a viral GTA 6 livestream clip for YouTube Shorts / Instagram Reels / TikTok.
Streamer: {streamer} on {platform}. Clip title: "{title}". Clip length: {duration:.0f} s.
The attached video is the clip. Watch AND listen to all of it.

1. GATE - judge honestly:
   - is_gta6_footage: true only if the game shown is real Grand Theft Auto VI (Leonida / Vice City, GTA VI HUD).
   - is_spoiler: true if it shows story spoilers (story cutscenes, plot reveals, endings, character deaths).
   - is_english: true if the streamer speaks English (or nobody talks). false for any other language.
   - unsafe: true ONLY for nudity, hate speech / slurs, real-world violence or doxxing. In-game violence, crime,
     police chases and swearing are normal GTA content and are NOT unsafe.
   - hype_score 1-10: how funny / wild / shareable the moment AND the reaction are (9-10 = everyone would send
     this to a friend; 5-6 = meh). Be strict.
2. transcript: what people SAY, as short phrases in order: {{"start": s, "end": e, "text": "..."}} - seconds of
   this clip, max 7 words per phrase, the exact words (censor swear words like f*** and s***). [] if nobody talks.
3. trim_start / trim_end: the part worth keeping - cut dead air and slow starts/ends, and keep AT MOST 90 seconds
   (pick the best 90 s if the clip is longer); the whole clip if it's already tight.
4. banner: max 6 words, ALL CAPS, a hook that makes people stay (e.g. "HE WAS NOT READY FOR THIS").
5. youtube_title (max 70 chars, mention GTA 6 and the streamer), caption (1-2 sentences ending with a question),
   x_text (max 200 chars), hashtags (4, include #GTA6). No emojis in the banner.

Return ONLY JSON: {{"is_gta6_footage": bool, "is_spoiler": bool, "is_english": bool, "unsafe": bool, "hype_score": int,
 "scene_summary": str, "transcript": [{{"start": number, "end": number, "text": str}}], "trim_start": number,
 "trim_end": number, "banner": str, "youtube_title": str, "caption": str, "x_text": str, "hashtags": [str]}}"""


def ai_review(path: Path, clip: dict) -> dict | None:
    """Gemini watches + listens (needed for the transcript). No AI that can hear it -> no post."""
    import gemini_writer
    if not env("GEMINI_API_KEY"):
        return None
    prompt = PROMPT.format(streamer=clip["streamer"], platform=clip["platform"].title(),
                           title=clip["title"], duration=probe(path)["duration"])
    try:
        uploaded = gemini_writer._upload_video(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("[streams] video upload to Gemini failed: %s", exc)
        return None
    try:
        return gemini_writer._generate_json([uploaded, prompt], temperature=0.6)
    finally:
        try:
            gemini_writer._client().files.delete(name=uploaded.name)
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- render + post

def subtitle_words(transcript: list, offset: float, length: float) -> list[dict]:
    from video_processor import _even_word_timings
    words = []
    for ph in transcript or []:
        try:
            a, b, text = float(ph["start"]) - offset, float(ph["end"]) - offset, str(ph["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not text or b <= 0 or a >= length:
            continue
        a, b = max(0.0, a), min(length, max(b, a + 0.4))
        words += [{**w, "start": w["start"] + a, "end": w["end"] + a}
                  for w in _even_word_timings(text, b - a, lead=0.0)]
    return words


def produce(clip: dict, dry_run: bool, state: dict) -> dict | None:
    from gemini_writer import _clean_copy
    from highlight import cut_segments
    from publisher import _available, publish_video, routes_for
    from source_fetcher import Candidate, _download
    from video_processor import render

    c = _cfg()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = RAW_DIR / f"stream_{re.sub(r'[^a-zA-Z0-9]', '_', clip['id'])[:40]}.mp4"
    cand = Candidate(key=clip["key"], url=clip["url"], title=clip["title"], uploader=clip["streamer"],
                     provider=f"{clip['platform']}_clip", duration=clip["duration"], kind="stream",
                     media_url=clip.get("media_url") or None)
    log.info("[streams] downloading %s clip by %s (%d views, %.0f/h): %s", clip["platform"], clip["streamer"],
             clip["views"], clip["velocity"], clip["title"][:60])
    a0, b0 = clip.get("section") or (0, int(clip["duration"]) + 1)
    if clip["platform"] == "youtube":
        cand.provider = "youtube"
    if not _download(cand, a0, b0, raw):
        log.warning("[streams] download failed")
        return None
    info = probe(raw)
    if min(info["width"], info["height"]) < int(c.get("min_height", 720)):
        log.info("[streams] skipping: only %dx%d", info["width"], info["height"])
        return None
    rep_frames = smoothness(raw)
    if rep_frames > float(c.get("max_repeated_frames", 0.2)):
        log.info("[streams] ✋ %s by %s is choppy (%.0f%% repeated frames) - skipped", clip["key"], clip["streamer"],
                 rep_frames * 100)
        state.setdefault("rejected", {})[clip["key"]] = time.time()
        raw.unlink(missing_ok=True)
        return None
    ai = ai_review(raw, clip)
    if not ai:
        log.warning("[streams] no AI could watch + hear the clip - not posting it")
        return {"skip": "ai unavailable"}
    reason = None
    if not ai.get("is_gta6_footage", False):
        reason = "not GTA 6"
    elif ai.get("is_spoiler"):
        reason = "story spoiler"
    elif ai.get("is_english") is False:
        reason = "not English"
    elif ai.get("unsafe"):
        reason = "unsafe content"
    elif int(ai.get("hype_score") or 0) < int(c.get("min_hype", 7)):
        reason = f"low hype ({ai.get('hype_score')})"
    if reason:
        log.info("[streams] ✋ rejected %s by %s: %s | %s", clip["key"], clip["streamer"], reason,
                 ai.get("scene_summary", ""))
        state.setdefault("rejected", {})[clip["key"]] = time.time()
        return None
    dur = info["duration"]
    try:
        a = max(0.0, float(ai.get("trim_start") or 0))
        b = min(dur, float(ai.get("trim_end") or dur))
    except (TypeError, ValueError):
        a, b = 0.0, dur
    if b - a < 6:
        a, b = 0.0, dur
    if b - a > float(c.get("max_output_sec", 90)):  # AI ignored the limit: keep the first 90 s of its pick
        b = a + float(c.get("max_output_sec", 90))
    # Whisper: exact word times (subtitles), clean cut points, spoken-language double check
    import transcribe
    heard, meta = transcribe.words(raw)
    if meta.get("language") and meta["language"] != "en" and meta.get("language_probability", 0) >= 0.7 and len(heard) > 5:
        log.info("[streams] ✋ %s by %s: Whisper hears %s, not English", clip["key"], clip["streamer"], meta["language"])
        state.setdefault("rejected", {})[clip["key"]] = time.time()
        raw.unlink(missing_ok=True)
        return None
    if heard:
        a, b = transcribe.smart_cut(raw, a, b, heard, max_len=float(c.get("max_output_sec", 90)))
    cut = cut_segments(raw, RAW_DIR / f"{raw.stem}_cut.mp4", [(a, b)])
    raw.unlink(missing_ok=True)
    if not cut:
        return None
    copy = _clean_copy({**ai, "commentary": []}, b - a)
    from gemini_writer import polish_packaging
    copy.update(polish_packaging(copy, f"{clip['streamer']} on {clip['platform']}: {ai.get('scene_summary', '')} | "
                                       + " ".join(str(p.get("text", "")) for p in ai.get("transcript") or [])[:600]))
    copy["format_tags"] = list(c.get("hashtags", ["#GTA6Stream", "#GTA6Funny"]))
    copy["keywords"] = f"GTA 6 {clip['streamer']} stream funny moment"
    copy["series"] = ""
    if heard:  # word-exact subtitles from Whisper
        words = transcribe.subtitle_words([{**w, "start": w["start"] - a, "end": w["end"] - a}
                                           for w in heard if a - 0.05 <= w["start"] < b])
    else:
        words = subtitle_words(ai.get("transcript"), a, b - a)
    clip_id = "stream_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M") + "_" + \
        hashlib.md5(clip["key"].encode()).hexdigest()[:6]
    rendered = render(cut, clip_id, copy["banner"], [], voiced=[], caption_words=words,
                      game_volume=float(c.get("audio_volume", 1.0)), fg_zoom=float(c.get("zoom", 1.6)))
    if not rendered:
        return None
    credit = f"{clip['streamer']} on {clip['platform'].title()}"
    platforms = [p for p in c.get("platforms", ["tiktok", "instagram", "x"])
                 if dry_run or any(_available(pr, p) for pr in routes_for(p))]
    results = {} if dry_run else publish_video(Path(rendered["path"]), platforms, copy, credit)
    ok = dry_run or any(r.get("ok") for r in results.values())
    record = {"key": clip["key"], "streamer": clip["streamer"], "created": clip["created"], "at": time.time(),
              "url": clip["url"], "views": clip["views"], "velocity": clip["velocity"], "banner": copy["banner"],
              "hype": ai.get("hype_score"), "results": results, "dry_run": dry_run,
              "rendered": rendered["path"], "subtitles": len(words)}
    if ok and not dry_run:
        state.setdefault("posted", []).append(record)
    return record


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=env("DRY_RUN").lower() == "true")
    args = ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))
    return run_once(args.dry_run)


def run_once(dry_run: bool) -> int:
    args = argparse.Namespace(dry_run=dry_run)
    c = _cfg()
    if not c.get("enabled", True):
        log.info("[streams] disabled in config")
        return 0
    READY_DIR.mkdir(parents=True, exist_ok=True)
    state = load_json(STATE_FILE, {"posted": [], "rejected": {}})
    now = time.time()
    state["posted"] = [p for p in state.get("posted", []) if now - p.get("at", 0) < 4 * 86400]
    state["rejected"] = {k: v for k, v in state.get("rejected", {}).items() if now - v < 2 * 86400}
    last24 = [p for p in state["posted"] if now - p.get("at", 0) < 86400]
    gap_min = float(c.get("min_gap_minutes", 90))
    if not args.dry_run:
        if len(last24) >= int(c.get("max_per_day", 4)):
            log.info("[streams] daily cap reached (%d in 24 h) - nothing to do", len(last24))
            return 0
        if last24 and now - max(p["at"] for p in last24) < gap_min * 60:
            log.info("[streams] last stream clip was under %.0f min ago - waiting", gap_min)
            return 0
    candidates = [x for x in find_viral(c, state) if x["key"] not in state["rejected"]]
    result = None
    for clip in candidates[: int(c.get("max_tries", 3))]:
        result = produce(clip, args.dry_run, state)
        if result and result.get("skip"):
            break  # AI down: every clip would fail the same way
        if result:
            break
    save_json(STATE_FILE, state)
    if result and not result.get("skip"):
        res = result.get("results") or {}
        write_summary("\n".join([f"## 🎮 Stream clip{' — DRY RUN' if args.dry_run else ''}",
                                 f"**{result['banner']}** — {result['streamer']} ({result['views']} views, "
                                 f"{result['velocity']:.0f}/h, hype {result['hype']}) [{result['url']}]({result['url']})"]
                                + [f"- {p}: {'✅' if r.get('ok') else '❌'} {r.get('url') or r.get('error') or ''}"
                                   for p, r in res.items()]))
        log.info("[streams] done ✅ %s", {p: r.get("url") or r.get("error") for p, r in res.items()})
    else:
        log.info("[streams] nothing posted this time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
