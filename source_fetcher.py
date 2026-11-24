"""Discover GTA VI footage from many places and cut a highlight clip from it.

Sources (in priority order, all configurable in config.json -> "sources"):
  1. local     - any .mp4 you drop in the repo's /clips folder (used once each)
  2. extra_urls - any URL yt-dlp understands (YouTube, Dailymotion, Twitch VODs,
                 Streamable, X, Reddit...) that you paste into config.json
  3. dailymotion - Dailymotion public search API (no key, works from GitHub runners)
  4. youtube   - YouTube search through yt-dlp (often bot-blocked on datacenter IPs;
                 set the optional YTDLP_COOKIES secret to make it reliable)
  5. official_trailers - Rockstar trailers, only as a last resort

Search results go through strict title filters (no GTA 5 mods, fan concepts,
leaks, reaction/news videos...) and are ranked. For long videos only a ~3 min
section is downloaded, then highlight.py picks the most intense seconds.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from common import (LOCAL_CLIPS_DIR, MANIFEST_FILE, RAW_DIR, WORK_DIR, cookies_to_netscape, env,
                    load_config, load_json, parse_cookies, probe, run, save_json)
from highlight import concat_files

log = logging.getLogger("autopilot")

DM_API = "https://api.dailymotion.com/videos"
# uploads titled "GTA 6" are often GTA V / mods: footage from these gets the focused GTA 6 check (authentic.py)
AUTHENTICITY_PROVIDERS = ("youtube", "dailymotion", "x_account", "extra")
# footage someone talks over (a YouTuber's commentary, nonstop dialogue) leaves no room for our narrator
TALKY_REASON = "someone already talks over this footage"
CALM_REASON = "nothing happens in this footage"


def ytdlp_format(max_h: int = 1080) -> str:
    """Quality between `min_height` (default 720p) and 1080p, preferring <=30fps
    (our videos are 30fps - 60fps sources are ~2x bigger for nothing)."""
    h = int(load_config().get("sources", {}).get("min_height", 720))
    q = f"[height>={h}][height<={max_h}]"
    return (f"bv*{q}[fps<=?30][vcodec!*=av01]+ba/bv*{q}[fps<=?30]+ba/b{q}[fps<=?30]/"
            f"bv*{q}+ba/b{q}")


class YouTubeFormatsLimited(Exception):
    """YouTube served no 720p+ stream for this video, even with the other player clients."""


class FormatsLimited(Exception):
    """The site (e.g. Dailymotion) lists no 720p+ stream for this video."""


# When YouTube serves only low-quality streams to one client (it does that for hours at a time),
# other clients usually still get 720p+. Tried in order; the last one goes without the login.
YT_FALLBACK_CLIENTS = [
    ["--extractor-args", "youtube:player_client=tv,web_safari"],
    ["--extractor-args", "youtube:player_client=android_vr,tv_simply", "--no-cookies"],
]

FULL_DOWNLOAD_MAX_SEC = 600      # only fetch whole videos (no section support) when they are short
FULL_DOWNLOAD_MAX_SIZE = "250M"
BOT_BLOCK_MARKERS = ("confirm you", "not a bot", "sign in to confirm", "http error 429", "precondition", "403: forbidden", "403 forbidden")
# a YouTube player client that worked after the default one was refused (bot check / no HD) - used for the rest of
# the run, so every later video doesn't fail first
_yt_client: list[str] = []


@dataclass
class Candidate:
    key: str
    url: str
    title: str
    uploader: str
    provider: str
    duration: float = 0.0
    created: float | None = None
    language: str = ""
    views: int = 0
    official: bool = False
    local_path: str | None = None
    score: float = 0.0
    kind: str = "gameplay"          # "gameplay" or "stream" (streamer clips / VODs with reactions)
    heatmap: list | None = None     # YouTube "most replayed" data [{start_time, end_time, value}]
    media_url: str | None = None    # direct video file (X posts)


@dataclass
class ClipSource:
    candidate: Candidate
    clip_path: Path
    section: tuple[int, int]
    segments: list = field(default_factory=list)
    sections: list = field(default_factory=list)   # all slices joined into this reel (tour format)
    is_reel: bool = False

    @property
    def credit(self) -> str:
        c = self.candidate
        if c.official:
            return "Rockstar Games"
        if c.provider in ("twitch_clip", "twitch_vod"):
            return f"{c.uploader} on Twitch" if c.uploader else "Twitch"
        return c.uploader or "original creator"


# --------------------------------------------------------------------------- state

def load_manifest() -> dict:
    data = load_json(MANIFEST_FILE, {})
    data.setdefault("version", 2)
    data.setdefault("sources", {})
    data.setdefault("items", [])
    return data


def save_manifest(data: dict) -> None:
    data["items"] = data["items"][-400:]
    save_json(MANIFEST_FILE, data)


# --------------------------------------------------------------------------- filters

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _term_regex(terms: list[str]) -> re.Pattern | None:
    terms = [t.strip().lower() for t in terms if t.strip()]
    if not terms:
        return None
    return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in terms) + r")(?![a-z0-9])")


class TitleFilter:
    def __init__(self, cfg: dict):
        s = cfg.get("sources", {})
        self.required = _term_regex(s.get("required_title_terms", []))
        self.context = _term_regex(s.get("context_title_terms", []))
        self.blocked = _term_regex(s.get("blocked_title_terms", []) + cfg.get("blocked_keywords", []))
        self.preferred = {t.lower(): w for t, w in s.get("preferred_title_terms", {}).items()}
        self.strong = _term_regex(s.get("strong_gameplay_terms", []))

    def check(self, title: str) -> tuple[bool, str]:
        t = _norm(title)
        if self.required and not self.required.search(t):
            return False, "not GTA VI"
        if self.blocked and (m := self.blocked.search(t)):
            return False, f"blocked term '{m.group(1)}'"
        if self.context and not self.context.search(t):
            return False, "no gameplay context"
        return True, ""

    def language_ok(self, title: str, language: str) -> bool:
        """Non-English uploads are only OK when the title clearly says it is raw gameplay
        (game audio is fine in any language; foreign news/talk shows are not)."""
        if language in ("", "en") or not self.strong:
            return True
        return bool(self.strong.search(_norm(title)))

    def bonus(self, title: str) -> float:
        t = _norm(title)
        return sum(w for term, w in self.preferred.items() if term in t)


# --------------------------------------------------------------------------- discovery

def _launch_ts(cfg: dict) -> float | None:
    s = cfg.get("sources", {})
    if not s.get("only_after_launch", True):
        return None
    try:
        return datetime.fromisoformat(s.get("launch_date", "2026-11-19")).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _min_created_ts(cfg: dict) -> float | None:
    """Only recent uploads: now - max_age_days (and never before launch if only_after_launch)."""
    s = cfg.get("sources", {})
    stamps = []
    if s.get("max_age_days"):
        stamps.append(time.time() - float(s["max_age_days"]) * 86400)
    if (lt := _launch_ts(cfg)):
        stamps.append(lt)
    return max(stamps) if stamps else None


def discover_local() -> list[Candidate]:
    out = []
    if LOCAL_CLIPS_DIR.exists():
        for mp4 in sorted(LOCAL_CLIPS_DIR.glob("*.mp4")):
            key = "local:" + hashlib.md5(f"{mp4.name}:{mp4.stat().st_size}".encode()).hexdigest()[:12]
            out.append(Candidate(key=key, url=str(mp4), title=mp4.stem.replace("_", " "),
                                 uploader="", provider="local", local_path=str(mp4), score=100))
    return out


def discover_extra_urls(cfg: dict) -> list[Candidate]:
    out = []
    for entry in cfg.get("sources", {}).get("extra_urls", []):
        if isinstance(entry, str):
            entry = {"url": entry}
        url = entry.get("url", "").strip()
        if not url:
            continue
        key = "url:" + hashlib.md5(url.encode()).hexdigest()[:12]
        out.append(Candidate(key=key, url=url, title=entry.get("title", "GTA VI gameplay"),
                             uploader=entry.get("credit", ""), provider="extra",
                             official=bool(entry.get("official")), score=50))
    return out


def discover_dailymotion(cfg: dict, queries: list[str], tf: TitleFilter) -> list[Candidate]:
    s = cfg.get("sources", {})
    after = _min_created_ts(cfg)
    out: dict[str, Candidate] = {}
    for q in queries:
        for sort in ("relevance", "recent"):
            params = {
                "search": q,
                "fields": "id,title,duration,owner.screenname,created_time,language,views_total,url,available_formats",
                "limit": 50, "sort": sort, "flags": "no_live,exportable",
                "longer_than": max(1, int(s.get("min_duration_sec", 50)) // 60),
            }
            if after:
                params["created_after"] = int(after)
            try:
                r = requests.get(DM_API, params=params, timeout=20)
                r.raise_for_status()
                items = r.json().get("list", [])
            except Exception as exc:  # noqa: BLE001
                log.warning("[sources] Dailymotion search '%s' failed: %s", q, exc)
                continue
            for v in items:
                if after and v.get("created_time") and float(v["created_time"]) < after:
                    continue  # the API filter is not trusted alone: older "GTA 6" uploads are fakes
                ok, why = tf.check(v.get("title", ""))
                ok = ok and tf.language_ok(v.get("title", ""), v.get("language") or "")
                dur = float(v.get("duration") or 0)
                if not ok or not (s.get("min_duration_sec", 50) <= dur <= s.get("max_duration_sec", 14400)):
                    continue
                key = f"dm:{v['id']}"
                fmts = v.get("available_formats") or []
                if not ({"hd720", "hd1080", "hd1440", "uhd2160"} & set(fmts)):
                    continue  # no 720p+ version -> skip (min quality 720p)
                quality = 1.5 if "hd1080" in fmts else 0.8
                out[key] = Candidate(score=quality,
                    key=key, url=v.get("url") or f"https://www.dailymotion.com/video/{v['id']}",
                    title=v.get("title", ""), uploader=v.get("owner.screenname", ""),
                    provider="dailymotion", duration=dur, created=v.get("created_time"),
                    language=v.get("language") or "", views=int(v.get("views_total") or 0),
                )
    log.info("[sources] Dailymotion: %d usable videos", len(out))
    return list(out.values())


def _ytdlp_base() -> list[str]:
    cmd = ["yt-dlp", "--no-warnings", "--no-playlist", "--socket-timeout", "45", "--retries", "3",
           "--extractor-retries", "3", "--concurrent-fragments", "4"]
    # YouTube (especially when logged in with cookies) needs a JavaScript engine to unlock the
    # video streams; Node.js is preinstalled on GitHub runners but yt-dlp only uses Deno by default.
    if shutil.which("node"):
        cmd += ["--js-runtimes", "node"]
    cookies = WORK_DIR / "yt_cookies.txt"
    if env("YTDLP_COOKIES") and not cookies.exists():
        parsed = parse_cookies(env("YTDLP_COOKIES"), ".youtube.com")
        if parsed:
            cookies.parent.mkdir(parents=True, exist_ok=True)
            cookies.write_text(cookies_to_netscape(parsed), encoding="utf-8")
        else:
            log.warning("[sources] YTDLP_COOKIES is set but could not be read")
    if cookies.exists():
        cmd += ["--cookies", str(cookies)]
    return cmd


def blocked_channel(name: str, cfg: dict | None = None) -> bool:
    """News sites / official accounts: their footage gets copyright-claimed fast - never use it."""
    blocked = (cfg or load_config()).get("sources", {}).get("blocked_channels", [])
    low = (name or "").lower().strip()
    return bool(low) and any(b.lower() == low or (len(b) > 4 and b.lower() in low) for b in blocked)


def _youtube_search(q: str, n: int = 30) -> tuple[list[dict], str]:
    """One YouTube search (newest first) -> (entries, error text). Entries carry an approximate upload time
    ("3 days ago") so clearly old videos are dropped before any download / date lookup."""
    # sp=CAI%3D -> YouTube's own "sort by upload date" (newest footage first)
    url = "https://www.youtube.com/results?search_query=" + requests.utils.quote(q) + "&sp=CAI%253D"
    res = run(_ytdlp_base() + ["--extractor-args", "youtubetab:approximate_date", "--flat-playlist", "-J",
                               "--playlist-end", str(n), url], timeout=90)
    if res.returncode != 0:
        return [], res.stderr[-300:] or "failed"
    try:
        return [e for e in json.loads(res.stdout).get("entries", []) or [] if e], ""
    except (json.JSONDecodeError, AttributeError):
        return [], "bad answer"


def _clearly_before(ts, cutoff: float | None) -> bool:
    """An approximate upload time ("N days ago") that is surely before the cutoff (1.5 days of margin)."""
    try:
        return bool(cutoff and ts and float(ts) < cutoff - 1.5 * 86400)
    except (TypeError, ValueError):
        return False


def discover_youtube(cfg: dict, queries: list[str], tf: TitleFilter, kind: str = "gameplay") -> list[Candidate]:
    from concurrent.futures import ThreadPoolExecutor
    s = cfg.get("sources", {})
    out: dict[str, Candidate] = {}
    cutoff, old = _min_created_ts(cfg), 0
    queries = list(dict.fromkeys(queries))
    # a few searches at a time (one by one took ~1 min per 8 queries); a bot check stops the rest
    workers = max(1, min(int(s.get("search_workers", 3)), len(queries) or 1))
    blocked = threading.Event()

    def search(q: str):
        if blocked.is_set():
            return q, [], "skipped"
        entries, err = _youtube_search(q)
        if err and any(m in err.lower() for m in BOT_BLOCK_MARKERS):
            blocked.set()
        return q, entries, err
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(search, queries))
    for q, entries, err in results:
        if err:
            if err != "skipped":
                log.warning("[sources] YouTube search '%s' failed: %s", q, err[-200:])
            continue
        for e in entries:
            if not e.get("id"):
                continue
            if _clearly_before(e.get("timestamp"), cutoff):
                old += 1
                continue
            ok, _ = tf.check(e.get("title", ""))
            dur = float(e.get("duration") or 0)
            if not ok or not (s.get("min_duration_sec", 50) <= dur <= s.get("max_duration_sec", 14400)):
                continue
            if e.get("live_status") in ("is_live", "is_upcoming"):
                continue
            if blocked_channel(e.get("channel") or e.get("uploader") or "", cfg):
                continue
            key = f"yt:{e['id']}"
            out[key] = Candidate(
                key=key, url=f"https://www.youtube.com/watch?v={e['id']}", title=e.get("title", ""),
                uploader=e.get("channel") or e.get("uploader") or "", provider="youtube",
                duration=dur, views=int(e.get("view_count") or 0), kind=kind,
            )
    log.info("[sources] YouTube (%s): %d usable videos%s", kind, len(out),
             f" ({old} older than the launch skipped)" if old else "")
    return list(out.values())


_twitch_token: str | None = None


def _twitch_headers() -> dict | None:
    global _twitch_token
    cid, secret = env("TWITCH_CLIENT_ID"), env("TWITCH_CLIENT_SECRET")
    if not (cid and secret):
        return None
    if _twitch_token is None:
        r = requests.post("https://id.twitch.tv/oauth2/token", timeout=20, params={
            "client_id": cid, "client_secret": secret, "grant_type": "client_credentials"})
        r.raise_for_status()
        _twitch_token = r.json()["access_token"]
    return {"Client-Id": cid, "Authorization": f"Bearer {_twitch_token}"}


def _twitch_duration(d: str) -> float:
    """'3h21m12s' -> seconds"""
    total = 0.0
    for num, unit in re.findall(r"(\d+)([hms])", d or ""):
        total += int(num) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def discover_twitch(cfg: dict, tf: "TitleFilter") -> list[Candidate]:
    """Most-viewed recent GTA VI clips + recorded streams (VODs) from Twitch's official API."""
    tcfg = cfg.get("sources", {}).get("twitch", {})
    try:
        headers = _twitch_headers()
    except Exception as exc:  # noqa: BLE001
        log.warning("[sources] Twitch login failed: %s", exc)
        return []
    if not headers:
        return []
    api = "https://api.twitch.tv/helix"
    out: list[Candidate] = []
    try:
        g = requests.get(f"{api}/games", headers=headers, timeout=20,
                         params={"name": tcfg.get("game_name", "Grand Theft Auto VI")}).json().get("data", [])
        if not g:
            log.warning("[sources] Twitch: game not found")
            return []
        game_id = g[0]["id"]
        langs = set(tcfg.get("languages", ["en"]))
        start = datetime.fromtimestamp(time.time() - float(tcfg.get("clip_days", 3)) * 86400, timezone.utc)
        clips = requests.get(f"{api}/clips", headers=headers, timeout=20, params={
            "game_id": game_id, "first": 100, "started_at": start.strftime("%Y-%m-%dT%H:%M:%SZ")}).json().get("data", [])
        for c in clips:
            if langs and c.get("language") not in langs:
                continue
            if int(c.get("view_count") or 0) < int(tcfg.get("min_clip_views", 300)):
                continue
            if float(c.get("duration") or 0) < float(tcfg.get("min_clip_sec", 12)):
                continue
            if tf.blocked and tf.blocked.search(_norm(c.get("title", ""))):
                continue
            out.append(Candidate(
                key=f"twc:{c['id']}", url=c["url"], title=c.get("title") or "Twitch clip",
                uploader=c.get("broadcaster_name", ""), provider="twitch_clip", kind="stream",
                duration=float(c.get("duration") or 0), views=int(c.get("view_count") or 0),
                created=datetime.fromisoformat(c["created_at"].replace("Z", "+00:00")).timestamp(),
                language=c.get("language", ""), score=float(tcfg.get("clip_bonus", 3.0)),
            ))
        if tcfg.get("include_vods", True):
            vods = requests.get(f"{api}/videos", headers=headers, timeout=20, params={
                "game_id": game_id, "first": 30, "sort": "views", "type": "archive",
                "period": tcfg.get("vod_period", "week")}).json().get("data", [])
            for v in vods:
                if langs and v.get("language") not in langs:
                    continue
                out.append(Candidate(
                    key=f"twv:{v['id']}", url=v["url"], title=v.get("title") or "Twitch stream",
                    uploader=v.get("user_name", ""), provider="twitch_vod", kind="stream",
                    duration=_twitch_duration(v.get("duration", "")), views=int(v.get("view_count") or 0),
                    created=datetime.fromisoformat(v["created_at"].replace("Z", "+00:00")).timestamp(),
                    language=v.get("language", ""),
                ))
    except Exception as exc:  # noqa: BLE001
        log.warning("[sources] Twitch API error: %s", exc)
    log.info("[sources] Twitch: %d clips/streams", len(out))
    return out


def discover_x_accounts(cfg: dict, tf: TitleFilter) -> list[Candidate]:
    """Video posts from the X accounts in sources.x_accounts (see x_source.py)."""
    import x_source
    s = cfg.get("sources", {})
    min_ts = _min_created_ts(cfg)
    min_sec, max_sec = float(s.get("x_min_sec", 15)), float(s.get("x_max_sec", 600))
    bonus = float(s.get("x_accounts_bonus", 3.0))
    out = []
    for acc, clip in x_source.account_clips(cfg):
        text = re.sub(r"https?://\S+", "", clip.get("text", "")).strip()
        if tf.blocked and (m := tf.blocked.search(_norm(text))):
            log.debug("[x-source] skip %s: blocked term '%s'", clip["id"], m.group(1))
            continue
        if not min_sec <= clip.get("duration", 0) <= max_sec:
            continue
        if min_ts and clip.get("created") and clip["created"] < min_ts:
            continue
        out.append(Candidate(key=f"x:{clip['id']}", url=f"https://x.com/{acc}/status/{clip['id']}",
                             title=text[:120] or f"GTA 6 clip from {acc}", uploader=f"{acc} on X",
                             provider="x_account", duration=float(clip["duration"]), created=clip.get("created"),
                             views=int(clip.get("views") or 0), score=bonus, kind="gameplay",
                             media_url=clip["media_url"]))
    log.info("[sources] X accounts: %d usable video posts", len(out))
    return out


def discover_trailers(cfg: dict) -> list[Candidate]:
    out = []
    for t in cfg.get("sources", {}).get("official_trailers", []):
        key = "trailer:" + hashlib.md5(t["url"].encode()).hexdigest()[:10]
        out.append(Candidate(key=key, url=t["url"], title=t["title"], uploader="Rockstar Games",
                             provider="official_trailers", duration=float(t.get("duration", 0)),
                             official=True, score=-50))
    return out


def rank(cands: list[Candidate], manifest: dict, tf: TitleFilter, boost: list[str] | None = None,
         prefer_min_sec: int = 0, talky_ok: bool = False) -> list[Candidate]:
    """talky_ok: this run may use footage someone talks over (a format that mutes the original voice)."""
    import ai_cache
    import authentic
    srcs = manifest["sources"]
    talky_after = int(load_config().get("sources", {}).get("talky_channel_after", 2))
    now = time.time()
    cutoff = _launch_ts(load_config())
    ranked, why = [], {}
    for c in cands:
        st = srcs.get(c.key, {})
        if st.get("rejected") or st.get("fail_count", 0) >= 2:
            why["rejected before"] = why.get("rejected before", 0) + 1
            continue
        if cutoff and c.created and c.created < cutoff and not c.official and c.provider != "local":
            why["before launch"] = why.get("before launch", 0) + 1  # "GTA 6" footage from before launch = fake
            continue
        if authentic.known_fake(c.key):
            why["not GTA 6 (checked before)"] = why.get("not GTA 6 (checked before)", 0) + 1
            continue
        ch_status, ch_adjust = authentic.channel_verdict(c.uploader) if c.provider in (
            "youtube", "dailymotion", "x_account", "extra") else ("", 0.0)
        if ch_status == "fake":
            why["GTA V / mod channel"] = why.get("GTA V / mod channel", 0) + 1
            continue
        if (not talky_ok and talky_after > 0 and c.provider in AUTHENTICITY_PROVIDERS
                and ai_cache.talky_count(c.uploader) >= talky_after):
            why["channel talks over its footage"] = why.get("channel talks over its footage", 0) + 1
            continue
        uses = len(st.get("uses", []))
        max_uses = 1 if c.duration < 600 else (2 if c.duration < 1800 else 4)
        if c.provider == "local":
            max_uses = 1
        if c.provider == "official_trailers":
            max_uses = 999
        if uses >= max_uses:
            continue
        score = c.score + tf.bonus(c.title) + ch_adjust  # channels whose videos passed the GTA 6 check first
        score -= unsafe_title_penalty(c.title)  # strip clubs etc. - the AI rejects them as unsafe anyway
        if boost and any(b in _norm(c.title) for b in boost):
            score += 4.0                       # title matches this run's video format
        if prefer_min_sec and c.duration and c.duration < prefer_min_sec:
            score -= 3.0                       # too short to fill a countdown
        score += math.log10(c.views + 10) * 0.6
        if c.language in ("en", ""):
            score += 1.0
        if c.created:
            age_days = (now - c.created) / 86400
            score += max(0.0, 2.0 - age_days / 7)  # fresh footage first
        score -= uses * 2.5                    # prefer never-used videos
        if c.provider == "dailymotion" and c.duration > FULL_DOWNLOAD_MAX_SEC:
            score -= 2.0                       # may need a full download -> slow
        score += random.uniform(0, 2.0)        # variety between runs
        c.score = score
        ranked.append(c)
    ranked.sort(key=lambda c: c.score, reverse=True)
    if why:
        log.info("[sources] skipped %s", why)
    return ranked


@functools.lru_cache(maxsize=4)
def _unsafe_rx(terms: tuple) -> re.Pattern | None:
    return _term_regex(list(terms))


def unsafe_title_penalty(title: str) -> float:
    """Titles that suggest content the AI rejects as unsafe (strip clubs, lap dances...) rank lower: each such
    clip costs a download + an AI review and is never used."""
    rx = _unsafe_rx(tuple(load_config().get("sources", {}).get("unsafe_title_terms", [])))
    if rx and rx.search(_norm(title)):
        return float(load_config().get("sources", {}).get("unsafe_title_penalty", 6.0))
    return 0.0


def _stream_filter(cfg: dict) -> TitleFilter:
    """Like the normal filter, but streams/reactions are allowed (that's the point)."""
    s = cfg.get("sources", {})
    allowed = {t.lower() for t in s.get("stream_allowed_terms", [])}
    c2 = json.loads(json.dumps(cfg))
    c2["sources"]["blocked_title_terms"] = [t for t in s.get("blocked_title_terms", []) if t.lower() not in allowed]
    c2["sources"]["context_title_terms"] = s.get("context_title_terms", []) + s.get("stream_context_terms", [])
    return TitleFilter(c2)


def pick_kind(cfg: dict) -> str:
    mix = cfg.get("sources", {}).get("content_mix", {"gameplay": 0.6, "stream": 0.4})
    kinds, weights = zip(*mix.items())
    return random.choices(kinds, weights=weights, k=1)[0]


def discover(cfg: dict, manifest: dict, kind: str | None = None, fmt: dict | None = None) -> list[Candidate]:
    s = cfg.get("sources", {})
    providers = s.get("providers", ["local", "extra_urls", "twitch", "dailymotion", "youtube"])
    if env("SOURCE_PROVIDERS").strip():  # manual test runs: only these providers
        providers = [p.strip() for p in env("SOURCE_PROVIDERS").split(",") if p.strip()]
        log.info("[sources] providers limited to %s for this run", providers)
    kind = (fmt or {}).get("kind") or kind or pick_kind(cfg)
    log.info("[sources] content type this run: %s%s", kind.upper(),
             f" | format: {fmt['name']}" if fmt else "")
    tf, stf = TitleFilter(cfg), _stream_filter(cfg)
    n = int(s.get("queries_per_run", 4))
    g_all, s_all = s.get("search_queries", ["GTA 6 gameplay"]), s.get("stream_search_queries", [])
    if fmt and fmt.get("queries"):
        # search for footage that fits this run's format (the other content type stays as a backup)
        if kind == "stream":
            s_all = fmt["queries"]
        else:
            g_all = fmt["queries"]
    boost = [t.lower() for t in (fmt or {}).get("title_terms", [])]
    # countdown formats (top 5s) need enough footage for 5 different moments
    prefer_min = int(s.get("tour", {}).get("whole_video_min_sec", 150)) if (fmt or {}).get("countdown") else 0
    g_queries = random.sample(g_all, k=min(len(g_all), n))
    s_queries = random.sample(s_all, k=min(len(s_all), max(2, n // 2)))

    gameplay: list[Candidate] = []
    stream: list[Candidate] = []
    if "local" in providers:
        gameplay += discover_local()
    if "extra_urls" in providers:
        gameplay += discover_extra_urls(cfg)
    if "x_accounts" in providers:
        try:
            gameplay += discover_x_accounts(cfg, tf)
        except Exception as exc:  # noqa: BLE001 - a broken X page must never stop the run
            log.warning("[sources] X accounts failed: %.200s", exc)
    if "twitch" in providers:
        stream += discover_twitch(cfg, stf)
    if "dailymotion" in providers:
        gameplay += discover_dailymotion(cfg, g_queries, tf)
    if "youtube" in providers:
        gameplay += discover_youtube(cfg, g_queries, tf, "gameplay")
        if s_queries:
            stream += discover_youtube(cfg, s_queries, stf, "stream")

    bad = [t.lower() for t in (fmt or {}).get("blocked_title_terms", [])]
    if bad:  # e.g. Money Tips: never "glitch" / "infinite money" videos (patched exploits, fake methods)
        n_before = len(gameplay) + len(stream)
        gameplay = [x for x in gameplay if not any(t in (x.title or "").lower() for t in bad)]
        stream = [x for x in stream if not any(t in (x.title or "").lower() for t in bad)]
        if n_before - len(gameplay) - len(stream):
            log.info("[sources] %d videos skipped for this format's blocked words (%s)",
                     n_before - len(gameplay) - len(stream), ", ".join(bad[:4]))
    first, second = (stream, gameplay) if kind == "stream" else (gameplay, stream)
    talky_ok = bool((fmt or {}).get("allow_narrated"))
    ranked = (rank(first, manifest, tf, boost, prefer_min, talky_ok)  # preferred type first
              + rank(second, manifest, tf, boost, prefer_min, talky_ok))
    if "official_trailers" in providers:
        ranked += rank(discover_trailers(cfg), manifest, tf)
    return ranked


# --------------------------------------------------------------------------- download

def _pick_section(c: Candidate, manifest: dict, length: int, avoid: list | None = None) -> tuple[int, int]:
    dur = int(c.duration or 0)
    if dur <= 0 or dur <= length + 10:
        return 0, dur or length
    used = [(u[0], u[1]) for u in manifest["sources"].get(c.key, {}).get("uses", [])] + list(avoid or [])
    if c.heatmap:
        # YouTube "most replayed": centre the section on the part viewers rewatch the most
        peaks = sorted((p for p in c.heatmap if p.get("start_time", 0) > dur * 0.04),
                       key=lambda p: p.get("value", 0), reverse=True)
        for p in peaks[:12]:
            mid = (float(p["start_time"]) + float(p.get("end_time", p["start_time"]))) / 2
            start = int(min(max(0, mid - length / 2), dur - length))
            if not any(start < b and start + length > a for a, b in used):
                log.info("[sources] using YouTube 'most replayed' peak around %ds", int(mid))
                return start, start + length
    lo = int(dur * 0.06) if dur > 300 else 3           # skip intros / "subscribe" talk
    hi = max(lo + 1, int(dur * 0.93) - length)         # skip outros
    for _ in range(25):
        start = random.randint(lo, hi)
        if not any(start < b and start + length > a for a, b in used):
            return start, start + length
    return random.randint(lo, hi), 0  # end fixed below


def _valid_video(path: Path, min_sec: float = 10) -> bool:
    return path.exists() and probe(path)["duration"] >= min_sec


def _is_blocked(err: str) -> bool:
    e = err.lower()
    return any(m in e for m in BOT_BLOCK_MARKERS) or "403" in e


def _download_media_url(c: Candidate, start: int, end: int, out: Path) -> bool:
    """Direct video file (X posts are short): download it whole, then cut the section."""
    import requests
    full = out.with_name(out.stem + "_full.mp4")
    if ".m3u8" in (c.media_url or ""):  # HLS stream (Kick clips): ffmpeg reads it directly
        res = run(["ffmpeg", "-y", "-user_agent", "Mozilla/5.0", "-i", c.media_url, "-c", "copy",
                   "-bsf:a", "aac_adtstoasc", str(full)], timeout=300)
        if res.returncode != 0 or not full.exists():
            log.warning("[sources] stream download failed: %.200s", res.stderr[-200:])
            full.unlink(missing_ok=True)
            return False
        dur = probe(full)["duration"]
        if start <= 0.5 and end >= dur - 0.5:
            full.rename(out)
        else:
            run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(full), "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "17", "-c:a", "aac", "-b:a", "192k", str(out)], timeout=300)
            full.unlink(missing_ok=True)
        return _valid_video(out, 6)
    try:
        with requests.get(c.media_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(full, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    except Exception as exc:  # noqa: BLE001
        log.warning("[sources] X video download failed: %.200s", exc)
        full.unlink(missing_ok=True)
        return False
    dur = probe(full)["duration"]
    if start <= 0.5 and end >= dur - 0.5:
        full.rename(out)
    else:
        run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(full), "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "17", "-c:a", "aac", "-b:a", "192k", str(out)], timeout=300)
        full.unlink(missing_ok=True)
    return _valid_video(out, 6)


_TIME_RE = re.compile(rb"time=(\d+):(\d\d):(\d\d(?:\.\d+)?)")


def _stop(p: subprocess.Popen) -> None:
    """Stop yt-dlp the way Ctrl+C does: it asks ffmpeg to finish the file, so the part already downloaded stays a
    playable video. Then make sure nothing of it (an ffmpeg child) keeps running in the background."""
    try:
        p.send_signal(signal.SIGINT)
        p.wait(timeout=25)
    except (subprocess.TimeoutExpired, OSError):
        pass
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run_ytdlp(cmd: list[str], timeout: int = 300, section_len: float = 0.0,
               keep_min: float = 0.0) -> subprocess.CompletedProcess:
    """Run yt-dlp in its own process group (a timeout used to leave its ffmpeg running in the background).
    Section downloads are watched through ffmpeg's progress: when the section can't finish before the timeout,
    the download stops early - as soon as `keep_min` seconds are there (kept), or at once if even that can't
    arrive in time. returncode 124 = stopped (stderr says why)."""
    log.debug("RUN %s", " ".join(cmd))
    t0 = time.time()
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))
    st = {"out": bytearray(), "err": bytearray(), "got": 0.0, "first": 0.0, "moved": t0}

    def pump(stream, name: str) -> None:
        tail = b""
        while True:
            chunk = stream.read1(8192) if hasattr(stream, "read1") else stream.read(8192)
            if not chunk:
                break
            buf = st[name]
            buf += chunk
            if len(buf) > 400_000:
                del buf[:-200_000]
            if name == "err":
                found = list(_TIME_RE.finditer(tail + chunk))
                tail = chunk[-40:]
                if found:
                    h, m, s = found[-1].groups()
                    got = int(h) * 3600 + int(m) * 60 + float(s)
                    now = time.time()
                    st["first"] = st["first"] or now
                    if got > st["got"]:
                        st["got"], st["moved"] = got, now

    pumps = [threading.Thread(target=pump, args=(p.stdout, "out"), daemon=True),
             threading.Thread(target=pump, args=(p.stderr, "err"), daemon=True)]
    for t in pumps:
        t.start()
    reason = ""
    while True:
        try:
            p.wait(timeout=2)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.time()
        left = t0 + timeout - now
        if left <= 0:
            reason = f"took longer than {timeout}s"
            break
        if section_len and st["first"]:
            got, el = st["got"], now - st["first"]
            if now - st["moved"] > 60:
                reason = "stalled for 60s"
                break
            if el >= 30 and got < section_len:
                rate = got / el
                if rate > 0 and (section_len - got) / rate > left:  # would not finish in time
                    if got + rate * left < keep_min:
                        reason = f"too slow ({rate:.2f}x real time)"
                        break
                    if got >= min(section_len, max(keep_min, 0.75 * section_len)):
                        reason = f"slow ({rate:.2f}x real time) - stopped with {got:.0f}s downloaded"
                        break
    if reason:
        _stop(p)
    for t in pumps:
        t.join(timeout=10)
    out_s = st["out"].decode("utf-8", "replace")
    err_s = st["err"].decode("utf-8", "replace").replace("\r", "\n")
    if reason:
        err_s += f"\nTIMEOUT: yt-dlp {reason}"
        return subprocess.CompletedProcess(cmd, 124, stdout=out_s, stderr=err_s)
    return subprocess.CompletedProcess(cmd, p.returncode, stdout=out_s, stderr=err_s)


_INFO: dict[str, tuple[dict, Path]] = {}  # url + client -> (yt-dlp info, its JSON file) - extracted once per run
_info_lock = threading.Lock()


def _info(c: Candidate, extra: list[str] | None = None) -> tuple[dict | None, Path | None]:
    """yt-dlp's info for a video (formats, date, heatmap) - fetched once per run; the JSON file lets the downloads
    skip extracting it again (--load-info-json). Raises PermissionError when YouTube refuses (bot check)."""
    extra = list(extra if extra is not None else (_yt_client if c.provider == "youtube" else []))
    k = c.url + "|" + " ".join(extra)
    with _info_lock:
        if k in _INFO:
            return _INFO[k]
    res = run(_ytdlp_base() + extra + ["-J", "--skip-download", c.url], timeout=90)
    if res.returncode != 0:
        if c.provider == "youtube" and _is_blocked(res.stderr):
            raise PermissionError("youtube-blocked")
        log.info("[sources] no video info for '%s': %s", c.title[:50], res.stderr.strip()[-160:])
        return None, None
    try:
        info = json.loads(res.stdout)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(info, dict):
        return None, None
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    f = RAW_DIR / f"info_{hashlib.md5(k.encode()).hexdigest()[:12]}.json"
    f.write_text(json.dumps(info), encoding="utf-8")
    with _info_lock:
        _INFO[k] = (info, f)
    return info, f


_ID_HEIGHT = re.compile(r"(?<![0-9])(\d{3,4})p?(?:@(\d{2,3}))?(?:-\d+)?$")


def _fmt_height(f: dict) -> int:
    """Height of a format - from yt-dlp, else from 'WxH', else from the id (Dailymotion: hls-1080, hls-720@60)."""
    if f.get("height"):
        return int(f["height"])
    m = re.search(r"(\d{3,4})x(\d{3,4})", str(f.get("resolution") or ""))
    if m:
        return int(m.group(2))
    m = _ID_HEIGHT.search(str(f.get("format_id") or ""))
    if m and 100 <= int(m.group(1)) <= 4320:
        return int(m.group(1))
    return 0


def _fmt_fps(f: dict) -> float:
    if f.get("fps"):
        return float(f["fps"])
    m = _ID_HEIGHT.search(str(f.get("format_id") or ""))
    return float(m.group(2)) if m and m.group(2) else 0.0


def _choose_format(info: dict, min_h: int, max_h: int = 1080) -> str | None:
    """Pick the download format ourselves when yt-dlp's filters find nothing (e.g. Dailymotion streams without
    height info): min_h..max_h (else the smallest bigger one), <=30 fps, H.264, with sound. None = no 720p+."""
    fmts = [f for f in info.get("formats") or [] if f.get("format_id") and f.get("url")]
    if fmts and not any(_fmt_height(f) for f in fmts if f.get("vcodec") != "none"):
        return "bv*+ba/b"  # the site gives no sizes at all: best stream (the resolution is checked after download)
    vids = [f for f in fmts if f.get("vcodec") != "none" and _fmt_height(f)]
    auds = [f for f in fmts if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")]
    ok = [f for f in vids if min_h <= _fmt_height(f) <= max_h]
    if not ok:
        bigger = sorted((f for f in vids if _fmt_height(f) > max_h), key=_fmt_height)
        ok = [f for f in bigger if _fmt_height(f) == _fmt_height(bigger[0])] if bigger else []
    if not ok:
        return None

    def score(f: dict) -> tuple:
        fps = _fmt_fps(f)
        muxed = f.get("acodec") != "none"
        return (fps <= 31, _fmt_height(f), "av01" not in str(f.get("vcodec") or ""), muxed or bool(auds),
                str(f.get("protocol") or "").startswith("http") and "m3u8" not in str(f.get("protocol") or ""),
                float(f.get("tbr") or 0))
    best = max(ok, key=score)
    if best.get("acodec") == "none" and auds:
        audio = max(auds, key=lambda a: (float(a.get("abr") or a.get("tbr") or 0)))
        return f"{best['format_id']}+{audio['format_id']}"
    return str(best["format_id"])


def _formats_summary(info: dict, n: int = 12) -> str:
    out = []
    for f in (info.get("formats") or [])[-n:]:
        v = "video" if f.get("vcodec") != "none" else "audio"
        if f.get("acodec") not in (None, "none") and v == "video":
            v = "video+audio"
        h = _fmt_height(f)
        out.append(f"{f.get('format_id')}:{h or '?'}p{('@' + str(int(_fmt_fps(f)))) if _fmt_fps(f) else ''} {v}")
    return ", ".join(out) or "no formats"


def _salvage(out: Path, keep_min: float) -> bool:
    """A section download stopped early (slow site): keep the part that arrived if it is long enough."""
    part = out.with_name(out.name + ".part")
    if not part.exists():
        return False
    got = probe(part)["duration"]
    if got >= keep_min > 0:
        part.replace(out)
        log.info("[sources] slow download - kept the first %.0fs of the section", got)
        return True
    part.unlink(missing_ok=True)
    return False


def _download(c: Candidate, start: int, end: int, out: Path, timeout: int = 300) -> bool:
    out.unlink(missing_ok=True)
    if c.media_url:
        return _download_media_url(c, start, end, out)
    if c.local_path:
        res = run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", c.local_path,
                   "-c", "copy", str(out)], timeout=300)
        return res.returncode == 0 and _valid_video(out)
    s = load_config().get("sources", {})
    min_h = int(s.get("min_height", 720))
    length = max(1.0, float(end - start))
    keep_min = min(length, float(s.get("salvage_min_sec", 45)))
    # the whole (short) clip is wanted, e.g. a Twitch clip: a normal download - much faster than ffmpeg sections
    whole = (start <= 0 and c.duration and end >= c.duration - 1 and c.duration <= FULL_DOWNLOAD_MAX_SEC)

    def attempt(fmt: str, extra: list[str] | None = None, info_file: Path | None = None, budget: int = timeout):
        out.unlink(missing_ok=True)
        out.with_name(out.name + ".part").unlink(missing_ok=True)
        cmd = _ytdlp_base() + list(extra or []) + ["-f", fmt, "--merge-output-format", "mp4"]
        cmd += ["--max-filesize", FULL_DOWNLOAD_MAX_SIZE] if whole else ["--download-sections", f"*{start}-{end}"]
        cmd += ["-o", str(out)] + (["--load-info-json", str(info_file)] if info_file else [c.url])
        res = _run_ytdlp(cmd, timeout=budget, section_len=0 if whole else length, keep_min=keep_min)
        if res.returncode == 0 and _valid_video(out, min(10.0, length * 0.8)):
            return True, res
        if res.returncode == 124 and not whole and _salvage(out, keep_min):
            return True, res
        return False, res

    yt_extra = list(_yt_client) if c.provider == "youtube" else []
    info_file, fmt = None, ytdlp_format()
    known = _INFO.get(c.url + "|" + " ".join(yt_extra))
    if c.provider == "dailymotion" or known:
        # Dailymotion: pick the stream ourselves (its HLS list often has no height info, so yt-dlp's filters found
        # nothing: "Requested format is not available"); YouTube: reuse the info fetched for the date check
        info, info_file = known or _info(c)
        if c.provider == "dailymotion":
            if not info:
                return False
            fmt = _choose_format(info, min_h)
            if not fmt:
                log.info("[sources] Dailymotion has no %dp+ stream for '%s' (%s)", min_h, c.title[:50],
                         _formats_summary(info))
                raise FormatsLimited(_formats_summary(info))
    ok, res = attempt(fmt, yt_extra, info_file)
    if ok:
        return True
    err = res.stderr[-600:]
    if "format is not available" in err.lower():
        if c.provider == "youtube":
            for extra in [x for x in YT_FALLBACK_CLIENTS if x != yt_extra]:
                log.info("[sources] YouTube offered no 720p+ stream - retrying with %s", " ".join(extra[1:2]))
                ok, res = attempt(fmt, extra)
                if ok:
                    log.info("[sources] ✅ got it with %s", " ".join(extra[1:2]))
                    return True
                err = res.stderr[-600:]
                if "format is not available" not in err.lower():
                    break
            if "format is not available" in err.lower():
                raise YouTubeFormatsLimited(err)
        else:
            info, info_file = _info(c)
            fmt2 = _choose_format(info, min_h) if info else None
            log.info("[sources] %s: yt-dlp found no %dp+ format - streams: %s -> %s", c.provider, min_h,
                     _formats_summary(info or {}), fmt2 or "none")
            if not fmt2 and info and c.provider == "dailymotion":
                raise FormatsLimited(_formats_summary(info))
            if fmt2:
                ok, res = attempt(fmt2, None, info_file)
                if ok:
                    return True
                err = res.stderr[-600:]
    if c.provider == "youtube" and _is_blocked(err):
        # bot check / 403 for this client: the other player clients often still work (no login needed)
        for extra in [x for x in YT_FALLBACK_CLIENTS if x != yt_extra]:
            ok, res = attempt(fmt, extra, budget=min(timeout, 240))
            if ok:
                _yt_client[:] = extra
                log.info("[sources] YouTube refused the default client - %s works, using it for this run",
                         " ".join(extra[1:2]))
                return True
        raise PermissionError("youtube-blocked")
    if res.returncode == 124:
        why = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "timeout"
        if c.provider == "youtube" and not whole and ("too slow" in why or "stalled" in why):
            # throttled stream: another client gets another server; 720p is half the data
            extra = next((x for x in YT_FALLBACK_CLIENTS if x != yt_extra), None)
            if extra:
                log.info("[sources] YouTube download %s - one more try at 720p with %s", why.split("yt-dlp ")[-1],
                         " ".join(extra[1:2]))
                ok, res = attempt(ytdlp_format(720), extra, budget=min(timeout, 200))
                if ok:
                    return True
        log.warning("[sources] section download too slow, skipping (%s): %s", why.split("yt-dlp ")[-1], c.title[:60])
        return False

    # fallback: a source that can't be section-downloaded -> fetch the whole video (if not huge) and cut locally.
    if whole:
        log.warning("[sources] download failed (%s): %s", c.provider, err[-300:])
        return False
    if c.duration and c.duration > FULL_DOWNLOAD_MAX_SEC:
        log.warning("[sources] section download failed (%s: %s) and the video is too long for a full download - "
                    "skipping", c.provider, err.strip().splitlines()[-1][:160] if err.strip() else "no error text")
        return False
    full = out.with_name(out.stem + "_full.mp4")
    full.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    res = _run_ytdlp(_ytdlp_base() + yt_extra + ["-f", fmt, "--merge-output-format", "mp4",
                                                 "--max-filesize", FULL_DOWNLOAD_MAX_SIZE, "-o", str(full)]
                     + (["--load-info-json", str(info_file)] if info_file else [c.url]), timeout=timeout)
    if res.returncode != 0 or not _valid_video(full):
        log.warning("[sources] download failed (%s): %s", c.provider, res.stderr[-400:])
        full.unlink(missing_ok=True)
        if c.provider == "youtube" and _is_blocked(res.stderr):
            raise PermissionError("youtube-blocked")
        return False
    run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(full),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-c:a", "aac", "-b:a", "192k", str(out)],
        timeout=600)
    full.unlink(missing_ok=True)
    return _valid_video(out)


def _fill_metadata(c: Candidate, cfg: dict) -> bool:
    """For URLs without duration/date (extra urls, youtube), fetch metadata once (the downloads reuse it)."""
    if c.local_path:
        c.duration = probe(Path(c.local_path))["duration"]
        return c.duration > 0
    need_date = c.provider == "youtube" and _min_created_ts(cfg)
    need_meta = need_date or (c.provider == "youtube" and c.duration > 300)  # heatmap for long videos
    if c.duration and not need_meta:
        return True
    try:
        meta, _ = _info(c)
    except PermissionError:
        if c.provider != "youtube" or _yt_client:
            raise
        meta = None
        for extra in YT_FALLBACK_CLIENTS:  # bot check for the default client: the others often still answer
            try:
                meta, _ = _info(c, extra)
            except PermissionError:
                continue
            if meta:
                _yt_client[:] = extra
                log.info("[sources] YouTube refused the default client - %s works, using it for this run",
                         " ".join(extra[1:2]))
                break
        if not meta:
            raise
    if not meta:
        return False
    c.duration = float(meta.get("duration") or c.duration or 0)
    c.heatmap = meta.get("heatmap") or None
    c.uploader = c.uploader or meta.get("channel") or meta.get("uploader") or ""
    if meta.get("timestamp"):
        c.created = float(meta["timestamp"])
    elif meta.get("upload_date"):
        c.created = datetime.strptime(meta["upload_date"], "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
    if need_date and (not c.created or c.created < _min_created_ts(cfg)):
        log.info("[sources] skip %s YouTube video (%s): %s", "old" if c.created else "undated",
                 datetime.fromtimestamp(c.created, timezone.utc).date() if c.created else "no date", c.title[:60])
        _note_failure(c)
        _note_failure(c)  # 2 strikes -> never retried
        return False
    return c.duration > 0


def iter_clips(target_sec: int, max_attempts: int = 6, skip_providers: set | None = None,
               fmt: dict | None = None):
    """Yield ClipSource objects (best first). Caller calls mark_used / mark_rejected.
    `skip_providers` may be modified by the caller while iterating (e.g. AI check is down)."""
    skip_providers = skip_providers if skip_providers is not None else set()
    cfg = load_config()
    s = cfg.get("sources", {})
    manifest = load_manifest()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    section_len = int(s.get("section_length_sec", 180))
    youtube_blocked = False
    yt_limited = 0  # YouTube videos that came back without any 720p+ stream this run
    dm_limited = 0  # same for Dailymotion

    cands = discover(cfg, manifest, fmt=fmt)
    log.info("[sources] %d ranked candidates; top: %s", len(cands),
             [f"{c.provider}:{c.title[:40]}" for c in cands[:5]])
    attempts = 0
    for c in cands:
        if attempts >= max_attempts:
            break
        if (c.provider == "youtube" and youtube_blocked) or c.provider in skip_providers:
            continue
        attempts += 1
        try:
            if not _fill_metadata(c, cfg):
                _note_failure(c)
                continue
            sec_file = RAW_DIR / f"section_{re.sub(r'[^a-z0-9]', '_', c.key.lower())}.mp4"
            tour = s.get("tour", {})
            n_slices = int(tour.get("slices", 3))
            slice_len = int(tour.get("slice_sec", 100))
            whole_min = int(tour.get("whole_video_min_sec", 150))
            whole_max = int(tour.get("whole_video_max_sec", 330))
            countdown = bool((fmt or {}).get("countdown"))
            if (tour.get("enabled", True) and countdown and c.kind == "gameplay" and c.duration
                    and whole_min <= c.duration < n_slices * slice_len * 2):
                # countdown from a 2.5-11 min video (most "funny moments" compilations): one long section,
                # up to ~5.5 min, becomes the reel the AI picks the 5 moments from
                length = min(whole_max, int(c.duration))
                a, b = _pick_section(c, manifest, length)
                b = min(b if b > a else a + length, int(c.duration))
                log.info("[sources] downloading countdown reel [%ds-%ds] %s", a, b, c.title[:70])
                if not _download(c, a, b, sec_file):
                    _note_failure(c)
                    continue
                ranges = [(a, b)]
                start, end = a, b
            elif (tour.get("enabled", True) and c.kind == "gameplay" and c.duration
                    and not (fmt or {}).get("continuous")
                    and c.duration >= n_slices * slice_len * 2):
                # feature-tour format: several slices from different parts -> one reel
                ranges, files, problem = _download_slices(c, manifest, sec_file, n_slices, slice_len, fmt)
                if problem or len(files) < 2:
                    for f in files:
                        f.unlink(missing_ok=True)
                    if problem:
                        log.info("[gate] ✋ rejected '%s' after its first slice: %s - the other slices are not "
                                 "downloaded", c.title[:60], problem[1])
                        mark_rejected(c, problem[0])
                    else:
                        _note_failure(c)
                    continue
                order = sorted(range(len(ranges)), key=lambda i: ranges[i][0])
                reel = concat_files([files[i] for i in order], sec_file)
                for f in files:
                    f.unlink(missing_ok=True)
                if not reel:
                    _note_failure(c)
                    continue
                ranges = [ranges[i] for i in order]
                start, end = ranges[0][0], ranges[-1][1]
            else:
                ranges = []
                start, end = _pick_section(c, manifest, section_len)
                if end <= start:
                    end = start + section_len
                if c.duration:
                    end = min(end, int(c.duration))
                log.info("[sources] downloading %s [%ds-%ds] %s", c.provider, start, end, c.title[:80])
                if not _download(c, start, end, sec_file):
                    _note_failure(c)
                    continue
        except FormatsLimited as exc:
            dm_limited += 1
            log.warning("[sources] %s has no %sp+ stream for '%s' - not trying it again (%s)", c.provider,
                        s.get("min_height", 720), c.title[:60], str(exc)[:160])
            _note_failure(c)
            _note_failure(c)
            if dm_limited >= 2:
                skip_providers.add(c.provider)
                log.warning("[sources] %s serves no HD streams right now - using the other sources this run", c.provider)
            continue
        except YouTubeFormatsLimited:
            yt_limited += 1
            log.warning("[sources] YouTube served no 720p+ stream for '%s' (all clients)", c.title[:60])
            if yt_limited >= 2:
                youtube_blocked = True
                log.warning("[sources] YouTube is limiting quality right now - using the other sources this run")
            continue
        except PermissionError:
            youtube_blocked = True
            if env("YTDLP_COOKIES"):
                log.warning("[sources] YouTube rejected the YTDLP_COOKIES (expired or logged out?) - skipping "
                            "YouTube for this run. Run the 'check_setup' workflow for details.")
            else:
                log.warning("[sources] YouTube is blocking this runner - skipping YouTube for this run "
                            "(add a YTDLP_COOKIES secret to fix).")
            continue

        min_h = int(s.get("min_height", 720))
        sec_info = probe(sec_file)
        if c.provider != "local" and min(sec_info["width"], sec_info["height"]) < min_h:
            log.info("[sources] skipping %s: only %dx%d (< %dp)", c.title[:50], sec_info["width"],
                     sec_info["height"], min_h)
            sec_file.unlink(missing_ok=True)
            mark_rejected(c, f"low resolution {sec_info['width']}x{sec_info['height']}")
            continue
        if sec_info["duration"] < 15:
            sec_file.unlink(missing_ok=True)
            _note_failure(c)
            continue
        talk = _talks_over(c, sec_file, fmt) if len(ranges) <= 1 else None  # tour slice 1 was checked already
        calm = None if talk else _nothing_happens(c, sec_file, fmt)
        if talk or calm:
            log.info("[gate] ✋ rejected '%s' before any AI watched it: %s", c.title[:60], talk or calm)
            sec_file.unlink(missing_ok=True)
            mark_rejected(c, TALKY_REASON if talk else CALM_REASON)
            continue
        import authentic  # the GTA 6 check remembers its answer for this source + window
        authentic.remember(sec_file, c.key, c.uploader, c.provider,
                           [x for r in ranges for x in r] if ranges else (start, end))
        # the AI editor (autopilot.py) watches this whole section / reel and picks the best part(s)
        yield ClipSource(candidate=c, clip_path=sec_file, section=(start, end), segments=[],
                         sections=ranges, is_reel=bool(ranges))


def _download_slices(c: Candidate, manifest: dict, sec_file: Path, n: int, length: int, fmt: dict | None = None
                     ) -> tuple[list[tuple[int, int]], list[Path], tuple[str, str] | None]:
    """A tour reel's slices: the first one alone - it is checked first (fake? talked over?), a bad video stops
    there - then the others at the same time (one after another they took 1-4 min each;
    sources.tour.parallel_slices). Returns the windows and files of the slices that arrived, plus the problem found
    in slice 1 as (rejection reason, detail) or None. A download error that rules the source out (bot check, no HD
    stream) is raised after every slice file is removed."""
    windows: list[tuple[int, int]] = []
    for _ in range(n):
        a, b = _pick_section(c, manifest, length, avoid=[(x - 20, y + 20) for x, y in windows])
        windows.append((a, min(b if b > a else a + length, int(c.duration))))
    paths = [RAW_DIR / f"slice{k}_{sec_file.name}" for k in range(n)]

    def fetch(k: int) -> bool:
        a, b = windows[k]
        log.info("[sources] downloading slice %d/%d [%ds-%ds] %s", k + 1, n, a, b, c.title[:70])
        return _download(c, a, b, paths[k])

    try:
        got = [0] if fetch(0) else []
        problem = _first_slice_problem(c, paths[0], windows[0], fmt) if got else None
        rest = list(range(1, n))
        if rest and not problem:
            if load_config().get("sources", {}).get("tour", {}).get("parallel_slices", True):
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=len(rest)) as pool:
                    oks = list(pool.map(fetch, rest))
            else:
                oks = [fetch(k) for k in rest]
            got += [k for k, ok in zip(rest, oks) if ok]
    except Exception:
        for f in paths:
            f.unlink(missing_ok=True)
        raise
    return [windows[k] for k in got], [paths[k] for k in got], problem


def _first_slice_problem(c: Candidate, path: Path, window: tuple[int, int],
                         fmt: dict | None) -> tuple[str, str] | None:
    """Why the first slice already rules the video out, as (rejection reason, detail) - or None."""
    fake = _first_slice_fake(c, path, window)
    if fake:
        return "not GTA VI footage", f"not GTA 6 ({fake})"
    talk = _talks_over(c, path, fmt)
    return (TALKY_REASON, talk) if talk else None


def _talks_over(c: Candidate, path: Path, fmt: dict | None) -> str | None:
    """Someone talks over most of this footage (a YouTuber's commentary, nonstop cutscene dialogue) - our narrator
    would have no room, and the AI gate rejects it anyway after a full download + a Gemini watch (a live run lost
    3 videos that way). Whisper runs on the runner: free, no AI quota, ~15 s for the first minute. Only gameplay
    from search-type sources, never when the format mutes the original voice (allow_narrated). '82% speech in the
    first 60 s' if over sources.speech_precheck.max_share, else None (also when Whisper isn't available)."""
    pc = load_config().get("sources", {}).get("speech_precheck", {})
    if (not pc.get("enabled", True) or c.kind != "gameplay" or c.provider not in AUTHENTICITY_PROVIDERS
            or (fmt or {}).get("allow_narrated")):
        return None
    import transcribe
    seconds = float(pc.get("seconds", 60))
    share = transcribe.speech_share(path, seconds)
    if share is None or share <= float(pc.get("max_share", 0.75)):
        return None
    return f"{share:.0%} speech in the first {seconds:.0f} s"


def _nothing_happens(c: Candidate, path: Path, fmt: dict | None) -> str | None:
    """Walking, roaming, calm driving: nothing for a narrator to talk about (owner: "you can't do narration on a
    walking clip"). Counts the sudden loud moments (crash, shot, explosion, landing) - free, local FFmpeg, before any
    AI watches it. Formats whose footage is calm by nature (secrets, things to do: calm_ok) and streams are not
    checked. 'nothing happens: ...' if under sources.action_check.min_per_min, else None (also without sound)."""
    ac = load_config().get("sources", {}).get("action_check", {})
    if (not ac.get("enabled", True) or c.kind != "gameplay" or c.provider not in AUTHENTICITY_PROVIDERS
            or (fmt or {}).get("calm_ok")):
        return None
    from highlight import action_events
    n = action_events(path)
    dur = probe(path)["duration"] or 0
    if n is None or dur <= 0 or n / (dur / 60) >= float(ac.get("min_per_min", 0.5)):
        return None
    return f"nothing happens: {n} sudden loud moment(s) (crash, shot, explosion...) in {dur:.0f} s"


def note_talky(c: Candidate) -> None:
    """Remember the uploader: after sources.talky_channel_after such videos, its videos aren't downloaded."""
    if c.provider in AUTHENTICITY_PROVIDERS and c.uploader:
        import ai_cache
        ai_cache.talky_put(c.uploader, c.key)


def _first_slice_fake(c: Candidate, path: Path, window: tuple[int, int]) -> str | None:
    """A tour reel downloads its slices one after another (1-4 min each). The first slice gets the GTA 6 check
    before the others are fetched: one confidently fake part (GTA V, a mod, another game) already rules the whole
    video out (ai_cache keeps it as fake_src), so the rest of the download would be wasted - a run spent 9 min
    on a video the check then called GTA V (99%). Returns 'GTA V 99%: <evidence>' if confidently fake, else None
    (unsure / no AI answer: download on - the whole reel is checked later as before)."""
    cfg = load_config()
    if (not cfg.get("sources", {}).get("tour", {}).get("check_first_slice", True)
            or not cfg.get("authenticity", {}).get("enabled", True) or c.provider not in AUTHENTICITY_PROVIDERS):
        return None
    import ai_cache
    import authentic
    res = authentic.check([path], [{"key": c.key, "channel": c.uploader, "provider": c.provider,
                                    "section": list(window)}])
    r = res[0] if res else None
    if not r or r.get("error") or ai_cache.classify(r["game"], r["confidence"]) != "fake":
        return None
    return f"{r['game']} {r['confidence'] * 100:.0f}%: {r['evidence'][:80]}"


def _note_failure(c: Candidate) -> None:
    m = load_manifest()
    st = m["sources"].setdefault(c.key, {"title": c.title, "uses": []})
    st["fail_count"] = st.get("fail_count", 0) + 1
    save_manifest(m)


def mark_rejected(c: Candidate, reason: str) -> None:
    if reason == TALKY_REASON:
        note_talky(c)
    m = load_manifest()
    st = m["sources"].setdefault(c.key, {"title": c.title, "uses": []})
    if c.provider in ("official_trailers",):
        return
    if c.duration and c.duration > 1200:
        # long video: only this section was bad, allow another try elsewhere
        st["fail_count"] = st.get("fail_count", 0) + 1
    else:
        st["rejected"] = True
    st["reject_reason"] = reason
    save_manifest(m)


def mark_used(src: ClipSource) -> None:
    m = load_manifest()
    st = m["sources"].setdefault(src.candidate.key, {"title": src.candidate.title, "uses": []})
    for a, b in (src.sections or [src.section]):
        st.setdefault("uses", []).append([a, b, int(time.time())])
    st["url"] = src.candidate.url
    save_manifest(m)
