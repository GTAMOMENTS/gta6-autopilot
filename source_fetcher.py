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

import hashlib
import json
import logging
import math
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from common import (LOCAL_CLIPS_DIR, MANIFEST_FILE, RAW_DIR, WORK_DIR, cookies_to_netscape, env,
                    load_config, load_json, parse_cookies, probe, run, save_json)
from highlight import concat_files, cut_highlight

log = logging.getLogger("autopilot")

DM_API = "https://api.dailymotion.com/videos"
def ytdlp_format() -> str:
    """Quality between `min_height` (default 720p) and 1080p, preferring <=30fps
    (our videos are 30fps - 60fps sources are ~2x bigger for nothing)."""
    h = int(load_config().get("sources", {}).get("min_height", 720))
    q = f"[height>={h}][height<=1080]"
    return (f"bv*{q}[fps<=?30][vcodec!*=av01]+ba/bv*{q}[fps<=?30]+ba/b{q}[fps<=?30]/"
            f"bv*{q}+ba/b{q}")
class YouTubeFormatsLimited(Exception):
    """YouTube served no 720p+ stream for this video, even with the other player clients."""


# When YouTube serves only low-quality streams to one client (it does that for hours at a time),
# other clients usually still get 720p+. Tried in order; the last one goes without the login.
YT_FALLBACK_CLIENTS = [
    ["--extractor-args", "youtube:player_client=tv,web_safari"],
    ["--extractor-args", "youtube:player_client=android_vr,tv_simply", "--no-cookies"],
]

FULL_DOWNLOAD_MAX_SEC = 600      # only fetch whole videos (no section support) when they are short
FULL_DOWNLOAD_MAX_SIZE = "250M"
BOT_BLOCK_MARKERS = ("confirm you", "not a bot", "sign in to confirm", "http error 429", "precondition", "403: forbidden", "403 forbidden")


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


def discover_youtube(cfg: dict, queries: list[str], tf: TitleFilter, kind: str = "gameplay") -> list[Candidate]:
    s = cfg.get("sources", {})
    out: dict[str, Candidate] = {}
    for q in queries:
        # sp=CAI%3D -> YouTube's own "sort by upload date" (newest footage first)
        url = "https://www.youtube.com/results?search_query=" + requests.utils.quote(q) + "&sp=CAI%253D"
        res = run(_ytdlp_base() + ["--flat-playlist", "-J", "--playlist-end", "30", url], timeout=90)
        if res.returncode != 0:
            log.warning("[sources] YouTube search '%s' failed: %s", q, res.stderr[-200:])
            if any(m in res.stderr.lower() for m in BOT_BLOCK_MARKERS):
                break
            continue
        try:
            entries = json.loads(res.stdout).get("entries", []) or []
        except json.JSONDecodeError:
            continue
        for e in entries:
            if not e or not e.get("id"):
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
    log.info("[sources] YouTube (%s): %d usable videos", kind, len(out))
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
         prefer_min_sec: int = 0) -> list[Candidate]:
    srcs = manifest["sources"]
    now = time.time()
    ranked = []
    for c in cands:
        st = srcs.get(c.key, {})
        if st.get("rejected") or st.get("fail_count", 0) >= 2:
            continue
        uses = len(st.get("uses", []))
        max_uses = 1 if c.duration < 600 else (2 if c.duration < 1800 else 4)
        if c.provider == "local":
            max_uses = 1
        if c.provider == "official_trailers":
            max_uses = 999
        if uses >= max_uses:
            continue
        score = c.score + tf.bonus(c.title)
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
    return ranked


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

    first, second = (stream, gameplay) if kind == "stream" else (gameplay, stream)
    ranked = (rank(first, manifest, tf, boost, prefer_min)            # preferred type first
              + rank(second, manifest, tf, boost, prefer_min))
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


def _download(c: Candidate, start: int, end: int, out: Path) -> bool:
    out.unlink(missing_ok=True)
    if c.media_url:
        return _download_media_url(c, start, end, out)
    if c.local_path:
        res = run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", c.local_path,
                   "-c", "copy", str(out)], timeout=300)
        return res.returncode == 0 and _valid_video(out)

    # 1) fast path: only download the section we need
    def section(extra: list[str]):
        return run(_ytdlp_base() + extra + [
            "-f", ytdlp_format(), "--merge-output-format", "mp4",
            "--download-sections", f"*{start}-{end}",
            "-o", str(out), c.url,
        ], timeout=300)
    res = section([])
    if res.returncode == 0 and _valid_video(out):
        return True
    err = res.stderr[-400:]
    if c.provider == "youtube" and "format is not available" in err.lower():
        for extra in YT_FALLBACK_CLIENTS:
            log.info("[sources] YouTube offered no 720p+ stream - retrying with %s", " ".join(extra[1:2]))
            out.unlink(missing_ok=True)
            res = section(extra)
            if res.returncode == 0 and _valid_video(out):
                log.info("[sources] ✅ got it with %s", " ".join(extra[1:2]))
                return True
            err = res.stderr[-400:]
            if "format is not available" not in err.lower():
                break
        if "format is not available" in err.lower():
            raise YouTubeFormatsLimited(err)
    if c.provider == "youtube" and _is_blocked(err):
        raise PermissionError("youtube-blocked")
    if res.returncode == 124:  # too slow - a full download would be even slower, skip this video
        log.warning("[sources] section download too slow, skipping: %s", c.title[:60])
        return False

    # 2) fallback: some HLS sources (Dailymotion with split audio/video) can't be
    #    section-downloaded -> fetch the whole video (if not huge) and cut locally.
    if c.duration and c.duration > FULL_DOWNLOAD_MAX_SEC:
        log.warning("[sources] section download failed and video too long for a full download - skipping")
        return False
    full = out.with_name(out.stem + "_full.mp4")
    full.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    res = run(_ytdlp_base() + ["-f", ytdlp_format(), "--merge-output-format", "mp4",
                               "--max-filesize", FULL_DOWNLOAD_MAX_SIZE, "-o", str(full), c.url], timeout=300)
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
    """For URLs without duration/date (extra urls, youtube), fetch metadata once."""
    if c.local_path:
        c.duration = probe(Path(c.local_path))["duration"]
        return c.duration > 0
    need_date = c.provider == "youtube" and _min_created_ts(cfg)
    need_meta = need_date or (c.provider == "youtube" and c.duration > 300)  # heatmap for long videos
    if c.duration and not need_meta:
        return True
    res = run(_ytdlp_base() + ["-J", "--skip-download", c.url], timeout=90)
    if res.returncode != 0:
        if c.provider == "youtube" and _is_blocked(res.stderr):
            raise PermissionError("youtube-blocked")
        return False
    try:
        meta = json.loads(res.stdout)
    except json.JSONDecodeError:
        return False
    c.duration = float(meta.get("duration") or c.duration or 0)
    c.heatmap = meta.get("heatmap") or None
    c.uploader = c.uploader or meta.get("channel") or meta.get("uploader") or ""
    if meta.get("timestamp"):
        c.created = float(meta["timestamp"])
    elif meta.get("upload_date"):
        c.created = datetime.strptime(meta["upload_date"], "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
    if need_date and c.created and c.created < _min_created_ts(cfg):
        log.info("[sources] skip old YouTube video (%s): %s",
                 datetime.fromtimestamp(c.created, timezone.utc).date(), c.title[:60])
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
    style = s.get("clip_style", "auto")
    youtube_blocked = False
    yt_limited = 0  # YouTube videos that came back without any 720p+ stream this run

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
                    and c.duration >= n_slices * slice_len * 2):
                # feature-tour format: several slices from different parts -> one reel
                ranges, files = [], []
                for k in range(n_slices):
                    a, b = _pick_section(c, manifest, slice_len, avoid=[(x - 20, y + 20) for x, y in ranges])
                    b = min(b if b > a else a + slice_len, int(c.duration))
                    f = RAW_DIR / f"slice{k}_{sec_file.name}"
                    log.info("[sources] downloading slice %d/%d [%ds-%ds] %s", k + 1, n_slices, a, b, c.title[:70])
                    if _download(c, a, b, f):
                        ranges.append((a, b))
                        files.append(f)
                if len(files) < 2:
                    for f in files:
                        f.unlink(missing_ok=True)
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
        # the AI editor (autopilot.py) watches this whole section / reel and picks the best part(s)
        yield ClipSource(candidate=c, clip_path=sec_file, section=(start, end), segments=[],
                         sections=ranges, is_reel=bool(ranges))


def _note_failure(c: Candidate) -> None:
    m = load_manifest()
    st = m["sources"].setdefault(c.key, {"title": c.title, "uses": []})
    st["fail_count"] = st.get("fail_count", 0) + 1
    save_manifest(m)


def mark_rejected(c: Candidate, reason: str) -> None:
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
