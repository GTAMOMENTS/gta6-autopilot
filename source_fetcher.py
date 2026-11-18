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
from highlight import cut_highlight

log = logging.getLogger("autopilot")

DM_API = "https://api.dailymotion.com/videos"
YTDLP_FORMAT = "bv*[height<=1080][vcodec!*=av01]+ba/b[height<=1080]/bv*+ba/b"
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


@dataclass
class ClipSource:
    candidate: Candidate
    clip_path: Path
    section: tuple[int, int]
    segments: list = field(default_factory=list)

    @property
    def credit(self) -> str:
        c = self.candidate
        if c.official:
            return "Rockstar Games"
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
    after = _launch_ts(cfg)
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
                quality = 1.5 if "hd1080" in fmts else (0.8 if "hd720" in fmts else -2.0)
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
           "--extractor-retries", "3"]
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


def discover_youtube(cfg: dict, queries: list[str], tf: TitleFilter) -> list[Candidate]:
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
            key = f"yt:{e['id']}"
            out[key] = Candidate(
                key=key, url=f"https://www.youtube.com/watch?v={e['id']}", title=e.get("title", ""),
                uploader=e.get("channel") or e.get("uploader") or "", provider="youtube",
                duration=dur, views=int(e.get("view_count") or 0),
            )
    log.info("[sources] YouTube: %d usable videos", len(out))
    return list(out.values())


def discover_trailers(cfg: dict) -> list[Candidate]:
    out = []
    for t in cfg.get("sources", {}).get("official_trailers", []):
        key = "trailer:" + hashlib.md5(t["url"].encode()).hexdigest()[:10]
        out.append(Candidate(key=key, url=t["url"], title=t["title"], uploader="Rockstar Games",
                             provider="official_trailers", duration=float(t.get("duration", 0)),
                             official=True, score=-50))
    return out


def rank(cands: list[Candidate], manifest: dict, tf: TitleFilter) -> list[Candidate]:
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
        score += math.log10(c.views + 10) * 0.6
        if c.language in ("en", ""):
            score += 1.0
        if c.created:
            age_days = (now - c.created) / 86400
            score += max(0.0, 2.0 - age_days / 7)  # fresh footage first
        score -= uses * 2.5                    # prefer never-used videos
        score += random.uniform(0, 2.0)        # variety between runs
        c.score = score
        ranked.append(c)
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked


def discover(cfg: dict, manifest: dict) -> list[Candidate]:
    s = cfg.get("sources", {})
    providers = s.get("providers", ["local", "extra_urls", "dailymotion", "youtube", "official_trailers"])
    tf = TitleFilter(cfg)
    all_queries = s.get("search_queries", ["GTA 6 gameplay"])
    queries = random.sample(all_queries, k=min(len(all_queries), int(s.get("queries_per_run", 4))))
    log.info("[sources] search queries this run: %s", queries)

    primary: list[Candidate] = []
    if "local" in providers:
        primary += discover_local()
    if "extra_urls" in providers:
        primary += discover_extra_urls(cfg)
    if "dailymotion" in providers:
        primary += discover_dailymotion(cfg, queries, tf)
    if "youtube" in providers:
        primary += discover_youtube(cfg, queries, tf)
    ranked = rank(primary, manifest, tf)
    if "official_trailers" in providers:
        ranked += rank(discover_trailers(cfg), manifest, tf)  # always last
    return ranked


# --------------------------------------------------------------------------- download

def _pick_section(c: Candidate, manifest: dict, length: int) -> tuple[int, int]:
    dur = int(c.duration or 0)
    if dur <= 0 or dur <= length + 10:
        return 0, dur or length
    used = [(u[0], u[1]) for u in manifest["sources"].get(c.key, {}).get("uses", [])]
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


def _download(c: Candidate, start: int, end: int, out: Path) -> bool:
    out.unlink(missing_ok=True)
    if c.local_path:
        res = run(["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", c.local_path,
                   "-c", "copy", str(out)], timeout=300)
        return res.returncode == 0 and _valid_video(out)

    # 1) fast path: only download the section we need
    res = run(_ytdlp_base() + [
        "-f", YTDLP_FORMAT, "--merge-output-format", "mp4",
        "--download-sections", f"*{start}-{end}",
        "-o", str(out), c.url,
    ], timeout=300)
    if res.returncode == 0 and _valid_video(out):
        return True
    err = res.stderr[-400:]
    if c.provider == "youtube" and _is_blocked(err):
        raise PermissionError("youtube-blocked")
    if res.returncode == 124:  # too slow - a full download would be even slower, skip this video
        log.warning("[sources] section download too slow, skipping: %s", c.title[:60])
        return False

    # 2) fallback: some HLS sources (Dailymotion with split audio/video) can't be
    #    section-downloaded -> fetch the whole video (if not huge) and cut locally.
    if c.duration and c.duration > 2400:
        log.warning("[sources] section download failed and video too long for full download: %s", err)
        return False
    full = out.with_name(out.stem + "_full.mp4")
    full.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    res = run(_ytdlp_base() + ["-f", YTDLP_FORMAT, "--merge-output-format", "mp4",
                               "--max-filesize", "900M", "-o", str(full), c.url], timeout=600)
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
    need_date = c.provider == "youtube" and _launch_ts(cfg)
    if c.duration and not need_date:
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
    c.uploader = c.uploader or meta.get("channel") or meta.get("uploader") or ""
    if meta.get("timestamp"):
        c.created = float(meta["timestamp"])
    elif meta.get("upload_date"):
        c.created = datetime.strptime(meta["upload_date"], "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
    if need_date and c.created and c.created < _launch_ts(cfg):
        log.info("[sources] skip pre-launch YouTube video: %s", c.title)
        return False
    return c.duration > 0


def iter_clips(target_sec: int, max_attempts: int = 6, skip_providers: set | None = None):
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

    cands = discover(cfg, manifest)
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
            start, end = _pick_section(c, manifest, section_len)
            if end <= start:
                end = start + section_len
            if c.duration:
                end = min(end, int(c.duration))
            sec_file = RAW_DIR / f"section_{re.sub(r'[^a-z0-9]', '_', c.key.lower())}.mp4"
            log.info("[sources] downloading %s [%ds-%ds] %s", c.provider, start, end, c.title[:80])
            if not _download(c, start, end, sec_file):
                _note_failure(c)
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

        clip_file = RAW_DIR / f"clip_{int(time.time())}.mp4"
        try:
            clip, segments = cut_highlight(sec_file, clip_file, target_sec, style)
        except Exception as exc:  # noqa: BLE001
            log.warning("[sources] highlight cut failed: %s", exc)
            clip, segments = None, []
        sec_file.unlink(missing_ok=True)
        if not clip:
            _note_failure(c)
            continue
        yield ClipSource(candidate=c, clip_path=clip, section=(start, end), segments=segments)


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
    st.setdefault("uses", []).append([src.section[0], src.section[1], int(time.time())])
    st["url"] = src.candidate.url
    save_manifest(m)
