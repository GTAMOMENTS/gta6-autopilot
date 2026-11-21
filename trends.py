"""Trend radar: what GTA 6 players search for and watch RIGHT NOW (all free, no keys needed).

Sources (each one optional - if a site blocks us, the others still work):
  1. YouTube search autocomplete ("gta 6 how to ..." -> the phrases people type most)
  2. this week's most viewed + the most relevant GTA 6 YouTube videos (titles + views) via yt-dlp
  3. Reddit r/GTA6 hot posts (best effort - Reddit often blocks cloud servers)
  4. Twitch: total viewers of English GTA 6 live streams -> "stream health" (is streaming still big?)
An AI turns this into a ranked list of topics: [{topic, angle, queries, money, demand}] saved in
queue/trends.json (refreshed every few hours, so runs don't repeat the work).

Used by:
  - the "trending" Short format (follows whatever is hot), the Money Tip topics, long-video guide topics
  - X text posts (talk about what people talk about)
  - stream health: when GTA 6 streams die down, the stream-clip formats give way to trend formats
"""
from __future__ import annotations

import json
import logging
import re
import time

import requests

from common import STATE_DIR, load_config, load_json, run, save_json

log = logging.getLogger("autopilot")
TRENDS_FILE = STATE_DIR / "trends.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0 Safari/537.36"}
GTA_RE = re.compile(r"\bgta ?(6|vi)\b|grand theft auto (6|vi)\b", re.I)


def _cfg() -> dict:
    return load_config().get("trends", {})


# --------------------------------------------------------------------------- sources

def autocomplete() -> list[str]:
    """Most searched phrases, best first (YouTube's own suggestions + Google's)."""
    seeds = _cfg().get("seeds", ["gta 6", "gta 6 how to", "gta 6 best", "gta 6 money", "gta 6 where",
                                 "gta 6 secret", "gta 6 cars", "gta 6 funny", "gta 6 tips", "gta 6 new"])
    out: list[str] = []
    for seed in seeds:
        for ds in ("yt", ""):
            try:
                r = requests.get("https://suggestqueries.google.com/complete/search", headers=UA, timeout=15,
                                 params={"client": "firefox", "ds": ds, "q": seed, "hl": "en", "gl": "us"})
                r.raise_for_status()
                out += [s for s in r.json()[1] if isinstance(s, str)]
            except Exception as exc:  # noqa: BLE001
                log.info("[trends] autocomplete '%s' failed: %.100s", seed, exc)
                break
    keep = [s.strip().lower() for s in out if GTA_RE.search(s)]
    return list(dict.fromkeys(keep))[:120]


def hot_videos() -> list[dict]:
    """This week's most viewed GTA 6 videos + YouTube's 'relevant' order (which favours what's trending)."""
    from source_fetcher import _ytdlp_base
    urls = {
        "week_top": "https://www.youtube.com/results?search_query=GTA+6&sp=CAMSAggD",  # this week, by views
        "relevant": "https://www.youtube.com/results?search_query=GTA+6",
        "today": "https://www.youtube.com/results?search_query=GTA+6&sp=EgIIAg%253D%253D",  # uploaded today
    }
    seen, out = set(), []
    for name, url in urls.items():
        res = run(_ytdlp_base() + ["--flat-playlist", "-J", "--playlist-end", "40", url], timeout=90)
        if res.returncode != 0:
            log.info("[trends] YouTube %s search failed: %.150s", name, res.stderr[-150:])
            continue
        try:
            entries = json.loads(res.stdout).get("entries") or []
        except json.JSONDecodeError:
            continue
        for e in entries:
            if not e or e.get("id") in seen or not GTA_RE.search(e.get("title") or ""):
                continue
            seen.add(e.get("id"))
            out.append({"title": e.get("title", ""), "views": int(e.get("view_count") or 0), "list": name,
                        "channel": e.get("channel") or e.get("uploader") or "", "duration": e.get("duration")})
    out.sort(key=lambda v: v["views"], reverse=True)
    return out[:80]


def reddit_hot() -> list[str]:
    titles = []
    for sub in _cfg().get("subreddits", ["GTA6", "GTA"]):
        try:
            r = requests.get(f"https://www.reddit.com/r/{sub}/hot.json", params={"limit": 40}, headers=UA, timeout=15)
            r.raise_for_status()
            for p in r.json().get("data", {}).get("children", []):
                d = p.get("data", {})
                if not d.get("stickied") and d.get("score", 0) >= 50:
                    titles.append(f"{d.get('title', '')[:120]} ({d.get('score')} upvotes)")
        except Exception as exc:  # noqa: BLE001
            log.info("[trends] Reddit r/%s unavailable: %.80s", sub, exc)
    return titles[:50]


def twitch_viewers() -> int | None:
    """Total viewers of English GTA 6 live streams right now (None = Twitch not configured / failed)."""
    from source_fetcher import _twitch_headers
    try:
        headers = _twitch_headers()
        if not headers:
            return None
        api = "https://api.twitch.tv/helix"
        game = load_config().get("stream_clips", {}).get("twitch_game", "Grand Theft Auto VI")
        games = requests.get(f"{api}/games", headers=headers, timeout=20, params={"name": game}).json().get("data", [])
        if not games:
            return None
        streams = requests.get(f"{api}/streams", headers=headers, timeout=20,
                               params={"game_id": games[0]["id"], "first": 100, "type": "live"}).json().get("data", [])
        return sum(int(s.get("viewer_count") or 0) for s in streams if s.get("language") in ("en", ""))
    except Exception as exc:  # noqa: BLE001
        log.info("[trends] Twitch viewers unavailable: %.100s", exc)
        return None


# --------------------------------------------------------------------------- AI: data -> topics

PROMPT = """You are the trend analyst of "{brand}", a GTA 6 channel (YouTube Shorts, TikTok, Instagram Reels, X and
long YouTube videos). {context}
Here is what GTA 6 players search for and watch RIGHT NOW:

MOST SEARCHED (YouTube/Google autocomplete, most popular first):
{suggest}

THIS WEEK'S BIGGEST GTA 6 VIDEOS (title - views):
{videos}

REDDIT HOT POSTS:
{reddit}

Find the {n} topics people want most right now that we can show with GTA 6 GAMEPLAY footage.
Rules:
- no story spoilers (no plot, mission endings, character deaths, story twists) and no leaks/rumours as facts
- a topic must be something you can SHOW in open-world gameplay (a place, car, activity, mechanic, money method,
  challenge, secret...), not news or drama, and NEVER settings / graphics options / menus / performance / HDR /
  console or PC specs / prices / release info (nothing to film)
- prefer topics that show up in several sources (search + big videos) and are rising
For each topic:
- topic: 2-6 words ("Fastest Way To Make Money")
- angle: one sentence, what our video shows and why people click
- queries: 3-5 YouTube searches to find footage of it, each starting with "GTA 6" (add "no commentary" to at
  least two of them)
- money: true if it is about making money / getting rich
- demand: 1-10, how strongly the data shows people want this NOW
- evidence: the search phrases / video titles that show it (short)
Return ONLY JSON: {{"topics": [{{"topic": str, "angle": str, "queries": [str], "money": bool, "demand": int,
"evidence": str}}], "summary": "one sentence: what GTA 6 players care about today"}}"""


def _analyse(data: dict) -> dict | None:
    import ai_fallback
    import gemini_writer
    brand = load_config().get("brand", {}).get("display_name", "GTA VI Top Moments")
    vids = "\n".join(f"- {v['title'][:100]} - {v['views']:,}" for v in data["videos"][:50]) or "(none)"
    prompt = PROMPT.format(brand=brand, context=gemini_writer._game_context(), n=int(_cfg().get("topics", 12)),
                           suggest="\n".join(f"- {s}" for s in data["suggest"][:80]) or "(none)", videos=vids,
                           reddit="\n".join(f"- {t}" for t in data["reddit"][:30]) or "(none)")
    out = gemini_writer._generate_json(prompt, temperature=0.4)
    if not out and ai_fallback.available():
        out = ai_fallback.chat_json(prompt, temperature=0.4)
    if not isinstance(out, dict) or not isinstance(out.get("topics"), list):
        return None
    topics = []
    for t in out["topics"]:
        if not isinstance(t, dict) or not t.get("topic"):
            continue
        queries = [q for q in (t.get("queries") or []) if isinstance(q, str) and GTA_RE.search(q)][:5]
        if not queries:
            queries = [f"GTA 6 {t['topic']}", f"GTA 6 {t['topic']} no commentary"]
        try:
            demand = max(1, min(10, int(t.get("demand") or 5)))
        except (TypeError, ValueError):
            demand = 5
        topics.append({"topic": str(t["topic"])[:60], "angle": str(t.get("angle") or "")[:220],
                       "queries": queries, "money": bool(t.get("money")), "demand": demand,
                       "evidence": str(t.get("evidence") or "")[:200]})
    topics.sort(key=lambda t: t["demand"], reverse=True)
    return {"topics": topics, "summary": str(out.get("summary") or "")[:300]}


# --------------------------------------------------------------------------- public

def refresh(force: bool = False) -> dict:
    """Fresh trends (cached for trends.refresh_hours). Never raises - returns the old data on failure."""
    state = load_json(TRENDS_FILE, {"topics": [], "used": {}, "viewers": []})
    c = _cfg()
    if not c.get("enabled", True):
        return state
    age_h = (time.time() - float(state.get("at", 0))) / 3600
    if not force and age_h < float(c.get("refresh_hours", 6)) and state.get("topics"):
        return state
    try:
        viewers = twitch_viewers()
        if viewers is not None:
            state["viewers"] = (state.get("viewers", []) + [[int(time.time()), viewers]])[-200:]
        data = {"suggest": autocomplete(), "videos": hot_videos(), "reddit": reddit_hot()}
        log.info("[trends] data: %d search phrases, %d videos, %d reddit posts, twitch viewers %s",
                 len(data["suggest"]), len(data["videos"]), len(data["reddit"]), viewers)
        if data["suggest"] or data["videos"]:
            res = _analyse(data)
            if res and res["topics"]:
                state.update(res)
                state["at"] = time.time()
                state["top_searches"] = data["suggest"][:25]
                log.info("[trends] %s | top topics: %s", res["summary"],
                         [f"{t['topic']} ({t['demand']})" for t in res["topics"][:8]])
            else:
                log.warning("[trends] no AI could turn the data into topics - keeping the previous trends")
        else:
            log.warning("[trends] no trend data reachable - keeping the previous trends")
    except Exception as exc:  # noqa: BLE001 - trends must never break a run
        log.warning("[trends] refresh failed: %.200s", exc)
    save_json(TRENDS_FILE, state)
    return state


def stream_health(state: dict | None = None) -> dict:
    """Are GTA 6 live streams still big? weak = viewers fell below trends.stream_weak_ratio of the 7-day peak
    (or below trends.stream_min_viewers). Unknown data = not weak."""
    state = state or load_json(TRENDS_FILE, {"viewers": []})
    c = _cfg()
    pts = [(t, v) for t, v in state.get("viewers", []) if time.time() - t < 7 * 86400]
    if not pts:
        return {"weak": False, "viewers": None, "peak": None}
    now_v = pts[-1][1]
    peak = max(v for _, v in pts)
    weak = now_v < float(c.get("stream_min_viewers", 20000)) or now_v < peak * float(c.get("stream_weak_ratio", 0.35))
    return {"weak": bool(weak), "viewers": now_v, "peak": peak}


def _recently_used(state: dict, topic: str, days: float) -> bool:
    return time.time() - float(state.get("used", {}).get(topic.lower(), 0)) < days * 86400


def pick(money: bool | None = None, min_demand: int = 1, exclude: list[str] | None = None) -> dict | None:
    """Best topic not used in the last trends.topic_cooldown_days. money=True only money topics, False none.
    exclude: words that disqualify a topic (e.g. glitch / exploit for Money Tips)."""
    state = load_json(TRENDS_FILE, {"topics": [], "used": {}})
    days = float(_cfg().get("topic_cooldown_days", 4))
    bad = [w.lower() for w in (exclude or [])]
    for t in state.get("topics", []):
        if money is not None and bool(t.get("money")) != money:
            continue
        text = f"{t.get('topic', '')} {t.get('angle', '')} {' '.join(t.get('queries') or [])}".lower()
        if bad and any(w in text for w in bad):
            continue
        if t.get("demand", 0) >= min_demand and not _recently_used(state, t["topic"], days):
            return t
    return None


def mark_used(topic: str) -> None:
    if not topic:
        return
    state = load_json(TRENDS_FILE, {"topics": [], "used": {}})
    used = state.setdefault("used", {})
    used[topic.lower()] = time.time()
    state["used"] = dict(sorted(used.items(), key=lambda kv: kv[1])[-200:])
    save_json(TRENDS_FILE, state)


def context_line() -> str:
    """One line for text prompts: what people search for today."""
    state = load_json(TRENDS_FILE, {})
    bits = []
    if state.get("summary"):
        bits.append(state["summary"])
    if state.get("topics"):
        bits.append("Hot topics: " + ", ".join(t["topic"] for t in state["topics"][:6]) + ".")
    return " ".join(bits)


def main() -> int:
    """Manual check: python autopilot.py --slot trends (prints the current trends)."""
    st = refresh(force=True)
    h = stream_health(st)
    log.info("[trends] stream health: %s", h)
    for t in st.get("topics", []):
        log.info("[trends] %2d  %-35s money=%s  %s | queries: %s", t["demand"], t["topic"], t["money"], t["angle"],
                 t["queries"])
    return 0 if st.get("topics") else 1


if __name__ == "__main__":
    from common import setup_logging
    setup_logging()
    raise SystemExit(main())
