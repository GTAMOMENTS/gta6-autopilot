"""Footage from X accounts (config.json -> sources.x_accounts), e.g. @GtaGameClips.

Opens the account's Media tab in a headless browser with the same X_COOKIES session used for posting,
and reads the timeline data X itself loads (video file URLs, length, date, views) - no page scraping.
Results are cached in queue/x_sources_cache.json so X is opened at most every few hours.
Every clip still goes through the normal AI checks (real GTA 6, no spoilers / watermark / facecam /
narrator, hype score, real script) and is credited to the account.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime

from common import STATE_DIR, env, load_json, save_json

log = logging.getLogger("autopilot")
CACHE_FILE = STATE_DIR / "x_sources_cache.json"


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _screen_name(tweet: dict) -> str:
    core = tweet.get("core") or {}
    for d in _walk(core):
        name = d.get("screen_name")
        if isinstance(name, str) and name:
            return name
    return ""


def parse_timeline(payloads: list[dict], account: str) -> list[dict]:
    """Video posts made by `account` in X timeline JSON (any layout: UserMedia / UserTweets)."""
    out: dict[str, dict] = {}
    why: dict[str, int] = {}
    for payload in payloads:
        for d in _walk(payload):
            legacy = d.get("legacy")
            if not (isinstance(legacy, dict) and d.get("rest_id") and "full_text" in legacy):
                continue
            if not legacy.get("extended_entities"):
                why["no media"] = why.get("no media", 0) + 1
                continue
            if _screen_name(d).lower() != account.lower():
                why[f"other account ({_screen_name(d) or '?'})"] = why.get(f"other account ({_screen_name(d) or '?'})", 0) + 1
                continue
            if "retweeted_status_result" in legacy or str(legacy.get("full_text", "")).startswith("RT @"):
                why["retweet"] = why.get("retweet", 0) + 1
                continue  # only the account's own posts, no retweets
            for media in legacy["extended_entities"].get("media", []):
                if media.get("type") != "video":
                    continue
                info = media.get("video_info") or {}
                mp4s = [v for v in info.get("variants", []) if v.get("content_type") == "video/mp4" and v.get("url")]
                if not mp4s:
                    continue
                best = max(mp4s, key=lambda v: v.get("bitrate") or 0)
                orig = media.get("original_info") or {}
                try:
                    created = datetime.strptime(legacy.get("created_at", ""), "%a %b %d %H:%M:%S %z %Y").timestamp()
                except ValueError:
                    created = None
                views = (d.get("views") or {}).get("count") or 0
                out[d["rest_id"]] = {
                    "id": d["rest_id"], "text": legacy.get("full_text", ""), "created": created,
                    "duration": (info.get("duration_millis") or 0) / 1000, "media_url": best["url"],
                    "width": orig.get("width", 0), "height": orig.get("height", 0),
                    "views": int(views) if str(views).isdigit() else 0,
                    "likes": legacy.get("favorite_count", 0),
                }
                break
            else:
                why["no video"] = why.get("no video", 0) + 1
    if why:
        log.info("[x-source] @%s: %d video posts kept, skipped: %s", account, len(out),
                 dict(sorted(why.items(), key=lambda kv: -kv[1])[:6]))
    return list(out.values())


def fetch_account(account: str, scrolls: int = 5) -> list[dict]:
    """Open x.com/<account>/media (logged in with X_COOKIES) and collect the timeline JSON it loads."""
    import x_browser
    from playwright.sync_api import sync_playwright

    responses = []  # read the bodies after loading - reading inside the event handler is unreliable

    def on_response(resp):
        if "/graphql/" in resp.url or "/i/api/" in resp.url:
            responses.append(resp)

    payloads: list[dict] = []
    with sync_playwright() as p:
        browser = x_browser._launch(p)
        try:
            ctx = browser.new_context(viewport={"width": 1280, "height": 900}, user_agent=x_browser.UA, locale="en-US")
            ctx.add_cookies(x_browser._browser_cookies())
            page = ctx.new_page()
            page.on("response", on_response)
            for path in ("media",):  # one page only - more page loads trigger X's bot check
                page.goto(f"https://x.com/{account}/{path}".rstrip("/"), wait_until="domcontentloaded",
                          timeout=60_000)
                page.wait_for_timeout(7000)
                if x_browser._needs_login(page):
                    log.warning("[x-source] X asks to log in again - X_COOKIES expired?")
                    return []
                for _ in range(scrolls):
                    page.mouse.wheel(0, 4000)
                    page.wait_for_timeout(2500)
                ops, unreadable = {}, {}
                for resp in responses:
                    op = resp.url.split("?")[0].rstrip("/").split("/")[-1]
                    ops[op] = ops.get(op, 0) + 1
                    try:
                        payloads.append(resp.json())
                    except Exception as exc:  # noqa: BLE001
                        unreadable[op] = str(exc)[:60]
                responses.clear()
                if unreadable:
                    log.info("[x-source] unreadable answers: %s", unreadable)
                clips = parse_timeline(payloads, account)
                if not clips:  # data format not understood -> take the video posts straight from the page
                    clips = _clips_from_page(page, account)
                n_tweets = page.locator('[data-testid="tweet"]').count()
                n_videos = page.locator("video").count()
                log.info("[x-source] x.com/%s/%s: api calls %s | %d posts, %d videos on page | %d own video posts parsed",
                         account, path, ops, n_tweets, n_videos, len(clips))
                if not n_tweets:
                    body = re.sub(r"\s+", " ", page.inner_text("body"))[:300]
                    log.info("[x-source] page text: %s", body)
                if clips:
                    return clips
        finally:
            browser.close()
    return parse_timeline(payloads, account)


def _clips_from_page(page, account: str, limit: int = 10) -> list[dict]:
    """Fallback: status links of the video posts visible on the page; yt-dlp then reads each post
    (length, date, views, direct video) with the same X login."""
    hrefs = page.eval_on_selector_all(
        'article[data-testid="tweet"]',
        r"""els => els.filter(e => e.querySelector('video, [data-testid="videoPlayer"]'))
                   .map(e => { const a = [...e.querySelectorAll('a[href*="/status/"]')]
                                 .map(x => x.getAttribute('href')).find(h => /\/status\/\d+$/.test(h));
                               const t = e.querySelector('[data-testid="tweetText"]');
                               return [a || '', t ? t.innerText : '']; })""")
    out, why = [], {}
    import json as _json
    from common import WORK_DIR, cookies_to_netscape, parse_cookies, run
    jar = WORK_DIR / "x_cookies.txt"
    jar.parent.mkdir(parents=True, exist_ok=True)
    jar.write_text(cookies_to_netscape(parse_cookies(env("X_COOKIES"), ".x.com")), encoding="utf-8")
    for href, text in hrefs:
        m = re.search(r"^/([^/]+)/status/(\d+)$", href or "")
        if not m:
            why["no link"] = why.get("no link", 0) + 1
            continue
        if m.group(1).lower() != account.lower():
            why["other account"] = why.get("other account", 0) + 1
            continue
        if any(c["id"] == m.group(2) for c in out) or len(out) >= limit:
            continue
        res = run(["yt-dlp", "--no-warnings", "--cookies", str(jar), "-J", "--skip-download",
                   f"https://x.com{href}"], timeout=60)
        try:
            meta = _json.loads(res.stdout)
        except (ValueError, TypeError):
            why["yt-dlp failed"] = why.get("yt-dlp failed", 0) + 1
            log.debug("[x-source] yt-dlp: %s", (res.stderr or "").strip()[-200:])
            continue
        if isinstance(meta, dict) and meta.get("_type") == "playlist":
            meta = (meta.get("entries") or [{}])[0] or {}
        fmts = [f for f in (meta.get("formats") or []) if f.get("ext") == "mp4" and f.get("vcodec") not in (None, "none")
                and str(f.get("url", "")).startswith("http") and ".m3u8" not in str(f.get("url", ""))]
        best = max(fmts, key=lambda f: (f.get("height") or 0, f.get("tbr") or 0)) if fmts else None
        out.append({"id": m.group(2), "text": text or meta.get("description") or meta.get("title") or "",
                    "created": float(meta.get("timestamp") or 0) or None, "duration": float(meta.get("duration") or 0),
                    "media_url": (best or {}).get("url") or "", "width": (best or {}).get("width") or 0,
                    "height": (best or {}).get("height") or 0, "views": int(meta.get("view_count") or 0),
                    "likes": int(meta.get("like_count") or 0), "page_url": f"https://x.com{href}"})
    jar.unlink(missing_ok=True)
    log.info("[x-source] @%s from the page: %d video posts read with yt-dlp (skipped: %s)", account, len(out), why)
    return out


def account_clips(cfg: dict, accounts: list[str] | None = None, cache_file=None,
                  cache_hours: float | None = None) -> list[tuple[str, dict]]:
    """[(account, clip)] for every configured X account, from cache when it is fresh."""
    s = cfg.get("sources", {})
    accounts = [a.lstrip("@") for a in (accounts if accounts is not None else s.get("x_accounts", [])) if a.strip()]
    if not accounts:
        return []
    if not x_browser_ready():
        log.info("[x-source] X_COOKIES not set - skipping X accounts")
        return []
    cache_file = cache_file or CACHE_FILE
    cache = load_json(cache_file, {})
    max_age = float(cache_hours if cache_hours is not None else s.get("x_cache_hours", 4)) * 3600
    out = []
    for acc in accounts:
        entry = cache.get(acc.lower(), {})
        if time.time() - entry.get("at", 0) > max_age:
            try:
                clips = fetch_account(acc)
                log.info("[x-source] @%s: %d video posts found", acc, len(clips))
                if clips:  # an empty answer is retried next run instead of being cached
                    entry = {"at": time.time(), "clips": clips}
                    cache[acc.lower()] = entry
            except Exception as exc:  # noqa: BLE001
                log.warning("[x-source] could not read @%s: %.200s", acc, exc)
        out += [(acc, c) for c in entry.get("clips", [])]
    save_json(cache_file, cache)
    return out


def x_browser_ready() -> bool:
    if env("X_SOURCE_DISABLED").lower() == "true":
        return False
    import x_browser
    return x_browser.configured()


if __name__ == "__main__":  # quick manual check: python x_source.py GtaGameClips
    import sys
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(fetch_account(sys.argv[1] if len(sys.argv) > 1 else "GtaGameClips"), indent=1)[:3000])
