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
    for payload in payloads:
        for d in _walk(payload):
            legacy = d.get("legacy")
            if not (isinstance(legacy, dict) and d.get("rest_id") and legacy.get("extended_entities")):
                continue
            if (_screen_name(d).lower() != account.lower() or "retweeted_status_result" in legacy
                    or str(legacy.get("full_text", "")).startswith("RT @")):
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
    return list(out.values())


def fetch_account(account: str, scrolls: int = 4) -> list[dict]:
    """Open x.com/<account>/media (logged in with X_COOKIES) and collect the timeline JSON it loads."""
    import x_browser
    from playwright.sync_api import sync_playwright

    payloads: list[dict] = []

    def on_response(resp):
        url = resp.url
        if "/graphql/" in url and any(k in url for k in ("UserMedia", "UserTweets")):
            try:
                payloads.append(resp.json())
            except Exception:  # noqa: BLE001
                pass

    with sync_playwright() as p:
        browser = x_browser._launch(p)
        try:
            ctx = browser.new_context(viewport={"width": 1280, "height": 900}, user_agent=x_browser.UA, locale="en-US")
            ctx.add_cookies(x_browser._browser_cookies())
            page = ctx.new_page()
            page.on("response", on_response)
            page.goto(f"https://x.com/{account}/media", wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(6000)
            if x_browser._needs_login(page):
                log.warning("[x-source] X asks to log in again - X_COOKIES expired?")
                return []
            for _ in range(scrolls):
                page.mouse.wheel(0, 4000)
                page.wait_for_timeout(2500)
        finally:
            browser.close()
    return parse_timeline(payloads, account)


def account_clips(cfg: dict) -> list[tuple[str, dict]]:
    """[(account, clip)] for every configured X account, from cache when it is fresh."""
    s = cfg.get("sources", {})
    accounts = [a.lstrip("@") for a in s.get("x_accounts", []) if a.strip()]
    if not accounts:
        return []
    if not x_browser_ready():
        log.info("[x-source] X_COOKIES not set - skipping X accounts")
        return []
    cache = load_json(CACHE_FILE, {})
    max_age = float(s.get("x_cache_hours", 4)) * 3600
    out = []
    for acc in accounts:
        entry = cache.get(acc.lower(), {})
        if time.time() - entry.get("at", 0) > max_age:
            try:
                clips = fetch_account(acc)
                log.info("[x-source] @%s: %d video posts found", acc, len(clips))
                if clips or not entry:
                    entry = {"at": time.time(), "clips": clips}
                    cache[acc.lower()] = entry
            except Exception as exc:  # noqa: BLE001
                log.warning("[x-source] could not read @%s: %.200s", acc, exc)
        out += [(acc, c) for c in entry.get("clips", [])]
    save_json(CACHE_FILE, cache)
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
