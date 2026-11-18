"""FREE X posting through a real (headless) browser using your logged-in session.

X has no free API any more, so this opens x.com with your login cookies, types the
post (with hashtags), attaches the video and clicks "Post" - like a person would.

Setup: export your x.com cookies with the free "Cookie-Editor" browser extension
(Export -> JSON) and save them as the GitHub secret X_COOKIES. See SETUP_GUIDE.md.

WARNING: X's rules forbid automating the website. X may ask you to log in again,
lock, or suspend the account. Opt-in only; nothing happens without X_COOKIES.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from common import READY_DIR, env, parse_cookies

log = logging.getLogger("autopilot")

COMPOSE_URL = "https://x.com/compose/post"
TEXTBOX = '[data-testid="tweetTextarea_0"]'
POST_BTN = '[data-testid="tweetButton"], [data-testid="tweetButtonInline"]'
FILE_INPUT = 'input[data-testid="fileInput"], input[type="file"]'
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/141.0.0.0 Safari/537.36")


def configured() -> bool:
    names = {c["name"] for c in parse_cookies(env("X_COOKIES"), ".x.com")}
    return {"auth_token", "ct0"} <= names


def _browser_cookies() -> list[dict]:
    out = []
    for c in parse_cookies(env("X_COOKIES"), ".x.com"):
        domain = c["domain"]
        if "twitter.com" in domain:  # old exports
            domain = domain.replace("twitter.com", "x.com")
        if "x.com" not in domain:
            continue
        ck = {"name": c["name"], "value": c["value"], "domain": domain, "path": c.get("path", "/"),
              "secure": True, "httpOnly": c.get("httpOnly", False), "sameSite": "None"}
        if c.get("expires"):
            ck["expires"] = c["expires"]
        out.append(ck)
    return out


LOGIN_MARKERS = 'input[autocomplete="username"], input[name="text"][autocomplete], [data-testid="loginButton"], [data-testid="google_sign_in_container"]'


def _needs_login(page) -> bool:
    if any(k in page.url for k in ("/login", "/i/flow", "/account/access", "onboarding", "mode=login",
                                   "redirect_after_login", "/logout")):
        return True
    return page.locator(LOGIN_MARKERS).count() > 0


def _wait_ready(page, timeout_ms: int = 45_000) -> str:
    """Wait until either the composer or a login wall shows. Returns 'compose' or 'login'."""
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        if _needs_login(page):
            return "login"
        if page.locator(TEXTBOX).count() and page.locator(TEXTBOX).first.is_visible():
            return "compose"
        page.wait_for_timeout(1000)
    return "timeout"


LOGIN_MSG = ("X is showing the login page - X_COOKIES are expired/logged out, or X locked the account. "
             "Log in to x.com in your browser, export fresh cookies (see SETUP_GUIDE.md) and update the secret.")


def _launch(p):
    try:
        return p.chromium.launch(channel="chrome", headless=True)  # Google Chrome is preinstalled on GitHub runners
    except Exception:  # noqa: BLE001
        return p.chromium.launch(headless=True)


def post(text: str, video: Path | None = None) -> dict:
    from playwright.sync_api import sync_playwright

    shot = READY_DIR / "x_browser_debug.png"
    with sync_playwright() as p:
        browser = _launch(p)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900}, user_agent=UA, locale="en-US")
        ctx.add_cookies(_browser_cookies())
        page = ctx.new_page()
        try:
            page.goto(COMPOSE_URL, wait_until="domcontentloaded", timeout=60_000)
            state = _wait_ready(page)
            if state == "login":
                raise RuntimeError(LOGIN_MSG)
            if state != "compose":
                raise RuntimeError(f"X composer did not load ({page.url})")
            box = page.locator(TEXTBOX).first

            if video:
                page.locator(FILE_INPUT).first.set_input_files(str(video))
                log.info("[x-browser] uploading video...")
                page.wait_for_timeout(3000)

            box.click()
            # trailing space closes the hashtag suggestion popup after the last #tag
            page.keyboard.type(text.rstrip() + " ", delay=12)

            btn = page.locator(POST_BTN).first
            deadline = time.time() + (420 if video else 30)
            while time.time() < deadline:
                if btn.is_visible() and btn.get_attribute("aria-disabled") != "true" and btn.is_enabled():
                    break
                page.wait_for_timeout(3000)
            else:
                raise RuntimeError("Post button never became clickable (video still processing or rejected)")
            page.wait_for_timeout(1500)
            btn.click()

            # success = toast with a link to the new post, or the composer closing
            url = ""
            try:
                toast = page.locator('[data-testid="toast"]').first
                toast.wait_for(state="visible", timeout=90_000 if video else 30_000)
                link = toast.locator('a[href*="/status/"]').first
                if link.count():
                    url = "https://x.com" + (link.get_attribute("href") or "")
                toast_text = toast.inner_text()
                if any(w in toast_text.lower() for w in ("wrong", "failed", "already said", "limit")):
                    raise RuntimeError(f"X said: {toast_text}")
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001
                if page.locator(TEXTBOX).count() and page.locator(TEXTBOX).first.inner_text().strip():
                    raise RuntimeError("Post did not go through (composer still open)")
            log.info("[x-browser] posted %s", url)
            return {"ok": True, "url": url}
        except Exception as exc:
            try:
                READY_DIR.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(shot))
                log.info("[x-browser] debug screenshot saved (download it from the run Artifacts)")
            except Exception:  # noqa: BLE001
                pass
            return {"ok": False, "error": f"x_browser: {exc}"}
        finally:
            browser.close()


def check_login() -> tuple[bool, str]:
    """For the setup check: are the cookies logged in?"""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = _launch(p)
        ctx = browser.new_context(user_agent=UA, locale="en-US")
        ctx.add_cookies(_browser_cookies())
        page = ctx.new_page()
        try:
            page.goto(COMPOSE_URL, wait_until="domcontentloaded", timeout=60_000)
            state = _wait_ready(page)
            if state == "compose":
                return True, "logged in - X posting will work"
            if state == "login":
                return False, LOGIN_MSG
            return False, f"X page did not load properly ({page.url})"
        finally:
            browser.close()
