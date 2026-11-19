"""'check_setup' slot: tests every secret and prints a plain-English report.

Run it from GitHub: Actions -> GTA6 Top Moments 24/7 Autopilot -> Run workflow ->
slot = check_setup. Then open the run and read the summary table. Nothing is posted.
"""
from __future__ import annotations

import json
import logging

import requests

from common import WORK_DIR, cookies_to_netscape, env, parse_cookies, run, write_summary

log = logging.getLogger("autopilot")
OK, BAD, OFF = "✅", "❌", "⚪"


def _gemini():
    if not env("GEMINI_API_KEY"):
        return OFF, "not set", "Optional but recommended: free key at aistudio.google.com → Get API key."
    try:
        import gemini_writer
        models = gemini_writer._discover_models()
        if models:
            return OK, f"works - models: {', '.join(models)}", "If you see quota errors in runs, it resets daily; backups take over."
        return BAD, "key set but no models returned", "Create a new key at aistudio.google.com."
    except Exception as exc:  # noqa: BLE001
        return BAD, f"error: {str(exc)[:150]}", "Create a new key at aistudio.google.com."


def _openrouter():
    if not env("OPENROUTER_API_KEY"):
        return OFF, "not set", "Free backup AI: openrouter.ai → Keys → Create key."
    r = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {env('OPENROUTER_API_KEY')}"}, timeout=20)
    if r.status_code != 200:
        return BAD, f"rejected ({r.status_code})", "Copy the key again from openrouter.ai → Keys (starts with sk-or-)."
    import ai_fallback
    models = [m for m, vision in ai_fallback._models("openrouter") if vision]
    return OK, f"works - free vision models: {', '.join(models[:3]) or 'none right now'}", ""


def _simple_key(name, url, hint):
    if not env(name):
        return OFF, "not set", hint
    r = requests.get(url, headers={"Authorization": f"Bearer {env(name)}"}, timeout=20)
    if r.status_code != 200:
        return BAD, f"rejected ({r.status_code})", hint
    import ai_fallback
    provider = next((p for p, v in ai_fallback.PROVIDERS.items() if v["key"] == name), None)
    models = ai_fallback._models(provider) if provider else []
    shown = ", ".join(f"{m}{' (sees images)' if vision else ''}" for m, vision in models[:3])
    return OK, f"works - will use: {shown or 'no suitable models found'}", ""


def _zernio():
    if not env("ZERNIO_API_KEY"):
        return OFF, "not set", "Needed for Instagram + TikTok (free for 2 accounts): zernio.com → API keys."
    r = requests.get("https://zernio.com/api/v1/accounts", headers={"Authorization": f"Bearer {env('ZERNIO_API_KEY')}"}, timeout=30)
    if r.status_code != 200:
        return BAD, f"rejected ({r.status_code}): {r.text[:120]}", "Create a new API key in Zernio and update ZERNIO_API_KEY."
    data = r.json()
    accounts = data.get("accounts", data.get("data", data if isinstance(data, list) else []))
    linked = [f"{a.get('platform')} ({a.get('username') or a.get('name') or a.get('displayName') or '?'})" for a in accounts]
    platforms = {a.get("platform") for a in accounts}
    tip = ""
    missing = {"instagram", "tiktok"} - platforms
    if missing:
        tip = f"Connect {', '.join(sorted(missing))} in Zernio (Accounts → Connect). "
    if len(accounts) > 2:
        tip += "More than 2 accounts may cost money on Zernio - keep only Instagram + TikTok."
    return (OK if not missing else BAD), f"linked: {', '.join(linked) or 'nothing'}", tip


def _youtube():
    keys = ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN")
    have = [k for k in keys if env(k)]
    if not have:
        return OFF, "not set", "Free YouTube uploads: follow 'YouTube' in SETUP_GUIDE.md."
    if len(have) < 3:
        return BAD, f"missing {', '.join(k for k in keys if not env(k))}", "Add all 3 secrets."
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": env("YT_CLIENT_ID"), "client_secret": env("YT_CLIENT_SECRET"),
        "refresh_token": env("YT_REFRESH_TOKEN"), "grant_type": "refresh_token"}, timeout=30)
    if r.status_code != 200:
        return BAD, f"Google refused: {r.text[:160]}", ("invalid_grant = token expired/revoked: redo step 5-6 of the YouTube guide, "
                                                         "and make sure the app is 'In production'.")
    tok = r.json()["access_token"]
    info = requests.get("https://oauth2.googleapis.com/tokeninfo", params={"access_token": tok}, timeout=30).json()
    scopes = info.get("scope", "")
    if "youtube.upload" not in scopes and "auth/youtube " not in scopes + " ":
        return BAD, f"token works but has no upload permission (scopes: {scopes or 'none'})", \
            "Redo Part 2 of the YouTube steps and paste the youtube.upload scope before 'Authorize APIs'."
    ch = requests.get("https://www.googleapis.com/youtube/v3/channels?part=snippet&mine=true",
                      headers={"Authorization": f"Bearer {tok}"}, timeout=30)
    names = [i["snippet"]["title"] for i in ch.json().get("items", [])] if ch.status_code == 200 else []
    if names:
        return OK, f"works - uploads go to channel: {', '.join(names)}", ""
    return OK, "works - upload permission OK (channel name hidden with upload-only permission; "\
               "videos go to the channel you picked when authorising)", ""


def _ytdlp_cookies():
    raw = env("YTDLP_COOKIES")
    if not raw:
        return OFF, "not set", "Optional: lets the bot use YouTube videos as clip sources. See SETUP_GUIDE.md."
    cookies = [c for c in parse_cookies(raw, ".youtube.com") if "youtube" in c["domain"] or "google" in c["domain"]]
    if not cookies:
        return BAD, "secret is set but no youtube.com cookies could be read from it", \
            "Export again from youtube.com with Cookie-Editor → Export → JSON, and paste the whole thing."
    names = {c["name"] for c in cookies}
    important = {"SID", "HSID", "SSID", "SAPISID", "__Secure-3PSID", "LOGIN_INFO"}
    if not names & important:
        return BAD, f"{len(cookies)} cookies read, but none are login cookies", "You were probably not logged in when exporting."
    path = WORK_DIR / "check_cookies.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cookies_to_netscape(cookies), encoding="utf-8")
    import shutil
    cmd = ["yt-dlp", "--cookies", str(path), "--skip-download", "--no-warnings", "-J"]
    if shutil.which("node"):
        cmd += ["--js-runtimes", "node"]
    res = run(cmd + ["https://www.youtube.com/watch?v=QdBZY2fkU-0"], timeout=150)
    if res.returncode == 0:
        # the login works - but which qualities does YouTube actually serve? (the bot needs 720p+)
        import json as _json
        best = {}
        for vid in ("QdBZY2fkU-0", "RuZUqpDuzfo"):
            r2 = res if vid == "QdBZY2fkU-0" else run(cmd + [f"https://www.youtube.com/watch?v={vid}"], timeout=150)
            try:
                fmts = _json.loads(r2.stdout).get("formats", [])
                best[vid] = max([f.get("height") or 0 for f in fmts if f.get("vcodec") not in (None, "none")] or [0])
            except Exception:  # noqa: BLE001
                best[vid] = f"error: {(r2.stderr.strip().splitlines() or ['?'])[-1][:120]}"
        ver = run(["yt-dlp", "--version"], timeout=30).stdout.strip()
        path.unlink(missing_ok=True)
        log.info("[check] yt-dlp %s - best video height per test video: %s", ver, best)
        heights = [h for h in best.values() if isinstance(h, int)]
        if heights and max(heights) < 720:
            return BAD, f"login accepted, but YouTube only serves up to {max(heights)}p (bot needs 720p+), yt-dlp {ver}", \
                "Usually a YouTube player change - needs a newer yt-dlp. Tell Claude."
        return OK, f"works - {len(cookies)} cookies, up to {max(heights or [0])}p available (yt-dlp {ver})", ""
    path.unlink(missing_ok=True)
    err = res.stderr.strip().splitlines()[-1][:180] if res.stderr.strip() else "unknown error"
    if "format is not available" in err:
        return BAD, "login accepted, but YouTube's video streams could not be unlocked", \
            "Not your fault - this needs a code fix (JavaScript engine for yt-dlp). Tell Claude."
    return BAD, f"YouTube rejected them: {err}", ("Cookies expire when you keep using that browser session. Use a private/incognito "
                                                  "window, log in, export, then CLOSE the window without logging out.")


def _x_cookies():
    if not env("X_COOKIES"):
        return OFF, "not set", "Optional FREE X posting through a browser. See 'X' in SETUP_GUIDE.md."
    import x_browser
    if not x_browser.configured():
        return BAD, "auth_token / ct0 cookies not found in the secret", "Export x.com cookies with Cookie-Editor → Export → JSON."
    try:
        ok, msg = x_browser.check_login()
    except Exception as exc:  # noqa: BLE001
        return BAD, f"browser error: {str(exc)[:150]}", ""
    return (OK if ok else BAD), msg, ""


def _x_api():
    keys = ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET")
    have = [k for k in keys if env(k)]
    if not have:
        return OFF, "not set", "Paid option (~$3/month). Not needed if X_COOKIES works."
    if len(have) < 4:
        return BAD, f"missing {', '.join(k for k in keys if not env(k))}", ""
    return OK, "all 4 keys set (not test-called, each call costs money)", ""


def _twitch():
    if not (env("TWITCH_CLIENT_ID") and env("TWITCH_CLIENT_SECRET")):
        return OFF, "not set", "Optional, free: viral Twitch clips + streams as sources. See section H of SETUP_GUIDE.md."
    import source_fetcher
    from common import load_config
    try:
        found = source_fetcher.discover_twitch(load_config(), source_fetcher._stream_filter(load_config()))
    except Exception as exc:  # noqa: BLE001
        return BAD, f"error: {str(exc)[:150]}", "Check TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET."
    clips = sum(1 for c in found if c.provider == "twitch_clip")
    vods = len(found) - clips
    return OK, f"works - {clips} viral GTA 6 clips + {vods} recorded streams found right now", ""


def _upload_post():
    if not env("UPLOAD_POST_API_KEY"):
        return OFF, "not set", "Optional last-resort fallback (10 free uploads/month)."
    return OK, "set (used only as a fallback)", ""


def _big_text_ai(provider: str):
    """Cerebras / SambaNova: list the models and ask one tiny JSON question with the best one."""
    import ai_fallback
    key = ai_fallback.PROVIDERS[provider]["key"]
    if not env(key):
        return OFF, "not set", f"Optional, free: big fast text models for titles/hooks. Create a key and add {key}."
    models = ai_fallback._models(provider)
    if not models:
        return BAD, "key set, but no models listed (wrong key?)", "Create a new API key and update the secret."
    for model, _ in models[:3]:
        try:
            data, status = ai_fallback._call(provider, model, 'Reply with this JSON only: {"ok": true}', None, 0.0,
                                             timeout=40)
        except Exception as exc:  # noqa: BLE001
            data, status = None, str(exc)[:60]
        if data:
            return OK, f"works - answered with {model} (models: {', '.join(m for m, _ in models[:4])})", ""
    return BAD, f"models listed but no answer (last status {status})", "Free tier busy or key limited - try again later."


CHECKS = [
    ("Gemini AI (GEMINI_API_KEY)", _gemini),
    ("Big text AI (CEREBRAS_API_KEY)", lambda: _big_text_ai("cerebras")),
    ("Big text AI (SAMBANOVA_API_KEY)", lambda: _big_text_ai("sambanova")),
    ("Backup AI (OPENROUTER_API_KEY)", _openrouter),
    ("Backup AI (GROQ_API_KEY)", lambda: _simple_key("GROQ_API_KEY", "https://api.groq.com/openai/v1/models",
                                                     "Optional: console.groq.com → API Keys.")),
    ("Backup AI (NVIDIA_API_KEY)", lambda: _simple_key("NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1/models",
                                                       "Optional, free, no card: build.nvidia.com → Get API Key (starts with nvapi-).")),
    ("Instagram + TikTok (ZERNIO_API_KEY)", _zernio),
    ("YouTube uploads (YT_CLIENT_ID / SECRET / REFRESH_TOKEN)", _youtube),
    ("X posting - free browser (X_COOKIES)", _x_cookies),
    ("X posting - paid API (X_API_KEY ...)", _x_api),
    ("YouTube as clip source (YTDLP_COOKIES)", _ytdlp_cookies),
    ("Twitch clips + streams (TWITCH_CLIENT_ID / SECRET)", _twitch),
    ("Upload-Post fallback (UPLOAD_POST_API_KEY)", _upload_post),
]


def run_checks() -> int:
    rows = ["## 🔧 Setup check", "", "| | What | Result | What to do |", "|---|---|---|---|"]
    for label, fn in CHECKS:
        try:
            icon, result, todo = fn()
        except Exception as exc:  # noqa: BLE001
            icon, result, todo = BAD, f"check crashed: {str(exc)[:150]}", ""
        log.info("%s %s: %s %s", icon, label, result, f"-> {todo}" if todo else "")
        rows.append(f"| {icon} | {label} | {result.replace('|', '/')} | {todo.replace('|', '/')} |")
    rows += ["", "✅ works · ❌ set but broken · ⚪ not set up (optional unless the platform needs it)"]
    write_summary("\n".join(rows))
    return 0


if __name__ == "__main__":
    from common import setup_logging
    setup_logging()
    run_checks()
    print(json.dumps({"done": True}))
