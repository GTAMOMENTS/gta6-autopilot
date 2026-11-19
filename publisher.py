"""Publish videos and text posts through the cheapest working provider per platform.

Providers (configure order per platform in config.json -> publishing.routes):
  youtube_api - official YouTube Data API. FREE (100 uploads/day). Secrets:
                YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN (see tools/youtube_auth.py)
  zernio      - Zernio API. First 2 connected accounts FREE with unlimited posts
                (use them for TikTok + Instagram). Secret: ZERNIO_API_KEY
  x_api       - official X API (pay-per-use, roughly $0.015 per post). Secrets:
                X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
  upload_post - Upload-Post (free plan = 10 uploads/month). Secrets:
                UPLOAD_POST_API_KEY, UPLOAD_POST_PROFILE

If a provider is not configured or fails, the next one in the list is tried.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path

import requests

from common import env, load_config

log = logging.getLogger("autopilot")

ZERNIO = "https://zernio.com/api/v1"
UPLOAD_POST = "https://api.upload-post.com/api"
X_API = "https://api.x.com/2"
ZERNIO_PLATFORM = {"youtube": "youtube", "instagram": "instagram", "tiktok": "tiktok", "x": "twitter"}

DEFAULT_ROUTES = {
    "youtube": ["youtube_api", "zernio", "upload_post"],
    "instagram": ["zernio", "upload_post"],
    "tiktok": ["zernio", "upload_post"],
    "x": ["x_api", "x_browser", "zernio", "upload_post"],
}

_disabled: set[str] = set()   # providers that hit a quota / auth error this run


def _pcfg() -> dict:
    return load_config().get("publishing", {})


def routes_for(platform: str) -> list[str]:
    return _pcfg().get("routes", {}).get(platform, DEFAULT_ROUTES.get(platform, []))


# --------------------------------------------------------------------------- captions

# Company / platform brand tags make a post look like undisclosed sponsored ("branded") content on
# Instagram - never use them. config.json -> publishing.blocked_hashtags can add more.
BRAND_TAGS = {"rockstargames", "rockstar", "rockstarnorth", "taketwo", "playstation", "ps5", "xbox",
              "xboxseriesx", "sony", "microsoft", "steam", "epicgames", "nvidia", "amd", "ad", "sponsored",
              "partner", "giveaway"}


def clean_hashtags(tags: list[str]) -> list[str]:
    blocked = BRAND_TAGS | {t.lower().lstrip("#") for t in _pcfg().get("blocked_hashtags", [])}
    out = [t for t in tags if t.lstrip("#").lower() not in blocked]
    return list(dict.fromkeys(out)) or ["#GTA6"]


def _closing_question(copy: dict) -> str:
    """The video's ending question - used as the first comment to get people talking."""
    lines = [ln.get("text", "") for ln in copy.get("commentary") or []]
    for text in reversed(lines):
        if text.strip().endswith("?"):
            return text.strip()
    qs = [x.strip() for x in re.findall(r"[^.!?]+\?", copy.get("caption", ""))]
    return qs[-1] if qs else ""


def build_captions(copy: dict, credit: str) -> dict:
    mx = _pcfg().get("max_hashtags", {})
    # 3-5 targeted tags beat 30 generic ones (YouTube ignores ALL tags above 15): #GTA6 + format + AI tags
    tags = clean_hashtags(["#GTA6"] + list(copy.get("format_tags") or []) + list(copy.get("hashtags") or []))

    def tags_for(platform: str, default: int) -> str:
        return " ".join(tags[: int(mx.get(platform, default))])

    credit_line = f"🎥 Gameplay by {credit}" if credit else ""
    keywords = (copy.get("keywords") or "").strip()  # plain search words (TikTok / YouTube search)
    series = (copy.get("series") or "").strip()
    base = (copy.get("youtube_title") or copy.get("banner", "GTA 6 moment").title()).strip()
    if "gta 6" not in base.lower() and "gta vi" not in base.lower():
        base = f"GTA 6: {base}"
    yt_title = f"{base} | {series}" if series and len(base) + len(series) + 3 <= 88 else base
    x_text = copy.get("x_text") or copy.get("caption", "")
    x_full = f"{x_text}\n\n{credit_line}".strip()
    x_tags = ""
    for t in tags[: int(mx.get("x", 3))]:
        if len(x_full) + len(x_tags) + len(t) + 2 <= 280:
            x_tags += " " + t
    if x_tags:
        x_full += "\n" + x_tags.strip()
    caption = copy.get("caption", "")
    return {
        "youtube_title": (f"{yt_title} #Shorts" if len(yt_title) <= 91 else yt_title)[:100],
        "youtube_description": "\n\n".join(x for x in (caption, f"{series} · {keywords}".strip(" ·"), credit_line,
                                                         f"{tags_for('youtube', 5)} #Shorts") if x).strip(),
        "instagram": "\n.\n".join(x for x in (caption, credit_line, tags_for("instagram", 5)) if x).strip(),
        "tiktok": "\n".join(x for x in (f"{caption} {keywords}".strip(), tags_for("tiktok", 5), credit_line)
                            if x).strip()[:2200],
        "x": x_full[:280],
        "tags": [t.lstrip("#") for t in tags] + ["GTA 6", "GTA VI", "Grand Theft Auto 6"]
                + ([keywords] if keywords else []),
        "first_comment": _closing_question(copy),
    }


# --------------------------------------------------------------------------- availability

def _available(provider: str, platform: str) -> bool:
    if provider in _disabled:
        return False
    if provider == "youtube_api":
        return platform == "youtube" and all(env(k) for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"))
    if provider == "x_api":
        return platform == "x" and all(env(k) for k in ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET"))
    if provider == "zernio":
        return bool(env("ZERNIO_API_KEY")) and _zernio_account(platform) is not None
    if provider == "upload_post":
        return bool(env("UPLOAD_POST_API_KEY"))
    if provider == "x_browser":
        if platform != "x":
            return False
        import x_browser
        try:
            import playwright  # noqa: F401
        except ImportError:
            return False
        return x_browser.configured()
    return False


# --------------------------------------------------------------------------- YouTube Data API

def _yt_token() -> str:
    r = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id": env("YT_CLIENT_ID"), "client_secret": env("YT_CLIENT_SECRET"),
        "refresh_token": env("YT_REFRESH_TOKEN"), "grant_type": "refresh_token",
    }, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"token refresh failed {r.status_code}: {r.text[:300]} "
                           "(if this says invalid_grant, re-run tools/youtube_auth.py and make sure the "
                           "OAuth consent screen is published 'In production')")
    return r.json()["access_token"]


def youtube_api_upload(video: Path, caps: dict) -> dict:
    p = _pcfg()
    token = _yt_token()
    meta = {
        "snippet": {
            "title": caps["youtube_title"], "description": caps["youtube_description"][:4900],
            "tags": caps["tags"][:15], "categoryId": str(p.get("youtube_category_id", "20")),
        },
        "status": {"privacyStatus": p.get("youtube_privacy", "public"), "selfDeclaredMadeForKids": False},
    }
    init = requests.post(
        "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8",
                 "X-Upload-Content-Type": "video/mp4", "X-Upload-Content-Length": str(video.stat().st_size)},
        json=meta, timeout=60)
    if init.status_code != 200 or "Location" not in init.headers:
        raise RuntimeError(f"upload init {init.status_code}: {init.text[:400]}")
    with open(video, "rb") as f:
        up = requests.put(init.headers["Location"], data=f,
                          headers={"Content-Type": "video/mp4"}, timeout=900)
    if up.status_code not in (200, 201):
        raise RuntimeError(f"upload {up.status_code}: {up.text[:400]}")
    data = up.json()
    vid = data.get("id")
    privacy = data.get("status", {}).get("privacyStatus")
    if privacy and privacy != meta["status"]["privacyStatus"]:
        log.warning("[youtube] video %s is '%s' (requested %s) - check YouTube Studio / API project status",
                    vid, privacy, meta["status"]["privacyStatus"])
    if vid and p.get("first_comment", True) and caps.get("first_comment"):
        _youtube_first_comment(token, vid, caps["first_comment"])
    return {"ok": True, "url": f"https://youtube.com/shorts/{vid}", "id": vid}


def youtube_set_thumbnail(video_id: str, image: Path) -> str:
    """Custom thumbnail (YouTube allows it once the channel is phone-verified)."""
    try:
        r = requests.post(f"https://www.googleapis.com/upload/youtube/v3/thumbnails/set?videoId={video_id}",
                          headers={"Authorization": f"Bearer {_yt_token()}", "Content-Type": "image/jpeg"},
                          data=image.read_bytes(), timeout=60)
        if r.status_code in (200, 201):
            log.info("[youtube] custom thumbnail set")
            return "set"
        hint = " - verify the channel with your phone at youtube.com/verify" if r.status_code == 403 else ""
        log.warning("[youtube] thumbnail not set (%s)%s: %.200s", r.status_code, hint, r.text)
        return f"failed {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        log.warning("[youtube] thumbnail upload failed: %.200s", exc)
        return "failed"


_comment_scope_missing = False


def _youtube_first_comment(token: str, video_id: str, text: str) -> None:
    """Post the ending question as the first comment (needs the youtube.force-ssl permission)."""
    global _comment_scope_missing
    if _comment_scope_missing:
        return
    try:
        r = requests.post("https://www.googleapis.com/youtube/v3/commentThreads?part=snippet",
                          headers={"Authorization": f"Bearer {token}"}, timeout=30,
                          json={"snippet": {"videoId": video_id,
                                            "topLevelComment": {"snippet": {"textOriginal": text[:500]}}}})
        if r.status_code in (200, 201):
            log.info("[youtube] first comment posted: %s", text)
        elif r.status_code == 403 and ("insufficient" in r.text.lower() or "scope" in r.text.lower()):
            _comment_scope_missing = True
            log.warning("[youtube] first comment skipped: the YouTube login has no comment permission. "
                        "Re-run tools/youtube_auth.py once (it now asks for it) and update YT_REFRESH_TOKEN.")
        else:
            log.warning("[youtube] first comment failed %s: %.200s", r.status_code, r.text)
    except Exception as exc:  # noqa: BLE001
        log.warning("[youtube] first comment failed: %.200s", exc)


# --------------------------------------------------------------------------- Zernio

_zernio_accounts: dict | None = None


def _zh() -> dict:
    return {"Authorization": f"Bearer {env('ZERNIO_API_KEY')}", "Content-Type": "application/json"}


def _zernio_account(platform: str) -> str | None:
    global _zernio_accounts
    if _zernio_accounts is None:
        _zernio_accounts = {}
        try:
            r = requests.get(f"{ZERNIO}/accounts", headers=_zh(), timeout=30)
            if r.status_code != 200:
                log.error("[zernio] accounts error %s: %s", r.status_code, r.text[:300])
                if r.status_code in (401, 403):
                    _disabled.add("zernio")
            else:
                data = r.json()
                accounts = data.get("accounts", data.get("data", data if isinstance(data, list) else []))
                for a in accounts:
                    if a.get("status") == "disconnected" or a.get("isActive") is False:
                        continue
                    _zernio_accounts.setdefault(a.get("platform"), a.get("_id") or a.get("id"))
                log.info("[zernio] linked accounts: %s", sorted(k for k in _zernio_accounts if k))
        except Exception as exc:  # noqa: BLE001
            log.error("[zernio] accounts lookup failed: %s", exc)
    return _zernio_accounts.get(ZERNIO_PLATFORM.get(platform, platform))


def _zernio_upload(video: Path) -> str:
    r = requests.post(f"{ZERNIO}/media/presign", headers=_zh(), json={
        "filename": video.name, "contentType": "video/mp4", "size": video.stat().st_size}, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"presign {r.status_code}: {r.text[:300]}")
    d = r.json().get("data", r.json())
    with open(video, "rb") as f:
        put = requests.put(d["uploadUrl"], data=f, headers={"Content-Type": "video/mp4"}, timeout=900)
    if put.status_code not in (200, 201):
        raise RuntimeError(f"storage upload {put.status_code}")
    return d["publicUrl"]


def _zernio_post(body: dict, platforms: list[str]) -> dict:
    r = requests.post(f"{ZERNIO}/posts", headers={**_zh(), "Idempotency-Key": str(uuid.uuid4())},
                      json=body, timeout=300)
    if r.status_code in (401, 403):
        _disabled.add("zernio")
    if r.status_code not in (200, 201, 207):
        return {p: {"ok": False, "error": f"zernio {r.status_code}: {r.text[:300]}"} for p in platforms}
    post = r.json().get("post", r.json())
    by_platform = {x.get("platform"): x for x in post.get("platforms", [])}
    out = {}
    for p in platforms:
        entry = by_platform.get(ZERNIO_PLATFORM[p], {})
        status = entry.get("status", post.get("status", "published"))
        if status in ("failed", "cancelled"):
            out[p] = {"ok": False, "error": f"zernio: {entry.get('errorMessage') or status}"}
        else:
            out[p] = {"ok": True, "url": entry.get("platformPostUrl") or "", "id": post.get("_id"), "status": status}
    return out


def zernio_video(video: Path, platforms: list[str], caps: dict) -> dict:
    p = _pcfg()
    media_url = _zernio_upload(video)
    entries = []
    for plat in platforms:
        e = {"platform": ZERNIO_PLATFORM[plat], "accountId": _zernio_account(plat)}
        if plat == "youtube":
            e["customContent"] = caps["youtube_description"]
            e["platformSpecificData"] = {"title": caps["youtube_title"], "visibility": p.get("youtube_privacy", "public"),
                                         "madeForKids": False, "categoryId": str(p.get("youtube_category_id", "20"))}
        elif plat == "instagram":
            e["customContent"] = caps["instagram"]
            e["platformSpecificData"] = {"shareToFeed": True}
        elif plat == "tiktok":
            e["customContent"] = caps["tiktok"]
        elif plat == "x":
            e["customContent"] = caps["x"]
        entries.append(e)
    body = {
        "content": caps["instagram"],
        "mediaItems": [{"url": media_url, "type": "video"}],
        "platforms": entries,
        "publishNow": True,
        "tags": caps["tags"][:10],
    }
    if "tiktok" in platforms:
        body["tiktokSettings"] = {
            "privacy_level": p.get("tiktok_privacy", "PUBLIC_TO_EVERYONE"),
            "allow_comment": True, "allow_duet": True, "allow_stitch": True,
            "content_preview_confirmed": True, "express_consent_given": True,
        }
    return _zernio_post(body, platforms)


def zernio_text(text: str) -> dict:
    return _zernio_post({"content": text, "publishNow": True,
                         "platforms": [{"platform": "twitter", "accountId": _zernio_account("x")}]}, ["x"])["x"]


# --------------------------------------------------------------------------- X API (OAuth 1.0a)

def _x_auth():
    from requests_oauthlib import OAuth1
    return OAuth1(env("X_API_KEY"), env("X_API_SECRET"), env("X_ACCESS_TOKEN"), env("X_ACCESS_TOKEN_SECRET"))


def _x_check(r: requests.Response, what: str) -> dict:
    if r.status_code in (401, 402, 403) and "duplicate" not in r.text.lower():
        _disabled.add("x_api")
    if r.status_code not in (200, 201, 202, 204):
        raise RuntimeError(f"X {what} {r.status_code}: {r.text[:300]}")
    return r.json() if r.text else {}


def _x_tweet(text: str, media_id: str | None = None) -> dict:
    body: dict = {"text": text}
    if media_id:
        body["media"] = {"media_ids": [media_id]}
    d = _x_check(requests.post(f"{X_API}/tweets", json=body, auth=_x_auth(), timeout=60), "tweet")
    tid = d.get("data", {}).get("id")
    return {"ok": True, "id": tid, "url": f"https://x.com/i/status/{tid}"}


def x_api_video(video: Path, caps: dict) -> dict:
    auth = _x_auth()
    size = video.stat().st_size
    d = _x_check(requests.post(f"{X_API}/media/upload/initialize", auth=auth, timeout=60, json={
        "media_type": "video/mp4", "total_bytes": size, "media_category": "tweet_video"}), "media init")
    media_id = d["data"]["id"]
    chunk = 4 * 1024 * 1024
    with open(video, "rb") as f:
        idx = 0
        while data := f.read(chunk):
            _x_check(requests.post(f"{X_API}/media/upload/{media_id}/append", auth=auth, timeout=120,
                                   data={"segment_index": idx}, files={"media": ("chunk", data, "application/octet-stream")}),
                     "media append")
            idx += 1
    d = _x_check(requests.post(f"{X_API}/media/upload/{media_id}/finalize", auth=auth, timeout=60), "media finalize")
    info = d.get("data", {}).get("processing_info")
    deadline = time.time() + 300
    while info and info.get("state") in ("pending", "in_progress") and time.time() < deadline:
        time.sleep(max(2, int(info.get("check_after_secs", 5))))
        d = _x_check(requests.get(f"{X_API}/media/upload", auth=auth, timeout=30,
                                  params={"command": "STATUS", "media_id": media_id}), "media status")
        info = d.get("data", {}).get("processing_info")
    if info and info.get("state") == "failed":
        raise RuntimeError(f"X media processing failed: {info}")
    return _x_tweet(caps["x"], media_id)


# --------------------------------------------------------------------------- Upload-Post

def _up_headers() -> dict:
    return {"Authorization": f"Apikey {env('UPLOAD_POST_API_KEY')}"}


def _up_result(r: requests.Response, platforms: list[str]) -> dict:
    if r.status_code in (401, 402, 403, 429) or (r.status_code >= 400 and "limit" in r.text.lower()):
        _disabled.add("upload_post")
    if r.status_code not in (200, 201, 202):
        return {p: {"ok": False, "error": f"upload-post {r.status_code}: {r.text[:300]}"} for p in platforms}
    data = r.json()
    if data.get("usage"):
        log.info("[upload-post] usage: %s", data["usage"])
    results = data.get("results") or {}
    out = {}
    for p in platforms:
        res = results.get(p) if isinstance(results, dict) else None
        if res is None:  # async / background upload accepted
            out[p] = {"ok": bool(data.get("success", True)), "url": "", "id": data.get("request_id")}
        else:
            out[p] = {"ok": bool(res.get("success")), "url": res.get("url", ""), "error": res.get("error")}
    return out


def upload_post_video(video: Path, platforms: list[str], caps: dict) -> dict:
    data = [("user", env("UPLOAD_POST_PROFILE", "default")), ("title", caps["x"]),
            ("async_upload", "true"), ("request_id", str(uuid.uuid4()))]
    data += [("platform[]", p) for p in platforms]
    if "youtube" in platforms:
        data += [("youtube_title", caps["youtube_title"]), ("youtube_description", caps["youtube_description"]),
                 ("privacyStatus", _pcfg().get("youtube_privacy", "public")), ("selfDeclaredMadeForKids", "false"),
                 ("categoryId", str(_pcfg().get("youtube_category_id", "20")))]
        data += [("tags[]", t) for t in caps["tags"][:10]]
    if "instagram" in platforms:
        data += [("instagram_title", caps["instagram"]), ("media_type", "REELS"), ("share_to_feed", "true")]
    if "tiktok" in platforms:
        data += [("tiktok_title", caps["tiktok"])]
    if "x" in platforms:
        data += [("x_title", caps["x"])]
    with open(video, "rb") as f:
        r = requests.post(f"{UPLOAD_POST}/upload", headers=_up_headers(), data=data,
                          files={"video": (video.name, f, "video/mp4")}, timeout=600)
    return _up_result(r, platforms)


def upload_post_text(text: str) -> dict:
    r = requests.post(f"{UPLOAD_POST}/upload_text", headers=_up_headers(), timeout=60, data=[
        ("user", env("UPLOAD_POST_PROFILE", "default")), ("platform[]", "x"), ("title", text)])
    return _up_result(r, ["x"])["x"]


# --------------------------------------------------------------------------- orchestration

def publish_video(video: Path, platforms: list[str], copy: dict, credit: str) -> dict:
    """Returns {platform: {"ok": bool, "provider": str, "url"/"error": ...}}."""
    caps = build_captions(copy, credit)
    results: dict[str, dict] = {}
    pending = {p: [prov for prov in routes_for(p)] for p in platforms}

    while pending:
        # assign each pending platform to its next available provider
        batches: dict[str, list[str]] = {}
        for plat, provs in list(pending.items()):
            while provs and not _available(provs[0], plat):
                provs.pop(0)
            if not provs:
                results.setdefault(plat, {"ok": False, "error": "no configured provider succeeded"})
                pending.pop(plat)
                continue
            batches.setdefault(provs.pop(0), []).append(plat)
        for provider, plats in batches.items():
            log.info("[publish] %s -> %s", provider, plats)
            try:
                if provider == "youtube_api":
                    out = {"youtube": youtube_api_upload(video, caps)}
                elif provider == "x_api":
                    out = {"x": x_api_video(video, caps)}
                elif provider == "x_browser":
                    import x_browser
                    out = {"x": x_browser.post(caps["x"], video)}
                elif provider == "zernio":
                    out = zernio_video(video, plats, caps)
                else:
                    out = upload_post_video(video, plats, caps)
            except Exception as exc:  # noqa: BLE001
                out = {p: {"ok": False, "error": f"{provider}: {exc}"} for p in plats}
            for plat in plats:
                res = out.get(plat, {"ok": False, "error": "no result"})
                res["provider"] = provider
                if res.get("ok"):
                    log.info("[publish] ✅ %s via %s %s", plat, provider, res.get("url", ""))
                    results[plat] = res
                    pending.pop(plat, None)
                else:
                    log.error("[publish] ❌ %s via %s: %s", plat, provider, res.get("error"))
                    results[plat] = res
    return results


def add_hashtags(text: str, tags: list[str] | None = None) -> str:
    """Make sure a text post carries a couple of hashtags (without passing 280 chars)."""
    tags = tags or load_config().get("x_text_hashtags", ["#GTA6", "#GTAVI"])
    missing = [t for t in tags if t.lower() not in text.lower()]
    if not missing or not text.split():
        return text
    sep = " " if text.split()[-1].startswith("#") else "\n\n"
    addition = ""
    for t in missing:
        if len(text) + len(sep) + len(addition) + len(t) + 1 <= 280:
            addition += (" " if addition else "") + t
    return text + sep + addition if addition else text


def publish_text(text: str) -> dict:
    provs = list(routes_for("x"))
    last = {"ok": False, "error": "no configured X provider"}
    for provider in provs:
        if not _available(provider, "x"):
            continue
        try:
            if provider == "x_api":
                res = _x_tweet(text)
            elif provider == "x_browser":
                import x_browser
                res = x_browser.post(text)
            elif provider == "zernio":
                res = zernio_text(text)
            else:
                res = upload_post_text(text)
        except Exception as exc:  # noqa: BLE001
            res = {"ok": False, "error": f"{provider}: {exc}"}
        res["provider"] = provider
        if res.get("ok"):
            log.info("[publish] ✅ X text via %s %s", provider, res.get("url", ""))
            return res
        log.error("[publish] ❌ X text via %s: %s", provider, res.get("error"))
        last = res
    return last
