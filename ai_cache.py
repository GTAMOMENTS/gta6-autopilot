"""Memory of AI verdicts between runs (queue/ai_verdicts.json, kept by the Actions state cache).

The free Gemini quota is small and every run used to re-check the same footage. Stored here:
  auth      "which game is this?" verdicts (authentic.py) per source key + window (the part that was checked)
  fake_src  sources with a confidently fake part (GTA V / mod / other game) -> never downloaded again
  channels  per uploader: which of its videos passed / failed that check -> trusted channels rank higher, channels
            that keep uploading GTA V / mods as "GTA 6" are skipped
  reviews   long-video batch review verdicts per (prompt, clip key, section)
  bad       clips a review found not GTA 6 / broken / spoiler / unsafe / not English / glitch -> never downloaded again

Rules: errors (no AI answer, a clip missing from the answer) are never stored; a prompt change invalidates (the prompt
hash is part of the entry); every entry expires (TTLs in config `authenticity` / `sources`).
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time

from common import STATE_DIR, load_config, load_json, save_json

log = logging.getLogger("autopilot")

CACHE_FILE = STATE_DIR / "ai_verdicts.json"
VERSION = 1
_lock = threading.RLock()
_data: dict | None = None

REAL_GAMES = {"GTAVI", "GTA6"}
FAKE_GAMES = {"GTAV", "GTA5", "GTAONLINE", "MOD", "OTHER"}
# uploaders whose name means something (Twitch / Kick clips come from the GTA VI category - trusted anyway)
CHANNEL_PROVIDERS = {"youtube", "dailymotion", "x_account", "x", "extra"}


def _acfg() -> dict:
    return load_config().get("authenticity", {})


def _scfg() -> dict:
    return load_config().get("sources", {})


def enabled() -> bool:
    """config authenticity.cache; env AI_VERDICT_CACHE=0 switches it off (e.g. local test harnesses)."""
    return bool(_acfg().get("cache", True)) and os.getenv("AI_VERDICT_CACHE", "1").strip().lower() not in (
        "0", "false", "off", "no")


def digest(*parts) -> str:
    return hashlib.md5("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:10]


def _load() -> dict:
    global _data
    with _lock:
        if _data is None:
            d = load_json(CACHE_FILE, {})
            if not isinstance(d, dict) or d.get("version") != VERSION:
                d = {"version": VERSION}
            for k in ("auth", "fake_src", "channels", "reviews", "bad"):
                if not isinstance(d.get(k), dict):
                    d[k] = {}
            _data = d
            _prune(d)
        return _data


def _save() -> None:
    with _lock:
        if _data is not None:
            try:
                save_json(CACHE_FILE, _data)
            except OSError as exc:  # a full disk must never stop a run
                log.warning("[cache] could not save AI verdicts: %s", exc)


def reset() -> None:
    """Forget the in-memory copy (tests)."""
    global _data
    with _lock:
        _data = None


def _prune(d: dict) -> None:
    now = time.time()
    keep_auth = 86400 * max(float(_acfg().get("cache_days_fake", 60)), float(_acfg().get("cache_days_real", 30)))
    for sect in ("auth", "fake_src"):
        d[sect] = {k: v for k, v in d[sect].items() if isinstance(v, dict) and now - v.get("at", 0) < keep_auth}
    rdays = 86400 * float(_scfg().get("review_cache_days", 10))
    d["reviews"] = {k: v for k, v in d["reviews"].items() if isinstance(v, dict) and now - v.get("at", 0) < rdays}
    bdays = 86400 * float(_scfg().get("bad_clip_days", 14))
    d["bad"] = {k: v for k, v in d["bad"].items() if isinstance(v, dict) and now - v.get("at", 0) < bdays}
    cdays = 86400 * float(_acfg().get("channel_memory_days", 90))
    for name, ch in list(d["channels"].items()):
        for kind in ("real", "fake"):
            ch[kind] = dict(sorted(((k, t) for k, t in (ch.get(kind) or {}).items() if now - t < cdays),
                                   key=lambda kv: kv[1])[-100:])
        if not ch["real"] and not ch["fake"]:
            d["channels"].pop(name, None)
    for sect, cap in (("auth", 6000), ("fake_src", 3000), ("reviews", 4000), ("bad", 4000)):
        if len(d[sect]) > cap:
            d[sect] = dict(sorted(d[sect].items(), key=lambda kv: kv[1].get("at", 0))[-cap:])


# --------------------------------------------------------------------------- authenticity

def classify(game: str, confidence: float, need: float | None = None) -> str:
    """'real' (confident GTA VI), 'fake' (confident GTA V / Online / mod / other game) or 'unsure'."""
    need = float(_acfg().get("min_confidence", 0.65)) if need is None else need
    g = str(game or "").replace(" ", "").upper()
    try:
        conf = float(confidence or 0)
    except (TypeError, ValueError):
        conf = 0.0
    if g in REAL_GAMES and conf >= need:
        return "real"
    if g in FAKE_GAMES and conf >= need:
        return "fake"
    return "unsure"


def _ttl_days(cls: str) -> float:
    a = _acfg()
    return float({"real": a.get("cache_days_real", 30), "fake": a.get("cache_days_fake", 60)}.get(
        cls, a.get("cache_days_unsure", 2)))


def _window(section) -> str:
    return "-".join(str(int(float(x))) for x in section) if section else "all"


def auth_get(key: str, pv: str, section=None) -> dict | None:
    """A still-valid cached verdict {game, confidence, evidence, ...} for exactly this source AND window (a pass
    for one part of a video says nothing about another part - compilations mix games), else None."""
    if not key or not enabled():
        return None
    with _lock:
        e = _load()["auth"].get(f"{key}@{_window(section)}")
    if not e or e.get("pv") != pv:
        return None
    if time.time() - e.get("at", 0) > _ttl_days(classify(e.get("game"), e.get("confidence"))) * 86400:
        return None
    return e


def auth_put(key: str, pv: str, verdict: dict, section=None, channel: str = "", provider: str = "") -> None:
    """Remember a REAL answer from the AI (callers never pass errors)."""
    if not key or not enabled() or not verdict:
        return
    entry = {"game": str(verdict.get("game") or "unsure"), "confidence": float(verdict.get("confidence") or 0),
             "evidence": str(verdict.get("evidence") or "")[:160], "pv": pv, "at": time.time(),
             "section": list(section) if section else None, "channel": channel or "", "provider": provider or ""}
    cls = classify(entry["game"], entry["confidence"])
    with _lock:
        d = _load()
        d["auth"][f"{key}@{_window(section)}"] = entry
        if cls == "fake":  # one confidently fake part = don't use this video again (stricter, never looser)
            d["fake_src"][key] = entry
        if channel and provider in CHANNEL_PROVIDERS and cls in ("real", "fake"):
            ch = d["channels"].setdefault(channel_id(channel), {"name": channel, "real": {}, "fake": {}})
            ch["name"] = channel
            other = "fake" if cls == "real" else "real"
            ch[other].pop(key, None)  # the newest answer for a video counts
            ch[cls][key] = time.time()
        _save()


def auth_fake(key: str) -> dict | None:
    """The cached verdict if this source was confidently found NOT to be GTA 6 (any prompt version), else None."""
    if not key or not enabled():
        return None
    with _lock:
        e = _load()["fake_src"].get(key)
    if not e or classify(e.get("game"), e.get("confidence")) != "fake":
        return None
    if time.time() - e.get("at", 0) > _ttl_days("fake") * 86400:
        return None
    return e


def channel_id(name: str) -> str:
    n = (name or "").lower().strip()
    n = n.removesuffix(" on x").removesuffix(" on twitch")
    return "".join(ch for ch in n if ch.isalnum())


def channel_stats(name: str) -> tuple[int, int]:
    """(videos that passed, videos that failed) the authenticity check for this uploader."""
    if not name or not enabled():
        return 0, 0
    with _lock:
        ch = _load()["channels"].get(channel_id(name))
    if not ch:
        return 0, 0
    return len(ch.get("real") or {}), len(ch.get("fake") or {})


def channels_summary(limit: int = 6) -> str:
    with _lock:
        chans = list(_load()["channels"].values())
    trusted = sorted((c for c in chans if c.get("real") and not c.get("fake")), key=lambda c: -len(c["real"]))
    fake = sorted((c for c in chans if c.get("fake")), key=lambda c: -len(c["fake"]))
    return (f"trusted: {[c['name'] for c in trusted[:limit]]} | fake/mod: "
            f"{[c['name'] + ' ' + str(len(c['fake'])) + 'x' for c in fake[:limit]]}")


# --------------------------------------------------------------------------- long-video review verdicts

def review_ns(prompt_text: str, volatile: str = "") -> str:
    """Prompt identity: the prompt without its volatile parts (today's date / days since launch)."""
    text = prompt_text.replace(volatile, "") if volatile else prompt_text
    return digest(text)


def _rkey(ns: str, key: str, section) -> str:
    return f"{ns}|{key}|{'-'.join(str(int(x)) for x in section) if section else ''}"


def review_get(ns: str, key: str, section=None) -> dict | None:
    if not key or not enabled():
        return None
    with _lock:
        e = _load()["reviews"].get(_rkey(ns, key, section))
    if not e or time.time() - e.get("at", 0) > float(_scfg().get("review_cache_days", 10)) * 86400:
        return None
    return dict(e.get("v") or {}) or None


BAD_FLAGS = (("is_gta6_footage", False, "not GTA 6"), ("broken", True, "broken picture"),
             ("is_spoiler", True, "spoiler"), ("unsafe", True, "unsafe"), ("is_english", False, "not English"),
             ("is_glitch", True, "glitch / exploit"))


def review_put(ns: str, key: str, section, verdict: dict) -> None:
    """Remember one clip's verdict (only real answers). Problems that belong to the clip itself (not GTA 6,
    broken, spoiler, unsafe, not English, glitch) also mark the clip as bad for every prompt."""
    if not key or not enabled() or not isinstance(verdict, dict) or not verdict:
        return
    v = {k: val for k, val in verdict.items() if k != "clip"}
    with _lock:
        d = _load()
        d["reviews"][_rkey(ns, key, section)] = {"v": v, "at": time.time()}
        for field, bad_value, reason in BAD_FLAGS:
            if field in v and v[field] is bad_value:
                d["bad"][key] = {"reason": reason, "at": time.time()}
                break
        _save()


def bad_reason(key: str) -> str | None:
    if not key or not enabled():
        return None
    with _lock:
        e = _load()["bad"].get(key)
    if not e or time.time() - e.get("at", 0) > float(_scfg().get("bad_clip_days", 14)) * 86400:
        return None
    return e.get("reason") or "rejected before"
