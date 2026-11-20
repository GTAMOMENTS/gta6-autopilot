"""Video format rotation: every video run uses the NEXT format in config.json -> formats.rotation
(top 5 funniest moments -> car chases -> top 5 things to do -> secrets -> ...).

The format decides what we search for, which titles rank higher, and what the AI is told to make.
The position is kept in queue/format_rotation.json (saved between runs by the Actions cache).
"""
from __future__ import annotations

import json
import logging
import random
import re

from common import STATE_DIR, env, load_config, load_json, save_json

log = logging.getLogger("autopilot")
ROTATION_FILE = STATE_DIR / "format_rotation.json"


def _all_formats(cfg: dict | None = None) -> list[dict]:
    f = (cfg or load_config()).get("formats", {})
    if not f.get("enabled", True):
        return []
    return [x for x in f.get("rotation", []) if x.get("enabled", True) and x.get("id")]


def _rotation(cfg: dict | None = None) -> list[dict]:
    """The normal rotation (Video A). slot_only formats (the Money Tip) are made only by their own slot."""
    return [x for x in _all_formats(cfg) if not x.get("slot_only")]


def format_by_id(fid: str, cfg: dict | None = None) -> dict | None:
    return next((x for x in _all_formats(cfg) if x["id"].lower() == (fid or "").lower()), None)


def current_format(cfg: dict | None = None, slot: dict | None = None) -> dict | None:
    """The format this run should make. FORMAT env (manual runs) overrides; a slot with "format" (Video B =
    money_tips) always makes that one; otherwise the rotation (+ learning, trends, stream health) decides."""
    cfg = cfg or load_config()
    rot = _rotation(cfg)
    wanted = env("FORMAT").strip().lower()
    if wanted and wanted != "auto":
        match = format_by_id(wanted, cfg)
        if match:
            return specialize(match)
        log.warning("[format] unknown FORMAT '%s' - using the rotation", wanted)
    if (slot or {}).get("format"):
        match = format_by_id(slot["format"], cfg)
        if match:
            return specialize(match)
        log.warning("[format] slot format '%s' not found / disabled - using the rotation", slot["format"])
    if not rot:
        return None
    state = load_json(ROTATION_FILE, {"next": 0, "history": []})
    in_turn = rot[int(state.get("next", 0)) % len(rot)]
    pick = _learned_choice(cfg, rot, state, in_turn)
    if pick.get("kind") == "stream" and pick["id"] != "trending":
        try:
            import trends
            health = trends.stream_health()
            if health["weak"]:  # streams died down -> follow what people search for instead
                log.info("[trends] GTA 6 streams are quiet (%s viewers, 7-day peak %s) - trending topic instead of %s",
                         health["viewers"], health["peak"], pick["id"])
                pick = format_by_id("trending", cfg) or pick
        except Exception as exc:  # noqa: BLE001
            log.info("[trends] stream health unknown: %.100s", exc)
    made = specialize(pick)
    if made is None:  # e.g. "trending" with no trend data yet -> the next format in the rotation
        nxt = rot[(rot.index(pick) + 1) % len(rot)] if pick in rot else in_turn
        made = specialize(nxt) if nxt is not pick else None
    return made


def specialize(fmt: dict | None) -> dict | None:
    """Fill in today's topic: Money Tip -> the next money topic (a hot money trend first); "trending" -> the
    hottest trend topic. Returns a copy (None = this format can't be made right now)."""
    if not fmt:
        return None
    fmt = json.loads(json.dumps(fmt))
    if fmt["id"] == "trending":
        import trends
        t = trends.pick(money=False, min_demand=int(fmt.get("min_demand", 5)))
        if not t:
            log.info("[trends] no fresh trend topic - skipping the trending format")
            return None
        _apply_topic(fmt, t, trend=True)
        return fmt
    topics = fmt.get("topics") or []
    if topics:
        t = None
        try:
            import trends
            t = trends.pick(money=True, min_demand=int(fmt.get("trend_min_demand", 7))) if fmt.get("use_trends") else None
        except Exception:  # noqa: BLE001
            t = None
        if t:
            _apply_topic(fmt, t, trend=True)
        else:
            state = load_json(ROTATION_FILE, {})
            recent = state.get("topics_used", {}).get(fmt["id"], [])[-(len(topics) - 1):] if len(topics) > 1 else []
            t = next((x for x in topics if x["topic"] not in recent), topics[0])
            _apply_topic(fmt, t, trend=False)
    return fmt


def _apply_topic(fmt: dict, t: dict, trend: bool) -> None:
    fmt["topic"], fmt["topic_from_trends"] = t["topic"], trend
    fmt["queries"] = list(dict.fromkeys(list(t.get("queries") or []) + list(fmt.get("queries") or [])))[:8]
    words = [w for w in re.findall(r"[a-z]{4,}", t["topic"].lower()) if w not in ("gta", "with", "your", "best")]
    fmt["title_terms"] = list(dict.fromkeys(words + list(fmt.get("title_terms") or [])))
    fmt["brief"] = (fmt.get("brief", "") + f' TODAY\'S TOPIC: "{t["topic"]}" - {t.get("angle", "")}').strip()
    if fmt["id"] == "trending":
        fmt["name"] = f"Trending: {t['topic']}"
        fmt["banner_example"] = f"GTA 6 {t['topic'].upper()}"[:40]
        fmt["title_example"] = f"GTA 6 {t['topic']} - What Nobody Tells You"
        fmt["keywords"] = f"GTA 6 {t['topic']}"
    else:
        fmt["name"] = f"{fmt['name']}: {t['topic']}"
    log.info("[format] topic: %s%s", t["topic"], " (trending now)" if trend else "")


def mark_topic(fmt: dict | None) -> None:
    """After a successful post: remember the topic so the next videos pick different ones."""
    if not fmt or not fmt.get("topic"):
        return
    if fmt.get("topic_from_trends"):
        import trends
        trends.mark_used(fmt["topic"])
        return
    state = load_json(ROTATION_FILE, {"next": 0, "history": []})
    used = state.setdefault("topics_used", {}).setdefault(fmt["id"], [])
    used.append(fmt["topic"])
    state["topics_used"][fmt["id"]] = used[-30:]
    save_json(ROTATION_FILE, state)


def _learned_choice(cfg: dict, rot: list[dict], state: dict, in_turn: dict) -> dict:
    """Favour the formats the audience watches most (YouTube views/likes/comments), but keep exploring:
    plain rotation until every format has enough data, then mostly the best recent performers."""
    lcfg = cfg.get("learning", {})
    if not lcfg.get("enabled", True):
        return in_turn
    try:
        import analytics
        scores = analytics.format_scores()
    except Exception as exc:  # noqa: BLE001
        log.info("[learn] no scores yet (%s)", exc)
        return in_turn
    need = int(lcfg.get("min_samples", 2))
    if any(scores.get(f["id"], {}).get("n", 0) < need for f in rot):
        log.info("[learn] still collecting data (%s) - normal rotation",
                 {f["id"]: scores.get(f["id"], {}).get("n", 0) for f in rot})
        return in_turn
    if random.random() < float(lcfg.get("explore", 0.25)):
        log.info("[learn] exploring - format in turn: %s", in_turn["id"])
        return in_turn
    recent = [h.get("format") for h in state.get("history", [])[-int(lcfg.get("no_repeat", 2)):]]
    ranked = sorted((f for f in rot if f["id"] not in recent), key=lambda f: scores[f["id"]]["score"], reverse=True)
    pick = ranked[0] if ranked else in_turn
    log.info("[learn] best performers: %s -> %s",
             [(f["id"], scores[f["id"]]["score"]) for f in ranked[:3]], pick["id"])
    return pick


def series_label(fmt: dict | None) -> str:
    """e.g. "GTA 6 Car Chases #4" - numbered series make people binge the channel."""
    if not fmt or not fmt.get("series"):
        return ""
    state = load_json(ROTATION_FILE, {"next": 0, "history": []})
    return f'{fmt["series"]} #{int(state.get("series", {}).get(fmt["id"], 0)) + 1}'


def bump_series(fmt: dict | None) -> None:
    if not fmt or not fmt.get("series"):
        return
    state = load_json(ROTATION_FILE, {"next": 0, "history": []})
    state.setdefault("series", {})[fmt["id"]] = int(state.get("series", {}).get(fmt["id"], 0)) + 1
    save_json(ROTATION_FILE, state)


def advance_format(fmt: dict | None, ok: bool) -> None:
    """Move the rotation on, so the next video run makes a different format."""
    rot = _rotation()
    if not fmt or not rot:
        return
    ids = [x["id"] for x in rot]
    state = load_json(ROTATION_FILE, {"next": 0, "history": []})
    pos = ids.index(fmt["id"]) if fmt["id"] in ids else int(state.get("next", 0))
    state["next"] = (pos + 1) % len(rot)
    state["history"] = (state.get("history", []) + [{"format": fmt["id"], "ok": ok}])[-50:]
    save_json(ROTATION_FILE, state)
    log.info("[format] next video will be: %s", rot[state["next"]]["name"])


def prompt_note(fmt: dict | None, tour: bool) -> str:
    """Instructions for the AI editor/writer about this run's format."""
    if not fmt:
        return ""
    lines = [f'THIS VIDEO\'S FORMAT: "{fmt["name"]}". {fmt.get("brief", "")}']
    if tour and fmt.get("countdown"):
        n = int(fmt.get("countdown"))
        word = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}.get(n, str(n))
        lines.append(f"Make it a countdown of exactly {n} numbered items: line 1 is the hook that promises the "
                     f"list, then \"Number {word}...\" down to \"Number one...\" (the best item last). Each item "
                     "gets 1-2 shots and says WHY it's on the list.")
    elif not tour:
        lines.append("This is ONE continuous moment, not a list: keep the format's theme in the banner, title "
                     "and lines, but don't number anything.")
    if fmt.get("banner_example"):
        lines.append(f'Banner like: "{fmt["banner_example"]}". YouTube title like: "{fmt.get("title_example", "")}".')
    lines.append("Pick the shots/part that fit this format best. If this footage really cannot honestly make "
                 "this format, set fits_format=false (and still fill everything in with the closest angle).")
    return "\n".join(lines)
