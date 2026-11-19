"""Video format rotation: every video run uses the NEXT format in config.json -> formats.rotation
(top 5 funniest moments -> car chases -> top 5 things to do -> secrets -> ...).

The format decides what we search for, which titles rank higher, and what the AI is told to make.
The position is kept in queue/format_rotation.json (saved between runs by the Actions cache).
"""
from __future__ import annotations

import logging
import random

from common import STATE_DIR, env, load_config, load_json, save_json

log = logging.getLogger("autopilot")
ROTATION_FILE = STATE_DIR / "format_rotation.json"


def _rotation(cfg: dict | None = None) -> list[dict]:
    f = (cfg or load_config()).get("formats", {})
    if not f.get("enabled", True):
        return []
    return [x for x in f.get("rotation", []) if x.get("enabled", True) and x.get("id")]


def current_format(cfg: dict | None = None) -> dict | None:
    """The format this run should make. FORMAT env (manual runs) overrides the rotation."""
    rot = _rotation(cfg)
    if not rot:
        return None
    wanted = env("FORMAT").strip().lower()
    if wanted and wanted != "auto":
        match = next((x for x in rot if x["id"].lower() == wanted), None)
        if match:
            return match
        log.warning("[format] unknown FORMAT '%s' - using the rotation", wanted)
    state = load_json(ROTATION_FILE, {"next": 0, "history": []})
    in_turn = rot[int(state.get("next", 0)) % len(rot)]
    return _learned_choice(cfg or load_config(), rot, state, in_turn)


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
