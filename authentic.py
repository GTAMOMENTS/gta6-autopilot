"""Is it really GTA 6? A focused check on sharp frames, before any clip is used.

Many videos titled "GTA 6" are GTA V footage, GTA V mods or other games (uploaded before AND after launch). The
general clip reviews (low-res proxy, many questions at once) were fooled - the owner and QA caught GTA V clips
in test videos. This asks ONE question on 4 sharp frames per clip: which game is this?

Answers are remembered per source (ai_cache.py): a video is checked once, not every run; channels whose videos
passed rank higher, channels that keep uploading GTA V / mods as "GTA 6" are skipped.
"""
from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ai_cache
from common import env, load_config, probe

log = logging.getLogger("autopilot")

PROMPT = """GAME FACTS (newer than your training data - trust them): {facts}

Above are sharp frames from {k} different video clips (the clip number is written before its frames).
For EACH clip decide which game it really is. GTA VI only came out on the date above, so you have never seen its
real HUD / menus - NEVER decide on the HUD, minimap or menus alone (GTA VI's may look similar to GTA V's).
Decide on things that are certain:
- GTA V: Los Santos / Blaine County places (the Vinewood sign, Maze Bank tower, Mount Chiliad, Del Perro pier),
  Michael / Franklin / Trevor / Lamar, GTA V's older graphics and car / character models
- GTA VI: Leonida places (Vice City, its beaches and neon, swamps, the Keys, Port Gellhorn, Ambrosia), Jason / Lucia,
  clearly newer-generation graphics
- "mod": GTA V's world / characters with new textures or lighting, or obviously fan-made assets
- "other": a different game; "unsure": nothing certain in the frames
Return ONLY JSON: {{"clips": [{{"clip": number, "game": "GTA VI" | "GTA V" | "GTA Online" | "mod" | "other" |
"unsure", "confidence": number (0-1), "evidence": str}}]}}"""

# path -> what it is (filled by the downloaders, so a check([path]) call knows which video / channel it is about)
_SOURCES: dict[str, dict] = {}


def _cfg() -> dict:
    return load_config().get("authenticity", {})


def prompt_version() -> str:
    """Changes when the question or the game facts change -> old answers are not reused."""
    return ai_cache.digest(PROMPT, load_config().get("game", {}).get("facts", ""), _cfg().get("frames", 4),
                           _cfg().get("width", 1280))


def remember(path: Path, key: str, channel: str = "", provider: str = "", section=None) -> None:
    """Tell the check which source a downloaded file is (its verdict is then cached per source key)."""
    if path and key:
        _SOURCES[str(Path(path))] = {"key": key, "channel": channel or "", "provider": provider or "",
                                     "section": list(section) if section else None}


def known_fake(key: str) -> str | None:
    """'GTA V 90%' etc. if this source was confidently found NOT to be GTA 6 before, else None."""
    e = ai_cache.auth_fake(key)
    return f"{e['game']} {float(e['confidence']) * 100:.0f}%" if e else None


def channel_verdict(name: str) -> tuple[str, float]:
    """('fake' | 'trusted' | 'suspect' | '', ranking adjustment) for an uploader, from earlier checks."""
    real, fake = ai_cache.channel_stats(name)
    c = _cfg()
    if fake >= int(c.get("block_channel_after_fakes", 2)) and fake > 2 * real:
        return "fake", -99.0
    if fake > real:
        return "suspect", -float(c.get("fake_channel_penalty", 3.0)) * (fake - real)
    if real and not fake:
        return "trusted", float(c.get("trusted_channel_bonus", 2.5)) * (1.0 if real < 3 else 1.4)
    return "", 0.0


def frames(path: Path, n: int = 4, width: int = 1280) -> list[bytes]:
    dur = probe(path)["duration"] or 0
    out = []
    for k in range(n):
        t = dur * (0.12 + 0.76 * k / max(1, n - 1))
        res = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                              "-vf", f"scale={width}:-2", "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                             capture_output=True, timeout=60)
        if res.returncode == 0 and len(res.stdout) > 2000:
            out.append(res.stdout)
    return out


def _result(game, conf, evidence, need: float, cached: bool = False, error: bool = False) -> dict:
    try:
        conf = float(conf or 0)
    except (TypeError, ValueError):
        conf = 0.0
    game = str(game or "unsure")
    return {"game": game, "confidence": conf, "evidence": str(evidence or "")[:160], "cached": cached,
            "error": error, "ok": (not error) and game.replace(" ", "").upper() in ("GTAVI", "GTA6") and conf >= need}


def _ask(paths: list[Path]) -> list[dict | None] | None:
    """One AI call for these clips: a raw verdict dict per path (None = the AI skipped that clip),
    or None if no AI could look at all."""
    import gemini_writer
    from google.genai import types
    c = _cfg()
    n, w = int(c.get("frames", 4)), int(c.get("width", 1280))
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(paths)))) as ex:  # ffmpeg seeks in parallel
        shots = list(ex.map(lambda p: frames(p, n, w), paths))
    contents, k = [], 0
    for i, imgs in enumerate(shots, 1):
        if not imgs:
            continue
        k += 1
        contents.append(f"Clip {i}:")
        contents += [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in imgs]
    if not k:
        return None
    contents.append(PROMPT.format(facts=gemini_writer._game_context(), k=len(paths)))
    gemini_writer._unavailable_until = 0.0
    data = gemini_writer._generate_json(contents, temperature=0.1, prefer_pro=True)
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
        return None
    by = {}
    for v in data["clips"]:
        try:
            by[int(v.get("clip"))] = v
        except (TypeError, ValueError, AttributeError):
            continue
    return [by.get(i) if shots[i - 1] else None for i in range(1, len(paths) + 1)]


def check(paths: list[Path], meta: list[dict] | None = None) -> list[dict] | None:
    """[{game, confidence, evidence, ok, cached, error}] per path (same order), or None if no AI could look.
    meta (optional, same order): {key, channel, provider, section} - otherwise taken from remember().
    Earlier answers for the same source are reused (no AI call); new answers are remembered. A clip the AI
    gave no answer for comes back as error=True / ok=False and is never remembered."""
    if not paths:
        return None
    c = _cfg()
    need = float(c.get("min_confidence", 0.65))
    pv = prompt_version()
    infos = [dict((meta or [])[i] if meta and i < len(meta) and meta[i] else _SOURCES.get(str(Path(p)), {}))
             for i, p in enumerate(paths)]
    out: list[dict | None] = [None] * len(paths)
    for i, info in enumerate(infos):
        e = ai_cache.auth_get(info.get("key", ""), pv, info.get("section"))
        if e:
            out[i] = _result(e["game"], e["confidence"], e.get("evidence"), need, cached=True)
    todo = [i for i, r in enumerate(out) if r is None]
    if todo:
        raw = _ask([paths[i] for i in todo]) if env("GEMINI_API_KEY") else None
        if raw is None:
            if len(todo) == len(paths):
                return None
            raw = [None] * len(todo)
        for i, v in zip(todo, raw):
            if not v:
                out[i] = _result("unsure", 0, "no AI verdict for this clip", need, error=True)
                continue
            out[i] = _result(v.get("game"), v.get("confidence"), v.get("evidence"), need)
            info = infos[i]
            ai_cache.auth_put(info.get("key", ""), pv, out[i], section=info.get("section"),
                              channel=info.get("channel", ""), provider=info.get("provider", ""))
    hits = sum(1 for r in out if r and r["cached"])
    if hits:
        log.info("[authentic] %d/%d answers reused from earlier runs (no AI call for them)", hits, len(paths))
    return out  # type: ignore[return-value]


def keep_real(clips: list[dict], tag: str = "[long]") -> list[dict]:
    """Long videos: drop every clip (dict with "path") that isn't confidently GTA 6. Batches of 6, a couple of
    batches at the same time (each waits ~1 min for the AI)."""
    if not _cfg().get("enabled", True):
        return clips
    kept = [m for m in clips if m.get("trusted")]  # GTA VI-category stream clips: real by definition
    todo = [m for m in clips if not m.get("trusted")]
    size = max(1, int(_cfg().get("batch", 6)))

    def meta(m: dict) -> dict:
        return {"key": m.get("key", ""), "channel": m.get("streamer", ""), "provider": m.get("platform", ""),
                "section": m.get("section")}

    # answers from earlier runs first: only the rest goes to the AI (in full batches)
    pv = prompt_version()
    known = [m for m in todo if ai_cache.auth_get(m.get("key", ""), pv, m.get("section"))]
    fresh = [m for m in todo if m not in known]
    batches = ([known] if known else []) + [fresh[b:b + size] for b in range(0, len(fresh), size)]
    if len(batches) > 1:
        try:  # model discovery once, before the threads start
            import gemini_writer
            gemini_writer._models()
            gemini_writer._pro_models()
        except Exception:  # noqa: BLE001
            pass
    for m in todo:  # check(paths) finds the source + window through remember()
        mm = meta(m)
        remember(Path(m["path"]), mm["key"], mm["channel"], mm["provider"], mm["section"])
    workers = max(1, int(_cfg().get("parallel_batches", 2)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda batch: (batch, check([Path(m["path"]) for m in batch])), batches))
    for batch, res in results:
        if res is None:
            log.warning("%s authenticity check unavailable for %d clips - not using them (strict)", tag, len(batch))
            continue
        for m, r in zip(batch, res):
            if r["ok"]:
                kept.append(m)
            elif r.get("error"):
                log.info("%s ✋ %s (%s): no authenticity answer - not using it (strict)", tag, m.get("key"),
                         m.get("streamer"))
            else:
                log.info("%s ✋ %s (%s): not GTA 6 - %s %.0f%% (%s)%s", tag, m.get("key"), m.get("streamer"),
                         r["game"], r["confidence"] * 100, r["evidence"], " [remembered]" if r.get("cached") else "")
                Path(m["path"]).unlink(missing_ok=True)
    order = {id(m): i for i, m in enumerate(clips)}
    kept.sort(key=lambda m: (not m.get("trusted"), order.get(id(m), 0)))  # trusted first, then the original order
    log.info("%s authenticity: %d/%d clips are real GTA 6 | channels %s", tag, len(kept), len(clips),
             ai_cache.channels_summary())
    return kept
