"""Is it really GTA 6? A focused check on sharp frames, before any clip is used.

Many videos titled "GTA 6" are GTA V footage, GTA V mods or other games (uploaded before AND after launch). The
general clip reviews (low-res proxy, many questions at once) were fooled - the owner and QA caught GTA V clips
in test videos. This asks ONE question on 4 sharp frames per clip: which game is this?
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from common import env, load_config, probe

log = logging.getLogger("autopilot")

PROMPT = """GAME FACTS (newer than your training data - trust them): {facts}

Above are sharp frames from {k} different video clips (the clip number is written before its frames).
For EACH clip decide which game it really is. Look closely at:
- the minimap / HUD (shape, position, style) - GTA V's minimap is a rounded rectangle bottom-left with green
  health / blue armour bars under it
- the city and landscape (Leonida: Vice City neon, beaches, swamps, the Keys - vs GTA V's Los Santos / Blaine
  County, the Vinewood sign, Mount Chiliad)
- the characters (Jason, Lucia vs Michael, Franklin, Trevor), car models, police, graphics generation
"mod" = GTA V modified to look like GTA 6 (new textures / lighting on GTA V's map and HUD). If it is GTA V with
better lighting, it is GTA V. Only say "GTA VI" if the frames really look like the new game.
Return ONLY JSON: {{"clips": [{{"clip": number, "game": "GTA VI" | "GTA V" | "GTA Online" | "mod" | "other" |
"unsure", "confidence": number (0-1), "evidence": str}}]}}"""


def _cfg() -> dict:
    return load_config().get("authenticity", {})


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


def check(paths: list[Path]) -> list[dict] | None:
    """[{game, confidence, evidence, ok}] per path (same order), or None if no AI could look."""
    import gemini_writer
    if not env("GEMINI_API_KEY") or not paths:
        return None
    from google.genai import types
    c = _cfg()
    contents, k = [], 0
    for i, p in enumerate(paths, 1):
        imgs = frames(p, int(c.get("frames", 4)), int(c.get("width", 1280)))
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
    need = float(c.get("min_confidence", 0.65))
    out = []
    for i in range(1, len(paths) + 1):
        v = by.get(i) or {}
        game = str(v.get("game") or "unsure")
        try:
            conf = float(v.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        out.append({"game": game, "confidence": conf, "evidence": str(v.get("evidence") or "")[:160],
                    "ok": game.replace(" ", "").upper() in ("GTAVI", "GTA6") and conf >= need})
    return out


def keep_real(clips: list[dict], tag: str = "[long]") -> list[dict]:
    """Long videos: drop every clip (dict with "path") that isn't confidently GTA 6. Batches of 6."""
    if not _cfg().get("enabled", True):
        return clips
    kept = []
    size = int(_cfg().get("batch", 6))
    for b in range(0, len(clips), size):
        batch = clips[b:b + size]
        res = check([Path(m["path"]) for m in batch])
        if res is None:
            log.warning("%s authenticity check unavailable for %d clips - not using them (strict)", tag, len(batch))
            continue
        for m, r in zip(batch, res):
            if r["ok"]:
                kept.append(m)
            else:
                log.info("%s ✋ %s (%s): not GTA 6 - %s %.0f%% (%s)", tag, m.get("key"), m.get("streamer"),
                         r["game"], r["confidence"] * 100, r["evidence"])
                Path(m["path"]).unlink(missing_ok=True)
    log.info("%s authenticity: %d/%d clips are real GTA 6", tag, len(kept), len(clips))
    return kept
