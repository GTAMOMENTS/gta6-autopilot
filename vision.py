"""Ask an AI about video frames - Gemini first, then the free backup AIs that can see images.

Every image check (is it really GTA 6? does the picture show what each line says? rewrite lines for their
footage) used Gemini only, so when Gemini's free quota ran out those checks stopped and strict mode blocked
every video. The owner's backup keys (Gemma, Mistral, OpenRouter, NVIDIA... - several can see
images) now take over.

ask(groups, prompt, key, num): groups = [(label, [jpeg bytes, ...])] in order. The answer is a dict with a list
under `key` (e.g. "clips" / "lines") whose items carry their 1-based group number under `num`.
"""
from __future__ import annotations

import base64
import logging

log = logging.getLogger("autopilot")


def _gemini(groups, prompt, temperature, prefer_pro):
    import gemini_writer
    from common import env
    if not env("GEMINI_API_KEY"):
        return None
    from google.genai import types
    contents = []
    for label, imgs in groups:
        contents.append(f"{label}:")
        contents += [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in imgs]
    contents.append(prompt)
    gemini_writer._unavailable_until = 0.0
    return gemini_writer._generate_json(contents, temperature=temperature, prefer_pro=prefer_pro)


def _backup(groups, prompt, key, num, temperature, max_images: int = 8):
    """Backups get a flat list of images (no labels between them) - the legend says which images belong to which
    group. Big requests are split into chunks that fit the providers' image limits, then numbered back."""
    import ai_fallback
    if not ai_fallback.available():
        return None
    per = max(1, min(max(len(g[1]) for g in groups), max_images // max(1, min(len(groups), max_images))))
    chunk = max(1, max_images // per)
    merged = []
    for c0 in range(0, len(groups), chunk):
        part = groups[c0:c0 + chunk]
        flat, legend, n = [], [], 0
        for i, (label, imgs) in enumerate(part, 1):
            use = imgs[:per]
            legend.append(f"images {n + 1}-{n + len(use)} = {num} {i} ({label})")
            flat += [base64.b64encode(b).decode() for b in use]
            n += len(use)
        head = ("You cannot see a video: you get still frames instead, in this order - " + "; ".join(legend) +
                f". Number the {key} 1-{len(part)} in that order.\n\n")
        data = ai_fallback.chat_json(head + prompt, frames=flat, temperature=temperature)
        if not isinstance(data, dict) or not isinstance(data.get(key), list):
            return None
        for v in data[key]:
            try:
                k = int(v.get(num))
            except (TypeError, ValueError, AttributeError):
                continue
            if 1 <= k <= len(part):
                merged.append({**v, num: k + c0})
    log.info("[vision] answered by a backup AI (%d group(s))", len(groups))
    return {key: merged, "_backup": True}


def ask(groups: list[tuple[str, list[bytes]]], prompt: str, key: str, num: str, temperature: float = 0.1,
        prefer_pro: bool = True) -> dict | None:
    groups = [(lbl, imgs) for lbl, imgs in groups if imgs]
    if not groups:
        return None
    data = _gemini(groups, prompt, temperature, prefer_pro)
    if isinstance(data, dict) and isinstance(data.get(key), list):
        return data
    try:
        return _backup(groups, prompt, key, num, temperature)
    except Exception as exc:  # noqa: BLE001 - a backup failure must never break a run
        log.warning("[vision] backup AIs failed: %.150s", exc)
        return None
