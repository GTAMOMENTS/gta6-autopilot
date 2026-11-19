"""The critic: a second, strong free text model (GPT-4.1 via GitHub Models first, then the other free backups)
reviews the packaging the video AI wrote - title, banner/thumbnail text, hook - and rewrites it for more clicks
without inventing anything. Anything that breaks the rules (too long, new claims, empty) keeps the original."""
from __future__ import annotations

import json
import logging
import re

from common import load_config

log = logging.getLogger("autopilot")

PROMPT = """You are the packaging expert of a fast-growing GTA 6 channel (YouTube Shorts / TikTok / Reels / long videos).
Another AI watched the video and wrote the packaging below. Make it get MORE CLICKS and MORE WATCH TIME:
- curiosity gap, strong verbs, specific details from the video, 1-2 CAPITALISED power words in titles max
- the hook must make people stay; titles must make people click; NEVER clickbait that the video doesn't deliver
- keep every fact from the description - DO NOT invent events, names or numbers that aren't in it
- respect each field's limit exactly; say "GTA 6", never "GTA VI"; no hashtags or emojis unless the field had them
What the video shows: {context}

Fields (rewrite each, or return it unchanged if it is already excellent):
{fields}

Return ONLY JSON with exactly these keys: {keys}"""


SENTENCE_FIELDS = {"caption", "intro_line", "x_text", "outro_line"}


def _invents(field: str, new: str, known: str) -> bool:
    """True if the rewrite adds facts: numbers that aren't in the original/description, or (in sentence-style
    fields, where it shows) capitalised names mid-sentence that the video description never mentions."""
    known_low = known.lower()
    if {n for n in re.findall(r"\d+", new)} - {n for n in re.findall(r"\d+", known)} - {"6"}:
        return True
    if field in SENTENCE_FIELDS:
        for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", new):
            if m.group(1).lower() not in known_low and m.group(1) not in ("GTA",):
                return True
    return False


def polish(fields: dict, limits: dict, context: str) -> dict:
    """fields: {name: text}; limits: {name: max chars}. Returns the improved fields (originals where rejected)."""
    cfg = load_config().get("critic", {})
    if not cfg.get("enabled", True) or not fields:
        return fields
    import ai_fallback
    if not ai_fallback.available():
        return fields
    listing = "\n".join(f'- {k} (max {limits.get(k, 100)} characters): "{v}"' for k, v in fields.items())
    data = ai_fallback.chat_json(PROMPT.format(context=context[:1500], fields=listing, keys=json.dumps(list(fields))),
                                 temperature=0.7)
    if not isinstance(data, dict):
        return fields
    out, changed = dict(fields), []
    for k, old in fields.items():
        new = re.sub(r"\s+", " ", str(data.get(k) or "")).strip().strip('"')
        if not new or len(new) > limits.get(k, 100) or "GTA VI" in new:
            continue
        if _invents(k, new, f"{old} {context}"):
            log.info("[critic] kept the original %s (the rewrite added names/numbers not in the video)", k)
            continue
        if new != old:
            out[k] = new
            changed.append(k)
    if changed:
        log.info("[critic] %s improved: %s", data.get("_ai_model", "AI"),
                 {k: f"{fields[k]!r} -> {out[k]!r}" for k in changed})
    return out
