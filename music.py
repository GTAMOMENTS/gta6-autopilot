"""Background music: a quiet beat under the narration, like real YouTubers use (free + safe).

Tracks come from (first that has any):
  1. music/            - your own royalty-free files (e.g. YouTube Audio Library, "no attribution required")
  2. queue/music/      - CC0 (public domain) beats downloaded once from Freesound (free API key in the
                         FREESOUND_API_KEY secret) and kept in the state cache. CC0 = no credit needed, fine
                         for monetised videos.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import requests

from common import BASE_DIR, STATE_DIR, env, load_config, load_json, save_json

log = logging.getLogger("autopilot")
CACHE = STATE_DIR / "music"
API = "https://freesound.org/apiv2"
AUDIO = (".mp3", ".m4a", ".wav", ".ogg")


def _cfg() -> dict:
    return load_config().get("music", {})


def _tracks(folder: Path) -> list[Path]:
    return [p for p in folder.glob("*") if p.suffix.lower() in AUDIO] if folder.exists() else []


def ensure_tracks() -> int:
    """Download CC0 beats from Freesound until the cache has music.tracks of them. Never raises."""
    key = env("FREESOUND_API_KEY")
    c = _cfg()
    if not key or not c.get("enabled", True):
        return len(_tracks(CACHE))
    want = int(c.get("tracks", 12))
    have = _tracks(CACHE)
    if len(have) >= want:
        return len(have)
    CACHE.mkdir(parents=True, exist_ok=True)
    index = load_json(CACHE / "index.json", {"tracks": {}})
    got = 0
    for q in c.get("queries", ["hip hop beat loop", "trap beat instrumental", "lofi beat loop",
                               "electronic beat loop", "chill beat instrumental", "phonk beat"]):
        if len(have) + got >= want:
            break
        try:
            r = requests.get(f"{API}/search/text/", timeout=25, params={
                "query": q, "token": key, "page_size": 15, "sort": "rating_desc",
                "filter": 'license:"Creative Commons 0" duration:[40 TO 300]',
                "fields": "id,name,username,license,duration,previews,avg_rating,num_ratings"})
            r.raise_for_status()
            results = r.json().get("results", [])
            log.info("[music] Freesound '%s': %d CC0 results", q, len(results))
        except Exception as exc:  # noqa: BLE001
            log.warning("[music] Freesound search '%s' failed: %.120s", q, exc)
            continue
        for x in results:
            if len(have) + got >= want:
                break
            lic = str(x.get("license", ""))
            # Freesound gives the licence as a URL (.../publicdomain/zero/1.0/) - CC0 only (the search filters too)
            if str(x.get("id")) in index["tracks"] or not ("publicdomain/zero" in lic or "Creative Commons 0" in lic):
                continue
            url = (x.get("previews") or {}).get("preview-hq-mp3")
            if not url:
                continue
            out = CACHE / f"fs_{x['id']}.mp3"
            try:
                a = requests.get(url, timeout=60, params={"token": key})
                a.raise_for_status()
                if len(a.content) < 100_000:
                    continue
                out.write_bytes(a.content)
            except Exception as exc:  # noqa: BLE001
                log.info("[music] download failed for %s: %.100s", x.get("name"), exc)
                continue
            index["tracks"][str(x["id"])] = {"name": x.get("name"), "by": x.get("username"), "query": q,
                                             "duration": x.get("duration"), "license": "CC0"}
            got += 1
    save_json(CACHE / "index.json", index)
    log.info("[music] %d new CC0 beats from Freesound (%d in the cache)", got, len(have) + got)
    return len(have) + got


def pick() -> Path | None:
    """A random track: your own music/ folder first, else the downloaded CC0 beats."""
    own = BASE_DIR / load_config().get("video", {}).get("fx", {}).get("music_dir", "music")
    tracks = _tracks(own) or _tracks(CACHE)
    return random.choice(tracks) if tracks else None
