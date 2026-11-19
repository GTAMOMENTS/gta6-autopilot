"""Learn from views: fetch each published video's YouTube stats (free, via yt-dlp - no extra API access)
and score the formats, so the rotation can favour what the audience actually watches."""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone

from common import load_config, run

log = logging.getLogger("autopilot")


def _items(m: dict) -> list[dict]:
    return [i for i in (m.get("items") or []) if isinstance(i, dict)]  # old/odd entries are skipped


def _age_h(item: dict) -> float:
    try:
        created = datetime.fromisoformat(str(item["created_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError, TypeError):
        return 0.0
    return (datetime.now(timezone.utc) - created).total_seconds() / 3600


def _yt_url(item: dict) -> str:
    results = item.get("results")
    yt = results.get("youtube") if isinstance(results, dict) else None
    return (yt.get("url") or "") if isinstance(yt, dict) else ""


def refresh_stats(max_videos: int | None = None) -> int:
    """Update views/likes/comments of recent published videos (at most every few hours each)."""
    from source_fetcher import _ytdlp_base, load_manifest, save_manifest
    cfg = load_config().get("learning", {})
    if not cfg.get("enabled", True):
        return 0
    max_videos = max_videos or int(cfg.get("max_fetch_per_run", 6))
    every = float(cfg.get("refresh_hours", 6)) * 3600
    m = load_manifest()
    todo = [i for i in _items(m) if i.get("status") == "published" and _yt_url(i) and not i.get("removed")
            and 3 <= _age_h(i) <= float(cfg.get("track_days", 10)) * 24
            and time.time() - (i.get("stats") or {}).get("at", 0) > every]
    todo.sort(key=lambda i: (i.get("stats") or {}).get("at", 0))
    done = 0
    for item in todo[:max_videos]:
        res = run(_ytdlp_base() + ["-J", "--skip-download", _yt_url(item)], timeout=60)
        try:
            meta = json.loads(res.stdout)
        except (json.JSONDecodeError, TypeError):
            meta = None
        if not isinstance(meta, dict):
            err = (res.stderr or "").strip().splitlines()[-1:] or [""]
            if "unavailable" in err[0].lower() or "removed" in err[0].lower():
                item["removed"] = True  # deleted video: stop checking it, keep it out of the scores
                done += 1
            log.info("[learn] could not read stats for %s (%.120s)", _yt_url(item), err)
            continue
        item["stats"] = {"views": int(meta.get("view_count") or 0), "likes": int(meta.get("like_count") or 0),
                         "comments": int(meta.get("comment_count") or 0), "age_h": round(_age_h(item), 1),
                         "at": time.time()}
        done += 1
    if done:
        save_manifest(m)
        log.info("[learn] refreshed YouTube stats for %d video(s)", done)
    return done


def format_scores() -> dict[str, dict]:
    """{format_id: {"score": float, "n": int, "views": [..]}} from videos at least min_age_hours old.
    Score = median of log10(views) with likes/comments as a bonus (engagement)."""
    from source_fetcher import load_manifest
    cfg = load_config().get("learning", {})
    min_age = float(cfg.get("min_age_hours", 12))
    out: dict[str, dict] = {}
    for item in _items(load_manifest()):
        st, fmt = item.get("stats"), item.get("format_id")
        if item.get("removed") or not isinstance(st, dict) or not fmt or st.get("age_h", 0) < min_age:
            continue
        v = st["views"]
        val = math.log10(v + 10) + 0.5 * math.log10(1 + 20 * (st["likes"] + 3 * st["comments"]) / max(v, 50))
        out.setdefault(fmt, {"vals": [], "views": []})
        out[fmt]["vals"].append(val)
        out[fmt]["views"].append(v)
    for fmt, d in out.items():
        vals = sorted(d.pop("vals"))
        d["n"] = len(vals)
        d["score"] = round(vals[len(vals) // 2], 3)
    return out
