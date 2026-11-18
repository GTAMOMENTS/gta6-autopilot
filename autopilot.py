"""GTA VI Top Moments - 24/7 autopilot entry point.

Each GitHub Actions run handles exactly ONE schedule slot:
  * scheduled runs are mapped to their slot via the cron expression that fired
    (GitHub often starts cron jobs 10-60+ min late - the old 45 min window missed them)
  * manual runs pick the slot from the workflow_dispatch input

Usage (locally):
  python autopilot.py --slot vertical_clip --dry-run
  python autopilot.py --slot x_text_morning
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from common import (HISTORY_FILE, READY_DIR, env, load_config, load_json, now_local, save_json,
                    setup_logging, tz, write_summary)

log = logging.getLogger("autopilot")
VIDEO_TYPES = ("vertical_clip", "x_video_clip", "test_all")
SEARCH_PROVIDERS = {"dailymotion", "youtube", "twitch_clip", "twitch_vod"}


# --------------------------------------------------------------------------- slot resolution

def _slot_from_cron(cron: str, slots: list[dict]) -> dict | None:
    try:
        minute, hour = cron.split()[:2]
        utc = datetime.now(timezone.utc).replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
    except (ValueError, IndexError):
        return None
    local_hhmm = utc.astimezone(tz()).strftime("%H:%M")
    return next((s for s in slots if s["time"] == local_hhmm), None)


def _nearest_due_slot(slots: list[dict], now: datetime) -> dict | None:
    cur = now.hour * 60 + now.minute
    best, best_delta = None, None
    for s in slots:
        hh, mm = map(int, s["time"].split(":"))
        delta = (cur - (hh * 60 + mm)) % 1440
        if delta <= 150 and (best_delta is None or delta < best_delta):
            best, best_delta = s, delta
    return best


def resolve_slot(cfg: dict, requested: str) -> dict | None:
    slots = [s for s in cfg.get("schedule", []) if s.get("enabled", True)]
    now = now_local()
    if requested == "test_all":
        return {"time": now.strftime("%H:%M"), "type": "test_all",
                "platforms": ["youtube", "instagram", "tiktok", "x"], "manual": True}
    if requested and requested != "auto":
        match = next((s for s in slots if s["type"] == requested), None)
        if match:
            return {**match, "manual": True}
        if requested.startswith("x_"):
            return {"time": now.strftime("%H:%M"), "type": requested, "platforms": ["x"], "manual": True}
        log.error("Unknown slot type '%s'", requested)
        return None
    cron = env("SCHEDULE_CRON")
    if cron:
        slot = _slot_from_cron(cron, slots)
        if slot:
            return slot
        log.warning("Cron '%s' matches no slot - falling back to nearest due slot", cron)
    return _nearest_due_slot(slots, now)


def slot_id(slot: dict) -> str:
    now = now_local()
    hh, mm = map(int, slot["time"].split(":"))
    day = now.date()
    if (hh * 60 + mm) > now.hour * 60 + now.minute + 5:  # slot belongs to yesterday (run delayed past midnight)
        day -= timedelta(days=1)
    return f"{day.isoformat()}_{slot['time']}_{slot['type']}"


# --------------------------------------------------------------------------- video slot

def run_video_slot(slot: dict, dry_run: bool) -> tuple[bool, dict]:
    from common import RAW_DIR, probe
    from gemini_writer import finalize_commentary, write_clip_copy
    from highlight import best_window, cut_segments, energy_curve, energy_hints
    from publisher import _available, publish_video, routes_for
    from source_fetcher import iter_clips, load_manifest, mark_rejected, mark_used, save_manifest
    from video_processor import render, synthesize_lines

    cfg = load_config()
    platforms = slot["platforms"]
    if not dry_run:
        usable = [p for p in platforms if any(_available(pr, p) for pr in routes_for(p))]
        missing = sorted(set(platforms) - set(usable))
        if missing:
            log.warning("No configured publisher for %s - they will be skipped", missing)
        if not usable:
            return True, {"skipped": "no publisher configured for " + ", ".join(platforms)}
        platforms = usable

    gate = cfg.get("quality_gate", {})
    vcfg = cfg.get("video", {})
    scfg = cfg.get("sources", {})
    min_hype = int(gate.get("min_hype_score", 7))
    backup_hype = int(gate.get("backup_hype_score", min_hype - 1))
    budget_end = time.time() + int(scfg.get("time_budget_min", 25)) * 60

    def publish_and_record(src, copy, clip_id, rendered, ranges) -> tuple[bool, dict]:
        c = src.candidate
        mark_used(src)
        results = {} if dry_run else publish_video(Path(rendered["path"]), platforms, copy, src.credit)
        ok = dry_run or any(r.get("ok") for r in results.values())
        record = {
            "id": clip_id, "slot": slot["type"], "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {"provider": c.provider, "kind": c.kind, "url": c.url, "title": c.title, "credit": src.credit,
                       "section": list(src.section), "segments": ranges},
            "copy": copy, "rendered": rendered["path"], "duration": rendered["duration"],
            "voice_lines": rendered.get("lines", []),
            "dry_run": dry_run, "results": results, "status": "published" if ok and not dry_run else
            ("dry_run" if dry_run else "failed"),
        }
        m = load_manifest()
        m["items"].append(record)
        save_manifest(m)
        return ok, record

    def produce_tour(src, copy, sec_dur) -> tuple[bool, dict] | None:
        """Feature tour: every shot lasts exactly as long as its narration line (the edit follows the voice)."""
        c, reel = src.candidate, src.clip_path
        shots = copy["shots"]
        clip_id = "gta6_" + now_local().strftime("%Y%m%d_%H%M") + "_" + hashlib.md5(c.key.encode()).hexdigest()[:6]
        voiced = synthesize_lines([{"t": 0.0, "text": sh["text"]} for sh in shots], clip_id)
        max_total = float(vcfg.get("tour_max_sec", 80))
        segments, cursor, kept = [], 0.0, []
        for sh, v in zip(shots, voiced):
            dur = min(max(v["dur"] + 0.45, 3.0), 12.0)
            if cursor + dur > max_total:
                break
            a = min(max(0.0, sh["start"] - 0.3), max(0.0, sec_dur - dur))
            segments.append((round(a, 2), round(a + dur, 2)))
            kept.append({**v, "t": cursor + 0.15})
            cursor += dur
        log.info("[tour] %d shots, %.0fs: %s", len(segments), cursor,
                 [f"{a:.0f}s '{v['text'][:40]}'" for (a, _), v in zip(segments, kept)])
        clip = cut_segments(reel, RAW_DIR / f"tour_{int(time.time())}.mp4", segments)
        reel.unlink(missing_ok=True)
        if not clip:
            return None
        copy["commentary"] = [{"t": round(v["t"], 1), "text": v["text"]} for v in kept]
        rendered = render(clip, clip_id, copy["banner"], copy["commentary"], voiced=kept)
        if not rendered:
            return None
        return publish_and_record(src, copy, clip_id, rendered, segments)

    def produce(src, copy, energy, sec_dur) -> tuple[bool, dict] | None:
        """AI editor's window -> cut -> voice lines -> render -> publish -> record."""
        c, section = src.candidate, src.clip_path
        min_len = min(float(vcfg.get("min_clip_sec", 25)), sec_dur)
        max_len = min(float(vcfg.get("duration_sec", 50)), sec_dur)
        start, end = copy.get("clip_start"), copy.get("clip_end")
        if start is None or end is None or not (0 <= start < end) or (end - start) < min_len * 0.8:
            a, b = best_window(energy, int(min(58, max_len)))
            start, end = float(a), float(b)
            log.info("[editor] AI gave no usable window - using the most intense %ds", int(end - start))
        else:
            end = min(end, sec_dur, start + max_len)
            log.info("[editor] AI picked %.0f-%.0fs of the slice (%.0fs)", start, end, end - start)
        clip = cut_segments(section, RAW_DIR / f"clip_{int(time.time())}.mp4", [(start, end)])
        section.unlink(missing_ok=True)
        if not clip:
            return None
        commentary = finalize_commentary(copy, start, end)
        copy["commentary"] = commentary
        log.info("[editor] voice lines: %s", [f"{ln['t']:.0f}s {ln['text']}" for ln in commentary])

        clip_id = "gta6_" + now_local().strftime("%Y%m%d_%H%M") + "_" + hashlib.md5(c.key.encode()).hexdigest()[:6]
        rendered = render(clip, clip_id, copy["banner"], commentary)
        if not rendered:
            return None
        return publish_and_record(src, copy, clip_id, rendered, [[start, end]])

    backup = None  # best "almost good enough" clip, used only if nothing reaches min_hype
    skip: set[str] = set()
    for src in iter_clips(target_sec=0, max_attempts=int(scfg.get("max_attempts", 8)), skip_providers=skip):
        if time.time() > budget_end:
            log.warning("Time budget exhausted")
            src.clip_path.unlink(missing_ok=True)
            break
        c = src.candidate
        section = src.clip_path                     # the whole downloaded slice (~2 min)
        sec_dur = probe(section)["duration"]
        energy = energy_curve(section)
        copy = write_clip_copy(section, c.title, src.credit, c.provider, c.kind, energy_hints(energy, 30, 3),
                               tour=src.is_reel)
        if not copy["verified_by_video"] and c.provider in SEARCH_PROVIDERS and gate.get("require_ai_check_for_search", True):
            # Can't confirm a random upload is real GTA VI without the AI watching it -> don't risk it.
            log.warning("[gate] AI video check unavailable - skipping search results, using trusted sources only")
            skip.update(SEARCH_PROVIDERS)
            section.unlink(missing_ok=True)
            continue
        if copy["verified_by_video"]:
            reason = None
            if not copy["is_gta6_footage"]:
                reason = "not GTA VI footage"
            elif copy.get("is_spoiler") and gate.get("reject_spoilers", True):
                reason = "story spoiler"
            elif copy.get("has_narrator") and gate.get("reject_narrator", True) and c.kind != "stream":
                reason = "someone already talks over this footage"
            elif copy["has_facecam"] and gate.get("reject_facecam", True) and c.kind != "stream":
                reason = "facecam / overlay"
            elif copy["hype_score"] < min_hype:
                reason = f"low hype ({copy['hype_score']})"
            if reason:
                log.info("[gate] ✋ rejected '%s': %s | %s", c.title[:60], reason, copy["scene_summary"])
                mark_rejected(c, reason)
                keep = (reason.startswith("low hype") and copy["hype_score"] >= backup_hype
                        and (backup is None or copy["hype_score"] > backup[1]["hype_score"]))
                if keep:
                    if backup:
                        backup[0].clip_path.unlink(missing_ok=True)
                    backup = (src, copy, energy, sec_dur)
                    log.info("[gate] kept as backup (hype %s) in case nothing better turns up", copy["hype_score"])
                else:
                    section.unlink(missing_ok=True)
                continue
            log.info("[gate] ✅ accepted (hype %s, %s): %s", copy["hype_score"], copy.get("content_type"),
                     copy["scene_summary"])

        if src.is_reel and len(copy.get("shots") or []) >= int(vcfg.get("tour_min_shots", 5)):
            result = produce_tour(src, copy, sec_dur)
        else:
            if src.is_reel:
                log.info("[tour] AI gave too few shots - making a single-moment video instead")
            result = produce(src, copy, energy, sec_dur)
        if result:
            if backup:
                backup[0].clip_path.unlink(missing_ok=True)
            return result

    if backup:
        log.info("[gate] nothing reached hype %d - posting the best backup (hype %s)", min_hype,
                 backup[1]["hype_score"])
        b_src, b_copy, b_energy, b_dur = backup
        if b_src.is_reel and len(b_copy.get("shots") or []) >= int(vcfg.get("tour_min_shots", 5)):
            result = produce_tour(b_src, b_copy, b_dur)
        else:
            result = produce(*backup)
        if result:
            return result
    return False, {"error": "no usable clip could be produced"}


# --------------------------------------------------------------------------- text slot

def run_text_slot(slot: dict, dry_run: bool) -> tuple[bool, dict]:
    from gemini_writer import write_text_post
    from publisher import add_hashtags, publish_text
    from source_fetcher import load_manifest

    history = load_json(HISTORY_FILE, {"completed_slots": [], "posts": []})
    recent = [p.get("text", "") for p in history.get("posts", []) if p.get("text")]
    today = now_local().date().isoformat()
    todays = [i.get("copy", {}).get("banner", "") for i in load_manifest()["items"]
              if i.get("status") == "published" and i.get("created_at", "")[:10] == today]
    text = add_hashtags(write_text_post(slot["type"], recent, todays))
    log.info("[text] %s", text)
    result = {"ok": True, "dry_run": True} if dry_run else publish_text(text)
    return bool(result.get("ok")), {"text": text, "result": result}


# --------------------------------------------------------------------------- main

def record_history(sid: str, slot: dict, ok: bool, detail: dict) -> None:
    h = load_json(HISTORY_FILE, {"completed_slots": [], "posts": []})
    if ok and not slot.get("manual"):
        h["completed_slots"].append(sid)
    h["completed_slots"] = h["completed_slots"][-300:]
    entry = {"slot_id": sid, "type": slot["type"], "ok": ok, "at": datetime.now(timezone.utc).isoformat()}
    if "text" in detail:
        entry["text"] = detail["text"]
    if "id" in detail:
        entry["clip_id"] = detail["id"]
        entry["results"] = {p: {k: v for k, v in r.items() if k in ("ok", "provider", "url", "error")}
                            for p, r in detail.get("results", {}).items()}
    h["posts"] = (h.get("posts", []) + [entry])[-400:]
    save_json(HISTORY_FILE, h)


def summarize(slot: dict, sid: str, ok: bool, detail: dict, dry_run: bool) -> None:
    lines = [f"## {'✅' if ok else '❌'} {slot['type']} ({sid}){' — DRY RUN' if dry_run else ''}"]
    if "text" in detail:
        lines.append(f"> {detail['text']}")
    if "copy" in detail:
        cp, s = detail["copy"], detail["source"]
        lines += [f"**Banner:** {cp['banner']}  ", f"**Source:** [{s['title']}]({s['url']}) by {s['credit']} "
                  f"({s['provider']}, section {s['section']})  ",
                  f"**AI check:** {cp.get('content_type', '?')}, hype {cp['hype_score']}, verified={cp['verified_by_video']} by {cp.get('ai_source', '?')} — {cp['scene_summary']}  ",
                  "**Voice lines:** " + " · ".join(f"[{ln['t']}s] {ln['text']}" for ln in
                                                   (detail.get('voice_lines') or cp.get('commentary', [])))]
    for plat, r in (detail.get("results") or {}).items():
        lines.append(f"- {plat}: {'✅' if r.get('ok') else '❌'} {r.get('provider', '')} {r.get('url') or r.get('error') or ''}")
    for key in ("error", "skipped"):
        if key in detail:
            lines.append(f"**{key}:** {detail[key]}")
    write_summary("\n".join(lines))


def _only_quota_errors(detail: dict) -> bool:
    results = detail.get("results") or ({"x": detail["result"]} if "result" in detail else {})
    errors = [str(r.get("error", "")).lower() for r in results.values() if not r.get("ok")]
    words = ("limit", "quota", "429", "no configured", "upgrade", "plan")
    return bool(errors) and all(any(w in e for w in words) for e in errors)


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", default=env("SLOT", "auto"))
    ap.add_argument("--dry-run", action="store_true", default=env("DRY_RUN").lower() == "true")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"))

    if args.slot == "check_setup":
        from setup_check import run_checks
        return run_checks()
    if args.slot == "voice_samples":
        from video_processor import voice_samples
        return voice_samples()

    cfg = load_config()
    slot = resolve_slot(cfg, args.slot)
    log.info("Run at %s | event=%s cron=%s requested=%s dry_run=%s",
             now_local().strftime("%Y-%m-%d %H:%M %Z"), os.getenv("GITHUB_EVENT_NAME", "local"),
             env("SCHEDULE_CRON") or "-", args.slot, args.dry_run)
    if not slot:
        log.info("No slot due right now - nothing to do.")
        return 0
    sid = slot_id(slot)
    log.info("Slot: %s -> %s", sid, slot["platforms"])

    history = load_json(HISTORY_FILE, {"completed_slots": [], "posts": []})
    if not slot.get("manual") and sid in history.get("completed_slots", []):
        log.info("Slot %s already completed - skipping duplicate run.", sid)
        return 0

    READY_DIR.mkdir(parents=True, exist_ok=True)
    if slot["type"] in VIDEO_TYPES:
        ok, detail = run_video_slot(slot, args.dry_run)
    else:
        ok, detail = run_text_slot(slot, args.dry_run)

    if not ok and _only_quota_errors(detail):
        detail["skipped"] = "all publishers are out of quota / not linked - see README to add a free route"
    if not args.dry_run and "skipped" not in detail:
        record_history(sid, slot, ok, detail)
    summarize(slot, sid, ok, detail, args.dry_run)
    if "skipped" in detail:
        log.warning("Slot skipped: %s", detail["skipped"])
        return 0
    if not ok:
        log.error("Slot %s FAILED: %s", sid, detail.get("error") or detail.get("results") or detail.get("result"))
        return 1
    log.info("Slot %s done ✅", sid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
