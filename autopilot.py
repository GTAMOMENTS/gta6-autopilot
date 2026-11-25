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
VIDEO_TYPES = ("vertical_clip", "video_b", "money_clip", "x_video_clip", "test_all")
SEARCH_PROVIDERS = {"dailymotion", "youtube", "twitch_clip", "twitch_vod", "x_account"}  # need the AI check
# QA problems a rewrite of the script fixes without new footage: lines the picture doesn't show, and approved
# lines the voice-over had no room for
REPAIRABLE = ("line_not_shown", "script_lines_dropped")


def repairable(issues: list[dict]) -> list[dict]:
    """The high QA issues a script rewrite can fix - [] if anything else blocks too (then the clip is skipped).
    The AI editor's own verdict / score / narration remarks restate the line problems, so they don't prevent a
    repair: the repaired video goes through the full QA again."""
    high = [i for i in issues if str(i.get("severity")) == "high"]
    fix = [i for i in high if i.get("type") in REPAIRABLE and isinstance(i.get("index"), int)]
    other = [i for i in high if i.get("type") not in REPAIRABLE + ("ai_verdict", "ai_score")
             and not any(w in str(i.get("type")) for w in ("narration", "mismatch"))]
    return fix if fix and not other else []


def repair_note(issues: list[dict], placed: list[dict]) -> str:
    """What the writer is told when it rewrites the whole script after QA."""
    wrong = [i for i in issues if i.get("type") == "line_not_shown" and 0 <= i["index"] < len(placed)]
    dropped = [i for i in issues if i.get("type") == "script_lines_dropped"]
    out = []
    if wrong:
        out.append("QA REJECTED these lines because the picture doesn't show them:\n" + "\n".join(
            f'- "{placed[i["index"]]["text"]}": {i.get("reason", "")}' for i in wrong))
    if dropped:
        out.append("THE SCRIPT IS TOO LONG FOR THE VIDEO - the voice-over had no room, so these lines were cut:\n"
                   + "\n".join(f'- "{i.get("text", "")}"' for i in dropped)
                   + "\nMake every line about a third shorter so ALL of them fit, the closing question included.")
    return "\n".join(out)


def _fx_at(item) -> float:
    """When an effect starts: {"t"} / {"a"} dicts or (start, ...) tuples."""
    if isinstance(item, dict):
        return float(item.get("t", item.get("a", 0.0)) or 0.0)
    return float(item[0])


def drop_calm_shots(shots: list[dict], energy: list[float], calm: float, keep_min: int) -> list[dict]:
    """A tour's shots that start in the calmest stretches of the footage (walking, roaming, calm driving: the energy
    of motion + game sound over the shot's first 4 s is under `calm`) - owner: "you can't do narration on a walking
    clip". Dropped, calmest first, while at least keep_min shots remain; the order of the rest is kept."""
    def level(sh: dict) -> float:
        a = max(0, int(float(sh.get("start", 0))))
        part = energy[a:a + 4]
        return sum(part) / len(part) if part else 1.0  # no data for it: never "calm"
    calm_ones = sorted((sh for sh in shots if level(sh) < calm), key=level)
    drop = {id(sh) for sh in calm_ones[:max(0, len(shots) - keep_min)]}
    return [sh for sh in shots if id(sh) not in drop]


def patient(write, vcfg: dict, deadline: float):
    """Gemini's 503 "high demand" comes and goes within minutes (a run's log: busy at 18:13, answering at 18:14,
    busy again at 18:15). A clip that already passed every gate is worth a short wait: a run that dropped a hype-9
    clip after one busy spell found nothing better - the next clip needs Gemini too, plus a download.
    write() -> the writer's answer or None; asked again after video.ai_busy_wait_sec, up to ai_busy_retries times,
    while the run's time budget (deadline) allows."""
    w = write()
    for _ in range(int(vcfg.get("ai_busy_retries", 2))):
        wait = float(vcfg.get("ai_busy_wait_sec", 60))
        if w or not env("GEMINI_API_KEY") or time.time() + wait > deadline:
            break
        log.info("[script] no AI could watch the video right now (Gemini busy?) - this clip passed every check: "
                 "waiting %.0fs and asking again", wait)
        time.sleep(wait)
        w = write()
    return w


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

def run_video_slot(slot: dict, dry_run: bool, fmt: dict | None = None) -> tuple[bool, dict]:
    from common import RAW_DIR, probe
    from gemini_writer import script_ok, write_clip_copy
    from highlight import best_window, cut_segments, energy_curve, energy_hints
    from publisher import _available, publish_video, routes_for
    from source_fetcher import (AUTHENTICITY_PROVIDERS, TALKY_REASON, iter_clips, load_manifest, mark_rejected,
                                mark_used, note_talky, save_manifest)
    from video_processor import HumanVoiceUnavailable, render, synthesize_lines

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

    def _evidence_ok(ev: str, f: dict) -> bool:
        ev = (ev or "").lower()
        return len(ev) >= 15 and any(t in ev for t in f.get("evidence_terms", []))

    def _mute(copy) -> float | None:
        """Footage with another YouTuber talking (allowed for some formats): their voice is muted."""
        return 0.0 if (fmt or {}).get("allow_narrated") and copy.get("has_narrator") else None

    def _approved(rendered: dict | None, texts: list[str]) -> dict | None:
        """Remember which script the judge approved - QA checks the video says all of it."""
        if rendered:
            rendered["approved"] = list(texts)
        return rendered

    def _script_ctx(copy, story="") -> dict:
        return {"fmt": f"{(fmt or {}).get('name', 'GTA 6 Short')}. {(fmt or {}).get('brief', '')}"[:500],
                "banner": copy.get("banner", ""), "title": copy.get("youtube_title", ""),
                "story": story or copy.get("scene_summary", "")}

    def _write_checked(video, copy, items, lead=0.0, spans=None, slots=False, feedback="",
                       tails: list | None = None) -> list[dict] | None:
        """Owner: "the writing shall occur after the AI watched the whole video, then write a good script for the
        whole video - not frame by frame". 1) the AI watches the WHOLE video and writes ONE script; 2) a different AI
        judges the script like a viewer (makes sense? promise kept? useful?); 3) the lines are checked against
        the picture. A failure goes back to the writer (whole video + the reasons) for a NEW script. None = no
        script good enough -> next clip, before any voice / render is spent.
        items: the current lines [{"t", "text"}]; slots=True keeps their times (tours: the edit is fixed).
        feedback: what the first attempt must fix already (e.g. a countdown that skips a number).
        tails: collects, from rejected scripts, the second the talking stopped when only the end was left silent
        (script_judge.quiet_tail) - the caller can end the video there and ask again."""
        import lessons
        import qa
        import script_judge
        from gemini_writer import write_final_script
        countdown = int(fmt["countdown"]) if slots and (fmt or {}).get("countdown") else None
        tries = max(1, int(vcfg.get("script_attempts", 3)))
        length = None if slots else probe(video)["duration"]
        last_issues = []
        for attempt in range(1, tries + 1):
            w = patient(lambda: write_final_script(
                video, copy, fmt, lead=lead, spans=spans, feedback=feedback,
                slots=[{"t": it["t"], "text": it["text"]} for it in items] if slots else None,
                n=1 if slots else max(1, int(vcfg.get("script_candidates", 3)))), vcfg, budget_end)
            if not w:
                log.warning("[script] no AI could watch the whole video and write the script")
                return None
            # several different scripts per attempt: the judge scores each, the best one that passes is used
            best, best_v = None, None
            for cand in [w] + list(w.get("alternatives") or []):
                log.info("[script] attempt %d%s - written after watching the whole video: %s", attempt,
                         f" ({cand['angle']})" if cand.get("angle") else "",
                         [f"{ln['t']:.0f}s {ln['text']}" for ln in cand["lines"]])
                trial = {**copy, "banner": cand.get("banner") or copy.get("banner", ""),
                         "youtube_title": cand.get("title") or copy.get("youtube_title", "")}
                v = script_judge.judge(cand["lines"], _script_ctx(trial, cand.get("story", "")),
                                       numbered_ok=bool(countdown), countdown=countdown,
                                       timing=None if slots else {"length": length, "lead": lead, "spans": spans})
                if not v["ok"] and tails is not None and length:
                    t_quiet = script_judge.quiet_tail(cand["lines"], length, lead, spans)
                    if t_quiet is not None:
                        tails.append(t_quiet)
                if best_v is None or (v["ok"], v["score"] or 0) > (best_v["ok"], best_v["score"] or 0):
                    best, best_v = cand, v
            w, verdict, lines = best, best_v, best["lines"]
            if not verdict["ok"]:
                feedback = script_judge.feedback(verdict) + (
                    "\nThe rejected script was:\n" + "\n".join(f'{ln["t"]:.0f}s: {ln["text"]}' for ln in lines))
                last_issues = [{"severity": "high", "type": "script_quality", "detail": p}
                               for p in verdict["problems"][:3]]
                continue
            if vcfg.get("precheck_lines", True):
                issues = qa.line_check(video, [{"text": ln["text"], "start": ln["t"],
                                                "dur": len(ln["text"].split()) / 2.6 + 0.4} for ln in lines], "short")
                bad = [i for i in issues if i.get("type") == "line_not_shown" and isinstance(i.get("index"), int)]
                if bad:
                    log.info("[precheck] %d line(s) don't match the picture at their moment: %s", len(bad),
                             [lines[i["index"]]["text"][:50] for i in bad if i["index"] < len(lines)])
                    feedback = ("THESE LINES DON'T MATCH WHAT IS ON SCREEN AT THEIR MOMENT:\n" + "\n".join(
                        f'- {lines[i["index"]]["t"]:.0f}s "{lines[i["index"]]["text"]}": {i.get("reason", "")}'
                        for i in bad if i["index"] < len(lines)) +
                        "\nWatch the video again and write a NEW script for the whole video where every line "
                        "matches its moment - still one story that makes sense.")
                    last_issues = bad
                    continue
            if w.get("banner") and w["banner"] != copy.get("banner"):
                log.info("[script] banner %r -> %r (the video delivers it)", copy.get("banner"), w["banner"])
                copy["banner"] = w["banner"]
            if w.get("title") and w["title"] != copy.get("youtube_title"):
                log.info("[script] title %r -> %r (the video delivers it)", copy.get("youtube_title"), w["title"])
                copy["youtube_title"] = w["title"]
            copy["script_review"] = {"score": verdict["score"], "by": verdict["by"], "story": w.get("story", "")}
            return lines
        lessons.record("short", (fmt or {}).get("id", ""), False, last_issues, stage="precheck")
        return None

    def _revise(video, copy, lines, why, lead=0.0, spans=None) -> list[dict] | None:
        """After the final QA: the AI watches the whole finished cut again and rewrites the WHOLE script (same
        times) with QA's reasons; judged again. None = can't fix it."""
        import script_judge
        from gemini_writer import write_final_script
        feedback = why + "\nRewrite the WHOLE script (one story) so every line matches its moment."
        w = patient(lambda: write_final_script(
            video, copy, fmt, lead=lead, spans=spans, slots=[{"t": ln["t"], "text": ln["text"]} for ln in lines],
            feedback=feedback), vcfg, budget_end)
        if not w:
            return None
        verdict = script_judge.judge(w["lines"], _script_ctx(copy, w.get("story", "")))
        if not verdict["ok"]:
            log.info("[repair] the rewritten script didn't pass the judge")
            return None
        copy["script_review"] = {"score": verdict["score"], "by": verdict["by"], "story": w.get("story", "")}
        return w["lines"]

    def publish_and_record(src, copy, clip_id, rendered, ranges, repair=None) -> tuple[bool, dict] | None:
        from formats import bump_series, series_label
        import lessons
        import qa
        c = src.candidate
        trust_note = ("FOOTAGE SOURCE: it passed a separate GTA VI authenticity check (or comes from the Twitch GTA VI "
                      "category) - flag 'not GTA 6' only if you clearly see GTA V places or characters. ")
        muted_note = ("The original creator's voice-over is muted on purpose (so there is no game sound) - only our "
                      "narrator speaks. " if _mute(copy) == 0.0 else "")
        tutorial_note = ("THIS IS A HOW-TO / MONEY TIP: flag as HIGH if the video does not clearly SHOW money being "
                         "earned (cash counter going up, payout or sale screen, picking up cash) or does not explain a "
                         "specific method a viewer can repeat, or if lines are vague filler that the footage doesn't "
                         "show. " if (fmt or {}).get("tutorial") else "")
        style_note = tutorial_note + ("INTENTIONAL EDIT (not errors): the banner pops in at the top on the first frame, captions show "
                      "the narration 2-4 words at a time with the spoken word highlighted, zoom punch-ins / a short shake "
                      "and flash on the big moment, and when the last line is a question it also appears as a card with a "
                      "call to action. ")
        teaser_note = trust_note + muted_note + style_note + ("The first ~1.2 s is an intentional cold open: a quick flash of the big moment, which then "
                       "plays again at its real place in the clip - that repeat is NOT a montage error. "
                       if float(vcfg.get("fx", {}).get("teaser_sec", 0)) > 0 and not src.is_reel else "")
        def _qa(r):
            return qa.check(Path(r["path"]), "short", lines=r.get("placed"), approved=r.get("approved"),
                            script_review=copy.get("script_review"),
                            context=f"{teaser_note}Format: {(fmt or {}).get('name', '')}. Banner: {copy.get('banner')}. "
                                    f"Title: {copy.get('youtube_title', '')}. "
                                    f"Narration lines with start times (also burned in as captions): "
                                    + " | ".join(f"[{ln['start']:.1f}s] {ln['text']}" for ln in r.get("placed", [])))
        src_info = {"provider": c.provider, "channel": c.uploader}
        verdict = _qa(rendered)
        repaired = False
        if not verdict["pass"] and repair is not None and vcfg.get("repair", True):
            bad_lines = repairable(verdict["issues"])
            if bad_lines:
                # repair instead of redo: the script is rewritten (same footage), re-voiced, re-rendered once
                lessons.record("short", (fmt or {}).get("id", ""), False, verdict["issues"], source=src_info)
                log.info("[repair] QA only found %d script problem(s) (%s) - rewriting the script", len(bad_lines),
                         ", ".join(sorted({i["type"] for i in bad_lines})))
                fixed = repair(rendered, bad_lines)
                if fixed:
                    rendered, repaired = fixed, True
                    verdict = _qa(rendered)
                    log.info("[repair] %s after fixing the lines", "✅ passed" if verdict["pass"] else "✋ still failing")
        lessons.record("short", (fmt or {}).get("id", ""), verdict["pass"], verdict["issues"],
                       stage="repair" if repaired else "final", source=src_info,
                       lines=[ln["text"] for ln in rendered.get("placed", [])], repaired=repaired)
        if not verdict["pass"]:
            qa_problems.append(verdict["report"].splitlines()[1] if len(verdict["report"].splitlines()) > 1 else "QA failed")
            mark_rejected(c, "failed QA")
            log.warning("[qa] ✋ not publishing this Short - trying the next clip")
            return None
        mark_used(src)
        if fmt:
            copy["format_tags"] = list(fmt.get("hashtags", []))
            copy["keywords"] = fmt.get("keywords", "")
            copy["series"] = series_label(fmt)
        results = {} if dry_run else publish_video(Path(rendered["path"]), platforms, copy, src.credit)
        ok = dry_run or any(r.get("ok") for r in results.values())
        if ok and not dry_run:
            bump_series(fmt)
        record = {
            "id": clip_id, "slot": slot["type"], "created_at": datetime.now(timezone.utc).isoformat(),
            "source": {"provider": c.provider, "kind": c.kind, "url": c.url, "title": c.title, "credit": src.credit,
                       "section": list(src.section), "segments": ranges},
            "format_id": (fmt or {}).get("id", ""), "copy": copy, "rendered": rendered["path"], "duration": rendered["duration"],
            "voice_lines": rendered.get("lines", []),
            "dry_run": dry_run, "results": results, "status": "published" if ok and not dry_run else
            ("dry_run" if dry_run else "failed"),
        }
        m = load_manifest()
        m["items"].append(record)
        save_manifest(m)
        return ok, record

    def produce_tour(src, copy, sec_dur, energy=None) -> tuple[bool, dict] | None:
        """Feature tour: every shot lasts exactly as long as its narration line (the edit follows the voice).
        energy (per second of the reel): shots in the calmest stretches are dropped - not for countdowns (every
        number must stay) or formats whose footage is calm by nature (calm_ok)."""
        c, reel = src.candidate, src.clip_path
        shots = copy["shots"]
        if energy and not (fmt or {}).get("countdown") and not (fmt or {}).get("calm_ok"):
            lively = drop_calm_shots(shots, energy, float(vcfg.get("tour_calm_energy", 0.25)),
                                     int(vcfg.get("tour_min_shots", 4)))
            if len(lively) < len(shots):
                log.info("[tour] dropped %d calm shot(s) - nothing happens there (walking / roaming / calm driving): %s",
                         len(shots) - len(lively), [f"{sh['start']:.0f}s" for sh in shots if sh not in lively])
                shots = copy["shots"] = lively
        if copy.get("ai_source") in ("built-in", "text-only", ""):
            log.warning("[script] ✋ not posting: no AI could watch the footage and write the script")
            reel.unlink(missing_ok=True)
            script_problems.append("no AI could write the script")
            return None
        fix = ""
        if (fmt or {}).get("countdown"):
            from formats import countdown_missing, countdown_note
            n = int(fmt["countdown"])
            missing = countdown_missing([sh["text"] for sh in shots], n)
            if missing and len(shots) < n + 1:  # no slot for every number: a Top 5 that skips one looks broken
                log.warning("[tour] ✋ countdown script skips number(s) %s and the edit has only %d shots - trying "
                            "the next clip", missing, len(shots))
                reel.unlink(missing_ok=True)
                script_problems.append(f"countdown skipped {missing}")
                return None
            if missing:
                # live run 2026-11-25: the editor's Top 5 left out number one over good footage (hype 9) and the
                # clip was dropped.
                # The script writer renumbers the same shots instead; the judge rejects any script that still
                # skips a number (static check), so a broken countdown is never voiced.
                log.info("[tour] countdown script skips number(s) %s - the script writer renumbers the %d shots",
                         missing, len(shots))
                fix = countdown_note(missing, n)
        written = _write_checked(reel, copy, [{"t": max(0.0, sh["start"] - 0.3), "text": sh["text"]} for sh in shots],
                                 slots=True, feedback=fix)
        if written is None:
            log.warning("[script] ✋ no script good enough for this footage - next clip (no voice / render spent)")
            reel.unlink(missing_ok=True)
            script_problems.append("no script good enough (judge / picture check)")
            return None
        shots = [{**sh, "text": ln["text"]} for sh, ln in zip(shots, written)]
        copy["shots"] = shots
        clip_id = "gta6_" + now_local().strftime("%Y%m%d_%H%M") + "_" + hashlib.md5(c.key.encode()).hexdigest()[:6]
        voiced = synthesize_lines([{"t": 0.0, "text": sh["text"]} for sh in shots], clip_id)
        max_total = float(vcfg.get("tour_max_sec", 80))
        segments, cursor, kept = [], 0.0, []
        for i_sh, (sh, v) in enumerate(zip(shots, voiced)):
            # room the renderer's line placement expects (0.35 s between lines + an end margin) - with less, it
            # dropped a "middle" line, i.e. a countdown number
            dur = min(max(v["dur"] + 0.5, 3.0), v["dur"] + 2.5) + (0.9 if i_sh == len(shots) - 1 else 0.0)
            if cursor + dur > max_total:
                if (fmt or {}).get("countdown"):
                    log.warning("[tour] ✋ the countdown doesn't fit in %.0f s - trying the next clip", max_total)
                    reel.unlink(missing_ok=True)
                    return None
                break
            a = min(max(0.0, sh["start"] - 0.3), max(0.0, sec_dur - dur))
            if segments and (fmt or {}).get("continuous"):
                a = max(a, segments[-1][1])  # one continuous scene: always forward, never the same footage twice
                if a + dur > sec_dur:
                    break
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
        # effects: opening shot starts punched in and settles, every cut gets a whoosh + a quick camera bump,
        # the last item lands with a boom, a punch-in, a shake and a flash (music lifts there too)
        cuts, acc = [], 0.0
        for a, b in segments[:-1]:
            acc += b - a
            cuts.append(acc)
        fx = {"zoom": [{"a": -0.2, "b": 0.5, "amt": 0.15, "in": 0.01, "out": 0.5}],
              "sfx": [{"t": max(0.0, t - 0.2), "kind": "whoosh"} for t in cuts], "flash": [], "shake": [], "action": []}
        fx["zoom"] += [{"a": t - 0.02, "b": t + 0.05, "amt": 0.08, "in": 0.02, "out": 0.45} for t in cuts[:-1]]
        if cuts:
            fx["sfx"].append({"t": cuts[-1], "kind": "boom"})
            fx["zoom"].append((cuts[-1], cuts[-1] + 1.0))
            fx["shake"].append((cuts[-1] + 0.02, 0.4))
            fx["flash"].append((cuts[-1], 0.5))
            fx["action"].append((cuts[-1] - 0.3, cuts[-1] + 2.5))
        rendered = _approved(render(clip, clip_id, copy["banner"], copy["commentary"], voiced=kept, fx=fx,
                                    game_volume=_mute(copy)), [sh["text"] for sh in shots])
        if not rendered:
            return None

        def repair_tour(r, bad):
            """QA found script problems: the whole script is rewritten by an AI that watches the whole cut (same
            shots - only the ones that fit - each line must still fit its shot), re-voiced, re-rendered once."""
            why = repair_note(bad, r.get("placed", [])) + "\nKeep each line as short as before or shorter."
            lines = _revise(clip, copy, [{"t": v["t"], "text": v["text"]} for v in kept], why)
            if not lines:
                return None
            new_v = synthesize_lines([{"t": 0.0, "text": ln["text"]} for ln in lines], clip_id + "_fix")
            if len(new_v) != len(kept):
                return None
            for k, v in enumerate(new_v):
                room = (segments[k][1] - segments[k][0]) - 0.35 - (0.9 if k == len(kept) - 1 else 0.0)
                if v["dur"] > room + 0.1:
                    log.info("[repair] the new line is too long for its shot: %s", v["text"][:50])
                    return None
            for k, v in enumerate(new_v):
                kept[k] = {**v, "t": kept[k]["t"]}
            copy["commentary"] = [{"t": round(v["t"], 1), "text": v["text"]} for v in kept]
            return _approved(render(clip, clip_id, copy["banner"], copy["commentary"], voiced=kept, fx=fx,
                                    game_volume=_mute(copy)), [v["text"] for v in kept])

        return publish_and_record(src, copy, clip_id, rendered, segments, repair=repair_tour)

    def produce(src, copy, energy, sec_dur) -> tuple[bool, dict] | None:
        """AI editor's window -> cut -> voice lines -> render -> publish -> record."""
        c, section = src.candidate, src.clip_path
        raw = sorted([ln for ln in copy.get("commentary_raw", []) if ln.get("t") is not None and ln.get("text")],
                     key=lambda ln: ln["t"])
        if _mute(copy) == 0.0 and len(raw) >= 4:
            # muted footage (another creator's voice removed): the narrator talks only ~1/3 of the time, the
            # rest would be dead silence (QA rejected 7-12 s gaps) -> the edit follows the voice: each line
            # gets its own piece of the same scene (section times from the AI), in order
            copy["shots"] = [{"start": float(ln["t"]), "text": ln["text"]} for ln in raw]
            log.info("[editor] muted footage - the edit follows the voice (%d lines)", len(copy["shots"]))
            return produce_tour(src, copy, sec_dur, energy)
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
        need_t = copy.get("required_t") if fmt and fmt.get("must_show") else None
        if need_t is not None and 0 <= need_t <= sec_dur and not start <= need_t <= end:
            # the proof (e.g. the cash counter going up) must be in the video: move the window over it
            length = end - start
            start = max(0.0, min(need_t - length * 0.6, sec_dur - length))
            end = min(sec_dur, start + length)
            log.info("[editor] moved the part to %.0f-%.0fs so it shows %s", start, end, fmt.get("must_show_short", ""))
        heard, heard_off = [], max(0.0, start - 3)
        if vcfg.get("fx", {}).get("smart_cut", True):
            # Whisper hears the footage's own speech (in-game dialogue): start / end on a pause, never mid-sentence
            # (QA: "starts with a jarring cut mid-sentence"), and the narration avoids talking over it
            try:
                import transcribe
                heard = [w for w in transcribe.words(section, heard_off, min(sec_dur, end + 3.5))[0]
                         if w.get("prob", 1) >= 0.4]
                if heard:
                    start, end = transcribe.smart_cut(section, start, end, heard, offset=heard_off,
                                                      max_len=max_len + 3)
            except Exception as exc:  # noqa: BLE001
                log.info("[cut] Whisper check of the footage's speech skipped: %.120s", exc)
        # the big moment: AI's payoff_t, else the most intense second of the picked part
        payoff = need_t if need_t is not None and start <= need_t <= end else copy.get("payoff_t")
        if payoff is None or not start <= payoff <= end:
            part = energy[int(start):int(end)] or [0]
            payoff = start + part.index(max(part)) + 0.5
        teaser = float(vcfg.get("fx", {}).get("teaser_sec", 0))
        segs = [(start, end)]
        if teaser > 0:  # cold open: 1.2 s flash of the payoff, then the clip from the start
            ta = max(0.0, min(payoff - teaser * 0.4, sec_dur - teaser))
            segs = [(ta, ta + teaser), (start, end)]
        clip = cut_segments(section, RAW_DIR / f"clip_{int(time.time())}.mp4", segs)
        section.unlink(missing_ok=True)
        if not clip:
            return None
        lead = teaser if teaser > 0 else 0.0
        hit = lead + payoff - start
        # hook + sound design: the cold open is punched in from frame 1 with a boom on its big moment and a
        # shake, a whoosh + white flash cut into the story; the payoff gets a riser, boom, punch-in, shake and an
        # impact flash; other sudden loud actions in the game sound (crash, shot) get a punchy hit + a small
        # punch-in; the music lifts a little on those beats
        fx = {"zoom": [], "sfx": [], "flash": [], "shake": [], "action": [(hit - 1.2, hit + 2.5)]}
        if lead:
            t_boom = max(0.0, payoff - segs[0][0] - 0.05)
            fx["zoom"].append({"a": -0.2, "b": lead - 0.1, "amt": None, "in": 0.01, "out": 0.12})
            fx["sfx"] += [{"t": t_boom, "kind": "boom"}, {"t": max(0.0, lead - 0.25), "kind": "whoosh"}]
            fx["shake"].append((t_boom + 0.03, 0.35))
            fx["flash"].append((lead, 0.9))
        fx["zoom"].append((max(lead, hit - 0.1), hit + 1.1))
        fx["sfx"].append({"t": max(lead, hit - 0.05), "kind": "boom"})
        fx["shake"].append((max(lead, hit - 0.02), 0.45))
        fx["flash"].append((max(lead, hit - 0.02), 0.45, 0.16))
        if hit - 1.05 >= lead + 0.4:
            fx["sfx"].append({"t": hit - 1.05, "kind": "riser"})
        try:
            from highlight import impacts
            for t in impacts(clip, count=3, min_gap=5.0, skip=[(0.0, lead + 0.6), (hit - 2.0, hit + 2.0)]):
                fx["sfx"].append({"t": max(0.0, t - 0.02), "kind": "hit"})
                fx["zoom"].append({"a": t - 0.04, "b": t + 0.3, "amt": 0.1, "in": 0.06, "out": 0.3})
                fx["action"].append((t - 0.3, t + 2.0))
        except Exception as exc:  # noqa: BLE001
            log.info("[fx] impact detection skipped: %.120s", exc)
        if heard:  # the footage's own speech on the final timeline: the narration slides into its gaps
            spans = []
            for w in heard:
                a = w["start"] + heard_off - start + lead
                b = w["end"] + heard_off - start + lead
                if b <= lead or a >= lead + (end - start):
                    continue
                if spans and a - spans[-1][1] < 0.5:
                    spans[-1][1] = b
                else:
                    spans.append([a, b])
            fx["dialogue"] = [tuple(x) for x in spans]
            log.info("[cut] the footage talks for %.0fs of %.0fs", sum(b - a for a, b in spans), end - start)
        talk = sum(b - a for a, b in fx.get("dialogue", []))
        if talk > float(vcfg.get("max_dialogue_share", 0.6)) * (end - start):
            # a cutscene / mission where the characters talk all the time: the narrator has no room (a posted Short
            # squeezed 7 lines between 57 s of dialogue) and story dialogue risks spoilers
            log.warning("[script] ✋ the characters talk for %.0f of %.0f s - no room for narration, next clip",
                        talk, end - start)
            clip.unlink(missing_ok=True)
            script_problems.append("dialogue-heavy footage")
            note_talky(c)  # an uploader whose videos keep doing this is skipped before download
            return None
        # the AI watches the WHOLE final video (cold open + story, with sound) and writes ONE script for it
        tails: list[float] = []
        commentary = _write_checked(clip, copy, [], lead=lead, spans=fx.get("dialogue"), tails=tails)
        if commentary is None and tails and vcfg.get("cut_quiet_tail", True):
            # every rejected script went quiet before the closing line: nothing happens at the end (owner: "you
            # can't do narration on a walking clip") - end the video where the talking stopped, write it again
            total = probe(clip)["duration"] or 0
            keep = round(max(tails) + float(vcfg.get("quiet_tail_room_sec", 3.5)), 2)
            if float(vcfg.get("min_short_sec", 20)) <= keep < total - 2:
                log.info("[editor] the scripts went quiet after %.0fs (nothing happens there) - the video ends at "
                         "%.0fs instead of %.0fs, script written again", max(tails), keep, total)
                short = cut_segments(clip, RAW_DIR / f"clip_{int(time.time())}_end.mp4", [(0.0, keep)])
                if short:
                    clip.unlink(missing_ok=True)
                    clip = short
                    segs[-1] = (segs[-1][0], segs[-1][0] + keep - lead)
                    fx = {k: [x for x in v if _fx_at(x) < keep - 0.3] for k, v in fx.items()}
                    commentary = _write_checked(clip, copy, [], lead=lead, spans=fx.get("dialogue"))
        if commentary is None:
            log.warning("[script] ✋ no script good enough for this footage - next clip (no voice / render spent)")
            clip.unlink(missing_ok=True)
            script_problems.append("no script good enough (judge / picture check)")
            return None
        copy["commentary"] = commentary
        log.info("[editor] voice lines: %s", [f"{ln['t']:.0f}s {ln['text']}" for ln in commentary])
        good, why = script_ok(commentary, c.kind, copy.get("ai_source", ""))
        if not good:
            log.warning("[script] ✋ not posting this clip: %s", why)
            clip.unlink(missing_ok=True)
            script_problems.append(why)
            return None

        clip_id = "gta6_" + now_local().strftime("%Y%m%d_%H%M") + "_" + hashlib.md5(c.key.encode()).hexdigest()[:6]
        rendered = _approved(render(clip, clip_id, copy["banner"], commentary, fx=fx, game_volume=_mute(copy)),
                             [ln["text"] for ln in commentary])
        if not rendered:
            return None

        def repair_moment(r, bad):
            """QA found script problems (lines the picture doesn't show / lines with no room): the AI watches the
            whole cut again and rewrites the WHOLE script (same times), judged again, re-rendered once."""
            lines = _revise(clip, copy, copy["commentary"], repair_note(bad, r.get("placed", [])), lead=lead,
                            spans=fx.get("dialogue"))
            if not lines:
                return None
            copy["commentary"] = lines
            return _approved(render(clip, clip_id, copy["banner"], lines, fx=fx, game_volume=_mute(copy)),
                             [ln["text"] for ln in lines])

        return publish_and_record(src, copy, clip_id, rendered, [list(x) for x in segs], repair=repair_moment)

    script_problems: list[str] = []  # clips skipped because no proper AI script could be written
    qa_problems: list[str] = []  # finished videos that failed the strict QA
    backup = None  # best "almost good enough" clip, used only if nothing reaches min_hype
    skip: set[str] = set()
    if fmt:
        log.info("[format] this video: %s", fmt["name"])
    for src in iter_clips(target_sec=0, max_attempts=int(scfg.get("max_attempts", 8)), skip_providers=skip,
                          fmt=fmt):
        if time.time() > budget_end:
            log.warning("Time budget exhausted")
            src.clip_path.unlink(missing_ok=True)
            break
        c = src.candidate
        section = src.clip_path                     # the whole downloaded slice (~2 min)
        sec_dur = probe(section)["duration"]
        try:  # laggy / frozen source footage looks broken (QA rejected such Shorts) - skip it before the AI
            import stream_clips
            rep = stream_clips.smoothness(section)
        except Exception:  # noqa: BLE001
            rep = 0.0
        if rep > float(gate.get("max_repeated_frames", 0.18)):
            log.info("[gate] ✋ rejected '%s': choppy footage (%.0f%% repeated frames)", c.title[:60], rep * 100)
            mark_rejected(c, "choppy")
            section.unlink(missing_ok=True)
            continue
        if c.provider in AUTHENTICITY_PROVIDERS and cfg.get("authenticity", {}).get("enabled", True):
            # "GTA 6" uploads are often GTA V / mods: one focused check on sharp frames before anything else
            import authentic
            real = authentic.check([section])
            if real is None or not real[0]["ok"]:
                why = ("no AI could check it is really GTA 6" if real is None else
                       f"not GTA 6 ({real[0]['game']} {real[0]['confidence'] * 100:.0f}%: {real[0]['evidence']})")
                log.info("[gate] ✋ rejected '%s': %s", c.title[:60], why)
                if real is not None:
                    mark_rejected(c, "not GTA VI footage")
                section.unlink(missing_ok=True)
                continue
        energy = energy_curve(section)
        copy = write_clip_copy(section, c.title, src.credit, c.provider, c.kind, energy_hints(energy, 30, 3),
                               tour=src.is_reel, fmt=fmt)
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
            elif copy.get("unsafe"):
                reason = "nudity / unsafe content"
            elif copy.get("has_narrator") and gate.get("reject_narrator", True) and c.kind != "stream" \
                    and not (fmt or {}).get("allow_narrated"):
                reason = TALKY_REASON
            elif copy["has_facecam"] and gate.get("reject_facecam", True) and c.kind != "stream":
                reason = "facecam / overlay"
            elif copy["hype_score"] < min_hype:
                reason = f"low hype ({copy['hype_score']})"
            elif fmt and not copy.get("fits_format", True):
                reason = f"doesn't fit the '{fmt['name']}' format"
            elif fmt and fmt.get("must_show") and not copy.get("shows_required"):
                reason = f"doesn't show {fmt.get('must_show_short', 'what the format needs')}"
            elif fmt and fmt.get("evidence_terms") and not _evidence_ok(copy.get("required_evidence", ""), fmt):
                # a posted Money Tip was mostly a bike chase with vague "grab the cash" lines: the proof must be concrete
                reason = (f"no concrete proof of {fmt.get('must_show_short', 'the format')} "
                          f"(AI: \"{copy.get('required_evidence', '')[:60]}\")")
            if reason:
                log.info("[gate] ✋ rejected '%s': %s | %s", c.title[:60], reason, copy["scene_summary"])
                mark_rejected(c, reason)
                keep = (reason.startswith(("low hype", "doesn't fit")) and copy["hype_score"] >= backup_hype
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

        try:
            if src.is_reel and len(copy.get("shots") or []) >= int(vcfg.get("tour_min_shots", 5)):
                result = produce_tour(src, copy, sec_dur, energy)
            else:
                if src.is_reel:
                    log.info("[tour] AI gave too few shots - making a single-moment video instead")
                result = produce(src, copy, energy, sec_dur)
        except HumanVoiceUnavailable as exc:
            # every clip would hit the same wall - stop instead of posting a robot voice
            for p in (src.clip_path, backup[0].clip_path if backup else None):
                if p:
                    p.unlink(missing_ok=True)
            return False, {"error": f"human voice unavailable, nothing posted: {exc}"}
        if result:
            if backup:
                backup[0].clip_path.unlink(missing_ok=True)
            return result

    if backup:
        log.info("[gate] nothing reached hype %d - posting the best backup (hype %s)", min_hype,
                 backup[1]["hype_score"])
        b_src, b_copy, b_energy, b_dur = backup
        try:
            if b_src.is_reel and len(b_copy.get("shots") or []) >= int(vcfg.get("tour_min_shots", 5)):
                result = produce_tour(b_src, b_copy, b_dur, b_energy)
            else:
                result = produce(*backup)
        except HumanVoiceUnavailable as exc:
            return False, {"error": f"human voice unavailable, nothing posted: {exc}"}
        if result:
            return result
    if qa_problems:
        return False, {"error": "nothing posted - every finished video failed QA: " + "; ".join(qa_problems[:3])}
    if script_problems:
        return False, {"error": "nothing posted - no proper AI script could be written (AI busy / out of quota?): "
                                + "; ".join(dict.fromkeys(script_problems))}
    return False, {"error": "no usable clip could be produced"}


# --------------------------------------------------------------------------- remake

def run_remake(dry_run: bool) -> tuple[bool, dict]:
    """Re-make an already published clip - same footage, same cut - with a freshly written AI script and a
    new Gemini voice-over, then post it again. REMAKE_CLIP = clip id (see the run summary / manifest);
    empty = the last published video."""
    from common import RAW_DIR, probe
    from formats import _rotation
    from gemini_writer import script_ok, write_clip_copy
    from highlight import cut_segments, energy_curve, energy_hints
    from publisher import _available, publish_video, routes_for
    from source_fetcher import Candidate, _download, _fill_metadata, load_manifest, save_manifest
    from video_processor import HumanVoiceUnavailable, render

    cfg = load_config()
    want = env("REMAKE_CLIP").strip()
    items = [i for i in load_manifest()["items"]
             if i.get("status") == "published" and (not want or i.get("id") == want)]
    if not items:
        return False, {"error": f"no published clip {want or ''} found to remake"}
    item = items[-1]
    s = item["source"]
    segs = s.get("segments") or []
    if len(segs) != 1:
        return False, {"error": "only single-moment clips can be remade (this one is a multi-shot tour)"}
    log.info("[remake] %s: %s %s cut %s", item["id"], s["title"][:70], s["section"], segs[0])
    key = next((k for k, st in load_manifest()["sources"].items() if st.get("url") == s["url"]), s["url"])
    c = Candidate(key=key, url=s["url"], title=s["title"], uploader=s.get("credit", ""), provider=s["provider"],
                  kind=s.get("kind", "gameplay"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    section = RAW_DIR / "remake_section.mp4"
    try:
        if not _fill_metadata(c, cfg) or not _download(c, int(s["section"][0]), int(s["section"][1]), section):
            return False, {"error": "could not download the original footage again"}
    except PermissionError:
        return False, {"error": "YouTube blocked the download (check YTDLP_COOKIES)"}
    clip = cut_segments(section, RAW_DIR / "remake_clip.mp4", [tuple(segs[0])])
    section.unlink(missing_ok=True)
    if not clip:
        return False, {"error": "could not cut the clip"}
    dur = probe(clip)["duration"]
    fmt = next((f for f in _rotation(cfg) if f["id"] == item.get("format_id")), None)
    copy = write_clip_copy(clip, c.title, s.get("credit", ""), c.provider, c.kind,
                           energy_hints(energy_curve(clip), 30, 3), tour=False, fmt=fmt)
    copy["clip_start"], copy["clip_end"] = 0.0, dur          # keep the exact same cut
    import script_judge
    from gemini_writer import write_final_script
    w = write_final_script(clip, copy, fmt)  # the AI watches the whole clip and writes ONE script for it
    verdict = script_judge.judge(w["lines"], {"fmt": (fmt or {}).get("name", "GTA 6 Short"), "story": w["story"],
                                              "banner": w["banner"] or copy.get("banner", ""),
                                              "title": w["title"] or copy.get("youtube_title", "")}) if w else None
    if not w or not verdict["ok"]:
        clip.unlink(missing_ok=True)
        return False, {"error": "nothing posted - no script good enough (whole-video writer + judge). Try again later."}
    copy["commentary"] = w["lines"]
    copy["banner"] = w["banner"] or copy.get("banner", "")
    copy["youtube_title"] = w["title"] or copy.get("youtube_title", "")
    log.info("[remake] script (%s/10): %s", verdict["score"], [f"{ln['t']:.0f}s {ln['text']}" for ln in
                                                               copy["commentary"]])
    good, why = script_ok(copy["commentary"], c.kind, copy.get("ai_source", ""))
    if not good:
        clip.unlink(missing_ok=True)
        return False, {"error": f"nothing posted - {why}. Try the remake again later."}
    clip_id = item["id"] + "_v2"
    try:
        rendered = render(clip, clip_id, copy["banner"], copy["commentary"])
    except HumanVoiceUnavailable as exc:
        return False, {"error": f"human voice unavailable, nothing posted: {exc}"}
    if not rendered:
        return False, {"error": "render failed"}
    platforms = [p for p in ("instagram", "tiktok", "youtube", "x") if any(_available(pr, p) for pr in routes_for(p))]
    results = {} if dry_run else publish_video(Path(rendered["path"]), platforms, copy, s.get("credit", ""))
    ok = dry_run or any(r.get("ok") for r in results.values())
    record = {"id": clip_id, "slot": "remake", "remake_of": item["id"], "format_id": item.get("format_id", ""),
              "created_at": datetime.now(timezone.utc).isoformat(), "source": s, "copy": copy,
              "rendered": rendered["path"], "duration": rendered["duration"],
              "voice_lines": rendered.get("lines", []), "dry_run": dry_run, "results": results,
              "status": "published" if ok and not dry_run else ("dry_run" if dry_run else "failed")}
    m = load_manifest()
    m["items"].append(record)
    save_manifest(m)
    return ok, record


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
    if detail.get("format"):
        lines.append(f"**Format:** {detail['format']}  ")
    if detail.get("moments"):
        lines.append(f"**Long video:** {detail['copy'].get('title', '')} ({detail['duration']:.0f}s)  ")
        lines += [f"- #{m['n']} {m['label']} ({m['streamer']}, hype {m['hype']})" for m in detail["moments"]]
    if "text" in detail:
        lines.append(f"> {detail['text']}")
    if "copy" in detail and not detail.get("moments"):  # Shorts (long videos are summarised above)
        cp, s = detail["copy"], detail["source"]
        lines += [f"**Banner:** {cp['banner']}  ", f"**Source:** [{s['title']}]({s['url']}) by {s['credit']} "
                  f"({s['provider']}, section {s['section']})  ",
                  f"**AI check:** {cp.get('content_type', '?')}, hype {cp['hype_score']}, verified={cp['verified_by_video']} by {cp.get('ai_source', '?')} — {cp['scene_summary']}  ",
                  "**Voice lines:** " + " · ".join(f"[{ln['t']}s] {ln['text']}" for ln in
                                                   (detail.get('voice_lines') or cp.get('commentary', [])))]
    if detail.get("streamer") and "copy" not in detail:  # Video A as a viral stream clip
        lines.append(f"**Stream clip:** {detail.get('banner', '')} — {detail['streamer']} [{detail.get('url', '')}]"
                     f"({detail.get('url', '')})  ")
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


def _fallback_stream_clip(cfg: dict, slot: dict, dry_run: bool, why: str) -> tuple[bool, dict] | None:
    """A video slot posted nothing -> post the most viral famous-streamer clip instead (streamer's own voice +
    subtitles: far less AI work than a narrated video, so it still works on a low-quota day). Owner: "if nothing got
    posted u should push a short clip". None = not tried (switched off / no time left)."""
    fb = cfg.get("fallback", {})
    if not fb.get("stream_clip_on_fail", True):
        return None
    left = float(fb.get("max_elapsed_min", 80)) * 60 - (time.time() - _T0)
    if left < float(fb.get("min_left_min", 8)) * 60:
        log.info("[fallback] no time left for a backup stream clip")
        return None
    log.warning("[fallback] nothing posted (%.150s) - posting the most viral streamer clip instead", why)
    import stream_clips
    try:
        ok, detail = stream_clips.run_for_slot(slot["platforms"], dry_run, deadline=time.time() + left)
    except Exception as exc:  # noqa: BLE001 - the backup must never crash the run
        log.warning("[fallback] stream clip failed: %.200s", exc)
        return None
    detail["fallback_for"] = str(why)[:300]
    log.info("[fallback] %s", "✅ backup stream clip done" if ok else "no stream clip passed either")
    return ok, detail


_T0 = time.time()


def main() -> int:
    global _T0
    _T0 = time.time()
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
    if args.slot == "approve_long":  # the owner watched the long video in the Artifacts and approves it
        from long_video import retry_pending
        res = retry_pending(False, approve=True)
        if not res:
            log.error("No long video is waiting for approval.")
            return 1
        if res.get("ok"):
            record_history("approve_long", {"type": "long_video_upload", "manual": True}, True, res)
            log.info("Long video published: %s", res["results"]["youtube"].get("url"))
            return 0
        log.error("Upload refused: %s", res.get("error"))
        return 1
    if args.slot == "trends":  # manual check of the trend radar (prints today's hot topics)
        import trends
        return trends.main()
    if args.slot == "stream_clip":  # manual test of the viral stream clips (normally streams.yml)
        import stream_clips
        return stream_clips.run_once(args.dry_run)
    if args.slot == "remake":
        READY_DIR.mkdir(parents=True, exist_ok=True)
        ok, detail = run_remake(args.dry_run)
        summarize({"type": "remake"}, detail.get("id", "remake"), ok, detail, args.dry_run)
        if not ok:
            log.error("Remake FAILED: %s", detail.get("error") or detail.get("results"))
        return 0 if ok else 1

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

    # a long video YouTube refused earlier (upload limit) is retried first, on any run
    try:
        from long_video import retry_pending
        pending = retry_pending(args.dry_run)
        if pending and pending.get("ok"):
            record_history(f"{sid}_pending_long", {"type": "long_video", "manual": True}, True, pending)
            history = load_json(HISTORY_FILE, {"completed_slots": [], "posts": []})
    except Exception as exc:  # noqa: BLE001
        log.warning("[long] pending upload retry failed: %.200s", exc)

    # "not before" date (e.g. the owner published one by hand - the next automatic one waits)
    nb = slot.get("not_before")
    if nb and not slot.get("manual") and datetime.now(timezone.utc) < datetime.fromisoformat(nb.replace("Z", "+00:00")):
        log.info("%s waits until %s (not_before in config.json) - skipping", slot["type"], nb)
        return 0

    # posting rhythm: e.g. a Short only every 2 days, a long video every 3 days (counted from the last success)
    every = float(slot.get("every_days", 0) or 0)
    if every and not slot.get("manual"):
        done = [p.get("at", "") for p in history.get("posts", []) if p.get("type") == slot["type"] and p.get("ok")]
        if done:
            last = datetime.fromisoformat(max(done).replace("Z", "+00:00"))
            gap_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600
            if gap_h < every * 24 - 4:
                log.info("%s posted %.0f h ago - this slot runs every %g days, skipping", slot["type"], gap_h, every)
                return 0

    # a platform can start later (e.g. Video A goes to YouTube Shorts only from week 2: upload limit)
    starts = slot.get("platform_from") or {}
    if starts:
        now_utc = datetime.now(timezone.utc)
        later = [p for p in slot["platforms"] if starts.get(p)
                 and now_utc < datetime.fromisoformat(starts[p].replace("Z", "+00:00"))]
        if later:
            log.info("%s: %s start later (%s)", slot["type"], later, {p: starts[p] for p in later})
            slot = {**slot, "platforms": [p for p in slot["platforms"] if p not in later]}

    READY_DIR.mkdir(parents=True, exist_ok=True)
    try:  # background music: CC0 beats from Freesound, downloaded once (FREESOUND_API_KEY)
        import music
        music.ensure_tracks()
    except Exception as exc:  # noqa: BLE001
        log.warning("[music] skipped: %.150s", exc)
    try:  # trend radar: what people search for / watch now (cached for a few hours, never breaks a run)
        import trends
        trends.refresh()
    except Exception as exc:  # noqa: BLE001
        log.warning("[trends] skipped: %.150s", exc)
    try:  # learn from views: refresh recent videos' YouTube stats (cheap, a few per run)
        import analytics
        analytics.refresh_stats()
        scores = analytics.format_scores()
        if scores:
            log.info("[learn] format scores: %s", {k: (v["score"], v["n"]) for k, v in scores.items()})
    except Exception as exc:  # noqa: BLE001
        log.warning("[learn] stats refresh skipped: %.150s", exc, exc_info=True)
    if slot["type"] == "long_video":  # long YouTube video: stream-moments countdown or a guide (money / trend)
        from long_video import run_long
        ok, detail = run_long(args.dry_run)
    elif slot["type"] in VIDEO_TYPES:
        try:
            import lessons
            log.info("[learn] QA history: %s", lessons.stats_line())
        except Exception:  # noqa: BLE001
            pass
        from formats import advance_format, current_format, format_by_id, mark_topic, specialize
        fmt = current_format(cfg, slot)
        tried_streams = False
        if fmt and fmt["id"] == "stream_reactions" and cfg.get("stream_clips", {}).get("in_main_schedule", True):
            # the most viral famous-streamer clip right now (streamer's own voice + subtitles)
            import stream_clips
            tried_streams = True
            ok, detail = stream_clips.run_for_slot(slot["platforms"], args.dry_run)
            if not ok and detail.get("no_stream_clip"):
                log.info("[streams] no viral stream clip today - following a trend instead")
                fmt = specialize(format_by_id("trending", cfg)) or specialize(format_by_id("top5_funny", cfg)) or fmt
                ok, detail = run_video_slot(slot, args.dry_run, fmt)
        else:
            ok, detail = run_video_slot(slot, args.dry_run, fmt)
        if fmt:
            detail.setdefault("format", fmt["name"])
            if not args.dry_run and "skipped" not in detail:
                if not slot.get("format"):  # Video B (fixed Money Tip) doesn't move Video A's rotation
                    advance_format(fmt, ok)
                if ok:
                    mark_topic(fmt)
        if not ok and "skipped" not in detail and not tried_streams:
            fb = _fallback_stream_clip(cfg, slot, args.dry_run,
                                       detail.get("error") or str(detail.get("results") or "failed"))
            if fb:
                ok = fb[0] or ok
                detail = fb[1] if fb[0] else {**detail, "fallback": fb[1].get("error", "failed")}
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
