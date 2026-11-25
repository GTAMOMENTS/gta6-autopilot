"""Smoke render of a whole Short through autopilot.run_video_slot (dry run): synthetic ~12 s footage ->
gates -> cut (cold open) -> fake Gemini voice -> 1080x1920 render with captions, SFX and a music bed -> QA ->
record. Only the network parts are faked (footage source, AI copy, authenticity AI, TTS, AI editor)."""
from __future__ import annotations

from pathlib import Path

import pytest

import autopilot
import common
import formats
import gemini_writer
import music
import publisher
import qa
import source_fetcher as sf
from fakes import ffmpeg, needs_ffmpeg, stream_info

pytestmark = [pytest.mark.slow, needs_ffmpeg]

LINES = ["He really thought this ramp was a good idea today.",
         "Five stars on him and he still goes for it.",
         "Watch the cop car try to follow him off the edge.",
         "Would you have tried that jump or taken the bridge?"]


@pytest.fixture
def short_world(monkeypatch, make_clip, fake_tts, sandbox):
    fake_tts.wps = 4.5  # a 12 s Short can't fit a 35+ word script at normal speed
    footage = make_clip("section.mp4", dur=12, size="1280x720", out=sandbox.raw / "section.mp4")
    beat = music.CACHE / "beat.mp3"
    beat.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg("-f", "lavfi", "-i", "sine=f=110:d=20,volume=4", "-ac", "2", str(beat))

    cand = sf.Candidate(key="yt:smoke1", url="https://youtube.com/watch?v=smoke1", title="GTA 6 insane ramp jump",
                        uploader="StuntChannel", provider="youtube", duration=600)
    monkeypatch.setattr(sf, "iter_clips", lambda **kw: iter([sf.ClipSource(candidate=cand, clip_path=footage,
                                                                            section=(100, 112))]))
    import authentic
    monkeypatch.setattr(authentic, "check", lambda paths: [{"game": "GTA VI", "confidence": 0.95, "evidence": "Vice City",
                                                            "ok": True} for _ in paths])
    seen = {}

    def write_copy(section, title, credit, provider, kind="gameplay", hints=None, tour=False, fmt=None):
        seen["fmt"] = fmt
        d = {"banner": "HE TOOK THE JUMP", "content_type": "action", "hype_score": 8, "is_gta6_footage": True,
             "clip_start": 0, "clip_end": 12, "payoff_t": 7, "scene_summary": "a car flies off a ramp",
             "youtube_title": "He took the jump in GTA 6", "caption": "He took the jump. Would you?",
             "commentary": [{"t": i * 3, "line": ln} for i, ln in enumerate(LINES)], "hashtags": ["#GTA6", "#Stunts"]}
        out = gemini_writer._clean_copy(d, 12)
        out.update(verified_by_video=True, ai_source="gemini-video")
        return out
    monkeypatch.setattr(gemini_writer, "write_clip_copy", write_copy)

    def final_script(clip, copy, fmt=None, lead=0.0, spans=None, feedback="", slots=None, n=1):
        seen.setdefault("writer", []).append({"feedback": feedback, "slots": slots, "lead": lead})
        if slots:
            return {"lines": [dict(s) for s in slots], "banner": "", "title": "", "story": "a car flies off a ramp"}
        return {"lines": [{"t": round(lead + 0.1 + i * 3, 2), "text": ln} for i, ln in enumerate(LINES)],
                "banner": "", "title": "", "story": "a car flies off a ramp"}
    monkeypatch.setattr(gemini_writer, "write_final_script", final_script)
    import script_judge
    monkeypatch.setattr(script_judge, "judge", lambda lines, ctx, numbered_ok=False, countdown=None, timing=None:
                        {"ok": True, "score": 8.0, "problems": [], "by": "test/judge", "static": []})
    monkeypatch.setattr(qa, "line_check", lambda path, lines, kind: [])  # its own tests: tests/test_qa_linecheck.py
    monkeypatch.setattr(qa, "ai_review", lambda path, context: (seen.__setitem__("qa_context", context) or
                                                                {"publish": True, "score": 8, "issues": [], "summary": "ok",
                                                                 "pro_level": 6, "improvements": ["tighter hook"]}))
    monkeypatch.setattr(publisher, "publish_video", lambda *a, **k: pytest.fail("a dry run must never publish"))
    real_technical = qa.technical

    def spy(path, still_ok=None):
        issues = real_technical(path, still_ok)
        seen.setdefault("technical", []).append({"path": Path(path), "issues": issues})
        return issues
    monkeypatch.setattr(qa, "technical", spy)
    return seen


def test_short_smoke_render(cfg, short_world, fake_tts, sandbox):
    fmt = formats.format_by_id("stunts_crashes", cfg)
    slot = {"type": "vertical_clip", "platforms": ["instagram", "x", "tiktok", "youtube"], "time": "19:40"}
    ok, rec = autopilot.run_video_slot(slot, dry_run=True, fmt=fmt)
    assert ok, rec
    teaser = cfg["video"]["fx"]["teaser_sec"]

    # output: vertical 1080x1920, ~12 s + cold open, sound and picture the same length
    video = Path(rec["rendered"])
    assert video.parent == sandbox.ready and video.exists()
    info = stream_info(video)
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["v_dur"] == pytest.approx(12 + teaser, abs=0.3)
    assert abs(info["v_dur"] - info["a_dur"]) <= cfg["qa"]["max_av_diff"], info
    assert rec["duration"] == pytest.approx(12 + teaser, abs=0.3)

    # nothing high in the technical QA (the pipeline's own call on the final file)
    technical = short_world["technical"]
    assert len(technical) == 1 and technical[0]["path"] == video
    high = [i for i in technical[0]["issues"] if i["severity"] == "high"]
    assert high == [], high
    report = (sandbox.ready / (video.stem + "_QA.txt")).read_text(encoding="utf-8")
    assert report.startswith("QA PASSED (short)") and "PRO LEVEL: 6/10" in report

    # voice: ONE Gemini request with only the script (owner's rule), every line placed after the cold open
    assert fake_tts.prompts == ["\n\n".join(LINES)]
    assert set(fake_tts.voices) == {cfg["voice"]["gemini_tts_voice"]}
    texts = [ln["text"] for ln in rec["voice_lines"]]
    assert texts[0] == LINES[0] and texts[-1] == LINES[-1], "hook and closing question always stay"
    assert rec["voice_lines"][0]["t"] >= teaser
    ass = (sandbox.voice / f"{rec['id']}.ass").read_text(encoding="utf-8")
    assert "HE TOOK THE JUMP" in ass and ass.count("Dialogue:") > 5

    # cold open = a flash of the payoff (payoff_t=7), then the clip from the start
    segs = rec["source"]["segments"]
    assert len(segs) == 2 and segs[1] == [0.0, 12.0]
    assert segs[0][0] <= 7 <= segs[0][1] and segs[0][1] - segs[0][0] == pytest.approx(teaser)

    # dry run: recorded (in the temp state, not the real queue/), nothing published, series not bumped
    assert rec["status"] == "dry_run" and rec["results"] == {}
    assert rec["copy"]["series"] == "GTA 6 Stunts #1" and rec["format_id"] == "stunts_crashes"
    m = common.load_json(sf.MANIFEST_FILE, {})
    assert sf.MANIFEST_FILE.parent == sandbox.state
    assert [i["id"] for i in m["items"]] == [rec["id"]]
    assert m["sources"]["yt:smoke1"]["uses"][0][:2] == [100, 112]
    assert "Narration lines with start times" in short_world["qa_context"]
    assert short_world["fmt"]["id"] == "stunts_crashes"


def test_short_repair_rewrites_the_whole_script_from_the_whole_video(cfg, short_world, fake_tts, sandbox, monkeypatch):
    """Final QA flags a line -> the AI watches the whole cut again and rewrites the WHOLE script (same times, one
    story), judged again, re-rendered once - never a frame-by-frame patch of one line."""
    import lessons
    calls = {"n": 0}

    def line_check(path, lines, kind):
        calls["n"] += 1
        if calls["n"] == 2:   # 1 = pre-check (ok), 2 = final QA on the rendered video (one bad line), 3 = after repair
            return [{"severity": "high", "type": "line_not_shown", "at": lines[1]["start"], "index": 1,
                     "reason": "frames show a parked car", "detail": "the picture doesn't show it"}]
        return []
    monkeypatch.setattr(qa, "line_check", line_check)
    real = gemini_writer.write_final_script

    def writer(clip, copy, fmt=None, lead=0.0, spans=None, feedback="", slots=None, n=1):
        if slots:
            short_world.setdefault("repair", []).append((feedback, [s["t"] for s in slots]))
            return {"lines": [{"t": s["t"], "text": "Look at that car flying straight off the ramp." if i == 1 else s["text"]}
                              for i, s in enumerate(slots)], "banner": "", "title": "", "story": "ramp jump"}
        return real(clip, copy, fmt, lead, spans, feedback, slots, n)
    monkeypatch.setattr(gemini_writer, "write_final_script", writer)
    slot = {"type": "vertical_clip", "platforms": ["x"], "time": "19:40"}
    ok, rec = autopilot.run_video_slot(slot, dry_run=True, fmt=formats.format_by_id("stunts_crashes", cfg))
    assert ok, rec
    assert calls["n"] == 3, "pre-check, final QA, QA after the repair"
    fb, times = short_world["repair"][0]
    assert "QA REJECTED" in fb and "WHOLE script" in fb and len(times) == len(LINES)
    assert any("car flying straight off the ramp" in ln["text"] for ln in rec["copy"]["commentary"])
    stages = [a["stage"] for a in lessons._load()["attempts"]]
    assert stages[-2:] == ["final", "repair"] and lessons._load()["attempts"][-1]["pass"] is True


def test_rejected_script_goes_back_to_the_writer(cfg, short_world, fake_tts, sandbox, monkeypatch):
    """The judge rejects the first script -> the writer watches the whole video again with the reasons and writes a
    NEW script; a line that doesn't match the picture does the same. Only then voice + render."""
    import script_judge
    verdicts = iter([{"ok": False, "score": 3.0, "problems": ["promises a car theft that never happens"], "by": "j",
                      "static": []},
                     {"ok": True, "score": 8.0, "problems": [], "by": "j", "static": []},
                     {"ok": True, "score": 8.0, "problems": [], "by": "j", "static": []}])
    monkeypatch.setattr(script_judge, "judge", lambda lines, ctx, numbered_ok=False, countdown=None, timing=None: next(verdicts))
    checks = {"n": 0}

    def line_check(path, lines, kind):
        checks["n"] += 1
        if checks["n"] == 1:  # the 2nd script: one line doesn't match its moment
            return [{"severity": "high", "type": "line_not_shown", "index": 2, "reason": "no cop car on screen"}]
        return []
    monkeypatch.setattr(qa, "line_check", line_check)
    slot = {"type": "vertical_clip", "platforms": ["x"], "time": "19:40"}
    ok, rec = autopilot.run_video_slot(slot, dry_run=True, fmt=formats.format_by_id("stunts_crashes", cfg))
    assert ok, rec
    fbs = [w["feedback"] for w in short_world["writer"]]
    assert fbs[0] == "" and "REJECTED by the head of content (3/10)" in fbs[1] and "car theft" in fbs[1]
    assert "DON'T MATCH WHAT IS ON SCREEN" in fbs[2] and "no cop car" in fbs[2]
    assert all(w["slots"] is None for w in short_world["writer"][:3]), "a Short's script is written for the whole video"
    assert len(fake_tts.prompts) == 1, "voice only for the script that passed"


def test_best_of_three_scripts_is_used(cfg, short_world, fake_tts, sandbox, monkeypatch):
    """Each attempt writes 3 different scripts; the judge scores all of them and the best one that passes is voiced
    (no second attempt needed)."""
    import script_judge
    # fits the 12 s clip (the first version of this test had 2 of its 5 lines silently cut by the renderer)
    alt_lines = ["He had one job today: land the jump.",
                 "He chose violence, and the ramp chose gravity.",
                 "The cops are still working out what happened.",
                 "Ten out of ten commitment, zero planning.",
                 "Jump or the boring bridge, what's your call?"]

    def writer(clip, copy, fmt=None, lead=0.0, spans=None, feedback="", slots=None, n=1):
        short_world.setdefault("n", []).append(n)
        base = {"banner": "", "title": "", "story": "ramp jump"}
        mk = lambda ls: [{"t": round(lead + 0.1 + i * 2.5, 2), "text": t} for i, t in enumerate(ls)]
        return {**base, "lines": mk(LINES), "angle": "roast",
                "alternatives": [{**base, "lines": mk(alt_lines), "angle": "deadpan"},
                                 {**base, "lines": mk(LINES[::-1]), "angle": "hype"}]}
    monkeypatch.setattr(gemini_writer, "write_final_script", writer)
    monkeypatch.setattr(script_judge, "judge", lambda lines, ctx, numbered_ok=False, countdown=None, timing=None: (
        {"ok": True, "score": 8.0, "problems": [], "by": "j", "static": []} if lines[0]["text"].startswith("He had")
        else {"ok": False, "score": 5.0, "problems": ["not funny"], "by": "j", "static": []}))
    slot = {"type": "vertical_clip", "platforms": ["x"], "time": "19:40"}
    ok, rec = autopilot.run_video_slot(slot, dry_run=True, fmt=formats.format_by_id("stunts_crashes", cfg))
    assert ok, rec
    assert short_world["n"] == [3], "one attempt, three scripts"
    assert [ln["text"] for ln in rec["copy"]["commentary"]][:2] == alt_lines[:2]
    assert [ln["text"] for ln in rec["voice_lines"]] == alt_lines, "the whole approved script is in the video"


def test_script_too_long_for_the_clip_is_rewritten_shorter(cfg, short_world, fake_tts, sandbox, monkeypatch):
    """The renderer drops lines the voice-over has no room for - the posted story would miss lines the judge
    approved. QA catches it before any slow check, the writer rewrites the whole script shorter, QA passes."""
    import lessons
    long_lines = [f"{w} and honestly this whole ramp situation keeps getting more ridiculous every second."
                  for w in ("He floors it", "Five stars chase him", "The cop car follows", "Everyone screams")] + [
        "Would you have taken the bridge or tried that jump yourself today?"]
    short_lines = ["He floors it.", "Five stars behind him.", "The cop car follows.", "Gravity wins.",
                   "Bridge or ramp?"]

    def writer(clip, copy, fmt=None, lead=0.0, spans=None, feedback="", slots=None, n=1):
        if slots:
            short_world.setdefault("repair", []).append(feedback)
            return {"lines": [{"t": s["t"], "text": t} for s, t in zip(slots, short_lines)], "banner": "", "title": "",
                    "story": "ramp jump"}
        return {"lines": [{"t": round(lead + 0.1 + i * 2.5, 2), "text": t} for i, t in enumerate(long_lines)],
                "banner": "", "title": "", "story": "ramp jump"}
    monkeypatch.setattr(gemini_writer, "write_final_script", writer)
    slot = {"type": "vertical_clip", "platforms": ["x"], "time": "19:40"}
    ok, rec = autopilot.run_video_slot(slot, dry_run=True, fmt=formats.format_by_id("stunts_crashes", cfg))
    assert ok, rec
    assert len(short_world["repair"]) == 1 and "TOO LONG FOR THE VIDEO" in short_world["repair"][0]
    assert [ln["text"] for ln in rec["voice_lines"]] == short_lines, "every approved line is in the posted video"
    assert len(short_world["technical"]) == 1, "the too-long render never reached the slow checks"
    first, last = lessons._load()["attempts"][-2:]
    assert first["stage"] == "final" and not first["pass"] and last["stage"] == "repair" and last["pass"]
    assert "script_lines_dropped" in first["types"]
    assert "too long for their video" in lessons.prompt_hints("stunts_crashes"), "the writer learns from it"


# --------------------------------------------------------------------------- a Top 5 tour that skips a number

TOUR = ["Five things you have to try in GTA 6 right now.",
        "Number five: you can dive off any pier you like.",
        "Number four: the cops follow you out on jet skis.",
        "Number three: the seaplane lands right on the beach.",
        "Number two: a bar fight spills into the street.",
        "And this outfit swap takes one second flat - which one would you try first?"]


@pytest.fixture
def top5_tour(monkeypatch, short_world, make_clip, sandbox):
    """A 3-slice reel whose editor script counts 5..2 and never says number one (live run 2026-11-25)."""
    def world(lines):
        reel = make_clip("reel.mp4", dur=30, size="1280x720", out=sandbox.raw / "reel.mp4")
        cand = sf.Candidate(key="yt:top5", url="https://youtube.com/watch?v=top5", title="GTA 6 free roam no commentary",
                            uploader="RoamChannel", provider="youtube", duration=3600)
        monkeypatch.setattr(sf, "iter_clips", lambda **kw: iter([sf.ClipSource(
            candidate=cand, clip_path=reel, section=(100, 110), sections=[(100, 110), (900, 910), (1700, 1710)],
            is_reel=True)]))

        def write_copy(section, title, credit, provider, kind="gameplay", hints=None, tour=False, fmt=None):
            d = {"banner": "5 THINGS TO TRY", "content_type": "action", "hype_score": 9, "is_gta6_footage": True,
                 "scene_summary": "combat, a seaplane, an outfit swap", "youtube_title": "GTA 6: 5 things to try",
                 "caption": "Which one first?", "hashtags": ["#GTA6"],
                 "shots": [{"start": 1 + 4.5 * i, "line": ln} for i, ln in enumerate(lines)]}
            out = gemini_writer._clean_copy(d, 30)
            out.update(verified_by_video=True, ai_source="gemini-video")
            return out
        monkeypatch.setattr(gemini_writer, "write_clip_copy", write_copy)

        def renumber(clip, copy, fmt=None, lead=0.0, spans=None, feedback="", slots=None, n=1):
            short_world.setdefault("writer", []).append({"feedback": feedback, "slots": slots})
            texts = [s["text"] for s in slots]
            if "SKIP NUMBER 1" in feedback:
                texts[-1] = texts[-1].replace("And this", "Number one: this", 1)
            return {"lines": [{"t": s["t"], "text": t} for s, t in zip(slots, texts)], "banner": "", "title": "",
                    "story": "five things to try"}
        monkeypatch.setattr(gemini_writer, "write_final_script", renumber)
        import script_judge  # the real numbering rule: a countdown that skips a number never passes
        monkeypatch.setattr(script_judge, "judge", lambda lines, ctx, numbered_ok=False, countdown=None, timing=None: (
            lambda st: {"ok": not st, "score": 8.0, "problems": st, "by": "test/judge", "static": st})(
            script_judge.static_problems([ln["text"] for ln in lines], numbered_ok, countdown)))
    return world


def test_countdown_that_skips_a_number_is_renumbered_not_dropped(cfg, top5_tour, short_world, fake_tts):
    top5_tour(TOUR)
    slot = {"type": "video_b", "platforms": ["x"], "time": "01:40"}
    ok, rec = autopilot.run_video_slot(slot, dry_run=True, fmt=formats.format_by_id("top5_things_to_do", cfg))
    assert ok, rec
    first = short_world["writer"][0]
    assert "SKIP NUMBER 1" in first["feedback"] and len(first["slots"]) == len(TOUR), "the writer is told at once"
    said = [ln["text"] for ln in rec["voice_lines"]]
    assert said[-1].startswith("Number one:") and formats.countdown_missing(said, 5) == []


def test_countdown_with_no_slot_for_every_number_is_still_dropped(cfg, top5_tour, short_world, fake_tts):
    top5_tour(TOUR[:5])  # hook + 5..2: no shot left for number one
    slot = {"type": "video_b", "platforms": ["x"], "time": "01:40"}
    ok, rec = autopilot.run_video_slot(slot, dry_run=True, fmt=formats.format_by_id("top5_things_to_do", cfg))
    assert not ok and "countdown skipped [1]" in rec["error"]
    assert "writer" not in short_world, "no script / voice spent on it"


def test_a_quiet_end_is_cut_off_and_the_script_written_again(cfg, short_world, fake_tts, sandbox, monkeypatch):
    """Every script went quiet before the closing question (nothing happens at the end): the video ends where the
    talking stopped and the writer gets the shorter video - instead of the clip being dropped."""
    import script_judge
    cfg["video"]["min_short_sec"] = 5  # the smoke footage is 12 s: a tiny script is enough here
    cfg["quality_gate"].update(min_script_lines=3, min_script_words=10)
    monkeypatch.setattr(script_judge, "judge", lambda lines, ctx, numbered_ok=False, countdown=None, timing=None: (
        lambda st: {"ok": not st, "score": 8.0, "problems": st, "by": "test/judge", "static": st})(
        script_judge.dead_air(lines, timing["length"], timing["lead"], timing["spans"]) if timing else []))
    lengths = []

    def writer(clip, copy, fmt=None, lead=0.0, spans=None, feedback="", slots=None, n=1):
        length = common.probe(clip)["duration"]
        lengths.append(round(length, 1))
        hook = [{"t": lead + 0.1, "text": LINES[0]}, {"t": lead + 3.6, "text": "Look at that."}]
        if length > 10:  # quiet from ~4.7 s after the cold open to the closing question
            lines = hook + [{"t": lead + 11.0, "text": "Would you try it?"}]
        else:
            lines = hook + [{"t": lead + 4.9, "text": "Would you try it?"}]
        return {"lines": lines, "banner": "", "title": "", "story": "a ramp jump"}
    monkeypatch.setattr(gemini_writer, "write_final_script", writer)
    slot = {"type": "vertical_clip", "platforms": ["x"], "time": "19:40"}
    ok, rec = autopilot.run_video_slot(slot, dry_run=True, fmt=formats.format_by_id("stunts_crashes", cfg))
    assert ok, rec
    teaser = cfg["video"]["fx"]["teaser_sec"]
    keep = teaser + 3.6 + 3 / 2.7 + cfg["video"]["quiet_tail_room_sec"]  # "Look at that." ends, + room
    assert lengths[:3] == [round(12 + teaser, 1)] * 3 and lengths[3] == pytest.approx(keep, abs=0.15)
    assert rec["duration"] == pytest.approx(keep, abs=0.3)
    assert rec["source"]["segments"][-1][1] == pytest.approx(keep - teaser, abs=0.05), "the record shows the new end"


def test_a_talking_guide_is_muted_not_dropped(cfg, short_world, fake_tts, sandbox, monkeypatch):
    """Live 2026-11-25: 3 accepted weapon guides were dropped as "the characters talk for 45-50 of 50 s" - it was
    the guide's YouTuber. In a guide format (allow_narrated) their voice is muted and our narrator talks."""
    import transcribe
    words = [{"text": "word", "start": 0.2 + 0.4 * i, "end": 0.5 + 0.4 * i, "prob": 0.9} for i in range(40)]
    real_words = transcribe.words
    monkeypatch.setattr(transcribe, "words", lambda path, start=0.0, end=None: (
        (words, {"language": "en"}) if "section" in str(path) else real_words(path, start, end)))  # the source only
    monkeypatch.setattr(transcribe, "smart_cut", lambda path, a, b, found=None, offset=0.0, max_len=50.0: (a, b))
    seen = {}

    def writer(clip, copy, fmt=None, lead=0.0, spans=None, feedback="", slots=None, n=1):
        seen["spans"] = spans
        return {"lines": [{"t": round(lead + 0.1 + i * 3, 2), "text": ln} for i, ln in enumerate(LINES)],
                "banner": "", "title": "", "story": "a guide"}
    monkeypatch.setattr(gemini_writer, "write_final_script", writer)
    fmt = formats.format_by_id("hidden_weapons", cfg)
    fmt.pop("must_show")  # this test is about the muting; "ends on the weapon" has its own test
    ok, rec = autopilot.run_video_slot({"type": "vertical_clip", "platforms": ["x"], "time": "19:40"}, dry_run=True,
                                       fmt=fmt)
    assert ok, rec
    assert not seen["spans"], "the guide's voice is not 'dialogue' to talk around - it is muted"
    assert rec["copy"]["has_narrator"] is True


def test_a_muted_guide_stays_one_continuous_part_and_ends_on_the_item(cfg, short_world, fake_tts, sandbox,
                                                                       monkeypatch):
    """Owner 2026-11-25: a weapon guide became a 29 s montage of the route (one 5 s piece per line) and never showed
    the weapon. Guides stay one continuous part, and the item stays on screen for payoff_hold_sec."""
    fmt = formats.format_by_id("hidden_weapons", cfg)
    assert fmt["continuous"] and fmt["must_show"]

    def write_copy(section, title, credit, provider, kind="gameplay", hints=None, tour=False, fmt=None):
        d = {"banner": "FREE WEAPON", "content_type": "explainer", "hype_score": 8, "is_gta6_footage": True,
             "has_narrator": True, "shows_required": True, "required_t": 8.5, "fits_format": True,
             "required_evidence": "the rifle is picked up", "clip_start": 0, "clip_end": 9,
             "scene_summary": "a rifle in a shack", "youtube_title": "GTA 6 hidden rifle", "caption": "Got it?",
             "commentary": [{"t": i * 2.5, "line": ln} for i, ln in enumerate(LINES)], "hashtags": ["#GTA6"]}
        out = gemini_writer._clean_copy(d, 12)
        out.update(verified_by_video=True, ai_source="gemini-video")
        return out
    monkeypatch.setattr(gemini_writer, "write_clip_copy", write_copy)
    cfg["video"]["min_clip_sec"] = 5
    ok, rec = autopilot.run_video_slot({"type": "vertical_clip", "platforms": ["x"], "time": "19:40"}, dry_run=True,
                                       fmt=fmt)
    assert ok, rec
    story = rec["source"]["segments"][-1]
    assert len(rec["source"]["segments"]) <= 2, "cold open + one continuous part - not a montage of pieces"
    assert story[1] >= min(12.0, 8.5 + cfg["video"]["payoff_hold_sec"]) - 0.3, "the rifle stays on screen"
