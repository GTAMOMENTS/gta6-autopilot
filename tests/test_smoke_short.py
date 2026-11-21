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


def test_short_repair_fixes_only_the_wrong_line(cfg, short_world, fake_tts, sandbox, monkeypatch):
    """Final QA flags one line -> only that line is rewritten + re-voiced, the video re-rendered once, then it passes
    (instead of throwing the clip away)."""
    import lessons
    calls = {"n": 0, "ground": []}

    def line_check(path, lines, kind):
        calls["n"] += 1
        if calls["n"] == 2:   # 1 = pre-check (ok), 2 = final QA on the rendered video (one bad line), 3 = after repair
            return [{"severity": "high", "type": "line_not_shown", "at": lines[1]["start"], "index": 1,
                     "reason": "frames show a parked car", "detail": "the picture doesn't show it"}]
        return []
    monkeypatch.setattr(qa, "line_check", line_check)

    def ground(clip, items, what, extra=""):
        calls["ground"].append([it["text"] for it in items])
        if "REJECTED" not in extra:  # first pass (before voice): keep the lines
            return [it["text"] for it in items]
        return ["Look at that car flying straight off the ramp." if "Five stars" in it["text"] else it["text"]
                for it in items]
    monkeypatch.setattr(gemini_writer, "ground_lines", ground)
    fmt = formats.format_by_id("stunts_crashes", cfg)
    slot = {"type": "vertical_clip", "platforms": ["x"], "time": "19:40"}
    ok, rec = autopilot.run_video_slot(slot, dry_run=True, fmt=fmt)
    assert ok, rec
    assert calls["n"] == 3, "pre-check, final QA, QA after the repair"
    assert calls["ground"][-1] == [LINES[1]], "only the rejected line is rewritten"
    assert any("car flying straight off the ramp" in ln["text"] for ln in rec["copy"]["commentary"])
    stages = [a["stage"] for a in lessons._load()["attempts"]]
    assert stages[-2:] == ["final", "repair"] and lessons._load()["attempts"][-1]["pass"] is True


def test_precheck_last_try_saves_the_clip(cfg, short_world, fake_tts, sandbox, monkeypatch):
    """A line that fails the pre-check twice gets a plain 'what is on screen' rewrite before the clip is thrown away
    (a real run lost its only good clip over one weapon-wheel line)."""
    calls = {"n": 0, "extra": []}

    def line_check(path, lines, kind):
        calls["n"] += 1
        if calls["n"] <= 2 and any("Five stars" in ln["text"] or "weapon wheel" in ln["text"] for ln in lines):
            k = next(i for i, ln in enumerate(lines) if "Five stars" in ln["text"] or "weapon wheel" in ln["text"])
            return [{"severity": "high", "type": "line_not_shown", "at": lines[k]["start"], "index": k,
                     "reason": "no weapon wheel in the frames", "detail": "not shown"}]
        return []
    monkeypatch.setattr(qa, "line_check", line_check)

    def ground(clip, items, what, extra=""):
        calls["extra"].append(extra)
        if "LAST TRY" in extra:
            return ["A car flies off the ramp with cops right behind." for _ in items]
        if "REJECTED" in extra:
            return ["He opens the weapon wheel mid jump." for _ in items]
        return [it["text"] for it in items]
    monkeypatch.setattr(gemini_writer, "ground_lines", ground)
    slot = {"type": "vertical_clip", "platforms": ["x"], "time": "19:40"}
    ok, rec = autopilot.run_video_slot(slot, dry_run=True, fmt=formats.format_by_id("stunts_crashes", cfg))
    assert ok, rec
    assert sum("LAST TRY" in e for e in calls["extra"]) == 1
    assert any("cops right behind" in ln["text"] for ln in rec["copy"]["commentary"])
