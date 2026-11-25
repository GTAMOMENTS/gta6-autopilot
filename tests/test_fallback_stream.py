"""A video slot that posted nothing posts the most viral streamer clip instead (owner: "if nothing got posted u
should push a short clip")."""
import sys

import pytest

import analytics
import autopilot
import formats
import music
import stream_clips
import trends


@pytest.fixture
def run_main(monkeypatch, cfg):
    # the owner switched the fallback off on 2026-11-25 (no streamer clips); the switch itself must still work
    cfg["fallback"]["stream_clip_on_fail"] = True
    monkeypatch.setattr(music, "ensure_tracks", lambda: 0)
    monkeypatch.setattr(trends, "refresh", lambda: None)
    monkeypatch.setattr(analytics, "refresh_stats", lambda: None)
    monkeypatch.setattr(analytics, "format_scores", lambda: {})
    seen = {"history": [], "streams": []}
    monkeypatch.setattr(autopilot, "record_history", lambda sid, slot, ok, d: seen["history"].append((ok, d)))
    monkeypatch.setattr(autopilot, "summarize", lambda *a, **k: None)

    def go(fmt_id="stunts_crashes", video=(False, {"error": "lines don't match the footage (pre-check)"}),
           stream=(True, {"id": "stream_x", "results": {"x": {"ok": True}}, "format": "Viral stream clip"})):
        monkeypatch.setattr(formats, "current_format", lambda c, slot: formats.format_by_id(fmt_id, c))
        monkeypatch.setattr(formats, "advance_format", lambda f, ok: seen.setdefault("advanced", (f["id"], ok)))
        monkeypatch.setattr(autopilot, "run_video_slot", lambda slot, dry, fmt=None: video)

        def streams(platforms, dry_run, deadline=None):
            seen["streams"].append(platforms)
            return stream
        monkeypatch.setattr(stream_clips, "run_for_slot", streams)
        monkeypatch.setattr(sys, "argv", ["autopilot.py", "--slot", "vertical_clip"])
        return autopilot.main()
    return go, seen


def test_the_fallback_is_off_in_config():
    import common
    assert common.load_config()["fallback"]["stream_clip_on_fail"] is False, "owner 2026-11-25: no streamer clips"


def test_failed_video_posts_a_stream_clip(run_main):
    go, seen = run_main
    assert go() == 0
    ok, detail = seen["history"][-1]
    assert ok and detail["format"] == "Viral stream clip" and "pre-check" in detail["fallback_for"]
    assert seen["advanced"] == ("stunts_crashes", False), "the rotation still knows the narrated format failed"


def test_no_fallback_when_the_video_posted(run_main):
    go, seen = run_main
    assert go(video=(True, {"id": "v1", "results": {"x": {"ok": True}}})) == 0
    assert seen["streams"] == []


def test_both_fail_reports_failure(run_main):
    go, seen = run_main
    assert go(stream=(False, {"error": "no viral stream clip passed today", "no_stream_clip": True})) == 1
    ok, detail = seen["history"][-1]
    assert not ok and detail["fallback"] == "no viral stream clip passed today"


def test_switch_off(run_main, cfg):
    cfg["fallback"]["stream_clip_on_fail"] = False
    go, seen = run_main
    assert go() == 1 and seen["streams"] == []
