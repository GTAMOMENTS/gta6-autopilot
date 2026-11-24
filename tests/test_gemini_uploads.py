"""One Gemini upload per file per run: the editor, every script attempt and a repair watch the same upload (a 5-min
tour reel took ~50 s to encode + upload, three times for one posted Short). A changed file is uploaded again."""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

import gemini_writer
import stream_clips


@pytest.fixture
def uploads(monkeypatch):
    """Fake Files API: records every proxy upload and every delete; the run's upload cache starts empty."""
    seen = SimpleNamespace(uploaded=[], deleted=[])

    def upload(clip):
        seen.uploaded.append(clip.name)
        return SimpleNamespace(name=f"files/{len(seen.uploaded)}")
    monkeypatch.setattr(gemini_writer, "_upload_proxy", upload)
    monkeypatch.setattr(gemini_writer, "_client",
                        lambda: SimpleNamespace(files=SimpleNamespace(delete=lambda name: seen.deleted.append(name))))
    monkeypatch.setattr(gemini_writer, "_cleanup_registered", True)  # no atexit hook from tests
    gemini_writer._uploads.clear()
    yield seen
    gemini_writer._uploads.clear()


def test_same_file_is_uploaded_once(uploads, tmp_path):
    clip = tmp_path / "reel.mp4"
    clip.write_bytes(b"x" * 100)
    first = gemini_writer._upload_video(clip)
    assert gemini_writer._upload_video(clip) is first and uploads.uploaded == ["reel.mp4"]


def test_changed_file_is_uploaded_again(uploads, tmp_path):
    clip = tmp_path / "final.mp4"
    clip.write_bytes(b"x" * 100)
    first = gemini_writer._upload_video(clip)
    clip.write_bytes(b"y" * 120)  # a repair re-rendered the video to the same path
    os.utime(clip, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    second = gemini_writer._upload_video(clip)
    assert second is not first and uploads.uploaded == ["final.mp4", "final.mp4"]


def test_release_deletes_every_version_of_a_file_or_everything(uploads, tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    gemini_writer._upload_video(a)
    a.write_bytes(b"11")
    os.utime(a, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
    gemini_writer._upload_video(a)
    gemini_writer._upload_video(b)
    gemini_writer.release_uploads(a)
    assert sorted(uploads.deleted) == ["files/1", "files/2"]
    gemini_writer.release_uploads()
    assert sorted(uploads.deleted) == ["files/1", "files/2", "files/3"] and not gemini_writer._uploads


def test_failed_upload_is_not_remembered(uploads, monkeypatch, tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(gemini_writer, "_upload_proxy", lambda c: None)  # e.g. still PROCESSING after 2 min
    assert gemini_writer._upload_video(clip) is None and not gemini_writer._uploads


def test_script_attempts_watch_one_upload(uploads, monkeypatch, make_clip):
    clip = make_clip("cut.mp4", dur=12)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    lines = [{"t": 0.5 + 3 * i, "text": f"Line number {i} about the chase on screen."} for i in range(3)]
    monkeypatch.setattr(gemini_writer, "_generate_json", lambda *a, **k: {"story": "a chase", "lines": lines})
    for feedback in ("", "YOUR LAST SCRIPT WAS REJECTED", "THESE LINES DON'T MATCH"):
        assert gemini_writer.write_final_script(clip, {"banner": "CHASE"}, feedback=feedback)
    assert uploads.uploaded == ["cut.mp4"] and uploads.deleted == [], "3 attempts, 1 upload, kept for the repair"


def test_fallback_stream_clip_asks_gemini_after_an_earlier_outage(uploads, monkeypatch, make_clip):
    """Runs of 2026-11-23 / 11-24: Gemini was overloaded during the narrated video, which switched Gemini off for
    the rest of the run - the fallback stream clip was uploaded and dropped without one question to Gemini."""
    clip = make_clip("stream.mp4", dur=6)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(gemini_writer, "_unavailable_until", time.time() + 3600)
    asked = []

    def gen(contents, temperature=0.9, prefer_pro=False):
        asked.append(gemini_writer._unavailable_until)
        return {"is_gta6": True, "hype_score": 8}
    monkeypatch.setattr(gemini_writer, "_generate_json", gen)
    out = stream_clips.ai_review(clip, {"streamer": "Somebody", "platform": "twitch", "title": "GTA 6 jump"})
    assert out == {"is_gta6": True, "hype_score": 8} and asked == [0.0]
    assert uploads.deleted == ["files/1"], "a reviewed stream clip's upload is deleted right away"
