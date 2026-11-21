"""Backup AIs take over the image checks when Gemini is out; the writer learns from our views."""
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

import ai_fallback
import analytics
import authentic
import gemini_writer
import qa
import source_fetcher as sf
import vision

JPG = b"\xff\xd8" + b"0" * 3000


@pytest.fixture
def gemini_down(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(gemini_writer, "_generate_json", lambda *a, **k: None)
    monkeypatch.setattr(ai_fallback, "available", lambda: True)


def test_backup_answers_when_gemini_is_out_and_numbers_map_back(monkeypatch, gemini_down):
    calls = []

    def chat(prompt, frames=None, temperature=0.9):
        calls.append((prompt, len(frames)))
        n = prompt.count("= clip ")
        return {"clips": [{"clip": i, "game": "GTA VI", "confidence": 0.9} for i in range(1, n + 1)]}
    monkeypatch.setattr(ai_fallback, "chat_json", chat)
    groups = [(f"Clip {i}", [JPG, JPG, JPG, JPG]) for i in range(1, 6)]   # 5 clips x 4 frames = too many for one call
    data = vision.ask(groups, "Which game?", key="clips", num="clip")
    assert data["_backup"] and sorted(v["clip"] for v in data["clips"]) == [1, 2, 3, 4, 5]
    assert all(n <= 8 for _, n in calls), "each backup call fits the providers' image limit"
    assert "You cannot see a video" in calls[0][0]


def test_authenticity_check_survives_gemini_out(monkeypatch, gemini_down, tmp_path):
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=6",
                    "-c:v", "libx264", "-preset", "ultrafast", str(clip)], check=True)
    monkeypatch.setattr(ai_fallback, "chat_json", lambda prompt, frames=None, temperature=0.9:
                        {"clips": [{"clip": 1, "game": "GTA VI", "confidence": 0.92, "evidence": "Vice City"}]})
    out = authentic.check([clip])
    assert out and out[0]["ok"] is True


def test_qa_editor_falls_back_to_frames(monkeypatch, gemini_down, tmp_path):
    clip = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=30:d=6",
                    "-c:v", "libx264", "-preset", "ultrafast", str(clip)], check=True)
    monkeypatch.setattr(gemini_writer, "_upload_video", lambda p: None)
    seen = {}

    def chat(prompt, frames=None, temperature=0.9):
        seen["n"], seen["prompt"] = len(frames), prompt
        return {"publish": True, "score": 8, "issues": [], "summary": "fine"}
    monkeypatch.setattr(ai_fallback, "chat_json", chat)
    r = qa.ai_review(clip, "narration [1.0s] hello")
    assert r["publish"] and r["summary"].startswith("(backup AI, frames only)")
    assert seen["n"] == 8 and "image 1 =" in seen["prompt"] and "[1.0s] hello" in seen["prompt"]


def test_view_examples_teach_hooks_that_worked(monkeypatch):
    now = datetime.now(timezone.utc)
    items = []
    for i, views in enumerate([12000, 900, 40, 15, 7000]):
        items.append({"created_at": (now - timedelta(days=2)).isoformat(), "format_id": "top5_funny",
                      "stats": {"views": views, "likes": 1, "comments": 0, "age_h": 48},
                      "copy": {"banner": f"BANNER {i}", "youtube_title": f"Title {i}",
                               "commentary": [{"t": 0, "text": f"hook line {i}"}]}})
    monkeypatch.setattr(sf, "load_manifest", lambda: {"items": items, "sources": {}})
    text = analytics.view_examples(n=2)
    assert "MOST VIEWS" in text and '12000 views | banner "BANNER 0" | opening line "hook line 0"' in text
    assert "FEWEST" in text and "BANNER 3" in text
    monkeypatch.setattr(sf, "load_manifest", lambda: {"items": items[:2], "sources": {}})
    assert analytics.view_examples() == "", "not enough data yet"
