"""qa.line_check: does the picture show what each narration line says (one focused AI question on frames)."""
import subprocess
from pathlib import Path

import pytest

import gemini_writer
import qa


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=30:d=8",
                    "-f", "lavfi", "-i", "sine=d=8", "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", str(p)], check=True)
    return p


LINES = [{"text": "This is the easiest money in GTA 6.", "start": 0.3, "dur": 2.0},
         {"text": "Grab the cash from the back alley.", "start": 2.8, "dur": 2.0},
         {"text": "How much did you make?", "start": 5.5, "dur": 1.8}]


def _ai(monkeypatch, answer):
    sent = {}

    def gen(contents, temperature=0.9, prefer_pro=False):
        sent["contents"] = contents
        return answer
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(gemini_writer, "_generate_json", gen)
    return sent


def test_line_not_shown_is_high(monkeypatch, video):
    sent = _ai(monkeypatch, {"lines": [{"line": 1, "verdict": "general"},
                                       {"line": 2, "verdict": "no", "reason": "a man rides a bike on a road"},
                                       {"line": 3, "verdict": "general"}]})
    issues = qa.line_check(video, LINES, "short")
    assert [i["type"] for i in issues] == ["line_not_shown"]
    assert issues[0]["severity"] == "high" and issues[0]["at"] == 2.8 and "back alley" in issues[0]["detail"]
    frames = [c for c in sent["contents"] if not isinstance(c, str)]
    assert len(frames) == 6, "2 frames per line for Shorts"
    assert "Line 2 frames:" in sent["contents"] and '2. "Grab the cash from the back alley."' in sent["contents"][-1]


def test_all_shown_passes(monkeypatch, video):
    _ai(monkeypatch, {"lines": [{"line": i, "verdict": v} for i, v in [(1, "general"), (2, "yes"), (3, "general")]]})
    assert qa.line_check(video, LINES, "short") == []


def test_long_uses_one_frame_per_line(monkeypatch, video):
    sent = _ai(monkeypatch, {"lines": []})
    assert qa.line_check(video, LINES, "long") == []
    assert len([c for c in sent["contents"] if not isinstance(c, str)]) == 3


def test_no_ai_answer_blocks_in_strict_mode(monkeypatch, video):
    _ai(monkeypatch, None)
    issues = qa.line_check(video, LINES, "short")
    assert [i["type"] for i in issues] == ["line_check_unavailable"] and issues[0]["severity"] == "high"


def test_no_key_or_no_lines_skips(video):
    assert qa.line_check(video, LINES, "short") == []   # conftest removes GEMINI_API_KEY
    assert qa.line_check(video, [], "short") == []
