"""gemini_writer.ground_lines: each narration line rewritten to match the frames it plays over."""
import subprocess

import pytest

import gemini_writer


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=12",
                    "-c:v", "libx264", "-preset", "ultrafast", str(p)], check=True)
    return p


ITEMS = [{"text": "You will not believe the chaos on the highway today.", "start": 0.2, "dur": 3.0},
         {"text": "Number four: running from cops on a cruise ship deck.", "start": 3.5, "dur": 3.0},
         {"text": "Would you have tried that jump?", "start": 8.0, "dur": 2.0}]


def _ai(monkeypatch, answer):
    sent = {}

    def gen(contents, temperature=0.9, prefer_pro=False):
        sent["contents"] = contents
        return answer
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(gemini_writer, "_generate_json", gen)
    monkeypatch.setattr(gemini_writer, "_unavailable_until", 0.0)
    return sent


def test_lines_rewritten_to_the_footage(monkeypatch, clip):
    sent = _ai(monkeypatch, {"lines": [
        {"line": 1, "text": "This street fight in GTA 6 got out of hand fast."},
        {"line": 2, "text": "Number four: this pickup flies straight off the cliff."},
        {"line": 3, "text": "Would you have driven off that cliff?"}]})
    out = gemini_writer.ground_lines(clip, ITEMS, "Top 5 funniest moments")
    assert out == ["This street fight in GTA 6 got out of hand fast.",
                   "Number four: this pickup flies straight off the cliff.", "Would you have driven off that cliff?"]
    assert len([c for c in sent["contents"] if not isinstance(c, str)]) == 9, "3 frames per line"
    assert "Line 2 frames:" in sent["contents"]


def test_guards_keep_number_question_and_no_invented_numbers(monkeypatch, clip):
    _ai(monkeypatch, {"lines": [
        {"line": 1, "text": "He crashed 500 cars in GTA 6 today."},             # invented number -> old line
        {"line": 2, "text": "This pickup flies straight off the cliff edge."},    # lost "Number four" -> re-added
        {"line": 3, "text": "That jump was wild."}]})                             # question lost -> old line
    out = gemini_writer.ground_lines(clip, ITEMS, "Top 5")
    assert out[0] == ITEMS[0]["text"]
    assert out[1].lower().startswith("number four") and "pickup" in out[1]
    assert out[2] == ITEMS[2]["text"]


def test_no_ai_keeps_the_lines(monkeypatch, clip):
    _ai(monkeypatch, None)
    assert gemini_writer.ground_lines(clip, ITEMS, "x") is None


def test_no_key_skips(clip):
    assert gemini_writer.ground_lines(clip, ITEMS, "x") is None
