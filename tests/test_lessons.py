"""lessons.py: the system's memory of its own QA results."""
import time

import formats
import lessons

BAD = [{"severity": "high", "type": "line_not_shown",
        "detail": 'the picture doesn\'t show what the narrator says: "chaos on the highway" (frames show a street fight)'}]


def test_record_and_hints_reach_the_writer(cfg):
    lessons.record("short", "top5_funny", False, BAD, source={"provider": "youtube", "channel": "X"})
    lessons.record("short", "top5_funny", True, [], lines=["Number five: this ragdoll is unreal", "Number four: ..."])
    hints = lessons.prompt_hints("top5_funny")
    assert "RECENT MISTAKES" in hints and "chaos on the highway" in hints
    assert "Number five: this ragdoll is unreal" in hints, "good examples for the format"
    fmt = formats.specialize(formats.format_by_id("top5_funny", cfg))
    assert "RECENT MISTAKES" in formats.prompt_note(fmt, tour=True), "the script writer sees the lessons"


def test_no_history_no_hints():
    assert lessons.prompt_hints("car_chases") == ""
    assert lessons.stats_line() == "no QA history yet"


def test_failing_format_is_rested(cfg):
    for _ in range(6):
        lessons.record("short", "secrets", False, BAD)
    assert lessons.should_rest("secrets")
    assert not lessons.should_rest("car_chases")
    cfg["formats"]["enabled"] = True
    ids = [f["id"] for f in formats._rotation(cfg)]
    state = {"next": ids.index("secrets"), "history": []}
    formats.save_json(formats.ROTATION_FILE, state)
    import analytics
    analytics.format_scores = lambda: {}
    assert formats.current_format(cfg, {"type": "vertical_clip"})["id"] != "secrets"


def test_stats_line_counts_first_pass_and_repairs():
    lessons.record("short", "a", False, BAD)                       # first try failed...
    lessons.record("short", "a", True, [], stage="repair", repaired=True)  # ...repaired, passed
    lessons.record("short", "b", True, [])                         # passed first time
    line = lessons.stats_line()
    assert "first-pass rate 1/2 (50%)" in line and "2 passed (1 after repair)" in line


def test_precheck_failures_do_not_count_as_finished_videos():
    lessons.record("short", "a", False, BAD, stage="precheck")
    assert lessons.pass_rates().get("a") is None
    d = lessons._load()
    assert d["attempts"][-1]["stage"] == "precheck" and d["attempts"][-1]["at"] <= time.time()
