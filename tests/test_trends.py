"""Trend radar (trends.py): stream health, topic picking + cooldown, AI answer cleaning, cached refresh."""
from __future__ import annotations

import time

import pytest

import common
import trends

NOW = time.time()


def save(state):
    common.save_json(trends.TRENDS_FILE, state)


# --------------------------------------------------------------------------- stream health

@pytest.mark.parametrize("viewers,weak", [
    ([], False),                                                    # no data = not weak
    ([(NOW - 3600, 300000), (NOW, 250000)], False),                 # big and steady
    ([(NOW - 3600, 300000), (NOW, 15000)], True),                   # under stream_min_viewers (20k)
    ([(NOW - 3600, 300000), (NOW, 90000)], True),                   # under 35% of the 7-day peak
    ([(NOW - 3600, 100000), (NOW, 40000)], False),                  # 40% of peak, above 20k
    ([(NOW - 8 * 86400, 900000), (NOW, 60000)], False),             # old peak (8 days) is ignored
])
def test_stream_health(viewers, weak):
    save({"viewers": [list(v) for v in viewers]})
    h = trends.stream_health()
    assert h["weak"] is weak
    if viewers:
        assert h["viewers"] == viewers[-1][1]
    else:
        assert h == {"weak": False, "viewers": None, "peak": None}


def test_stream_health_thresholds_from_config(cfg):
    cfg["trends"]["stream_min_viewers"] = 100
    cfg["trends"]["stream_weak_ratio"] = 0.05
    assert trends.stream_health({"viewers": [[NOW - 60, 300000], [NOW, 16000]]})["weak"] is False


# --------------------------------------------------------------------------- pick / cooldown

TOPICS = [
    {"topic": "Hidden Beaches", "money": False, "demand": 9, "queries": ["GTA 6 beaches"]},
    {"topic": "Fast Money", "money": True, "demand": 8, "queries": ["GTA 6 money"]},
    {"topic": "Cheap Cars", "money": False, "demand": 4, "queries": ["GTA 6 cars"]},
]


def test_pick_filters_money_and_demand():
    save({"topics": TOPICS, "used": {}})
    assert trends.pick()["topic"] == "Hidden Beaches"
    assert trends.pick(money=True)["topic"] == "Fast Money"
    assert trends.pick(money=False)["topic"] == "Hidden Beaches"
    assert trends.pick(money=True, min_demand=9) is None
    assert trends.pick(money=False, min_demand=5)["topic"] == "Hidden Beaches"


def test_used_topic_cools_down_then_returns(cfg):
    save({"topics": TOPICS, "used": {}})
    trends.mark_used("Hidden Beaches")
    assert trends.pick(money=False)["topic"] == "Cheap Cars"
    days = cfg["trends"]["topic_cooldown_days"]
    st = common.load_json(trends.TRENDS_FILE, {})
    st["used"]["hidden beaches"] = time.time() - (days + 0.1) * 86400
    save(st)
    assert trends.pick(money=False)["topic"] == "Hidden Beaches"


def test_mark_used_ignores_empty_and_lowercases():
    save({"topics": TOPICS, "used": {}})
    trends.mark_used("")
    trends.mark_used("Fast Money")
    assert list(common.load_json(trends.TRENDS_FILE, {})["used"]) == ["fast money"]


def test_context_line():
    save({"summary": "Players want money.", "topics": TOPICS})
    line = trends.context_line()
    assert line.startswith("Players want money.") and "Hot topics: Hidden Beaches, Fast Money, Cheap Cars." in line
    save({})
    assert trends.context_line() == ""


# --------------------------------------------------------------------------- AI answer -> topics

def test_analyse_cleans_ai_answer(monkeypatch):
    import gemini_writer
    answer = {"summary": "s" * 400, "topics": [
        {"topic": "Low Demand", "demand": 2, "queries": ["GTA 6 low"]},
        {"topic": "No Queries", "demand": "15", "queries": ["how to drift", 5]},        # demand clamped to 10
        {"topic": "Bad Demand", "demand": "lots", "money": 1, "queries": ["gta vi cash", "GTA 6 cash"]},
        {"angle": "no topic - dropped"},
        "junk",
    ]}
    monkeypatch.setattr(gemini_writer, "_generate_json", lambda prompt, temperature=0.9, prefer_pro=False: answer)
    out = trends._analyse({"suggest": ["gta 6 money"], "videos": [{"title": "GTA 6 cash", "views": 10}], "reddit": []})
    names = [t["topic"] for t in out["topics"]]
    assert names == ["No Queries", "Bad Demand", "Low Demand"], "sorted by demand, junk dropped"
    nq = out["topics"][0]
    assert nq["demand"] == 10
    assert nq["queries"] == ["GTA 6 No Queries", "GTA 6 No Queries no commentary"], "non-GTA queries replaced"
    bd = out["topics"][1]
    assert bd["demand"] == 5 and bd["money"] is True and bd["queries"] == ["gta vi cash", "GTA 6 cash"]
    assert len(out["summary"]) == 300


def test_analyse_no_ai_answer(monkeypatch):
    import ai_fallback
    import gemini_writer
    monkeypatch.setattr(gemini_writer, "_generate_json", lambda *a, **k: None)
    monkeypatch.setattr(ai_fallback, "available", lambda: False)
    assert trends._analyse({"suggest": [], "videos": [], "reddit": []}) is None


# --------------------------------------------------------------------------- refresh (cache + failures)

def fake_sources(monkeypatch, viewers=123456, topics=None):
    calls = {"analyse": 0}
    monkeypatch.setattr(trends, "twitch_viewers", lambda: viewers)
    monkeypatch.setattr(trends, "autocomplete", lambda: ["gta 6 money fast"])
    monkeypatch.setattr(trends, "hot_videos", lambda: [{"title": "GTA 6 money", "views": 5}])
    monkeypatch.setattr(trends, "reddit_hot", lambda: [])

    def analyse(data):
        calls["analyse"] += 1
        return {"topics": topics if topics is not None else TOPICS, "summary": "today"}
    monkeypatch.setattr(trends, "_analyse", analyse)
    return calls


def test_refresh_saves_topics_and_viewers(monkeypatch):
    calls = fake_sources(monkeypatch)
    st = trends.refresh()
    assert calls["analyse"] == 1
    assert [t["topic"] for t in st["topics"]] == [t["topic"] for t in TOPICS]
    assert st["viewers"][-1][1] == 123456 and st["top_searches"] == ["gta 6 money fast"]
    assert common.load_json(trends.TRENDS_FILE, {})["summary"] == "today"
    trends.refresh()  # cached for refresh_hours
    assert calls["analyse"] == 1
    trends.refresh(force=True)
    assert calls["analyse"] == 2


def test_refresh_keeps_old_trends_when_ai_fails(monkeypatch):
    save({"topics": TOPICS, "used": {}, "viewers": [], "at": 0})
    fake_sources(monkeypatch, viewers=None, topics=[])
    st = trends.refresh()
    assert [t["topic"] for t in st["topics"]] == [t["topic"] for t in TOPICS]


def test_refresh_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("twitch down")
    monkeypatch.setattr(trends, "twitch_viewers", boom)
    st = trends.refresh(force=True)
    assert st["topics"] == []


def test_refresh_offline_sources_fail_softly():
    """The real source functions with no network (blocked) must return empty data, not crash."""
    assert trends.autocomplete() == []
    assert trends.reddit_hot() == []
    assert trends.twitch_viewers() is None


def test_trends_disabled(cfg, monkeypatch):
    cfg["trends"]["enabled"] = False
    calls = fake_sources(monkeypatch)
    trends.refresh(force=True)
    assert calls["analyse"] == 0
