"""Video A rotation, Video B money topics, the trending format and the stream-health switch (formats.py)."""
from __future__ import annotations

import time

import pytest

import common
import formats
import trends


@pytest.fixture
def slots(cfg):
    return {s["type"]: s for s in cfg["schedule"]}


@pytest.fixture(autouse=True)
def no_learning(monkeypatch):
    import analytics
    monkeypatch.setattr(analytics, "format_scores", lambda: {})


def save_trends(topics=(), viewers=(), used=None):
    common.save_json(trends.TRENDS_FILE, {"at": time.time(), "topics": list(topics), "used": used or {},
                                          "viewers": [list(v) for v in viewers]})


def set_next(fid, cfg):
    ids = [x["id"] for x in formats._rotation(cfg)]
    st = common.load_json(formats.ROTATION_FILE, {"next": 0, "history": []})
    st["next"] = ids.index(fid)
    common.save_json(formats.ROTATION_FILE, st)


STRONG = [(time.time() - 86400, 300000), (time.time(), 250000)]
HOT_TOPICS = [
    {"topic": "Best Cars To Steal", "angle": "rare cars", "queries": ["GTA 6 rare cars"], "money": False, "demand": 9},
    {"topic": "Fastest Money After Launch", "angle": "the method everyone uses", "queries": ["GTA 6 fastest money"],
     "money": True, "demand": 8},
]


# --------------------------------------------------------------------------- Video B: GTA 6 Money Tip

# money content was dropped from the schedule (2026-11-21) but the format must still work when a slot asks for it
MONEY_SLOT = {"time": "01:40", "type": "money_clip", "format": "money_tips", "platforms": ["x", "tiktok", "youtube"]}


def test_video_b_is_normal_rotation_now(cfg, slots):
    assert "format" not in slots["video_b"]
    ids = [x["id"] for x in formats._rotation(cfg)]
    assert formats.current_format(cfg, slots["video_b"])["id"] in ids, "Video B takes the next format of the rotation"


def test_money_slot_always_money_tips_and_topics_rotate(cfg, slots):
    f1 = formats.current_format(cfg, MONEY_SLOT)
    assert f1["id"] == "money_tips" and f1["topic"] == "Fastest Early-Game Money"
    assert f1["topic_from_trends"] is False
    assert f1["name"].endswith(": Fastest Early-Game Money")
    formats.mark_topic(f1)
    f2 = formats.current_format(cfg, MONEY_SLOT)
    assert f2["topic"] == "Selling Cars For Cash"


def test_money_topics_cycle_through_all_then_repeat(cfg, slots):
    topics = [t["topic"] for t in formats.format_by_id("money_tips", cfg)["topics"]]
    seen = []
    for _ in range(len(topics) + 1):
        f = formats.current_format(cfg, MONEY_SLOT)
        seen.append(f["topic"])
        formats.mark_topic(f)
    assert seen[:len(topics)] == topics, "every topic once, in config order"
    assert seen[-1] == topics[0], "then the oldest one comes back"


def test_hot_money_trend_jumps_the_queue_once(cfg, slots):
    save_trends(HOT_TOPICS, STRONG)
    f3 = formats.current_format(cfg, MONEY_SLOT)
    assert f3["topic"] == "Fastest Money After Launch" and f3["topic_from_trends"] is True
    assert f3["queries"][0] == "GTA 6 fastest money", "the trend's own searches come first"
    assert len(f3["queries"]) <= 8
    formats.mark_topic(f3)
    f4 = formats.current_format(cfg, MONEY_SLOT)
    assert f4["topic"] == "Fastest Early-Game Money", "a used trend must not repeat"
    assert f4["topic_from_trends"] is False


def test_weak_money_trend_does_not_jump_the_queue(cfg, slots):
    save_trends([{**HOT_TOPICS[1], "demand": 6}], STRONG)  # money_tips.trend_min_demand = 7
    f = formats.current_format(cfg, MONEY_SLOT)
    assert f["topic"] == "Fastest Early-Game Money"


def test_format_env_overrides_everything(cfg, slots, monkeypatch):
    monkeypatch.setenv("FORMAT", "car_chases")
    assert formats.current_format(cfg, MONEY_SLOT)["id"] == "car_chases"
    monkeypatch.setenv("FORMAT", "no_such_format")
    assert formats.current_format(cfg, MONEY_SLOT)["id"] == "money_tips"


# --------------------------------------------------------------------------- Video A rotation

def test_rotation_is_the_informative_channel(cfg):
    """Owner 2026-11-25: "quality is bad, video is nonsense, just bro" - the channel makes informative Shorts
    (money, police, easter eggs, tips, cars, robberies, what happens if, weapons, fastest money, things you didn't
    know, map secrets). Crash / funny / streamer / generic formats are out of the rotation but still exist."""
    ids = [x["id"] for x in formats._rotation(cfg)]
    assert ids == ["money_tips", "police_wanted", "easter_eggs", "beginner_tips", "best_cars", "robberies",
                   "what_happens_if", "hidden_weapons", "fastest_money", "didnt_know", "map_secrets"]
    for old in ("top5_funny", "car_chases", "stunts_crashes", "trending", "top5_things_to_do", "secrets",
                "stream_reactions"):
        assert old not in ids and formats.format_by_id(old, cfg), f"{old}: out of the rotation, still callable"
    assert formats.format_by_id("MONEY_TIPS", cfg)["id"] == "money_tips", "lookup ignores case"
    for f in formats._rotation(cfg):
        assert f["kind"] == "gameplay" and f["streams"] is False and f["allow_narrated"] and f["continuous"]
        assert "x_accounts" not in f["providers"] and "twitch" not in f["providers"]
        blocked = f["blocked_title_terms"]
        for word in ("ragdoll", "trainer", "cheat", "compilation", "funny moments", "glitch"):
            assert word in blocked, f"{f['id']} must block '{word}'"
        title = "gta 6 - grand theft auto vi gameplay (gta vi)"
        assert not [t for t in blocked if t in title], "a block word must never match a real GTA 6 title"


def test_every_video_slot_posts_everywhere_and_long_publishes_itself(cfg, slots):
    for kind in ("vertical_clip", "video_b"):
        assert set(slots[kind]["platforms"]) == {"instagram", "x", "tiktok", "youtube"}
        assert not slots[kind].get("platform_from"), f"{kind} posts on every platform from today"
    assert slots["long_video"]["every_days"] == 3 and cfg["long_video"]["require_approval"] is False


def test_rotation_advances_and_wraps(cfg, slots):
    ids = [x["id"] for x in formats._rotation(cfg)]
    save_trends(HOT_TOPICS, STRONG)
    made = []
    for _ in range(len(ids) + 1):
        f = formats.current_format(cfg, slots["vertical_clip"])
        made.append(f["id"])
        formats.advance_format(f, ok=True)
    assert made[:len(ids)] == ids
    assert made[-1] == ids[0]
    hist = common.load_json(formats.ROTATION_FILE, {})["history"]
    assert [h["format"] for h in hist] == made and all(h["ok"] for h in hist)


def test_trending_fills_from_trend_radar(cfg):
    save_trends(HOT_TOPICS, STRONG)
    t = formats.specialize(formats.format_by_id("trending", cfg))
    assert t["topic"] == "Best Cars To Steal", "trending ignores money topics"
    assert t["name"] == "Trending: Best Cars To Steal"
    assert t["queries"][0] == "GTA 6 rare cars"
    assert "TODAY'S TOPIC" in t["brief"] and len(t["banner_example"]) <= 40
    assert "cars" in t["title_terms"] and "steal" in t["title_terms"]


def test_specialize_returns_copy(cfg):
    original = formats.format_by_id("money_tips", cfg)
    made = formats.specialize(original)
    assert made is not original and "topic" not in original


def put_streams_back(cfg):
    """The stream-health switch still works if the owner brings streamer clips back into the rotation."""
    formats.format_by_id("stream_reactions", cfg)["slot_only"] = False


def test_stream_reactions_stays_when_streams_strong(cfg, slots):
    put_streams_back(cfg)
    save_trends(HOT_TOPICS, STRONG)
    set_next("stream_reactions", cfg)
    assert formats.current_format(cfg, slots["vertical_clip"])["id"] == "stream_reactions"


def test_stream_reactions_becomes_trending_when_streams_quiet(cfg, slots):
    put_streams_back(cfg)
    save_trends(HOT_TOPICS, STRONG + [(time.time(), 15000)])
    assert trends.stream_health()["weak"] is True
    set_next("stream_reactions", cfg)
    a = formats.current_format(cfg, slots["vertical_clip"])
    assert a["id"] == "trending" and a["name"] == "Trending: Best Cars To Steal"


def test_no_trend_data_skips_trending(cfg, slots):
    formats.format_by_id("trending", cfg)["slot_only"] = False  # out of the rotation since 2026-11-25; still works
    save_trends([], [])
    set_next("trending", cfg)
    a = formats.current_format(cfg, slots["vertical_clip"])
    ids = [x["id"] for x in formats._rotation(cfg)]
    assert a is not None and a["id"] != "trending"
    assert a["id"] == ids[(ids.index("trending") + 1) % len(ids)], "the next format in the rotation"


# --------------------------------------------------------------------------- series + prompt notes

def test_series_numbers_count_up(cfg):
    f = formats.format_by_id("money_tips", cfg)
    assert formats.series_label(f) == "GTA 6 Money Tip #1"
    formats.bump_series(f)
    formats.bump_series(f)
    assert formats.series_label(f) == "GTA 6 Money Tip #3"
    assert formats.series_label(formats.format_by_id("trending", cfg)) == "", "trending has no series"


def test_prompt_note_countdown_and_required_proof(cfg):
    top5 = formats.format_by_id("top5_funny", cfg)
    note = formats.prompt_note(top5, tour=True)
    assert "exactly 5 numbered items" in note and '"Number five...' in note
    single = formats.prompt_note(top5, tour=False)
    assert "don't number anything" in single
    money = formats.specialize(formats.format_by_id("money_tips", cfg))
    assert "REQUIRED FOR THIS FORMAT" in formats.prompt_note(money, tour=False)
    assert formats.prompt_note(None, tour=False) == ""


def test_formats_disabled_means_no_rotation(cfg, slots):
    cfg["formats"]["enabled"] = False
    assert formats._rotation(cfg) == []
    assert formats.current_format(cfg, slots["vertical_clip"]) is None


def test_countdown_missing_numbers():
    # QA blocked Top 5s that went 5 -> 3 (a number dropped) - a countdown must announce every number
    assert formats.countdown_missing(["Top 5. Number five: a", "Number four: b", "Number three: c",
                                      "Number two: d"], 5) == [1]
    assert formats.countdown_missing(["Top five. Number 5 a", "number four", "#3 lol", "Number two b",
                                      "And number one, the best"], 5) == []
    assert formats.countdown_missing(["Here are five fails", "Number three: c"], 5) == [5, 4, 2, 1]


def test_every_rotation_format_must_show_what_it_talks_about(cfg):
    """Owner 2026-11-25: "you can't just talk about something in a video without it being shown"."""
    for f in formats._rotation(cfg):
        assert f.get("must_show") and f.get("must_show_short"), f"{f['id']} has no must_show"
