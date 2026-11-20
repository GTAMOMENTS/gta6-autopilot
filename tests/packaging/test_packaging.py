"""Packaging & SEO: titles, captions, descriptions, hashtags, tags, pinned comments per platform + the critic's guards.
Offline: every AI answer is mocked, the trend radar reads a temp trends.json. Runs on its own
(python -m pytest tests/packaging) and inside the full suite."""
from __future__ import annotations

import copy as _copy
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import ai_fallback  # noqa: E402
import common  # noqa: E402
import critic  # noqa: E402
import gemini_writer  # noqa: E402
import long_video  # noqa: E402
import publisher  # noqa: E402
import trends  # noqa: E402

SEARCHES = ["gta 6 how to make money", "gta 6 money glitch", "gta 6 make money fast", "gta 6 police chase",
            "gta 6 release date pc", "gta 6 funny moments", "gta 6 best cars", "gta 6 cheats"]


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    """Own config copy, a trends.json with today's top searches, no AI unless a test mocks one."""
    monkeypatch.setattr(common, "_CONFIG_CACHE", _copy.deepcopy(json.loads(common.CONFIG_FILE.read_text("utf-8"))))
    tf = tmp_path / "trends.json"
    tf.write_text(json.dumps({"top_searches": SEARCHES, "topics": []}), encoding="utf-8")
    monkeypatch.setattr(trends, "TRENDS_FILE", tf)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(ai_fallback, "available", lambda: False)
    return common._CONFIG_CACHE


def mock_ai(monkeypatch, *answers):
    """ai_fallback.chat_json returns the answers in order; the prompts are recorded."""
    prompts, queue = [], list(answers)
    monkeypatch.setattr(ai_fallback, "available", lambda: True)

    def chat(prompt, frames=None, temperature=0.9):
        prompts.append(prompt)
        return {**queue.pop(0), "_ai_model": "mock/model"} if queue else None
    monkeypatch.setattr(ai_fallback, "chat_json", chat)
    return prompts


def hashtags(text: str) -> list[str]:
    return re.findall(r"(?<!\w)#[A-Za-z]\w*", text)  # "#3" (series / countdown number) is not a hashtag


# --------------------------------------------------------------------------- hashtags

def test_clean_hashtags_blocks_brands_cheats_and_dupes():
    got = publisher.clean_hashtags(["#GTA6", "#RockstarGames", "#PS5", "#GTA6PS5", "#PlayStation5", "#XboxSeriesX",
                                    "#Lamborghini", "#GTA6Ferrari", "#ad", "#GTA6MoneyGlitch", "#GTA6Mods", "#fyp",
                                    "#gta6", "#Shorts", "#GTA6Audio", "#GTA6Squad", "#ViceCity", "Leonida"])
    assert got == ["#GTA6", "#GTA6Audio", "#GTA6Squad", "#ViceCity", "#Leonida"]
    assert publisher.clean_hashtags(["#Rockstar", "#xbox"]) == ["#GTA6"]


def test_add_hashtags_text_post_gets_3_clean_tags():
    out = publisher.add_hashtags("Who wins this race? #PS5 https://t.co/abc")
    assert "#PS5" not in out and "http" not in out
    assert hashtags(out) == ["#GTA6", "#GTAVI", "#GTA6Gameplay"]
    assert len(publisher.add_hashtags("x" * 270)) <= 280


# --------------------------------------------------------------------------- title rules

def test_fix_count_only_touches_the_countdown_number():
    assert critic.fix_count("Top 10 CRAZIEST GTA 6 Moments", 8) == "Top 8 CRAZIEST GTA 6 Moments"
    assert critic.fix_count("GTA 6: 10 BEST Ways To Make $4,800 In 10 Minutes", 8) == \
        "GTA 6: 8 BEST Ways To Make $4,800 In 10 Minutes"
    assert critic.fix_count("GTA 6 Money Tip #3: 5 Stars Escape", 8) == "GTA 6 Money Tip #3: 5 Stars Escape"
    assert critic.fix_count("TOP 5 MONEY METHODS", 7) == "TOP 7 MONEY METHODS"


def test_seo_front_and_gta_vi():
    assert critic.fix_title("Nobody Expected The Police To Actually Do This In GTA 6!") == \
        "GTA 6: Nobody Expected The Police To Actually Do This!"
    assert critic.fix_title("Top 10 CRAZIEST GTA VI Moments Today (IShowSpeed)", 12) == \
        "Top 12 CRAZIEST GTA 6 Moments Today (IShowSpeed)"
    assert critic.fix_title("Selling cars is easy money #GTA6") == "GTA 6: Selling cars is easy money"
    long = "GTA 6: " + "Very Long Words " * 10 + "(No Spoilers)"
    assert len(critic.fix_title(long, max_len=60)) <= 60 and "(" not in critic.fix_title(long, max_len=60)


def test_money_titles_never_sound_like_cheats():
    assert critic.fix_title("GTA 6: 8 INSANE Money Hacks", 8, money=True) == "GTA 6: 8 INSANE Money Tips"
    assert critic.sanitize("UNLIMITED MONEY GLITCH") == "FAST MONEY METHOD"
    assert critic.sanitize("Use A Mod Menu For Cash") is None
    assert critic.fix_title("Modded Cash Drop", money=True, fallback="GTA 6: Top 5 BEST Ways To Make Money") == \
        "GTA 6: Top 5 BEST Ways To Make Money"


# --------------------------------------------------------------------------- critic guards (mocked AI)

ORIGINAL = "GTA 6: Top 8 BEST Ways To Make Money (No Spoilers)"
LISTING = "#8: Pawn Shop - sell stolen items for cash\n#1: Warehouse Loot - $4,800 payout on screen"


def test_critic_rejects_the_hacks_rewrite_seen_today(monkeypatch):
    mock_ai(monkeypatch, {"title": "GTA 6: 8 INSANE Money Hacks – From Pawn Shop Cash to $4,800 Warehouse Loot"})
    out = critic.polish({"title": ORIGINAL}, {"title": 90}, LISTING, money=True, keep=["make money"], count=8)
    assert out["title"] == ORIGINAL


def test_critic_accepts_an_honest_better_title(monkeypatch):
    prompts = mock_ai(monkeypatch, {"title": "GTA 6: 8 BEST Ways To Make Money FAST (No Spoilers)"})
    out = critic.polish({"title": ORIGINAL}, {"title": 90}, LISTING, money=True, count=8)
    assert out["title"] == "GTA 6: 8 BEST Ways To Make Money FAST (No Spoilers)"
    p = prompts[0]
    assert "MONEY VIDEO" in p and "hack" in p and "gta 6 make money fast" in p
    assert "money glitch" not in p and "cheats" not in p.split("search right now")[1].split("\n")[0]


@pytest.mark.parametrize("rewrite, why", [
    ("GTA 6: Top 10 BEST Ways To Make Money (No Spoilers)", "numbers"),
    ("GTA VI: Top 8 BEST Ways To Make Money (No Spoilers)", "GTA VI"),
    ("Top 8 BEST Ways To Make Money In The New Game (No Spoilers)", "keywords"),
    ("GTA 6: Top 8 BEST Ways To Get Rich (No Spoilers)", "keywords"),
    ("GTA 6: Top 8 BEST Ways To Make Money 💰 (No Spoilers)", "emojis"),
    ("GTA 6: Top 8 BEST Ways To Make Money Before The Ending", "spoiler"),
    ("GTA 6: Top 8 BEST INSANE EASY Ways To Make Money (No Spoilers)", "CAPS"),
    ("GTA 6: Top 8 BEST Ways To Make Money #GTA6 (No Spoilers)", "hashtags"),
])
def test_critic_guards(monkeypatch, rewrite, why):
    mock_ai(monkeypatch, {"title": rewrite})
    assert critic.polish({"title": ORIGINAL}, {"title": 90}, LISTING, money=True, count=8)["title"] == ORIGINAL, why


def test_critic_may_not_add_glitch_words_to_any_video(monkeypatch):
    mock_ai(monkeypatch, {"youtube_title": "GTA 6 Police Chase Glitch Goes WRONG"})
    out = critic.polish({"youtube_title": "GTA 6 Police Chase Goes WRONG"}, {"youtube_title": 70}, "cops chase a car")
    assert out["youtube_title"] == "GTA 6 Police Chase Goes WRONG"


def test_rules_apply_without_any_critic_ai():
    out = critic.polish({"title": "Top 10 Money Hacks In GTA VI", "thumbnail_text": "TOP 10 HACKS"},
                        {"title": 90, "thumbnail_text": 28}, LISTING, money=True, count=8)
    assert out == {"title": "Top 8 Money Tips In GTA 6", "thumbnail_text": "TOP 8 TIPS"}


# --------------------------------------------------------------------------- Shorts: AI copy -> captions per platform

CLIP_AI = {
    "is_gta6_footage": True, "unsafe": False, "has_facecam": False, "has_narrator": False, "is_spoiler": False,
    "fits_format": True, "shows_required": True, "required_t": 30, "hype_score": 8,
    "scene_summary": "The player sells a stolen car at the docks and the payout screen shows the cash.",
    "clip_start": 5, "clip_end": 45, "payoff_t": 30, "content_type": "explainer", "banner": "SELL CARS FOR CASH",
    "commentary": [{"t": 5, "line": "This is the easiest money in GTA 6 right now."},
                   {"t": 12, "line": "Grab any nice car and drive it straight to the docks."},
                   {"t": 20, "line": "Park it in the marked spot and wait for the buyer to call."},
                   {"t": 30, "line": "And there it is, the cash lands right on the payout screen."},
                   {"t": 40, "line": "How much did you make with this?"}],
    "youtube_title": "Selling Stolen Cars Is A Money Hack",
    "caption": "GTA 6 car selling is the easiest cash right now. How much did you make?",
    "description": "GTA 6 money tip: how to make money fast by selling stolen cars at the docks. The payout screen "
                   "shows the cash land instantly.",
    "x_text": "Bro stole a car and got paid in seconds #GTA6 https://x.com/somewhere",
    "hashtags": ["#GTA6", "#GTA6Money", "#Lamborghini", "#GTA6Cars"]}


def money_format(cfg):
    return _copy.deepcopy(next(f for f in cfg["formats"]["rotation"] if f["id"] == "money_tips"))


def test_short_copy_end_to_end(monkeypatch, env, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    monkeypatch.setattr(common, "probe", lambda p: {"duration": 60.0})
    monkeypatch.setattr(ai_fallback, "extract_frames", lambda clip, n: ["frame"] * 3)
    prompts = mock_ai(monkeypatch, CLIP_AI, {"youtube_title": "GTA 6 Money Tip: Sell Stolen Cars For FAST Cash",
                                             "banner": "SELL CARS FOR CASH",
                                             "caption": "Selling cars is easy cash. How much did you make?",
                                             "x_text": "Stole a car, got paid in seconds. Too easy."})
    fmt = money_format(env)
    out = gemini_writer.write_clip_copy(clip, "GTA 6 money guide", "SomeCreator", "youtube", fmt=fmt)
    assert "SEARCH WORDS" in prompts[0] and "gta 6 how to make money" in prompts[0] and "money glitch" not in prompts[0]
    assert out["youtube_title"] == "GTA 6 Money Tip: Sell Stolen Cars For FAST Cash"
    assert out["caption"].startswith("GTA 6"), "the rewrite dropped 'GTA 6' from the caption - original kept"
    assert out["x_text"] == "Stole a car, got paid in seconds. Too easy."
    assert out["description"].startswith("GTA 6 money tip")

    out.update(format_tags=fmt["hashtags"], keywords=fmt["keywords"], series="GTA 6 Money Tip #3",
               commentary=[{"t": ln["t"], "text": ln["text"]} for ln in out["commentary_raw"]])
    caps = publisher.build_captions(out, "SomeCreator")
    # YouTube: search words first, numbered series, no hashtags in the title; keyword-rich description
    assert caps["youtube_title"] == "GTA 6 Money Tip #3: Sell Stolen Cars For FAST Cash"
    first_two = caps["youtube_description"].split("\n\n")[0]
    assert first_two.startswith("GTA 6 money tip: how to make money fast")
    assert 3 <= len(hashtags(caps["youtube_description"])) <= 5 and "#Shorts" in caps["youtube_description"]
    # TikTok: keywords in the first line, 3-5 hashtags
    assert caps["tiktok"].splitlines()[0].startswith("GTA 6") and 3 <= len(hashtags(caps["tiktok"])) <= 5
    # X: short line, no link, credit, 3 hashtags at the end only
    assert "http" not in caps["x"] and caps["x"].startswith("Stole a car") and len(caps["x"]) <= 280
    assert len(hashtags(caps["x"])) == 3 and "Gameplay by SomeCreator" in caps["x"]
    # Instagram (paused) still builds
    assert "Gameplay by SomeCreator" in caps["instagram"] and 3 <= len(hashtags(caps["instagram"])) <= 5
    every = " ".join(v if isinstance(v, str) else " ".join(v) for v in caps.values())
    assert "Lamborghini" not in every and "glitch" not in every.lower() and "hack" not in every.lower()
    assert caps["tags"][0] == "GTA 6" and "gta 6 make money fast" in caps["tags"] and "GTA VI" in caps["tags"]
    assert sum(len(t) + 2 for t in caps["tags"]) <= 500
    assert caps["first_comment"] == "How much did you make with this?"


def test_captions_without_ai_description_or_series():
    caps = publisher.build_captions({"youtube_title": "This Police Chase Went CRAZY", "banner": "THE COPS GAVE UP",
                                     "caption": "Six cops and he still got away. Would you?",
                                     "x_text": "He really lost six cops with a pickup.", "hashtags": ["#GTA6"],
                                     "format_tags": ["#GTA6PoliceChase"], "keywords": "GTA 6 police chase car chase"},
                                    "")
    assert caps["youtube_title"] == "GTA 6: This Police Chase Went CRAZY"
    assert caps["tiktok"].splitlines()[0] == "GTA 6: This Police Chase Went CRAZY"  # caption lacks the keywords
    assert caps["youtube_description"].startswith("Six cops and he still got away")
    assert len(hashtags(caps["tiktok"])) >= 3 and caps["first_comment"] == "Would you?"


# --------------------------------------------------------------------------- long videos

ITEMS = [{"label": label, "facts": facts} for label, facts in (
    ("Pawn Shop", "sell stolen items for cash"), ("Car Export", "deliver cars to the docks"),
    ("Store Robbery", "cash from the register"), ("Warehouse Loot", "$4,800 payout on screen"))]
MONEY_SPEC = {"kind": "money_guide", "topic": "", "subject": "the best ways to make money in GTA 6 (no story spoilers)",
              "short": "money tips", "example": "Top 6 BEST Ways To Make Money (No Spoilers)",
              "hashtags": "#GTA6 #GTA6Money #GTA6Tips", "comment": "Which money method are you trying first? 👇"}


def test_package_guide_money(monkeypatch):
    seen = {}

    def gen(prompt, temperature=0.9, prefer_pro=False):
        seen["prompt"] = prompt
        return {"title": "GTA 6: Top 4 BEST Ways To Make Money (No Spoilers)", "thumbnail_text": "MONEY HACKS",
                "intro_line": "Number one made me rich in ten minutes.", "outro_line": "Which one will you try?",
                "summary": "GTA 6 how to make money: four legit methods.", "tags": ["gta 6 money guide"]}
    monkeypatch.setattr(gemini_writer, "_generate_json", gen)
    mock_ai(monkeypatch, {"title": "GTA 6: 4 INSANE Money Hacks – From Pawn Shop Cash to $4,800 Warehouse Loot",
                          "thumbnail_text": "EASY MONEY", "intro_line": "Number one pays the most."})
    pkg = long_video.package_guide(ITEMS, MONEY_SPEC, "GTA VI Top Moments")
    assert "make money fast" in seen["prompt"] and "never words that sound like cheats" in seen["prompt"]
    assert pkg["title"] == "GTA 6: Top 4 BEST Ways To Make Money (No Spoilers)"
    assert pkg["thumbnail_text"] == "EASY MONEY" and pkg["intro_line"] == "Number one pays the most."


def test_package_guide_title_without_money_falls_back(monkeypatch):
    monkeypatch.setattr(gemini_writer, "_generate_json", lambda *a, **k: {"title": "GTA 6: 4 Ways To Get Rich"})
    pkg = long_video.package_guide(ITEMS, MONEY_SPEC, "GTA VI Top Moments")
    assert pkg["title"] == "GTA 6: Top 4 BEST Ways To Make Money (No Spoilers)"


def test_package_moments_count_is_true(monkeypatch):
    moments = [{"label": f"Moment {i}", "streamer": s} for i, s in enumerate(["caseoh", "Kai Cenat", "IShowSpeed"] * 3)]
    monkeypatch.setattr(gemini_writer, "_generate_json", lambda *a, **k: {
        "title": "Top 12 CRAZIEST GTA VI Moments Today (IShowSpeed, Kai Cenat)", "thumbnail_text": "TOP 12 FAILS",
        "summary": "GTA 6 funny moments from today."})
    pkg = long_video.package(moments, "GTA VI Top Moments")
    assert pkg["title"] == "Top 9 CRAZIEST GTA 6 Moments Today (IShowSpeed, Kai Cenat)"
    assert pkg["thumbnail_text"] == "TOP 9 FAILS"


def test_long_captions():
    rendered = {"moments": [{}] * 4, "chapters": ["0:00 Intro", "0:20 #4 Pawn Shop", "1:30 #3 Car Export",
                                                  "2:40 #2 Store Robbery", "3:50 #1 Warehouse Loot"],
                "credits": ["#4 Pawn Shop - gameplay by A: https://youtu.be/a"]}
    pkg = {"title": "GTA 6: Top 4 BEST Ways To Make Money (No Spoilers)",
           "summary": "GTA 6 how to make money: four legit methods.", "tags": ["gta 6 money guide", "gta 6 cheats"]}
    caps = publisher.build_long_captions(pkg, rendered, MONEY_SPEC, "money_guide")
    lines = caps["youtube_description"].splitlines()
    assert lines[0].startswith("GTA 6 how to make money") and "make money in GTA 6" in lines[1]
    assert "0:20 #4 Pawn Shop" in lines and lines[-1] == "#GTA6 #GTA6Money #GTA6Tips"
    assert caps["tags"][0] == "GTA 6" and "gta 6 cheats" not in caps["tags"] and "GTA 6 make money fast" in caps["tags"]
    assert caps["first_comment"] == MONEY_SPEC["comment"] and caps["youtube_title"] == pkg["title"]
    moments = publisher.build_long_captions({"title": "Top 4 CRAZIEST GTA 6 Moments", "summary": "GTA 6 funny moments."},
                                            rendered, None, "moments", names=["caseoh", "Kai Cenat"])
    assert "(caseoh, Kai Cenat)" in moments["youtube_description"].splitlines()[1]


# --------------------------------------------------------------------------- prompts + text posts

def test_prompts_format_without_key_errors():
    kw = dict(brand="B", context="C", title="T", credit="X", provider="youtube", duration=60, min_len=25, max_len=50,
              hints="1-2s", kind_note="", format_note="", video_note="", seo_note=gemini_writer.packaging_note(None))
    for tpl in (gemini_writer.CLIP_PROMPT, gemini_writer.TOUR_PROMPT):
        text = tpl.format(**kw)
        assert '"description": str' in text and "{seo_note}" not in text
    long_video.PACKAGE_PROMPT.format(brand="B", n=5, moments="#5: x", searches="")
    long_video.GUIDE_PACKAGE_PROMPT.format(brand="B", subject="s", n=5, items="#5: x", example="e", short="tips",
                                           part="", search_hint="", searches="", honesty="")


def test_text_post_is_cleaned(monkeypatch):
    mock_ai(monkeypatch, {"text": "GTA VI money tip: what's the first car you're selling? https://t.co/x"})
    assert gemini_writer.write_text_post("x_text_morning", [], []) == "GTA 6 money tip: what's the first car you're selling?"
    mock_ai(monkeypatch, {"text": "Who saw the ending coming? The final mission is wild"})
    assert gemini_writer.write_text_post("x_text_morning", [], []) in gemini_writer._TEXT_FALLBACK["x_text_morning"]
