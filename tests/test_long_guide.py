"""Long videos (long_video.py): kind rotation, guide research (twins -> extra angles, proof of money on screen,
made-up numbers dropped), footage search filters, and the guide's countdown selection."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

import common
import long_video as lv
import source_fetcher as sf
import trends
from fakes import needs_ffmpeg

STRONG = [[time.time() - 86400, 300000], [time.time(), 250000]]
QUIET = [[time.time() - 86400, 300000], [time.time(), 9000]]


def streams(viewers):
    common.save_json(trends.TRENDS_FILE, {"topics": [], "used": {}, "viewers": viewers})


# --------------------------------------------------------------------------- kinds rotation

@pytest.mark.parametrize("fmt", ["moments", "money_guide", "trend_guide", "MONEY_GUIDE"])
def test_format_env_forces_one_kind(monkeypatch, fmt):
    monkeypatch.setenv("FORMAT", fmt)
    assert lv._long_kinds({"last_kind": "moments"}) == [fmt.lower()]


def test_format_env_not_a_long_kind_is_ignored(monkeypatch):
    monkeypatch.setenv("FORMAT", "car_chases")
    assert lv._long_kinds({})[0] == "moments"


def test_rotation_moments_with_trend_fallback(cfg):
    """Owner 2026-11-21: money guides dropped - long videos are moments, a trend guide as fallback."""
    streams(STRONG)
    assert lv._long_kinds({}) == ["moments", "trend_guide"]
    assert lv._long_kinds({"last_kind": "moments"}) == ["moments", "trend_guide"]


def test_money_guide_rotation_still_works_when_configured(cfg):
    streams(STRONG)
    cfg["long_video"]["rotation"] = ["moments", "money_guide"]
    assert lv._long_kinds({}) == ["moments", "money_guide", "trend_guide"]
    assert lv._long_kinds({"last_kind": "moments"}) == ["money_guide", "moments", "trend_guide"]
    assert lv._long_kinds({"last_kind": "trend_guide"})[0] == "moments", "unknown last kind = start of rotation"


def test_quiet_streams_trend_guide_replaces_moments(cfg):
    streams(QUIET)
    order = lv._long_kinds({"last_kind": "moments"})
    assert order[0] == "trend_guide" and order[-1] == "moments", "moments only as the last resort"


def test_no_viewer_data_counts_as_strong():
    streams([])
    assert lv._long_kinds({})[0] == "moments"


def test_rotation_from_config(cfg):
    cfg["long_video"]["rotation"] = ["money_guide", "bogus"]
    assert lv._long_kinds({"last_kind": "money_guide"})[0] == "money_guide"
    cfg["long_video"]["rotation"] = ["bogus"]
    assert lv._long_kinds({})[0] == "moments"


# --------------------------------------------------------------------------- rank helpers

@pytest.mark.parametrize("line,want", [
    ("Number three, grab a car.", "grab a car."),
    ("#7: the docks pay well", "the docks pay well"),
    ("At number 3 we have races", "we have races"),
    ("No. 12 - store robberies", "store robberies"),
    ("Numbers go up when you sell", "Numbers go up when you sell"),
])
def test_strip_rank(line, want):
    assert lv._strip_rank(line) == want


def test_ranked_says_the_card_number():
    assert lv._ranked(4, "number two. selling cars") == "Number four. Selling cars"
    assert lv._ranked(25, "Street races") == "Number 25. Street races"
    assert lv._ranked(3, "Number three") == ""


# --------------------------------------------------------------------------- guide research (review_guide)

MONEY = {"kind": "money_guide", "subject": "money", "noun": "way to make money", "proof": "cash going up"}
TREND = {"kind": "trend_guide", "subject": "beaches", "noun": "example"}


def clip(i, streamer, dur=110.0):
    return {"key": f"yt:v{i}", "streamer": streamer, "dur": dur, "path": Path(f"/raw/v{i}.mp4"), "platform": "youtube",
            "title": f"GTA 6 video {i}", "url": f"https://youtube.com/watch?v=v{i}", "views": 1000 * i}


def verdict(item, use=7, lines=None, facts="the money counter goes up", proof_t=20, **kw):
    v = {"is_gta6_footage": True, "broken": False, "is_spoiler": False, "unsafe": False, "is_glitch": False,
         "shows_item": True, "item": item, "facts": facts, "usefulness": use, "trim_start": 5, "trim_end": 45,
         "best_t": 12, "has_narrator": True, "proof_t": proof_t,
         "lines": lines if lines is not None else ["Grab a car and drive it to the docks.",
                                                   "The buyer pays you on the spot, right away."]}
    v.update(kw)
    return v


@pytest.fixture
def research(monkeypatch):
    """research(pairs, spec) -> review_guide's items, with the AI's verdicts given directly (no upload)."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    prompts = []

    def run(pairs, spec=MONEY):
        def batches(moments, prompt_for):
            prompts.append(prompt_for(len(moments), "1: 0-30 s (x on Youtube)"))
            yield moments, list(pairs)
        monkeypatch.setattr(lv, "_batch_verdicts", batches)
        return lv.review_guide([m for m, _ in pairs], spec, "Brand")
    run.prompts = prompts
    return run


def test_twins_merge_into_extra_angles(research):
    items = research([
        (clip(1, "LeonidaTips"), verdict("Street Races", 9)),
        (clip(2, "TowTrucker"), verdict("Street Race Wins", 6)),
        (clip(3, "MoneyGuru"), verdict("Selling Stolen Cars", 8)),
        (clip(4, "Copycat"), verdict("selling stolen car", 5)),
        (clip(5, "CashKing"), verdict("Store Robberies", 6)),
    ])
    by = {it["label"]: it for it in items}
    assert sorted(by) == ["Selling Stolen Cars", "Store Robberies", "Street Races"]
    races = by["Street Races"]
    assert races["streamer"] == "LeonidaTips", "the most useful clip is the main footage"
    assert races["sources"] == 2 and [x["streamer"] for x in races["extra"]] == ["TowTrucker"]
    assert by["Selling Stolen Cars"]["sources"] == 2 and by["Store Robberies"]["sources"] == 1
    assert "extra" not in by["Store Robberies"]
    assert [it["hype"] for it in items] == sorted((it["hype"] for it in items), reverse=True)


def test_money_guide_needs_proof_on_screen(research):
    items = research([(clip(1, "SideHustle"), verdict("Taxi Side Job", 9, proof_t=None)),
                      (clip(2, "Racer"), verdict("Street Races", 7, proof_t=30))])
    assert [it["label"] for it in items] == ["Street Races"]
    assert "proof_t: the second within the clip where you SEE cash going up" in research.prompts[0]


def test_trend_guide_needs_no_proof(research):
    items = research([(clip(1, "Beachy"), verdict("Hidden Beach", 8, proof_t=None))], spec=TREND)
    assert [it["label"] for it in items] == ["Hidden Beach"]
    assert "- proof_t: null" in research.prompts[0]


def test_made_up_numbers_are_dropped(research):
    lines = ["Walk into a corner store and point your gun at the cashier.",
             "He fills the bag and it pays 9,999 dollars.",            # never shown -> dropped
             "The HUD shows $4,500 after the sale, nice.",            # in facts -> kept
             "GTA 6 makes this one really easy, trust me.",           # "6" is always fine
             "Number two, run out the back door and lose the cops."]  # rank prefix stripped
    items = research([(clip(1, "CashKing"), verdict("Store Robberies", 7, lines=lines,
                                                    facts="the HUD shows $4,500 after the sale"))])
    got = items[0]["lines"]
    assert not any("9,999" in ln for ln in got)
    assert "The HUD shows $4,500 after the sale, nice." in got
    assert "GTA 6 makes this one really easy, trust me." in got
    assert "run out the back door and lose the cops." in got


def test_too_few_lines_left_drops_the_item(research):
    lines = ["It pays 9,999 dollars every time.", "You get 500 for every car."]  # both made up
    assert research([(clip(1, "CashKing"), verdict("Store Robberies", 7, lines=lines))]) == []


@pytest.mark.parametrize("flag", [{"is_gta6_footage": False}, {"broken": True}, {"is_spoiler": True},
                                  {"unsafe": True}, {"is_glitch": True}, {"shows_item": False}, {"item": ""}])
def test_rejections(research, flag):
    v = verdict("Street Races", 9)
    v.update(flag)
    assert research([(clip(1, "X"), v)]) == []


def test_missing_verdict_is_skipped(research):
    assert research([(clip(1, "X"), None), (clip(2, "Y"), verdict("Street Races"))])[0]["streamer"] == "Y"


def test_trim_window_rules(research):
    items = research([
        (clip(1, "A"), verdict("Street Races", 7, trim_start=10, trim_end=18, proof_t=12)),          # < 15 s -> 40 s
        (clip(2, "B"), verdict("Store Robberies", 7, trim_start=5, trim_end=30, proof_t=80)),         # proof outside
        (clip(3, "C"), verdict("Selling Cars", 7, trim_start=0, trim_end=100, proof_t=20)),          # max 45 s
        (clip(4, "D"), verdict("Tow Truck Job", 7, trim_start="soon", trim_end=None, proof_t=20)),   # junk numbers
    ])
    by = {it["streamer"]: it for it in items}
    assert (by["A"]["a"], by["A"]["b"]) == (10.0, 50.0)
    b = by["B"]
    assert b["a"] <= 80 <= b["b"] and b["b"] - b["a"] == pytest.approx(25.0), "moved over the proof, same length"
    assert by["C"]["b"] - by["C"]["a"] == 45.0
    assert (by["D"]["a"], by["D"]["b"]) == (0.0, 45.0)


def test_usefulness_clamped_and_labels_title_cased(research):
    items = research([(clip(1, "A"), verdict("street races", 14)), (clip(2, "B"), verdict("Tow Jobs", "high"))])
    by = {it["streamer"]: it for it in items}
    assert by["A"]["hype"] == 10 and by["A"]["label"] == "Street Races"
    assert by["B"]["hype"] == 1


def test_review_guide_needs_gemini_key():
    assert lv.review_guide([clip(1, "A")], MONEY, "Brand") == []


# --------------------------------------------------------------------------- footage search (gather_guide)

VIDS = [("MoneyGuru", "GTA 6 How To Make Money FAST", 600, 900000),
        ("CashKing", "GTA 6 Best Money Method (No Commentary)", 420, 500000),
        ("GlitchGod", "GTA 6 UNLIMITED MONEY GLITCH", 300, 2000000),
        ("ModMaster", "GTA 6 mod menu money", 300, 1500000),
        ("OldTimer", "GTA 5 money guide", 300, 800000),
        ("ShortOne", "GTA 6 money in 20 seconds", 20, 700000),
        ("RichPlayer", "GTA 6 Selling Cars Guide", 90, 300000),
        ("CashKing", "GTA 6 easy money 2", 400, 80000),
        ("CashKing", "GTA 6 easy money 3", 400, 70000)]


@pytest.fixture
def search(monkeypatch):
    import stream_clips as sc
    monkeypatch.setattr(sf, "discover_youtube", lambda cfg, queries, tf, kind="gameplay": [
        sf.Candidate(key=f"yt:v{i}", url=f"https://youtube.com/watch?v=v{i}", title=t, uploader=ch, provider="youtube",
                     duration=d, views=v) for i, (ch, t, d, v) in enumerate(VIDS)])
    monkeypatch.setattr(sf, "discover_dailymotion", lambda cfg, q, tf: [])
    monkeypatch.setattr(sc, "twitch_clips", lambda c: [])
    monkeypatch.setattr(sc, "kick_clips", lambda c: [])


def test_gather_guide_filters(search):
    pool = lv.gather_guide(lv.guide_spec("money_guide"), used={"yt:v0"})
    names = [(x["streamer"], x["title"]) for x in pool]
    assert all(s not in ("GlitchGod", "ModMaster", "OldTimer", "ShortOne") for s, _ in names), names
    assert ("MoneyGuru", VIDS[0][1]) not in names, "already used"
    assert [s for s, _ in names].count("CashKing") == 2, "max 2 videos per channel"
    assert names[0] == ("CashKing", VIDS[1][1]), "no-commentary footage first"
    rich = next(x for x in pool if x["streamer"] == "RichPlayer")
    assert rich["section"] == (0, 91), "short video: all of it"
    cash = next(x for x in pool if x["title"] == VIDS[1][1])
    assert cash["section"] == (50, 160), "long video: skip the intro, ~110 s section"


def test_guide_spec(cfg):
    m = lv.guide_spec("money_guide")
    assert m["proof"] and m["queries"] and len(m["queries"]) == len(set(m["queries"]))
    assert lv.guide_spec("trend_guide") is None, "no trend data yet"
    common.save_json(trends.TRENDS_FILE, {"topics": [{"topic": "Hidden Beaches", "angle": "a", "queries": ["GTA 6 beach"],
                                                      "money": False, "demand": 8}], "used": {}})
    t = lv.guide_spec("trend_guide")
    assert t["topic"] == "Hidden Beaches" and "proof" not in t
    assert lv.guide_spec("moments") is None


# --------------------------------------------------------------------------- final GTA 6 check on the shown windows

@pytest.fixture
def realcheck(monkeypatch, make_clip):
    """Items whose parts are real tiny clips; the authenticity AI is faked: install(verdict) with
    verdict(window_index) -> True (GTA VI) / False (GTA V) / None (the AI could not look)."""
    import authentic
    asked = []

    def install(verdict):
        def check(paths):
            batch = []
            for p in paths:
                k = int(p.stem.rsplit("_", 1)[1])
                asked.append((k, p.exists(), common.probe(p)["duration"] if p.exists() else 0))
                ok = verdict(k)
                batch.append(None if ok is None else {"game": "GTA VI" if ok else "GTA V", "confidence": 0.9,
                                                      "evidence": "Michael in Los Santos", "ok": bool(ok)})
            return None if any(r is None for r in batch) else batch
        monkeypatch.setattr(authentic, "check", check)
        return asked
    src = make_clip("src.mp4", dur=4, size="320x240", audio=False)

    def part(key, streamer, a=0.5, b=2.5):
        return {"key": key, "streamer": streamer, "path": src, "a": a, "b": b}

    def item(label, main, *extra):
        return {**main, "label": label, "rank": 5.0, "sources": 1 + len(extra), "extra": list(extra)}
    install.part, install.item = part, item
    return install


@needs_ffmpeg
def test_final_check_all_real(realcheck):
    p, it = realcheck.part, realcheck.item
    items = [it("Street Races", p("yt:1", "A"), p("yt:2", "B")), it("Store Robberies", p("yt:3", "C"))]
    asked = realcheck(lambda k: True)
    out = lv._final_real_check(items)
    assert [x["label"] for x in out] == ["Street Races", "Store Robberies"]
    assert [x["streamer"] for x in out[0]["extra"]] == ["B"]
    assert [k for k, _, _ in asked] == [0, 1, 2], "main clip and every extra angle are checked"
    assert all(exists and dur == pytest.approx(2.0, abs=0.2) for _, exists, dur in asked), "exactly the shown window"
    assert not list(lv.LONG_DIR.glob("realcheck_*.mp4")), "windows are cleaned up"


@needs_ffmpeg
def test_final_check_drops_fake_extra_angle(realcheck):
    p, it = realcheck.part, realcheck.item
    realcheck(lambda k: k != 1)
    out = lv._final_real_check([it("Street Races", p("yt:1", "A"), p("yt:2", "GTA5Guy"), p("yt:3", "C"))])
    assert out[0]["streamer"] == "A" and [x["streamer"] for x in out[0]["extra"]] == ["C"]


@needs_ffmpeg
def test_final_check_promotes_extra_angle_when_main_is_fake(realcheck):
    p, it = realcheck.part, realcheck.item
    realcheck(lambda k: k != 0)
    item = it("Street Races", p("yt:1", "GTA5Guy"), p("yt:2", "B"), p("yt:3", "C"))
    item.update(rank=9.5, sources=3)
    out = lv._final_real_check([item])
    assert len(out) == 1
    top = out[0]
    assert top["streamer"] == "B" and top["key"] == "yt:2", "the next real angle becomes the main footage"
    assert (top["label"], top["rank"], top["sources"]) == ("Street Races", 9.5, 3)
    assert [x["streamer"] for x in top["extra"]] == ["C"]


@needs_ffmpeg
def test_final_check_drops_item_without_real_footage(realcheck):
    p, it = realcheck.part, realcheck.item
    realcheck(lambda k: k == 2)
    out = lv._final_real_check([it("Fake Item", p("yt:1", "A"), p("yt:2", "B")), it("Real Item", p("yt:3", "C"))])
    assert [x["label"] for x in out] == ["Real Item"]


@needs_ffmpeg
def test_final_check_is_strict_when_ai_unavailable(realcheck):
    p, it = realcheck.part, realcheck.item
    realcheck(lambda k: None)
    assert lv._final_real_check([it("Street Races", p("yt:1", "A")), it("Store Robberies", p("yt:2", "B"))]) == []


@needs_ffmpeg
def test_final_check_batches_of_six(realcheck, monkeypatch):
    import authentic
    p, it = realcheck.part, realcheck.item
    realcheck(lambda k: True)
    inner, sizes = authentic.check, []

    def counting(paths):
        sizes.append(len(paths))
        return inner(paths)
    monkeypatch.setattr(authentic, "check", counting)
    items = [it(f"Item {i}", p(f"yt:{i}a", "A"), p(f"yt:{i}b", "B")) for i in range(4)]  # 8 parts
    assert len(lv._final_real_check(items)) == 4
    assert sizes == [6, 2]
