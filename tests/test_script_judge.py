"""The script judge and the whole-video writer - built from the two posted Shorts the owner rated 0/10."""
import pytest

import ai_fallback
import gemini_writer
import script_judge as sj

DIRT_BIKE = ["I tried stealing this dirt bike right from the NPC and it worked in GTA 6!",
             "This dirt bike handles the track jumps surprisingly well right off the line.",
             "Let's see if we can hit these dirt track jumps without crashing.",
             "Bro, that mid-air flip on the dirt track was actually clean!",
             "Now we gotta go back and steal another dirt bike from this guy.",
             "The suspension on this dirt bike handles the dirt track bumps so smoothly.",
             "Would you try this dirt track trial or stick to sports cars?"]
COP_CAR = ["You will not believe how easy it is to steal a cop car in GTA 6.",
           "Number two: Checking out the trunk reveals cocaine right next to an assault rifle.",
           "Number three: Opening the car trunk to check stored gear in GTA 6.",
           "Number four: Throwing punches at the police officer right next to the patrol car trunk.",
           "Number five: Beating down another police officer in the parking lot during the shootout.",
           "Number six: Sprinting away from the scene across the street past local storefronts.",
           "Number seven: Running across the sunlit road toward a parked pickup truck.",
           "Would you risk your cash to save a stolen ride in GTA 6?"]
GOOD = ["Jason thought one cop car would be an easy grab.",
        "Then the trunk pops open - and that is NOT standard police gear.",
        "Five officers later, the plan is officially falling apart.",
        "So he does the only reasonable thing: he runs.",
        "Would you have kept the car or grabbed the trunk and bailed?"]


def test_the_two_bad_shorts_fail_the_certain_checks():
    dirt = sj.static_problems(DIRT_BIKE)
    assert any("fake first person" in p for p in dirt) and any('"dirt"' in p for p in dirt)
    cop = sj.static_problems(COP_CAR)
    assert any("numbered items over ONE continuous scene" in p for p in cop)
    assert any("broken numbering" in p for p in sj.static_problems(COP_CAR, numbered_ok=True))


def test_a_good_story_passes_the_certain_checks():
    assert sj.static_problems(GOOD) == []
    assert sj.static_problems(["Oh my, look at him go.", "He actually lands it.", "Clean.", "Would you try it?"]) == []
    countdown = ["Top 3 GTA 6 hideouts.", "Number three: the swamp shack.", "Number two: the rooftop.",
                 "Number one: the yacht.", "Which one is yours?"]
    assert sj.static_problems(countdown, numbered_ok=True, countdown=3) == []
    assert "numbering out of order" in sj.static_problems(
        ["Hook.", "Number one: a.", "Number two: b.", "Number three: c.", "Which?"], numbered_ok=True)[0]


def test_judge_needs_a_good_score_and_a_kept_promise(monkeypatch):
    monkeypatch.setattr(ai_fallback, "available", lambda: True)
    answer = {"score": 8, "makes_sense": True, "promise_kept": True, "entertaining": True, "problems": [],
              "_ai_model": "cerebras/gpt-oss-120b"}
    monkeypatch.setattr(ai_fallback, "chat_json", lambda prompt, frames=None, temperature=0.9: dict(answer))
    lines = [{"t": i * 3.0, "text": t} for i, t in enumerate(GOOD)]
    v = sj.judge(lines, {"fmt": "Trending", "banner": "COP CAR GONE WRONG", "story": "a cop car theft goes wrong"})
    assert v["ok"] and v["score"] == 8 and v["by"] == "cerebras/gpt-oss-120b"
    answer.update(score=9, promise_kept=False, problems=["hook promises a car theft, no car is stolen"])
    v = sj.judge(lines, {})
    assert not v["ok"] and "no car is stolen" in sj.feedback(v)
    answer.update(score=9, promise_kept=True)
    assert not sj.judge([{"t": i * 3.0, "text": t} for i, t in enumerate(COP_CAR)], {})["ok"], "certain problems block"


def test_no_judge_means_no_post(monkeypatch):
    monkeypatch.setattr(ai_fallback, "available", lambda: False)
    monkeypatch.setattr(gemini_writer, "_generate_json", lambda *a, **k: None)
    assert sj.judge([{"t": 0, "text": t} for t in GOOD], {})["ok"] is False


def test_writer_watches_the_whole_video_and_returns_a_fitted_script(monkeypatch, make_clip):
    clip = make_clip("cut.mp4", dur=20)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    seen = {}

    class Up:
        name = "files/1"
    monkeypatch.setattr(gemini_writer, "_upload_video", lambda p: Up())
    monkeypatch.setattr(gemini_writer, "_client", lambda: type("C", (), {"files": type("F", (), {
        "delete": staticmethod(lambda name: None)})()})())

    def gen(contents, temperature=0.9, prefer_pro=False):
        seen["contents"] = contents
        return {"story": "Jason steals a cop car", "banner": "COP CAR CHAOS", "title": "GTA 6 Cop Car Chaos",
                "lines": [{"t": 0.0, "text": GOOD[0]}, {"t": 4.0, "text": GOOD[1]}, {"t": 4.5, "text": GOOD[2]},
                          {"t": 9.0, "text": GOOD[3]}, {"t": 18.5, "text": GOOD[4]}]}
    monkeypatch.setattr(gemini_writer, "_generate_json", gen)
    out = gemini_writer.write_final_script(clip, {"banner": "OLD", "youtube_title": "GTA 6 old"}, lead=1.2,
                                           spans=[(5.0, 7.0)], feedback="YOUR LAST SCRIPT WAS REJECTED")
    up, prompt = seen["contents"]
    assert isinstance(up, Up) and "WATCH THE WHOLE VIDEO" in prompt and "REJECTED" in prompt and "5.0-7.0s" in prompt
    ts = [ln["t"] for ln in out["lines"]]
    assert ts[0] == 1.3, "the hook starts right after the cold open"
    assert all(b > a for a, b in zip(ts, ts[1:])), "lines never start before the previous one ended"
    words_end = ts[-1] + len(out["lines"][-1]["text"].split()) / 2.7
    assert words_end <= 20 and out["lines"][-1]["text"].endswith("?"), "the closing question fits and stays"
    assert out["banner"] == "COP CAR CHAOS" and out["title"] == "GTA 6 Cop Car Chaos"


def test_writer_returns_three_different_scripts(monkeypatch, make_clip):
    clip = make_clip("cut3.mp4", dur=20)
    monkeypatch.setenv("GEMINI_API_KEY", "x")

    class Up:
        name = "files/2"
    monkeypatch.setattr(gemini_writer, "_upload_video", lambda p: Up())
    monkeypatch.setattr(gemini_writer, "_client", lambda: type("C", (), {"files": type("F", (), {
        "delete": staticmethod(lambda name: None)})()})())
    seen = {}

    def gen(contents, temperature=0.9, prefer_pro=False):
        seen["prompt"] = contents[1]
        one = [{"t": i * 3.0, "text": t} for i, t in enumerate(GOOD)]
        return {"story": "a cop car theft goes wrong", "scripts": [
            {"angle": "roast", "lines": one, "banner": "COP CAR GONE WRONG", "title": "GTA 6 Cop Car Gone Wrong"},
            {"angle": "hype", "lines": one[::-1]}, {"angle": "deadpan", "lines": one[:2]}]}
    monkeypatch.setattr(gemini_writer, "_generate_json", gen)
    out = gemini_writer.write_final_script(clip, {"banner": "OLD"}, fmt={"name": "Top 5 funniest moments"}, n=3)
    assert "3 DIFFERENT complete scripts" in seen["prompt"] and "NOT A LIST" in seen["prompt"]
    assert "step-by-step guide" in seen["prompt"] and "ONE concrete fact or tip" in seen["prompt"]
    assert "roast /" not in seen["prompt"] and "set-up -> punchline" not in seen["prompt"], "owner: no bro / roasts"
    assert out["angle"] == "roast" and [a["angle"] for a in out["alternatives"]] == ["hype"], "too-short scripts dropped"
    assert out["story"] == out["alternatives"][0]["story"] == "a cop car theft goes wrong"


def test_judge_prompt_carries_the_channel_rules_and_anchors(monkeypatch):
    seen = {}
    monkeypatch.setattr(ai_fallback, "available", lambda: True)
    monkeypatch.setattr(ai_fallback, "chat_json", lambda prompt, frames=None, temperature=0.9:
                        seen.setdefault("p", prompt) and {"score": 8, "makes_sense": True, "promise_kept": True})
    sj.judge([{"t": 0, "text": t} for t in GOOD], {"fmt": "Top 5 funniest moments"})
    p = seen["p"]
    assert "last line is a short question to the viewer (required" in p
    assert "it is correct that nothing is numbered" in p and "THEME, not a demand for numbers" in p
    assert "2/10" in p and "8/10" in p
    seen.clear()
    sj.judge([{"t": 0, "text": t} for t in GOOD], {}, numbered_ok=True, countdown=3)
    assert "this is a countdown" in seen["p"]


MARINA = [(1.7, "Bro rented a scooter in Leonida and chose immediate violence."),
          (7.0, "Tourists just wanted a relaxing morning by the bay."),
          (13.0, "Instead, they got mowed down by a guy in a tank top."),
          (20.0, "Parking enforcement on these docks is getting out of hand."),
          (32.0, "He's treating the marina like his personal bowling alley."),
          (41.0, "Look at him casually drifting past another crime scene."),
          (55.2, "Would you trust this guy with a rental deposit?")]


def test_dead_air_is_caught():
    lines = [{"t": t, "text": x} for t, x in MARINA]
    probs = sj.dead_air(lines, 59.2, lead=1.2)
    assert any("(24-32s)" in p for p in probs) and any("(44-55s)" in p for p in probs)
    assert any("talks only" in p for p in probs)
    # the characters talking in the gap is fine - nobody needs to talk over them
    assert not any("24-32s" in p for p in sj.dead_air(lines, 59.2, lead=1.2, spans=[(22.0, 32.0)]))
    full = [{"t": 1.3 + i * 4.5, "text": "He keeps finding new ways to make this marina regret everything today."}
            for i in range(13)]
    assert sj.dead_air(full, 59.2, lead=1.2) == []


def test_closing_question_must_be_short_and_on_its_own():
    buried = ("Bro swaps for a delivery van and tears straight across the tarmac while dodging cruisers, but would "
              "you survive this airport chase?")  # the last line of a posted Short (2026-11-23)
    probs = sj.static_problems(GOOD[:-1] + [buried])
    assert any("closing question is buried in a 22-word line" in p for p in probs)
    assert not any("buried" in p for p in sj.static_problems(GOOD[:-1] + ["Would you survive this airport chase?"]))
    # a countdown's last slot is "Number one: ..." - the question shares that line, it isn't buried
    countdown = ["Number two: he parks the bike on a palm tree, which is honestly a talent.",
                 "Number one: the cop car follows him off the pier and both of them go swimming, would you?"]
    assert not any("buried" in p for p in sj.static_problems(countdown, numbered_ok=True, countdown=2))


def test_quiet_tail_is_where_the_talking_stops():
    """Live 2026-11-25: a 34 s crash clip, the rider walks around from ~15 s - every script went quiet there."""
    lines = [{"t": 1.0, "text": "Bro really decided to play chicken with a luxury sedan on a dirt bike."},
             {"t": 7.0, "text": "That is a ten out of ten superhero launch right into orbit."},
             {"t": 11.0, "text": "He just bounced off the asphalt like a dropped sandwich."},
             {"t": 28.0, "text": "Would you try this stunt?"}]
    t = sj.quiet_tail(lines, 34.0, lead=1.2)
    assert t == pytest.approx(11.0 + 10 / 2.7, abs=0.05)
    middle = [lines[0], {"t": 16.0, "text": "Look at the car roll over after that."},
              {"t": 20.0, "text": "And the rider just gets up and walks away."}, lines[3]]  # quiet 7-16 s
    assert sj.quiet_tail(middle, 34.0, lead=1.2) is None, "a gap in the middle is a real script problem"
    full = lines[:3] + [{"t": 16.0, "text": "Now he walks it off like nothing happened at all, legend."},
                        {"t": 21.0, "text": "And the car is still spinning back there on the bridge."}, lines[3]]
    assert sj.quiet_tail(full, 34.0, lead=1.2) is None, "no long silence: nothing to cut"


def test_bro_is_rejected_without_an_ai():
    """Owner 2026-11-25: "quality is bad, video is nonsense, just bro"."""
    lines = ["This is the fastest way to lose the cops in GTA 6.", "Bro really thought he could outrun them.",
             "Ditch the car where no camera can see you.", "What's the highest wanted level you've escaped?"]
    assert any("slang" in p for p in sj.static_problems(lines))
    assert not any("slang" in p for p in sj.static_problems([lines[0], lines[2], lines[3]]))


def test_fewer_lines_than_slots_go_to_their_nearest_slots():
    """Live 2026-11-25: a backup wrote 5 lines for an 8-shot edit and the whole clip was dropped."""
    slots = [{"t": float(t), "text": "x"} for t in (0, 5, 10, 15, 20, 25, 30, 35)]
    lines = [{"t": 0.5, "text": "a"}, {"t": 9.0, "text": "b"}, {"t": 21.0, "text": "c"}, {"t": 26.0, "text": "d"},
             {"t": 36.0, "text": "e"}]
    out = gemini_writer._fewer_slots(lines, slots)
    assert [ln["t"] for ln in out] == [0, 10, 20, 25, 35] and [ln["text"] for ln in out] == list("abcde")
    crowded = [{"t": 34.0, "text": t} for t in "abc"]  # all near the end: still in order, one slot each
    assert [ln["t"] for ln in gemini_writer._fewer_slots(crowded, slots)] == [25, 30, 35]
    assert gemini_writer._fewer_slots(lines[:2], slots) is None, "under 3 lines is no script"
