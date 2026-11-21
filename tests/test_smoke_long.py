"""Long-video money guide end to end (long_video.py), ported from the scratchpad harnesses test_guide.py /
test_long2.py: footage search -> downloads + checks -> authenticity -> AI research in batches -> countdown
selection -> package -> 16:9 render -> QA -> upload text. Only network parts are faked (search, downloads,
Gemini, TTS). The real render uses a tiny guide (3 items, short clips) to stay fast."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import authentic
import gemini_writer
import long_video as lv
import music
import qa
import source_fetcher as sf
import stream_clips as sc
from fakes import ffmpeg, needs_ffmpeg, stream_info

pytestmark = needs_ffmpeg

PKG = {"title": "GTA 6: Top 5 BEST Ways To Make Money (No Spoilers)",
       "intro_line": "These are the best money methods in GTA 6, number one is easy.",
       "outro_line": "Which one are you trying first? Subscribe for more.",
       "thumbnail_text": "EASY MONEY", "summary": "The best ways to make money in GTA 6.", "tags": ["GTA 6 money"]}


def guide_world(monkeypatch, vids, items, clip_sec, size="854x480", gta5=(), too_old=()):
    """Fake YouTube search results `vids` [(channel, title, dur, views)], downloads (synthetic footage), the AI
    researcher (`items`: channel -> (item, usefulness, lines, facts, has_narrator, proof_t)) and the packager."""
    monkeypatch.setenv("FORMAT", "money_guide")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(sf, "discover_youtube", lambda cfg, queries, tf, kind="gameplay": [
        sf.Candidate(key=f"yt:v{i}", url=f"https://youtube.com/watch?v=v{i}", title=t, uploader=ch, provider="youtube",
                     duration=d, views=v, kind="gameplay") for i, (ch, t, d, v) in enumerate(vids)])
    monkeypatch.setattr(sf, "discover_dailymotion", lambda cfg, q, tf: [])
    monkeypatch.setattr(sc, "twitch_clips", lambda c: [])
    monkeypatch.setattr(sc, "kick_clips", lambda c: [])
    monkeypatch.setattr(sf, "_fill_metadata", lambda cand, cfg: cand.uploader not in too_old)
    downloads = []

    def download(cand, a, b, out):
        downloads.append(cand.uploader)
        ffmpeg("-f", "lavfi", "-i", f"testsrc2=s={size}:r=30:d={clip_sec}",
               "-f", "lavfi", "-i", f"sine=f={300 + 10 * len(cand.uploader)}:d={clip_sec}", "-shortest",
               "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out))
        return True
    monkeypatch.setattr(sf, "_download", download)
    monkeypatch.setattr(authentic, "check", lambda paths: [
        {"game": "GTA V" if any(f"yt_v{i}." in p.name for i in gta5) else "GTA VI", "confidence": 0.9,
         "evidence": "test", "ok": not any(f"yt_v{i}." in p.name for i in gta5)} for p in paths])
    monkeypatch.setattr(gemini_writer, "_upload_video", lambda path: SimpleNamespace(name="files/fake"))
    monkeypatch.setattr(gemini_writer, "_client", lambda: SimpleNamespace(files=SimpleNamespace(delete=lambda name: None)))
    import ai_fallback
    monkeypatch.setattr(ai_fallback, "chat_json", lambda *a, **k: None)  # critic: no rewrite
    seen = SimpleNamespace(review_prompts=[], package_prompts=[], downloads=downloads)

    def gen(contents, temperature=0.9, prefer_pro=False):
        if isinstance(contents, list):  # research batch: the video + the guide prompt
            prompt = contents[1]
            assert "countdown about" in prompt, "guide research prompt expected"
            seen.review_prompts.append(prompt)
            who = dict(re.findall(r"^(\d+): [\d.]+-[\d.]+ s \((\S+) on", prompt, re.M))
            out = []
            for i, ch in who.items():
                item, use, lines, facts, narr, proof = items.get(ch, ("Nothing", 1, [], "", True, None))
                out.append({"clip": int(i), "is_gta6_footage": True, "broken": False, "is_spoiler": False,
                            "unsafe": False, "is_glitch": False, "shows_item": item != "Nothing", "item": item,
                            "facts": facts, "lines": lines, "usefulness": use, "trim_start": 0, "trim_end": clip_sec,
                            "best_t": clip_sec / 2, "has_narrator": narr, "proof_t": proof})
            return {"clips": out}
        assert "countdown guide" in contents, "package prompt expected"
        seen.package_prompts.append(contents)
        return dict(PKG)
    monkeypatch.setattr(gemini_writer, "_generate_json", gen)
    return seen


def spy_technical(monkeypatch) -> list:
    """Record what the pipeline's own qa.technical() call found (instead of running the checks twice)."""
    calls, real = [], qa.technical

    def spy(path, still_ok=None):
        issues = real(path, still_ok)
        calls.append({"path": Path(path), "still_ok": list(still_ok or []), "issues": issues})
        return issues
    monkeypatch.setattr(qa, "technical", spy)
    return calls


# --------------------------------------------------------------------------- selection (render stubbed)

VIDS = [("MoneyGuru", "GTA 6 How To Make Money FAST (No Commentary)", 600, 900000),
        ("CashKing", "GTA 6 Best Money Method Right Now", 420, 500000),
        ("GlitchGod", "GTA 6 UNLIMITED MONEY GLITCH", 300, 2000000),     # filtered by title (glitch)
        ("RichPlayer", "GTA 6 Selling Cars Guide", 90, 300000),           # GTA V footage (authenticity)
        ("LeonidaTips", "GTA 6 Money Making No Commentary", 1200, 250000),
        ("SideHustle", "GTA 6 side jobs that pay", 500, 120000),          # no money on screen (no proof)
        ("EarlyCash", "GTA 6 make money early game", 700, 90000),         # uploaded before launch
        ("TowTrucker", "GTA 6 tow truck job no commentary", 300, 85000),  # same method as LeonidaTips
        ("Wallets", "GTA 6 wallets money", 300, 84000),                   # usefulness 4: below the bar
        ("CashKing", "GTA 6 easy money 2", 400, 80000),
        ("CashKing", "GTA 6 easy money 3", 400, 70000)]                   # 3rd from the same channel: dropped
ITEMS = {
    "MoneyGuru": ("Selling Stolen Cars", 8, ["Grab any parked sports car and drive it to the docks.",
                                            "The buyer pays you on the spot, the cash shows up right away."],
                  "the HUD shows $4,500 after the sale", True, 5),
    "CashKing": ("Store Robberies", 6, ["Walk into a corner store and point your gun at the cashier.",
                                        "He fills the bag and it pays 9,999 dollars, then you run.",
                                        "Run out the back door and lose the cops in the alleys."],
                 "the cash counter goes up", True, 5),
    "RichPlayer": ("Car Delivery Jobs", 7, ["Take a nice car to the dock buyer.", "You get paid right away for it."],
                   "sold at the docks", True, 5),
    "LeonidaTips": ("Street Races", 9, ["Find the race marker on the map and pull up in a fast car.",
                                        "Win the race and the prize money lands in your account."],
                    "the race win screen shows $10,000", False, 5),
    "SideHustle": ("Taxi Side Job", 9, ["Hop in a taxi and press the job button.",
                                        "Every passenger you drop off pays a small fare."], "fares", True, None),
    "TowTrucker": ("Street Race Wins", 6, ["Here is another driver winning the same kind of race.",
                                           "Cross the line first and the prize lands straight away."],
                   "win screen", True, 5),
    "EarlyCash": ("Collecting Lost Wallets", 4, ["Wallets lie around.", "Pick them up."], "", True, 5),
    "Wallets": ("Beach Wallets", 4, ["Wallets lie around the beach area, just walk over them.",
                                     "Pick them up for a little cash every single time."], "cash", True, 5),
}


@pytest.mark.slow
def test_guide_selection(monkeypatch, cfg, sandbox):
    seen = guide_world(monkeypatch, VIDS, ITEMS, clip_sec=14, gta5=(3,), too_old=("EarlyCash",))
    captured = {}

    def fake_render(moments, pkg, clip_id, mode="moments", outro_text=""):
        captured.update(moments=moments, pkg=pkg, mode=mode, outro=outro_text)
        return {"path": "/nonexistent.mp4"}
    monkeypatch.setattr(lv, "render_long", fake_render)
    made = lv.make_guide("money_guide", "GTA VI Top Moments", used=set(), state={})
    assert not made.get("error"), made
    good = captured["moments"]
    labels = [m["label"] for m in good]
    streamers = {m["streamer"] for m in good} | {x["streamer"] for m in good for x in m.get("extra", [])}

    assert "GlitchGod" not in seen.downloads and "EarlyCash" not in streamers, "glitch title / pre-launch upload"
    assert seen.downloads.count("CashKing") == 2, "max 2 videos per channel"
    assert "Car Delivery Jobs" not in labels, "GTA V footage is dropped"
    assert "Taxi Side Job" not in labels, "no money earned on screen = not a money method"
    assert "Beach Wallets" not in labels, "usefulness 4 is below guide_min_usefulness"
    assert labels == ["Store Robberies", "Selling Stolen Cars", "Street Races"], "countdown: weakest first, best last"
    top = good[-1]
    assert top["sources"] == 2 and [x["streamer"] for x in top["extra"]] == ["TowTrucker"], "twin = extra angle"
    robbery = good[0]
    assert not any("9,999" in ln for ln in robbery["lines"]), "made-up number dropped"
    assert captured["mode"] == "guide" and "WHICH ONE WILL YOU TRY" in captured["outro"]
    assert made["pkg"]["title"].startswith("GTA 6: Top 3 BEST"), "the title's number is the real item count"
    assert len(seen.review_prompts) == 2, "7 usable clips reviewed in batches of 6"
    assert "#1: Street Races" in seen.package_prompts[0]


def test_guide_with_too_little_footage_fails_cleanly(monkeypatch, cfg, sandbox):
    guide_world(monkeypatch, VIDS[:3], ITEMS, clip_sec=14)
    monkeypatch.setattr(lv, "render_long", lambda *a, **k: pytest.fail("must not render"))
    made = lv.make_guide("money_guide", "Brand", used=set(), state={})
    assert made["error"].startswith("only 2 usable gameplay videos")
    assert not list(sandbox.raw.glob("*.mp4")), "downloads are cleaned up"


# --------------------------------------------------------------------------- tiny real render

TINY_VIDS = [("LeonidaTips", "GTA 6 Money Making No Commentary", 600, 250000),
             ("MoneyGuru", "GTA 6 How To Make Money FAST", 600, 200000),
             ("CashKing", "GTA 6 Best Money Method", 400, 150000),
             ("TowTrucker", "GTA 6 tow truck job no commentary", 300, 85000)]
TINY_ITEMS = {
    "LeonidaTips": ("Street Races", 9, ["Pull up to the race marker in a fast car.",
                                        "Win and the prize lands right away."],
                    "the race win screen shows $10,000", True, 6),
    "TowTrucker": ("Street Race Wins", 6, ["Here is another driver winning a race.",
                                           "First across the line gets paid."], "win screen", True, 6),
    "MoneyGuru": ("Selling Stolen Cars", 8, ["Drive any sports car to the docks.",
                                            "The buyer pays $4,500 on the spot."],
                  "the HUD shows $4,500 after the sale", True, 6),
    "CashKing": ("Store Robberies", 6, ["Point your gun at the cashier.", "It pays 9,999 dollars every time.",
                                        "Then run before the cops arrive."], "the cash counter goes up", True, 6),
}


@pytest.mark.slow
def test_tiny_guide_render(monkeypatch, cfg, sandbox, fake_tts):
    # 12.5 s clips: the shortest footage the guide research accepts (12 s) - every second costs render time
    seen = guide_world(monkeypatch, TINY_VIDS, TINY_ITEMS, clip_sec=12.5, size="1280x720")
    cfg["long_video"].update(outro_sec=5.0, preset="ultrafast")
    beat = music.CACHE / "beat.mp3"
    beat.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg("-f", "lavfi", "-i", "sine=f=110:d=30,volume=4", "-ac", "2", str(beat))
    qa_seen = {}

    def ai_editor(path, context):
        qa_seen["context"] = context
        return {"publish": True, "score": 8, "issues": [], "summary": "clean", "pro_level": 7, "improvements": []}
    monkeypatch.setattr(qa, "line_check", lambda path, lines, kind: [])  # its own tests: tests/test_qa_linecheck.py
    monkeypatch.setattr(qa, "ai_review", ai_editor)
    technical = spy_technical(monkeypatch)
    import publisher
    monkeypatch.setattr(publisher, "youtube_api_upload", lambda *a, **k: pytest.fail("a dry run must never upload"))

    ok, rec = lv.run_long(dry_run=True)
    assert ok, rec
    assert rec["long_kind"] == "money_guide" and rec["status"] == "dry_run"

    video = Path(rec["rendered"])
    assert video.parent == sandbox.ready
    info = stream_info(video)
    assert (info["width"], info["height"]) == (1920, 1080)
    assert abs(info["v_dur"] - info["a_dur"]) <= cfg["qa"]["max_av_diff"], info
    assert info["v_dur"] == pytest.approx(rec["duration"], abs=0.2)

    assert len(technical) == 1 and technical[0]["path"] == video
    assert technical[0]["still_ok"][0] == pytest.approx((rec["duration"] - 5.0, rec["duration"]), abs=0.2), "outro card"
    high = [i for i in technical[0]["issues"] if i["severity"] == "high"]
    assert high == [], high
    assert rec["qa"].startswith("QA PASSED (long)") and "- [high]" not in rec["qa"]

    # countdown: 3 items, best last, the twin as an extra angle with its own credit
    assert [(m["n"], m["label"]) for m in rec["moments"]] == [(3, "Store Robberies"), (2, "Selling Stolen Cars"),
                                                              (1, "Street Races")]
    upload = next(sandbox.ready.glob("*_YOUTUBE_UPLOAD.txt")).read_text(encoding="utf-8")
    assert "#1 Street Races - gameplay by TowTrucker" in upload
    assert "#1 Street Races - gameplay by LeonidaTips" in upload
    assert upload.startswith("TITLE:\nGTA 6: Top 3 BEST Ways To Make Money")
    assert re.search(r"^0:\d\d #1 Street Races$", upload, re.M), "chapters in the description"

    # narration: the numbers the cards show, no made-up amounts, Fenrir only, one TTS request
    ctx = qa_seen["context"]
    assert "Number one. Street Races." in ctx and "Number three. Store Robberies." in ctx
    assert "9,999" not in ctx and "$4,500" in ctx
    assert len(fake_tts.prompts) == 1 and "9,999" not in fake_tts.prompts[0]
    assert set(fake_tts.voices) == {"Fenrir"}

    thumbs = sorted(p.name for p in sandbox.ready.glob("*_thumb*.jpg"))
    assert len(thumbs) == 3 and rec["thumbnail"].endswith("_thumb.jpg")
    assert (sandbox.ready / (video.stem + "_QA.txt")).exists()
    assert not lv.STATE_FILE.exists(), "dry runs never save long-video state"
    assert len(seen.review_prompts) == 1
