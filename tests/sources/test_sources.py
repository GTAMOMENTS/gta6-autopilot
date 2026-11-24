"""Offline tests for footage sources, downloads and the AI-verdict cache (no YouTube / Dailymotion / Gemini needed).

Run from the repo root:  python -m pytest -q tests/sources   (or: python tests/sources/test_sources.py)
The download tests use a local throttled HTTP server + the real yt-dlp / ffmpeg (skipped if they are missing).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ai_cache  # noqa: E402
import authentic  # noqa: E402
import common  # noqa: E402
import source_fetcher as sf  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="srctest_"))
HAVE_TOOLS = bool(shutil.which("ffmpeg") and shutil.which("yt-dlp"))


def _isolate():
    """Fresh state files in a temp dir for every test."""
    d = Path(tempfile.mkdtemp(dir=TMP))
    ai_cache.CACHE_FILE = d / "ai_verdicts.json"
    ai_cache.reset()
    sf.MANIFEST_FILE = d / "manifest.json"
    sf.RAW_DIR = d / "raw"
    sf.RAW_DIR.mkdir()
    sf._INFO.clear()
    sf._yt_client[:] = []
    authentic._SOURCES.clear()
    return d


def _video(path: Path, sec: float = 12, size: str = "1280x720", rate: int = 30) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(sec), "-c:v", "libx264",
                    "-preset", "ultrafast", "-g", str(rate), "-c:a", "aac", "-movflags", "+faststart", str(path)],
                   check=True)
    return path


# --------------------------------------------------------------------------- format picking (Dailymotion)

def _fmt(fid, h=None, w=None, fps=None, v="avc1", a="mp4a", proto="m3u8_native", tbr=1000):
    return {"format_id": fid, "url": f"https://x/{fid}", "height": h, "width": w, "fps": fps, "vcodec": v,
            "acodec": a, "protocol": proto, "tbr": tbr}


def test_choose_format_variants():
    nores = {"formats": [_fmt("hls-380", v=None, a=None), _fmt("hls-720", v=None, a=None),
                         _fmt("hls-1080", v=None, a=None), _fmt("hls-2160", v=None, a=None)]}
    assert sf._choose_format(nores, 720) == "hls-1080"  # heights only in the ids
    at60 = {"formats": [_fmt("hls-1080@60", 1080, 1920, 60), _fmt("hls-720@60", 720, 1280, 60),
                        _fmt("hls-480", 480, 848, 30)]}
    assert sf._choose_format(at60, 720) == "hls-1080@60"  # 60 fps is fine when there is nothing else
    mixed = {"formats": [_fmt("hls-1080@60", 1080, 1920, 60), _fmt("hls-720", 720, 1280, 30)]}
    assert sf._choose_format(mixed, 720) == "hls-720"  # <=30 fps preferred
    split = {"formats": [_fmt("hls-aud", v="none", a="mp4a", tbr=128), _fmt("hls-1080", 1080, 1920, 25, a="none"),
                         _fmt("hls-720", 720, 1280, 25, a="none")]}
    assert sf._choose_format(split, 720) == "hls-1080+hls-aud"  # video-only + the audio track
    only4k = {"formats": [_fmt("hls-2160", 2160), _fmt("hls-1440", 1440), _fmt("hls-480", 480)]}
    assert sf._choose_format(only4k, 720) == "hls-1440"  # smallest above 1080 (downscaled later)
    low = {"formats": [_fmt("hls-380", 360), _fmt("hls-480", 480)]}
    assert sf._choose_format(low, 720) is None  # no HD at all
    sizeless = {"formats": [_fmt("mp4", v=None, a=None)]}
    assert sf._choose_format(sizeless, 720) == "bv*+ba/b"  # no sizes anywhere: best, resolution checked later
    prog = {"formats": [_fmt("hls-1080", 1080, 1920, 30), _fmt("http-1080", 1080, 1920, 30, proto="https")]}
    assert sf._choose_format(prog, 720) == "http-1080"  # progressive file: cheap seeking for sections


def test_dailymotion_download_uses_picked_format():
    _isolate()
    src = _video(TMP / "dm_src.mp4") if HAVE_TOOLS else None
    info = {"id": "xby15e6", "formats": [_fmt("hls-480", v=None, a=None), _fmt("hls-720", v=None, a=None),
                                         _fmt("hls-1080", v=None, a=None)]}
    calls = []
    orig_info, orig_run = sf._info, sf._run_ytdlp

    def fake_info(c, extra=None):
        f = sf.RAW_DIR / "info.json"
        f.write_text(json.dumps(info))
        return info, f

    def fake_run(cmd, timeout=300, section_len=0.0, keep_min=0.0):
        calls.append(cmd)
        out = Path(cmd[cmd.index("-o") + 1])
        if src:
            shutil.copy(src, out)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    sf._info, sf._run_ytdlp = fake_info, fake_run
    try:
        c = sf.Candidate(key="dm:xby15e6", url="https://www.dailymotion.com/video/xby15e6", title="GTA 6 gameplay",
                         uploader="someone", provider="dailymotion", duration=3000)
        ok = sf._download(c, 100, 220, sf.RAW_DIR / "sec.mp4")
        cmd = calls[0]
        assert cmd[cmd.index("-f") + 1] == "hls-1080", cmd
        assert "--load-info-json" in cmd and c.url not in cmd  # no second extraction
        assert "--download-sections" in cmd
        if src:
            assert ok
        info["formats"] = [_fmt("hls-380", 360), _fmt("hls-480", 480)]
        try:
            sf._download(c, 100, 220, sf.RAW_DIR / "sec2.mp4")
            raise AssertionError("expected FormatsLimited")
        except sf.FormatsLimited:
            pass
    finally:
        sf._info, sf._run_ytdlp = orig_info, orig_run


def test_whole_clip_is_a_normal_download():
    _isolate()
    calls = []
    orig = sf._run_ytdlp

    def fake_run(cmd, timeout=300, section_len=0.0, keep_min=0.0):
        calls.append((cmd, section_len))
        return subprocess.CompletedProcess(cmd, 1, "", "ERROR: boom")
    sf._run_ytdlp = fake_run
    try:
        c = sf.Candidate(key="twc:abc", url="https://clips.twitch.tv/abc", title="clip", uploader="KaiCenat",
                         provider="twitch_clip", duration=45, kind="stream")
        assert not sf._download(c, 0, 46, sf.RAW_DIR / "c.mp4")
        cmd, section_len = calls[0]
        assert "--download-sections" not in cmd and section_len == 0 and "--max-filesize" in cmd
        assert len(calls) == 1  # no pointless full-download retry
    finally:
        sf._run_ytdlp = orig


def test_youtube_bot_block_switches_client_for_the_run():
    _isolate()
    src = _video(TMP / "yt_src.mp4") if HAVE_TOOLS else None
    seen = []
    orig = sf._run_ytdlp

    def fake_run(cmd, timeout=300, section_len=0.0, keep_min=0.0):
        seen.append(cmd)
        if "youtube:player_client=tv,web_safari" in cmd and src:
            shutil.copy(src, Path(cmd[cmd.index("-o") + 1]))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", "ERROR: Sign in to confirm you're not a bot")
    sf._run_ytdlp = fake_run
    try:
        c = sf.Candidate(key="yt:a", url="https://www.youtube.com/watch?v=a", title="GTA 6", uploader="ch",
                         provider="youtube", duration=900)
        if src:
            assert sf._download(c, 10, 130, sf.RAW_DIR / "y.mp4")
            assert sf._yt_client and "youtube:player_client=tv,web_safari" in sf._yt_client
            n = len(seen)
            sf._download(c, 200, 320, sf.RAW_DIR / "y2.mp4")
            assert "youtube:player_client=tv,web_safari" in seen[n]  # used straight away for later videos
    finally:
        sf._run_ytdlp = orig


# --------------------------------------------------------------------------- real yt-dlp: sections, salvage

class _Slow(SimpleHTTPRequestHandler):
    rate = 400_000

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.translate_path(self.path.split("?")[0])
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start, end = int(a or 0), int(b) if b else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    data = f.read(min(self.rate // 10, left))
                    if not data:
                        break
                    self.wfile.write(data)
                    left -= len(data)
                    time.sleep(len(data) / self.rate)
            except (BrokenPipeError, ConnectionResetError):
                pass


def _serve(root: Path, rate: int):
    # the shared conftest points every proxy at a dead port (offline tests); the local server is reached directly
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = "127.0.0.1,localhost"
    handler = type("H", (_Slow,), {"rate": rate})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: handler(*a, directory=str(root), **k))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# real throttled downloads against a local server: they wait out real timeouts (~40 s each) -> slow
@pytest.mark.slow
def test_section_from_info_json_and_slow_salvage():
    if not HAVE_TOOLS:
        print("  (skipped: no ffmpeg / yt-dlp)")
        return
    d = _isolate()
    www = d / "www"
    www.mkdir()
    # ~2.6 Mbit/s 720p video, served at ~2.4 Mbit/s -> ffmpeg gets about 0.9x real time
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-f", "lavfi",
                    "-i", "sine=frequency=440:sample_rate=48000", "-t", "150", "-c:v", "libx264", "-preset",
                    "ultrafast", "-b:v", "2500k", "-g", "60", "-c:a", "aac", "-movflags", "+faststart",
                    str(www / "v.mp4")], check=True)
    srv, base = _serve(www, 300_000)
    try:
        c = sf.Candidate(key="url:v", url=f"{base}/v.mp4", title="GTA 6 gameplay", uploader="", provider="extra",
                         duration=150)
        info, f = sf._info(c)
        assert info and f.exists()
        # normal section download, reusing the info JSON (no new extraction), with a hard 25 s limit:
        # the ~0.9x stream can't finish 60 s in time -> stopped gracefully, the part that arrived is kept
        cfg = common.load_config()
        old = cfg["sources"].get("salvage_min_sec")
        cfg["sources"]["salvage_min_sec"] = 10
        try:
            t0 = time.time()
            ok = sf._download(c, 30, 90, sf.RAW_DIR / "s.mp4", timeout=25)
            took = time.time() - t0
        finally:
            cfg["sources"]["salvage_min_sec"] = old
        dur = common.probe(sf.RAW_DIR / "s.mp4")["duration"] if ok else 0
        print(f"  slow section: ok={ok} kept {dur:.1f}s in {took:.0f}s")
        assert ok and 10 <= dur < 60 and took < 60
        # no ffmpeg / yt-dlp left running in the background
        time.sleep(1)
        ps = subprocess.run(["pgrep", "-f", f"{base}/v.mp4"], capture_output=True, text=True).stdout.strip()
        assert not ps, f"orphan downloader: {ps}"
    finally:
        srv.shutdown()


# real throttled downloads against a local server: they wait out real timeouts (~40 s each) -> slow
@pytest.mark.slow
def test_run_ytdlp_gives_up_early_when_hopeless():
    if not HAVE_TOOLS:
        return
    d = _isolate()
    www = d / "www"
    www.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30", "-t", "120",
                    "-c:v", "libx264", "-preset", "ultrafast", "-b:v", "4000k", "-g", "60", "-movflags", "+faststart",
                    str(www / "v.mp4")], check=True)
    srv, base = _serve(www, 100_000)  # ~0.2x real time
    try:
        out = sf.RAW_DIR / "h.mp4"
        t0 = time.time()
        res = sf._run_ytdlp(["yt-dlp", "--no-warnings", "-f", "b", "--download-sections", "*10-100", "-o", str(out),
                             f"{base}/v.mp4"], timeout=200, section_len=90, keep_min=45)
        took = time.time() - t0
        print(f"  hopeless section: rc={res.returncode} after {took:.0f}s: {res.stderr.strip().splitlines()[-1]}")
        assert res.returncode == 124 and "too slow" in res.stderr and took < 60
    finally:
        srv.shutdown()


# --------------------------------------------------------------------------- AI verdict cache

def test_authenticity_cache_window_and_errors():
    _isolate()
    asked = []

    def fake_ask(paths):
        asked.append([p.name for p in paths])
        out = []
        for p in paths:
            if "noanswer" in p.name:
                out.append(None)  # the AI skipped this clip
            elif "fake" in p.name:
                out.append({"game": "GTA V", "confidence": 0.95, "evidence": "Vinewood sign"})
            else:
                out.append({"game": "GTA VI", "confidence": 0.9, "evidence": "Vice City beach"})
        return out
    orig_ask, orig_env = authentic._ask, authentic.env
    authentic._ask, authentic.env = fake_ask, (lambda n, d="": "x" if n == "GEMINI_API_KEY" else d)
    try:
        p1, p2, p3 = TMP / "real.mp4", TMP / "fake.mp4", TMP / "noanswer.mp4"
        meta = [{"key": "yt:r", "channel": "GoodChannel", "provider": "youtube", "section": [100, 210]},
                {"key": "yt:f", "channel": "ModChannel", "provider": "youtube", "section": [50, 160]},
                {"key": "yt:n", "channel": "Other", "provider": "youtube", "section": [0, 60]}]
        r = authentic.check([p1, p2, p3], meta)
        assert [x["ok"] for x in r] == [True, False, False] and r[2]["error"] and not r[0]["cached"]
        assert set(r[0]) >= {"game", "confidence", "evidence", "ok"}  # the shape _final_real_check expects
        assert len(asked) == 1
        # same sources + same windows: answered from memory; the clip without an answer is asked again
        r2 = authentic.check([p1, p2, p3], meta)
        assert r2[0]["cached"] and r2[1]["cached"] and not r2[2]["cached"]
        assert asked[-1] == ["noanswer.mp4"]
        # another window of the "real" video: NOT reused (compilations mix games)
        n = len(asked)
        authentic.check([p1], [{**meta[0], "section": [400, 510]}])
        assert len(asked) == n + 1
        # the fake video is never used again, whatever window
        assert authentic.known_fake("yt:f") and not authentic.known_fake("yt:r")
        # a prompt / facts change invalidates
        orig_pv = authentic.prompt_version
        authentic.check([p1], [{**meta[0], "key": "yt:pv"}])
        authentic.prompt_version = lambda: "other"
        try:
            n = len(asked)
            authentic.check([p1], [{**meta[0], "key": "yt:pv"}])
            assert len(asked) == n + 1
        finally:
            authentic.prompt_version = orig_pv
        # unregistered paths (e.g. _final_real_check's windows) are always checked fresh, same return shape
        n = len(asked)
        r3 = authentic.check([TMP / "realcheck_0.mp4"])
        assert len(asked) == n + 1 and r3[0]["ok"] and isinstance(r3, list)
        # remember() registers a downloaded file for later check([path]) calls (Shorts path in autopilot.py)
        authentic.remember(TMP / "section_yt_r.mp4", "yt:r", "GoodChannel", "youtube", (100, 210))
        n = len(asked)
        assert authentic.check([TMP / "section_yt_r.mp4"])[0]["cached"] and len(asked) == n
        # no AI at all -> None (unchanged API)
        authentic._ask = lambda paths: None
        assert authentic.check([TMP / "x_new.mp4"]) is None
    finally:
        authentic._ask, authentic.env = orig_ask, orig_env
    # persisted across runs
    ai_cache.reset()
    assert ai_cache.auth_fake("yt:f")


def test_channel_memory_and_rank():
    _isolate()
    pv = authentic.prompt_version()
    for i in range(2):
        ai_cache.auth_put(f"yt:m{i}", pv, {"game": "mod", "confidence": 0.9}, channel="ModMaker", provider="youtube")
    ai_cache.auth_put("yt:g0", pv, {"game": "GTA VI", "confidence": 0.9}, channel="Real Gamer", provider="youtube")
    ai_cache.auth_put("yt:u0", pv, {"game": "unsure", "confidence": 0.2}, channel="Unknown", provider="youtube")
    # Twitch streamers are not judged as channels (category clips)
    ai_cache.auth_put("twc:x", pv, {"game": "GTA V", "confidence": 0.9}, channel="KaiCenat", provider="twitch_clip")
    assert authentic.channel_verdict("ModMaker")[0] == "fake"
    assert authentic.channel_verdict("Real Gamer")[0] == "trusted"
    assert authentic.channel_verdict("Unknown") == ("", 0.0)
    assert authentic.channel_verdict("KaiCenat") == ("", 0.0)
    cfg = common.load_config()
    tf = sf.TitleFilter(cfg)
    now = time.time()
    cands = [
        sf.Candidate("yt:new1", "u1", "GTA 6 free roam gameplay", "ModMaker", "youtube", 900, now - 3600),
        sf.Candidate("yt:new2", "u2", "GTA 6 free roam gameplay", "Real Gamer", "youtube", 900, now - 3600),
        sf.Candidate("yt:new3", "u3", "GTA 6 free roam gameplay", "Neutral", "youtube", 900, now - 3600),
        sf.Candidate("yt:old", "u4", "GTA 6 free roam gameplay", "Neutral", "youtube", 900,
                     sf._launch_ts(cfg) - 86400),
        sf.Candidate("yt:m0", "u5", "GTA 6 chase", "Someone", "youtube", 900, now - 3600),
        sf.Candidate("yt:strip", "u6", "GTA 6 strip club funny moments", "Neutral", "youtube", 900, now - 3600),
    ]
    ranked = sf.rank(cands, {"sources": {}}, tf)
    keys = [c.key for c in ranked]
    assert "yt:new1" not in keys and "yt:old" not in keys and "yt:m0" not in keys, keys
    assert keys.index("yt:new2") < keys.index("yt:strip")
    sc = {c.key: c.score for c in ranked}
    assert sc["yt:new2"] > sc["yt:new3"] - 2.0 + 2.5 - 0.01  # trusted bonus (random part is 0-2)


def test_youtube_search_drops_clearly_old_results():
    _isolate()
    cfg = common.load_config()
    launch = sf._launch_ts(cfg)
    entries = [{"id": "new", "title": "GTA 6 free roam gameplay", "duration": 900, "timestamp": time.time() - 86400},
               {"id": "old", "title": "GTA 6 free roam gameplay", "duration": 900, "timestamp": launch - 20 * 86400},
               {"id": "edge", "title": "GTA 6 free roam gameplay", "duration": 900, "timestamp": launch - 86400},
               {"id": "nodate", "title": "GTA 6 free roam gameplay", "duration": 900}]
    seen = []
    orig = sf.run

    def fake_run(cmd, timeout=300, check=False):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"entries": entries}), "")
    sf.run = fake_run
    try:
        out = sf.discover_youtube(cfg, ["GTA 6 gameplay", "GTA 6 free roam"], sf.TitleFilter(cfg))
    finally:
        sf.run = orig
    ids = sorted(c.key for c in out)
    # 'edge' (approx. 1 day before launch, within the margin) and undated ones go on to the exact date check
    assert ids == ["yt:edge", "yt:new", "yt:nodate"], ids
    assert all("youtubetab:approximate_date" in c for c in seen) and len(seen) == 2


def test_fill_metadata_rejects_undated_and_old():
    _isolate()
    cfg = common.load_config()
    orig = sf._info
    try:
        for meta, want in (({"duration": 600, "upload_date": "20260101"}, False), ({"duration": 600}, False),
                           ({"duration": 600, "timestamp": time.time() - 3600}, True)):
            sf._info = lambda c, extra=None, m=meta: (m, None)
            c = sf.Candidate("yt:z", "u", "GTA 6", "ch", "youtube", 0)
            assert sf._fill_metadata(c, cfg) is want, meta
    finally:
        sf._info = orig


def test_review_cache_in_batches():
    _isolate()
    import gemini_writer
    import long_video as lv
    if not HAVE_TOOLS:
        return
    clips = []
    for i in range(3):
        p = _video(TMP / f"rv{i}.mp4", sec=4, size="640x360")
        clips.append({"key": f"twitch:c{i}", "path": p, "streamer": "KaiCenat", "platform": "twitch",
                      "title": f"clip {i}", "dur": 4.0})
    calls = {"gen": 0}
    orig = (gemini_writer._upload_video, gemini_writer._generate_json, gemini_writer._client, lv.LONG_DIR)

    class _Files:
        def delete(self, name):
            pass

    class _Cl:
        files = _Files()

    def fake_gen(contents, temperature=0.9, prefer_pro=False):
        calls["gen"] += 1
        n = contents[1].count(" s (")
        return {"clips": [{"clip": i, "is_gta6_footage": i != 2, "hype_score": 7} for i in range(1, n + 1)]}
    gemini_writer._upload_video = lambda p: type("F", (), {"name": "files/x"})()
    gemini_writer._generate_json = fake_gen
    gemini_writer._client = lambda: _Cl()
    lv.LONG_DIR = TMP / "long"
    try:
        prompt = lambda k, ranges: f"review {k} clips\n{ranges}\nfacts: {lv._game_facts()}"  # noqa: E731
        got = [pair for _, pairs in lv._batch_verdicts(clips, prompt) for pair in pairs]
        assert calls["gen"] == 1 and all(v for _, v in got)
        assert ai_cache.bad_reason("twitch:c1") == "not GTA 6" and not ai_cache.bad_reason("twitch:c0")
        ai_cache.reset()  # next run
        got2 = [pair for _, pairs in lv._batch_verdicts(clips, prompt) for pair in pairs]
        assert calls["gen"] == 1, "cached verdicts must not call the AI"
        assert [v.get("hype_score") for _, v in got2] == [7, 7, 7]
        # a different prompt (e.g. another guide subject) is asked again
        list(lv._batch_verdicts(clips, lambda k, r: "other prompt " + r))
        assert calls["gen"] == 2
    finally:
        gemini_writer._upload_video, gemini_writer._generate_json, gemini_writer._client, lv.LONG_DIR = orig


def test_per_minute_limit_does_not_kill_model():
    import gemini_writer

    class _Models:
        def __init__(self, msg):
            self.msg = msg

        def generate_content(self, **k):
            raise RuntimeError(self.msg)

    orig = gemini_writer._client
    try:
        gemini_writer._dead_models.discard("m1")
        gemini_writer._client = lambda: type("C", (), {"models": _Models(
            "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 12.5s")})()
        from google.genai import types
        data, busy = gemini_writer._try_model(types, "m1", ["x"], 0.1)
        assert data is None and busy and "m1" not in gemini_writer._dead_models
        assert 10 < gemini_writer._busy_until["m1"] - time.time() <= 16
        gemini_writer._client = lambda: type("C", (), {"models": _Models(
            "429 RESOURCE_EXHAUSTED You exceeded your current quota. quotaId: GenerateRequestsPerDayPerProjectPerModel")})()
        data, busy = gemini_writer._try_model(types, "m1", ["x"], 0.1)
        assert "m1" in gemini_writer._dead_models  # daily quota gone: skip it for the rest of the run
    finally:
        gemini_writer._client = orig
        gemini_writer._dead_models.discard("m1")


def test_iter_clips_remembers_dailymotion_without_hd():
    _isolate()
    cands = [sf.Candidate(f"dm:{i}", f"https://www.dailymotion.com/video/{i}", "GTA 6 gameplay", "u",
                          "dailymotion", 900) for i in range(4)]
    tried = []
    orig = (sf.discover, sf._download)
    sf.discover = lambda cfg, manifest, fmt=None: cands

    def fake_dl(c, a, b, out, timeout=300):
        tried.append(c.key)
        raise sf.FormatsLimited("hls-380:360p video+audio")
    sf._download = fake_dl
    try:
        assert list(sf.iter_clips(0, max_attempts=4)) == []
    finally:
        sf.discover, sf._download = orig
    assert tried == ["dm:0", "dm:1"], tried  # after 2 such videos Dailymotion is skipped for the run
    st = sf.load_manifest()["sources"]
    assert st["dm:0"]["fail_count"] >= 2  # never tried again


def test_tour_stops_after_a_fake_first_slice():
    """A run spent 9 min downloading 3 slices of a video the GTA 6 check then called GTA V (99%): the first slice
    is checked before the others are fetched. Only a confident 'fake' stops it - unsure keeps downloading."""
    _isolate()
    cands = [sf.Candidate(f"dm:{k}", f"https://www.dailymotion.com/video/{k}", "GTA 6 free roam", "u", "dailymotion",
                          900) for k in ("fake", "unsure")]
    verdicts = {"dm:fake": {"game": "GTA V", "confidence": 0.99, "evidence": "Vinewood sign"},
                "dm:unsure": {"game": "unsure", "confidence": 0.3, "evidence": "dark tunnel"}}
    fetched, checked = [], []

    def fake_dl(c, a, b, out, timeout=300):
        fetched.append(c.key)
        Path(out).write_bytes(b"not a real video")
        return True

    def fake_check(paths, meta=None):
        checked.append((meta[0]["key"], tuple(meta[0]["section"])))
        v = verdicts[meta[0]["key"]]
        return [authentic._result(v["game"], v["confidence"], v["evidence"], 0.65)]
    orig = (sf.discover, sf._download, authentic.check)
    sf.discover, sf._download, authentic.check = (lambda cfg, manifest, fmt=None: cands), fake_dl, fake_check
    try:
        assert list(sf.iter_clips(0, max_attempts=2)) == []  # the "unsure" reel is garbage bytes: can't be joined
    finally:
        sf.discover, sf._download, authentic.check = orig
    assert fetched == ["dm:fake", "dm:unsure", "dm:unsure", "dm:unsure"], fetched
    assert [k for k, _ in checked] == ["dm:fake", "dm:unsure"] and all(b - a <= 110 for _, (a, b) in checked)
    st = sf.load_manifest()["sources"]
    assert st["dm:fake"]["reject_reason"] == "not GTA VI footage" and "reject_reason" not in st["dm:unsure"]
    assert not list(sf.RAW_DIR.glob("slice*")), "slice files are cleaned up"


if __name__ == "__main__":
    common.setup_logging()
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and all(a in name for a in sys.argv[1:]):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s)")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {exc}")
    print("ALL PASSED" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
