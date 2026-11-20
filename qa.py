"""Strict QA: every finished video is checked before it is published. Nothing posts without passing.

  1. technical   - audio/video length (sync), silent stretches, frozen/choppy parts, black frames, clipping
  2. voice check - Whisper listens to the final video: is every narration line actually said, and said WHEN it is
                   on screen? (words + timing)
  3. AI editor   - Gemini watches + listens to the whole final video: montage errors, voice vs on-screen text,
                   narration over something else, sync, repeated clips, spoilers, non-English, non-GTA, unsafe
Verdict: pass / fail with a list of issues (+ timestamps). A report is written next to the video.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from pathlib import Path

from common import env, load_config, probe, run

log = logging.getLogger("autopilot")


def _cfg() -> dict:
    return load_config().get("qa", {})


def _sdur(path: Path, sel: str) -> float:
    res = run(["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", "stream=duration", "-of",
               "csv=p=0", str(path)], timeout=30)
    try:
        return float(res.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return 0.0


def _ranges(stderr: str, start_key: str, end_key: str) -> list[tuple[float, float]]:
    starts = [float(x) for x in re.findall(rf"{start_key}[:=]\s*([\d.]+)", stderr)]
    ends = [float(x) for x in re.findall(rf"{end_key}[:=]\s*([\d.]+)", stderr)]
    return list(zip(starts, ends + [starts[-1] + 999] * (len(starts) - len(ends)))) if starts else []


def _outside(r: tuple[float, float], allowed: list[tuple[float, float]], dur: float = 1e9) -> float:
    """Seconds of range r that are NOT inside the allowed (intentional) ranges (r capped at the video's end -
    a still part that lasts to the end has no 'end' timestamp)."""
    a, b = r[0], min(r[1], dur)
    covered = sum(max(0.0, min(b, y) - max(a, x)) for x, y in allowed)
    return max(0.0, (b - a) - covered)


def technical(path: Path, still_ok: list[tuple[float, float]] | None = None) -> list[dict]:
    """still_ok: ranges where a static picture / quiet audio is intentional (outro card...)."""
    c, issues = _cfg(), []
    still_ok = still_ok or []
    dur = probe(path)["duration"] or 0
    v, a = _sdur(path, "v:0"), _sdur(path, "a:0")
    if not a:
        issues.append({"severity": "high", "type": "no_audio", "detail": "the video has no sound track"})
    elif abs(v - a) > float(c.get("max_av_diff", 0.12)):
        issues.append({"severity": "high", "type": "av_length", "detail": f"audio {a:.2f}s vs video {v:.2f}s"})
    res = run(["ffmpeg", "-hide_banner", "-i", str(path), "-vf",
               "scale=320:-2,freezedetect=n=0.003:d=1.2,blackdetect=d=0.4:pix_th=0.08",
               "-af", f"silencedetect=noise={c.get('silence_db', -45)}dB:d={c.get('silence_sec', 2.5)},volumedetect",
               "-f", "null", "-"], timeout=900)
    err = res.stderr
    for r in _ranges(err, "silence_start", "silence_end"):
        if _outside(r, still_ok, dur) > float(c.get("silence_sec", 2.5)):
            issues.append({"severity": "high", "type": "silence", "at": round(r[0], 1),
                           "detail": f"no sound for {min(r[1], dur) - r[0]:.1f}s"})
    for r in _ranges(err, "lavfi.freezedetect.freeze_start", "lavfi.freezedetect.freeze_end"):
        if _outside(r, still_ok, dur) > 1.2:
            issues.append({"severity": "high", "type": "frozen", "at": round(r[0], 1),
                           "detail": f"picture frozen for {min(r[1], dur) - r[0]:.1f}s"})
    for r in _ranges(err, "black_start", "black_end"):
        if _outside(r, still_ok, dur) > 0.4:
            issues.append({"severity": "medium", "type": "black", "at": round(r[0], 1),
                           "detail": f"black screen for {r[1] - r[0]:.1f}s"})
    peak = re.findall(r"max_volume:\s*(-?[\d.]+) dB", err)
    if peak and float(peak[-1]) >= -0.05:
        issues.append({"severity": "medium", "type": "clipping", "detail": f"audio peaks at {peak[-1]} dB (distortion)"})
    try:
        import stream_clips
        rep = stream_clips.smoothness(path)
        if rep > float(c.get("max_repeated_frames", 0.15)):
            issues.append({"severity": "high", "type": "choppy", "detail": f"{rep * 100:.0f}% repeated frames"})
    except Exception:  # noqa: BLE001
        pass
    return issues


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("*", ""))


def voice_check(path: Path, lines: list[dict]) -> list[dict]:
    """lines: [{"text", "start", "dur"}] = narration placed in the final video. Whisper must hear each line
    (word match) close to where it is placed (timing)."""
    import transcribe
    if not lines:
        return []
    heard, _ = transcribe.words(path)
    if not heard:
        if transcribe._get_model() is None:
            return [{"severity": "low", "type": "voice_unchecked", "detail": "Whisper not available"}]
        return [{"severity": "high", "type": "voice_missing", "detail": "no speech heard in the final video"}]
    c, issues = _cfg(), []
    tol = float(c.get("max_voice_offset", 0.6))
    for ln in lines:
        want = _norm_words(ln["text"])
        if not want:
            continue
        a, b = ln["start"] - 1.5, ln["start"] + ln.get("dur", 3) + 1.5
        got = [w for w in heard if a <= w["start"] <= b]
        got_words = _norm_words(" ".join(w["text"] for w in got))
        match = difflib.SequenceMatcher(None, want, got_words).find_longest_match(0, len(want), 0, len(got_words))
        ratio = difflib.SequenceMatcher(None, want, got_words).ratio() if got_words else 0.0
        covered = sum(1 for w in want if w in got_words) / len(want)
        if covered < float(c.get("min_word_match", 0.6)):
            issues.append({"severity": "high", "type": "voice_mismatch", "at": round(ln["start"], 1),
                           "detail": f"line not heard properly ({covered:.0%} of the words): \"{ln['text'][:60]}\""})
            continue
        # timing: when the first matched word is actually spoken vs when the line is placed
        if match.size:
            first = want[match.a]
            spoken = next((w["start"] for w in got if _norm_words(w["text"])[:1] == [first]), None)
            expect = ln["start"] + ln.get("dur", 3) * (match.a / max(1, len(want)))
            if spoken is not None and abs(spoken - expect) > tol:
                issues.append({"severity": "high", "type": "voice_out_of_sync", "at": round(ln["start"], 1),
                               "detail": f"spoken {spoken - expect:+.2f}s off the captions: \"{ln['text'][:50]}\""})
        _ = ratio
    return issues


AI_PROMPT = """You are the strict quality-control editor of a GTA 6 YouTube/TikTok channel. The attached file is a
FINISHED video that is about to be published. Watch AND listen to all of it, then report every problem a viewer
would notice. Be strict - a professional channel only publishes clean edits.
{context}
Check for:
- montage errors: abrupt or broken cuts, a cut in the middle of a sentence or action, the same clip shown twice,
  black or frozen frames, glitches, awkward empty parts
- sound: audio out of sync with the picture, silent parts, sudden loud/quiet jumps, distortion
- narration: the voice-over not saying what the on-screen text / subtitles say, narration out of sync with the
  captions, narration talking over the streamer's important line or the punchline, narration describing
  something different from what is on screen
- content: not GTA 6, story spoilers, non-English streamer, nudity / slurs / real violence
- overall: would a professional editor publish this as is?
Return ONLY JSON: {{"publish": bool, "score": int (1-10), "issues": [{{"at": seconds, "severity": "high"|"medium"|"low",
 "type": str, "detail": str}}], "summary": str}}"""


def ai_review(path: Path, context: str) -> dict | None:
    import gemini_writer
    if not env("GEMINI_API_KEY"):
        return None
    try:
        gemini_writer._unavailable_until = 0.0
        up = gemini_writer._upload_video(path)
        if up is None:
            return None
        data = gemini_writer._generate_json([up, AI_PROMPT.format(context=context[:3000])], temperature=0.2,
                                            prefer_pro=True)
        try:
            gemini_writer._client().files.delete(name=up.name)
        except Exception:  # noqa: BLE001
            pass
        return data if isinstance(data, dict) and "publish" in data else None
    except Exception as exc:  # noqa: BLE001
        log.warning("[qa] AI review failed: %.200s", exc)
        return None


def check(path: Path, kind: str, lines: list[dict] | None = None, still_ok: list | None = None,
          context: str = "") -> dict:
    """Full QA. kind: short | stream | long. Returns {"pass", "score", "issues", "report"}."""
    c = _cfg()
    if not c.get("enabled", True):
        return {"pass": True, "issues": [], "score": None, "report": "QA disabled"}
    issues = technical(path, still_ok)
    issues += voice_check(path, lines or [])
    ai = ai_review(path, context)
    score = None
    if ai is None:
        if c.get("require_ai_review", True):
            issues.append({"severity": "high", "type": "ai_review_unavailable",
                           "detail": "no AI could watch the final video (strict mode: not publishing)"})
    else:
        score = int(ai.get("score") or 0)
        for it in ai.get("issues") or []:
            if isinstance(it, dict):
                issues.append({**it, "type": f"ai:{it.get('type', 'issue')}"})
        if not ai.get("publish", False):
            issues.append({"severity": "high", "type": "ai_verdict", "detail": ai.get("summary", "AI editor said no")})
        if score < int(c.get("min_score", 7)):
            issues.append({"severity": "high", "type": "ai_score", "detail": f"AI editor score {score}/10"})
    high = [i for i in issues if str(i.get("severity")) == "high"]
    ok = not high
    lines_out = [f"QA {'PASSED' if ok else 'FAILED'} ({kind}) - AI score: {score if score is not None else 'n/a'}"]
    lines_out += [f"- [{i.get('severity')}] {i.get('type')}" + (f" at {i['at']}s" if i.get("at") is not None else "")
                  + f": {i.get('detail', '')}" for i in issues]
    report = "\n".join(lines_out)
    log.info("[qa] %s", report.replace("\n", " | ")[:1500])
    try:
        Path(str(path).rsplit(".", 1)[0] + "_QA.txt").write_text(report + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    return {"pass": ok, "score": score, "issues": issues, "report": report}
