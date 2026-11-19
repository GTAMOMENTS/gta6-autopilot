"""Whisper (open-source speech-to-text, runs locally on the GitHub runner - free, no key, no limits).

  words(path)       -> every spoken word with its exact start/end time (+ the detected language)
  smart_cut(...)    -> move cut points so a clip starts at the beginning of a sentence / a scene change and ends
                       after the punchline - never in the middle of a word

The model (default small.en, ~250 MB) is downloaded from Hugging Face on the first run and kept in the Actions
cache. If Whisper isn't available the callers fall back to what they did before.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from common import WORK_DIR, load_config, probe, run

log = logging.getLogger("autopilot")
_model = None
_failed = False

# subtitles are burned into the video: censor the worst words like f***
_SWEARS = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|nigg\w*|fag\w*|cunt\w*|dick|pussy|motherf\w*|retard\w*)\b", re.I)


def _cfg() -> dict:
    return load_config().get("whisper", {})


def censor(word: str) -> str:
    return _SWEARS.sub(lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), word)


def _get_model():
    global _model, _failed
    if _model is not None or _failed or not _cfg().get("enabled", True):
        return _model
    try:
        from faster_whisper import WhisperModel
        _model = WhisperModel(_cfg().get("model", "small.en"), device="cpu", compute_type="int8",
                              cpu_threads=int(_cfg().get("threads", 4)))
        log.info("[whisper] model %s loaded", _cfg().get("model", "small.en"))
    except Exception as exc:  # noqa: BLE001
        _failed = True
        log.warning("[whisper] not available (%.150s) - using the AI's timings instead", exc)
    return _model


def words(path: Path, start: float = 0.0, end: float | None = None) -> tuple[list[dict], dict]:
    """[{"text", "start", "end"}] relative to `start`, plus {"language", "language_probability"}."""
    model = _get_model()
    if model is None:
        return [], {}
    wav = WORK_DIR / "whisper_in.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.2f}"] + (["-to", f"{end:.2f}"] if end else []) + \
        ["-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(wav)]
    if run(cmd, timeout=120).returncode != 0 or not wav.exists():
        return [], {}
    try:
        multilingual = not str(_cfg().get("model", "small.en")).endswith(".en")
        segments, info = model.transcribe(str(wav), word_timestamps=True, vad_filter=True, beam_size=1,
                                          language=None if multilingual else "en")
        out = []
        for seg in segments:
            for w in seg.words or []:
                text = (w.word or "").strip()
                if text:
                    out.append({"text": text, "start": float(w.start), "end": float(w.end),
                                "prob": float(getattr(w, "probability", 1.0) or 1.0)})
        meta = {"language": getattr(info, "language", ""), "language_probability":
                float(getattr(info, "language_probability", 0) or 0)}
        return out, meta
    except Exception as exc:  # noqa: BLE001
        log.warning("[whisper] transcription failed: %.150s", exc)
        return [], {}
    finally:
        wav.unlink(missing_ok=True)


def subtitle_words(found: list[dict]) -> list[dict]:
    """Whisper words -> subtitle words (censored, low-confidence noise dropped)."""
    return [{"text": censor(w["text"]), "start": w["start"], "end": max(w["end"], w["start"] + 0.12)}
            for w in found if w.get("prob", 1) >= 0.25]


def scene_changes(path: Path, start: float, end: float, threshold: float = 0.35) -> list[float]:
    """Hard cuts / big camera changes between start and end (seconds of the file)."""
    res = run(["ffmpeg", "-hide_banner", "-ss", f"{max(0.0, start):.2f}", "-to", f"{end:.2f}", "-i", str(path),
               "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"], timeout=120)
    return [max(0.0, start) + float(t) for t in re.findall(r"pts_time:([\d.]+)", res.stderr)]


def smart_cut(path: Path, a: float, b: float, found: list[dict] | None = None, offset: float = 0.0,
              max_len: float = 50.0) -> tuple[float, float]:
    """Cut points that look edited by a human. `found` = Whisper words with times relative to `offset`."""
    dur = probe(path)["duration"] or b
    w = [{**x, "start": x["start"] + offset, "end": x["end"] + offset} for x in (found or [])]

    def sentence_starts():
        for i, x in enumerate(w):
            if i == 0 or x["start"] - w[i - 1]["end"] > 0.55 or re.search(r"[.!?]$", w[i - 1]["text"]):
                yield x["start"]

    def sentence_ends():
        for i, x in enumerate(w):
            if i == len(w) - 1 or w[i + 1]["start"] - x["end"] > 0.55 or re.search(r"[.!?]$", x["text"]):
                yield x["end"]

    new_a, new_b = a, b
    starts = [s for s in sentence_starts() if a - 2.5 <= s <= a + 0.4]
    if starts:
        new_a = max(0.0, max(starts) - 0.25)  # start right before a sentence
    else:
        inside = [x for x in w if x["start"] < a < x["end"]]
        if inside:
            new_a = max(0.0, inside[0]["start"] - 0.15)  # never start mid-word
        else:
            cuts = [t for t in scene_changes(path, a - 2.0, a + 0.6) if a - 2.0 <= t <= a + 0.6]
            if cuts:
                new_a = min(cuts, key=lambda t: abs(t - a))  # start on a scene change
    ends = [e for e in sentence_ends() if b - 0.4 <= e <= b + 3.0]
    if ends:
        new_b = min(dur, min(ends) + 0.45)  # let the punchline finish
    else:
        inside = [x for x in w if x["start"] < b < x["end"]]
        if inside:
            new_b = min(dur, inside[0]["end"] + 0.3)
        else:
            cuts = [t for t in scene_changes(path, b - 0.6, b + 2.0) if b - 0.6 <= t <= b + 2.0]
            if cuts:
                new_b = min(cuts, key=lambda t: abs(t - b))
    if new_b - new_a > max_len:
        new_b = new_a + max_len
    if new_b - new_a < 6:
        return a, b
    if (new_a, new_b) != (a, b):
        log.info("[cut] %.1f-%.1f -> %.1f-%.1f (sentence/scene aligned)", a, b, new_a, new_b)
    return round(new_a, 2), round(new_b, 2)
