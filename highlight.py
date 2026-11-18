"""Find the most action-packed part of a downloaded gameplay section.

Every second of the section gets an "energy" score built from:
  * motion  - FFmpeg scene-change score (fast cuts, explosions, chases)
  * loudness - EBU R128 momentary loudness (gunfire, engines, music)
  * a penalty for near-black / frozen frames (loading screens, menus, fades)

Then we either keep the single best contiguous window, or stitch the best
2-3 non-overlapping windows into a quick-cut montage, whichever scores higher.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from common import WORK_DIR, probe, run

log = logging.getLogger("autopilot")


def _per_second(values: list[tuple[float, float]], seconds: int, default: float) -> list[float]:
    buckets: list[list[float]] = [[] for _ in range(seconds)]
    for t, v in values:
        i = int(t)
        if 0 <= i < seconds:
            buckets[i].append(v)
    return [sum(b) / len(b) if b else default for b in buckets]


def _parse_metadata(path: Path, key: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not path.exists():
        return out
    t = None
    for line in path.read_text(errors="ignore").splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            t = float(m.group(1))
            continue
        if line.startswith(key + "=") and t is not None:
            try:
                out.append((t, float(line.split("=", 1)[1])))
            except ValueError:
                pass
    return out


def _rank_normalize(xs: list[float]) -> list[float]:
    """Map values to 0..1 by rank, robust to outliers and different sources."""
    if not xs:
        return xs
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    for rank, i in enumerate(order):
        out[i] = rank / max(1, len(xs) - 1)
    return out


def energy_curve(video: Path) -> list[float]:
    info = probe(video)
    seconds = int(info["duration"])
    if seconds <= 0:
        return []
    tmp = WORK_DIR / "analysis"
    tmp.mkdir(parents=True, exist_ok=True)
    scene_f, luma_f, loud_f = tmp / "scene.txt", tmp / "luma.txt", tmp / "loud.txt"
    for f in (scene_f, luma_f, loud_f):
        f.unlink(missing_ok=True)

    vf = (
        "fps=4,scale=192:-2,"
        f"select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file={scene_f},"
        f"signalstats,metadata=print:key=lavfi.signalstats.YAVG:file={luma_f}"
    )
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-vf", vf]
    if info["has_audio"]:
        cmd += ["-af", f"ebur128=metadata=1,ametadata=print:key=lavfi.r128.M:file={loud_f}"]
    else:
        cmd += ["-an"]
    cmd += ["-f", "null", "-"]
    res = run(cmd, timeout=600)
    if res.returncode != 0:
        log.warning("[highlight] analysis failed: %s", res.stderr[-300:])
        return [0.5] * seconds

    motion = _per_second(_parse_metadata(scene_f, "lavfi.scene_score"), seconds, 0.0)
    luma = _per_second(_parse_metadata(luma_f, "lavfi.signalstats.YAVG"), seconds, 60.0)
    loud = _per_second(
        [(t, max(-70.0, v)) for t, v in _parse_metadata(loud_f, "lavfi.r128.M")], seconds, -70.0
    )

    m_n, l_n = _rank_normalize(motion), _rank_normalize(loud)
    energy = []
    for i in range(seconds):
        e = 0.6 * m_n[i] + 0.4 * l_n[i]
        if luma[i] < 18:           # black / fade / loading screen
            e *= 0.1
        if motion[i] < 0.002:      # frozen frame, pause menu
            e *= 0.5
        energy.append(e)
    return energy


def _best_window(energy: list[float], length: int, banned: list[tuple[int, int]]) -> tuple[int, float]:
    best_start, best_score = -1, -1.0
    for s in range(0, len(energy) - length + 1):
        if any(s < b_end and s + length > b_start for b_start, b_end in banned):
            continue
        score = sum(energy[s:s + length]) / length
        if score > best_score:
            best_start, best_score = s, score
    return best_start, best_score


def choose_segments(energy: list[float], target: int, style: str = "auto") -> list[tuple[int, int]]:
    """Return [(start, end), ...] in seconds, chronologically ordered."""
    n = len(energy)
    if n <= target + 1:
        return [(0, n)]

    single_start, single_score = _best_window(energy, target, [])
    single = [(single_start, single_start + target)]
    if style == "single":
        return single

    pieces = 3 if n >= target * 2 else 2
    seg_len = target // pieces
    chosen: list[tuple[int, int]] = []
    total = 0.0
    for _ in range(pieces):
        banned = [(a - 2, b + 2) for a, b in chosen]
        s, sc = _best_window(energy, seg_len, banned)
        if s < 0:
            break
        chosen.append((s, s + seg_len))
        total += sc
    if len(chosen) < pieces:
        return single
    montage_score = total / pieces
    chosen.sort()

    if style == "montage":
        return chosen
    # auto: a montage only wins if it is clearly more intense than one continuous shot
    # (continuous shots keep cutscenes / story moments intact).
    return chosen if montage_score > single_score * 1.12 else single


def energy_hints(energy: list[float], length: int = 30, count: int = 3) -> list[tuple[int, int]]:
    """Top non-overlapping high-energy windows - given to the AI editor as hints."""
    out: list[tuple[int, int]] = []
    if len(energy) <= length:
        return [(0, len(energy))] if energy else []
    for _ in range(count):
        s, sc = _best_window(energy, length, [(a - 5, b + 5) for a, b in out])
        if s < 0:
            break
        out.append((s, s + length))
    return sorted(out)


def best_window(energy: list[float], length: int) -> tuple[int, int]:
    n = len(energy)
    if n <= length:
        return 0, n
    s, _ = _best_window(energy, length, [])
    return s, s + length


def cut_segments(src: Path, out: Path, segments: list[tuple[float, float]]) -> Path | None:
    """Cut [(start, end), ...] from src into one normalised <=1080p / 30fps clip with stereo audio."""
    info = probe(src)
    parts, labels = [], ""
    for i, (a, b) in enumerate(segments):
        parts.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS,"
                     f"scale=-2:'min(1080,ih)',fps=30,format=yuv420p,setsar=1[v{i}]")
        if info["has_audio"]:
            parts.append(f"[0:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS,aresample=48000,"
                         f"aformat=channel_layouts=stereo[a{i}]")
        else:
            parts.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{b - a}[a{i}]")
        labels += f"[v{i}][a{i}]"
    if len(segments) > 1:
        parts = [p.replace("scale=-2:'min(1080,ih)'", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                           "pad=1920:1080:(ow-iw)/2:(oh-ih)/2") for p in parts]
    fc = ";".join(parts) + f";{labels}concat=n={len(segments)}:v=1:a=1[vo][ao]"
    res = run(["ffmpeg", "-y", "-hide_banner", "-i", str(src), "-filter_complex", fc,
               "-map", "[vo]", "-map", "[ao]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-c:a", "aac", "-b:a", "192k", str(out)], timeout=600)
    if res.returncode != 0 or not out.exists():
        log.error("[highlight] cut failed: %s", res.stderr[-500:])
        return None
    return out


def cut_highlight(src: Path, out: Path, target_sec: int, style: str = "auto") -> tuple[Path | None, list]:
    """Meter-only fallback: cut the best `target_sec` seconds from `src`."""
    info = probe(src)
    if info["duration"] < 15:
        return None, []
    energy = energy_curve(src)
    segments = choose_segments(energy, target_sec, style) if energy else [(0, int(info["duration"]))]
    log.info("[highlight] %s -> segments %s", src.name, segments)
    return cut_segments(src, out, segments), segments


def concat_files(files: list[Path], out: Path) -> Path | None:
    """Join several downloaded slices into one normalised 1080p/30fps 'reel' (16:9, padded)."""
    inputs, parts, labels = [], [], ""
    for i, f in enumerate(files):
        inputs += ["-i", str(f)]
        info = probe(f)
        parts.append(f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
                     f"fps=30,format=yuv420p,setsar=1[v{i}]")
        if info["has_audio"]:
            parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
        else:
            parts.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{info['duration']:.2f}[a{i}]")
        labels += f"[v{i}][a{i}]"
    fc = ";".join(parts) + f";{labels}concat=n={len(files)}:v=1:a=1[vo][ao]"
    res = run(["ffmpeg", "-y", "-hide_banner", *inputs, "-filter_complex", fc, "-map", "[vo]", "-map", "[ao]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(out)],
              timeout=900)
    if res.returncode != 0 or not out.exists():
        log.error("[highlight] reel concat failed: %s", res.stderr[-400:])
        return None
    return out
