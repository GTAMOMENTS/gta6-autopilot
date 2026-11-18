"""Shared paths, config/state helpers and small utilities used by every module."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.json"

# Persistent state (restored/saved by actions/cache between runs)
STATE_DIR = BASE_DIR / "queue"
MANIFEST_FILE = STATE_DIR / "clips_manifest.json"
HISTORY_FILE = STATE_DIR / "posted_history.json"

# Scratch work dirs (wiped every run on a fresh runner)
WORK_DIR = BASE_DIR / "work"
RAW_DIR = WORK_DIR / "raw"
VO_DIR = WORK_DIR / "voice"
READY_DIR = WORK_DIR / "ready"
LOCAL_CLIPS_DIR = BASE_DIR / "clips"

log = logging.getLogger("autopilot")


def setup_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # The google/httpx libs are very chatty at INFO.
    for noisy in ("httpx", "httpcore", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        log.warning("Corrupt JSON in %s - starting fresh", path)
        return default


def save_json(path: Path, data) -> None:
    """Atomic write so a crash mid-write never corrupts the cached state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


_CONFIG_CACHE: dict | None = None


def load_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_json(CONFIG_FILE, {})
    return _CONFIG_CACHE


def tz() -> ZoneInfo:
    return ZoneInfo(os.getenv("TIMEZONE") or load_config().get("timezone", "Europe/Istanbul"))


def now_local() -> datetime:
    return datetime.now(tz())


def env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def run(cmd: list[str], timeout: int = 300, check: bool = False) -> subprocess.CompletedProcess:
    log.debug("RUN %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({res.returncode}): {res.stderr[-800:]}")
    return res


def probe(path: Path) -> dict:
    """Return duration/width/height/has_audio for a media file."""
    res = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "format=duration:stream=codec_type,width,height",
        str(path),
    ], timeout=60)
    info = {"duration": 0.0, "width": 0, "height": 0, "has_audio": False}
    try:
        data = json.loads(res.stdout or "{}")
    except json.JSONDecodeError:
        return info
    info["duration"] = float(data.get("format", {}).get("duration") or 0.0)
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and not info["width"]:
            info["width"], info["height"] = int(s.get("width") or 0), int(s.get("height") or 0)
        elif s.get("codec_type") == "audio":
            info["has_audio"] = True
    return info


def write_summary(markdown: str) -> None:
    """Append to the GitHub Actions job summary (no-op locally)."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")
