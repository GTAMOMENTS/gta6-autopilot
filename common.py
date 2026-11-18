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
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # never let a slow download/encode crash the whole run - report it as a normal failure
        log.warning("%s timed out after %ss", cmd[0], timeout)
        res = subprocess.CompletedProcess(cmd, 124, stdout="", stderr=f"TIMEOUT: {cmd[0]} took longer than {timeout}s")
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


def parse_json_loose(text: str) -> dict | None:
    """Parse an AI answer into a dict, repairing the usual mistakes:
    code fences, <think> blocks, chatter around the JSON, trailing commas,
    and unquoted string values like  "scene_summary": A man walks ...  """
    import re
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    end = text.rfind("}")
    body = text[start:end + 1] if end > start else text[start:]
    for candidate in (body, re.sub(r",\s*([}\]])", r"\1", body)):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Key-by-key salvage: only TOP-LEVEL keys (depth 1), so nested lists like
    # "commentary": [{"t": 0, "line": ...}] stay together even when the JSON is broken.
    keys = []
    depth, in_str, esc = 0, False, False
    prev = "{"  # last significant character outside strings
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                prev = '"'
            continue
        if ch == '"':
            # a quote only OPENS a string where a key/value can start; a stray closing quote after
            # an unquoted value (e.g.  "summary": A man runs.", "next": ...) is just ignored
            if prev in "{[,:":
                if depth == 1 and prev in "{,":
                    m = re.match(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*', body[i:])
                    if m:
                        keys.append((m.group(1), i, i + m.end()))
                in_str = True
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if not ch.isspace():
            prev = ch
    if not keys:
        return None
    out: dict = {}
    for n, (name, _, vstart) in enumerate(keys):
        stop = keys[n + 1][1] if n + 1 < len(keys) else len(body)
        raw = body[vstart:stop].strip().rstrip(",").strip()
        if n + 1 == len(keys):
            raw = raw[:-1].strip().rstrip(",").strip() if raw.endswith("}") else raw
        low = raw.lower()
        if low in ("true", "false"):
            val = low == "true"
        elif re.fullmatch(r"-?\d+(\.\d+)?", raw):
            val = float(raw) if "." in raw else int(raw)
        elif raw.startswith("["):
            try:
                val = json.loads(raw)
            except json.JSONDecodeError:
                val = _salvage_list(raw)
        else:
            if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
                raw = raw[1:-1]
            val = raw.strip().strip('"').replace('\\"', '"')
        out[name] = val
    return out or None


def _salvage_list(raw: str) -> list:
    """Repair a broken JSON list: list of {t, line} objects, or list of strings (hashtags)."""
    import re
    items = []
    for obj in re.findall(r"\{(.*?)\}", raw, re.S):
        t = re.search(r'"(?:t|time)"\s*:\s*"?([\d.:]+)"?', obj)
        text = re.search(r'"(?:line|text)"\s*:\s*"?(.*?)"?\s*(?:,\s*"[A-Za-z_]+"\s*:|$)', obj.strip(), re.S)
        if text and text.group(1).strip():
            items.append({"t": t.group(1) if t else None, "line": text.group(1).strip().strip('"')})
    if items:
        return items
    return re.findall(r'"([^"]+)"', raw) or re.findall(r"#\w+", raw)


def parse_cookies(raw: str, default_domain: str) -> list[dict]:
    """Read cookies pasted in ANY common format into a list of dicts
    (name, value, domain, path, secure, expires, httpOnly).

    Accepts: JSON export from the "Cookie-Editor" / "EditThisCookie" browser extensions,
    Netscape cookies.txt (from "Get cookies.txt LOCALLY"), or a raw "a=b; c=d" header string.
    """
    raw = (raw or "").strip().lstrip("﻿")
    out: list[dict] = []
    if not raw:
        return out
    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                data = data.get("cookies", [data])
            for c in data:
                if not c.get("name"):
                    continue
                out.append({
                    "name": c["name"], "value": str(c.get("value", "")),
                    "domain": c.get("domain") or default_domain, "path": c.get("path") or "/",
                    "secure": bool(c.get("secure", True)), "httpOnly": bool(c.get("httpOnly", False)),
                    "expires": int(c.get("expirationDate") or c.get("expires") or 0),
                })
            return out
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    tabbed = [ln.split("\t") for ln in lines if ln.count("\t") >= 6 and not ln.startswith("# ")]
    if not tabbed:  # tabs often become spaces when copy-pasted
        tabbed = [ln.split(None, 6) for ln in lines if not ln.startswith("# ") and len(ln.split(None, 6)) == 7
                  and ln.split()[1].upper() in ("TRUE", "FALSE")]
    if tabbed:
        for parts in tabbed:
            domain = parts[0].removeprefix("#HttpOnly_")
            out.append({"name": parts[5], "value": parts[6].strip(), "domain": domain, "path": parts[2] or "/",
                        "secure": parts[3].upper() == "TRUE", "httpOnly": parts[0].startswith("#HttpOnly_"),
                        "expires": int(parts[4]) if parts[4].strip().isdigit() else 0})
        return out
    for pair in raw.replace("\n", ";").split(";"):  # raw header string
        if "=" in pair:
            k, v = pair.split("=", 1)
            out.append({"name": k.strip(), "value": v.strip(), "domain": default_domain, "path": "/",
                        "secure": True, "httpOnly": False, "expires": 0})
    return out


def cookies_to_netscape(cookies: list[dict]) -> str:
    lines = ["# Netscape HTTP Cookie File"]
    for c in cookies:
        domain = c["domain"] if c["domain"].startswith(".") or c["domain"].count(".") > 1 else "." + c["domain"]
        sub = "TRUE" if domain.startswith(".") else "FALSE"
        expires = c.get("expires") or 2147483647
        lines.append("\t".join([domain, sub, c.get("path", "/"), "TRUE" if c.get("secure") else "FALSE",
                                str(int(expires)), c["name"], c["value"]]))
    return "\n".join(lines) + "\n"


def write_summary(markdown: str) -> None:
    """Append to the GitHub Actions job summary (no-op locally)."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")
