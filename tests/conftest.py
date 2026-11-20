"""Shared test setup - keep it small and generic (feature folders like tests/shorts/ add their own conftest).

Every test automatically runs:
  * with its own copy of config.json (mutate `cfg` freely, nothing leaks into other tests),
  * with every queue/ / work/ / queue_streams/ path in the product modules pointed at a temp folder
    (the real state cache and work dirs are never touched),
  * offline: no secrets in the environment, Python sockets blocked, proxies for child processes pointed at a
    dead port, Whisper reported as "not installed" (override transcribe.words / _get_model in a test to fake it).

Generic helpers: `make_clip` (synthetic FFmpeg footage), `fake_tts` (stands in for Gemini TTS / Fenrir).
"""
from __future__ import annotations

import copy
import importlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import common  # noqa: E402
from fakes import WORDS_PER_SEC, fake_tts_wav, ffmpeg  # noqa: E402

# folders the product writes into at runtime - tests must never write there
PROTECTED = ("queue", "work", "queue_streams")

SECRET_AND_RUN_ENV = (
    "GEMINI_API_KEY", "FREESOUND_API_KEY", "GITHUB_TOKEN", "GITHUB_MODELS_TOKEN", "OPENROUTER_API_KEY", "GROQ_API_KEY",
    "MISTRAL_API_KEY", "NVIDIA_API_KEY", "CEREBRAS_API_KEY", "SAMBANOVA_API_KEY", "ZERNIO_API_KEY", "YT_CLIENT_ID",
    "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN", "X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET",
    "UPLOAD_POST_API_KEY", "UPLOAD_POST_PROFILE", "YTDLP_COOKIES", "X_COOKIES", "TWITCH_CLIENT_ID",
    "TWITCH_CLIENT_SECRET", "TIMEZONE", "SCHEDULE_CRON", "SLOT", "DRY_RUN", "FORMAT", "REMAKE_CLIP",
    "SOURCE_PROVIDERS", "TTS_VOICE", "GITHUB_STEP_SUMMARY",
)


def _product_modules() -> list:
    """Import every top-level product module (they have no import side effects) so their path constants exist."""
    mods = []
    for f in sorted(REPO.glob("*.py")):
        try:
            mods.append(importlib.import_module(f.stem))
        except Exception:  # noqa: BLE001 - a broken module is reported by its own tests, not here
            continue
    return mods


class NetworkBlocked(OSError):
    """Raised when code under test tries to open a network connection (tests are offline)."""


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No secrets, no network. Product code treats the errors like the blocked sandbox network."""
    for name in SECRET_AND_RUN_ENV:
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # child processes (yt-dlp, curl) fail fast too
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        if self.family == getattr(socket, "AF_UNIX", None):
            return real_connect(self, address)
        raise NetworkBlocked(f"network access is disabled in tests ({address!r})")

    def guarded_getaddrinfo(host, *a, **k):
        raise NetworkBlocked(f"network access is disabled in tests ({host!r})")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)


@pytest.fixture(autouse=True)
def cfg(monkeypatch):
    """A private, mutable copy of config.json for this test (what load_config() returns everywhere)."""
    data = copy.deepcopy(json.loads(common.CONFIG_FILE.read_text(encoding="utf-8")))
    monkeypatch.setattr(common, "_CONFIG_CACHE", data)
    return data


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Point every queue/ / work/ / queue_streams/ path constant of every product module at tmp_path/_repo/..."""
    root = tmp_path / "_repo"
    real = {name: (REPO / name).resolve() for name in PROTECTED}
    fake = {name: root / name for name in PROTECTED}

    def remap(p: Path) -> Path | None:
        try:
            rp = p.resolve()
        except OSError:
            return None
        for name, base in real.items():
            if rp == base or base in rp.parents:
                return fake[name] / rp.relative_to(base)
        return None

    for mod in _product_modules():
        for attr, val in list(vars(mod).items()):
            if isinstance(val, Path):
                new = remap(val)
                if new is not None:
                    monkeypatch.setattr(mod, attr, new)
    fake["work"].mkdir(parents=True, exist_ok=True)
    fake["queue"].mkdir(parents=True, exist_ok=True)
    ac = sys.modules.get("ai_cache")
    if ac is not None and hasattr(ac, "reset"):
        # the AI verdict cache keeps an in-memory copy: one test's saved verdicts must not leak into the next
        monkeypatch.setattr(ac, "CACHE_FILE", fake["queue"] / "ai_verdicts.json")
        ac.reset()
    return SimpleNamespace(root=root, state=fake["queue"], work=fake["work"], ready=fake["work"] / "ready",
                           raw=fake["work"] / "raw", voice=fake["work"] / "voice")


@pytest.fixture(autouse=True)
def no_whisper(monkeypatch):
    """Whisper is "not installed" unless a test fakes it (the real model is a 500 MB download)."""
    import transcribe
    monkeypatch.setattr(transcribe, "_get_model", lambda: None)
    monkeypatch.setattr(transcribe, "words", lambda *a, **k: ([], {}))


@pytest.fixture(autouse=True, scope="session")
def real_dirs_untouched():
    """Safety net: fail the run if anything was written into the real queue/ or work/ folders."""
    def snapshot():
        out = {}
        for name in PROTECTED:
            base = REPO / name
            if base.exists():
                out.update({str(p): p.stat().st_mtime_ns for p in base.rglob("*") if p.is_file()})
        return out
    before = snapshot()
    yield
    after = snapshot()
    changed = sorted(k for k in after if before.get(k) != after[k])
    assert not changed, f"tests wrote into the real state/work folders: {changed[:10]}"


# --------------------------------------------------------------------------- generic media helpers

@pytest.fixture
def make_clip(tmp_path):
    """make_clip(name, dur=8, size="1280x720", audio=True, freq=220, still=False) -> Path of synthetic footage:
    moving testsrc2 picture (or a still frame) + a sine tone (or no audio track)."""
    folder = tmp_path / "footage"
    folder.mkdir(exist_ok=True)

    def _make(name: str = "clip.mp4", dur: float = 8, size: str = "1280x720", audio: bool = True,
              freq: int = 220, still: bool = False, out: Path | None = None) -> Path:
        path = out or folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        src = f"color=c=0x336699:s={size}:r=30:d={dur}" if still else f"testsrc2=s={size}:r=30:d={dur}"
        args = ["-f", "lavfi", "-i", src]
        if audio:
            args += ["-f", "lavfi", "-i", f"sine=f={freq}:d={dur}:sample_rate=48000", "-shortest"]
        args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        args += (["-c:a", "aac"] if audio else [])
        ffmpeg(*args, str(path))
        return path
    return _make


@pytest.fixture
def fake_tts(monkeypatch):
    """Gemini TTS replaced by tones (records every prompt it got in .prompts; set .wps to speak faster).
    Sets a dummy GEMINI_API_KEY (the voice code needs one) - AI calls still have to be faked by the test."""
    import video_processor as vp
    calls = SimpleNamespace(prompts=[], voices=[], wps=WORDS_PER_SEC)

    def tts(prompt, model, voice, out_wav):
        calls.prompts.append(prompt)
        calls.voices.append(voice)
        return fake_tts_wav(prompt, model, voice, Path(out_wav), wps=calls.wps)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(vp, "_gemini_tts_models", lambda: ["fake-tts"])
    monkeypatch.setattr(vp, "_gemini_tts_wav", tts)
    return calls
