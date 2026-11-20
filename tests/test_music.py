"""Background music (music.py): CC0 beats from Freesound (requests faked) and track picking."""
from __future__ import annotations

import pytest

import common
import music

CC0_URL = "http://creativecommons.org/publicdomain/zero/1.0/"
RESULTS = [
    {"id": 1, "name": "trap loop", "username": "a", "duration": 60, "license": CC0_URL,
     "previews": {"preview-hq-mp3": "https://cdn.test/1.mp3"}},
    {"id": 2, "name": "attribution beat", "username": "b", "duration": 60,
     "license": "http://creativecommons.org/licenses/by/4.0/", "previews": {"preview-hq-mp3": "https://cdn.test/2.mp3"}},
    {"id": 3, "name": "no preview", "username": "c", "duration": 60, "license": CC0_URL, "previews": {}},
    {"id": 4, "name": "tiny file", "username": "d", "duration": 60, "license": CC0_URL,
     "previews": {"preview-hq-mp3": "https://cdn.test/4.mp3"}},
    {"id": 5, "name": "old licence text", "username": "e", "duration": 90, "license": "Creative Commons 0",
     "previews": {"preview-hq-mp3": "https://cdn.test/5.mp3"}},
    {"id": 6, "name": "one too many", "username": "f", "duration": 90, "license": CC0_URL,
     "previews": {"preview-hq-mp3": "https://cdn.test/6.mp3"}},
]


class Resp:
    def __init__(self, data=None, content=b""):
        self._data, self.content = data, content

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


@pytest.fixture
def freesound(monkeypatch, cfg):
    """Fake Freesound API; returns the list of requests made."""
    monkeypatch.setenv("FREESOUND_API_KEY", "test-key-not-real")
    cfg["music"].update(tracks=2, queries=["trap beat", "lofi beat"])
    calls = []

    def get(url, timeout=None, params=None):
        calls.append((url, dict(params or {})))
        if url.endswith("/search/text/"):
            return Resp({"results": RESULTS})
        return Resp(content=b"\0" * (5_000 if url.endswith("/4.mp3") else 150_000))
    monkeypatch.setattr(music.requests, "get", get)
    return calls


def test_downloads_cc0_beats_given_as_licence_url(freesound, sandbox):
    assert music.CACHE.parent == sandbox.state, "cached with the state, in the test's temp folder"
    assert music.ensure_tracks() == 2
    assert sorted(p.name for p in music.CACHE.glob("*.mp3")) == ["fs_1.mp3", "fs_5.mp3"]
    downloads = [u for u, _ in freesound if not u.endswith("/search/text/")]
    assert "https://cdn.test/2.mp3" not in downloads, "CC BY (attribution) is never downloaded"
    search = next(p for u, p in freesound if u.endswith("/search/text/"))
    assert 'license:"Creative Commons 0"' in search["filter"] and search["token"] == "test-key-not-real"
    index = common.load_json(music.CACHE / "index.json", {})
    assert index["tracks"]["1"] == {"name": "trap loop", "by": "a", "query": "trap beat", "duration": 60, "license": "CC0"}
    assert set(index["tracks"]) == {"1", "5"}


def test_enough_tracks_means_no_requests(freesound):
    music.ensure_tracks()
    freesound.clear()
    assert music.ensure_tracks() == 2
    assert freesound == []


def test_already_known_ids_are_skipped(freesound, cfg):
    music.ensure_tracks()
    cfg["music"]["tracks"] = 3
    assert music.ensure_tracks() == 3
    assert (music.CACHE / "fs_6.mp3").exists()


def test_no_key_no_requests(monkeypatch, freesound):
    monkeypatch.delenv("FREESOUND_API_KEY")
    assert music.ensure_tracks() == 0
    assert freesound == []


def test_search_failure_never_raises(monkeypatch, freesound):
    def broken(url, timeout=None, params=None):
        raise ConnectionError("freesound down")
    monkeypatch.setattr(music.requests, "get", broken)
    assert music.ensure_tracks() == 0


def test_offline_never_raises(monkeypatch, cfg):
    """With the real requests library and no network (blocked in tests): nothing downloaded, no crash."""
    monkeypatch.setenv("FREESOUND_API_KEY", "test-key-not-real")
    cfg["music"]["queries"] = ["trap beat"]
    assert music.ensure_tracks() == 0


def test_pick_prefers_own_music_folder(cfg, tmp_path):
    own = tmp_path / "own_music"
    cfg["video"]["fx"]["music_dir"] = str(own)  # absolute path: BASE_DIR / abs = abs
    assert music.pick() is None
    music.CACHE.mkdir(parents=True, exist_ok=True)
    (music.CACHE / "fs_1.mp3").write_bytes(b"x")
    (music.CACHE / "index.json").write_text("{}")
    assert music.pick() == music.CACHE / "fs_1.mp3", "downloaded CC0 beats (json index is not a track)"
    own.mkdir()
    (own / "mine.WAV").write_bytes(b"x")
    (own / "README.md").write_text("not audio")
    assert music.pick() == own / "mine.WAV"
