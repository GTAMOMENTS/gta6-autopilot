"""Publishing (publisher.py): paused platforms (Instagram), provider fallback, captions + hashtag rules.
Every provider call is faked - nothing is ever posted."""
from __future__ import annotations

from pathlib import Path

import pytest

import publisher

VIDEO = Path("/nonexistent/short.mp4")
COPY = {"youtube_title": "He jumped the whole bridge", "banner": "HE JUMPED", "caption": "Bridge jump. Would you try it?",
        "x_text": "He really jumped it", "hashtags": ["#GTA6", "#RockstarGames", "#PS5", "#Stunts", "#GTA6Stunts"],
        "commentary": [{"t": 0, "text": "Watch this."}, {"t": 9, "text": "Would you have tried that jump?"}]}


@pytest.fixture
def providers(monkeypatch):
    """All providers 'configured'; records which platforms each one was asked to post to."""
    calls = []

    def ok(name, plats):
        calls.append((name, list(plats)))
        return {p: {"ok": True, "url": f"https://{name}/{p}"} for p in plats}
    monkeypatch.setattr(publisher, "_disabled", set())
    monkeypatch.setattr(publisher, "_available", lambda prov, plat: True)
    monkeypatch.setattr(publisher, "youtube_api_upload", lambda video, caps: ok("youtube_api", ["youtube"])["youtube"])
    monkeypatch.setattr(publisher, "x_api_video", lambda video, caps: ok("x_api", ["x"])["x"])
    monkeypatch.setattr(publisher, "zernio_video", lambda video, plats, caps: ok("zernio", plats))
    monkeypatch.setattr(publisher, "upload_post_video", lambda video, plats, caps: ok("upload_post", plats))
    return calls


def posted_to(calls):
    return sorted(p for _, plats in calls for p in plats)


def test_instagram_is_paused(providers, cfg):
    assert "instagram" in cfg["publishing"]["paused_platforms"]
    results = publisher.publish_video(VIDEO, ["instagram", "x", "tiktok", "youtube"], dict(COPY), "SomeCreator")
    assert "instagram" not in posted_to(providers)
    assert "instagram" not in results
    assert sorted(results) == ["tiktok", "x", "youtube"] and all(r["ok"] for r in results.values())
    assert ("zernio", ["tiktok"]) in providers


def test_unpausing_posts_to_instagram_again(providers, cfg):
    cfg["publishing"]["paused_platforms"] = []
    results = publisher.publish_video(VIDEO, ["instagram", "tiktok"], dict(COPY), "SomeCreator")
    assert ("zernio", ["instagram", "tiktok"]) in providers
    assert results["instagram"]["ok"] and results["instagram"]["provider"] == "zernio"


def test_only_paused_platforms_posts_nothing(providers):
    assert publisher.publish_video(VIDEO, ["instagram"], dict(COPY), "") == {}
    assert providers == []


def test_falls_back_to_next_provider(providers, monkeypatch):
    def zernio_down(video, plats, caps):
        providers.append(("zernio", list(plats)))
        return {p: {"ok": False, "error": "quota"} for p in plats}
    monkeypatch.setattr(publisher, "zernio_video", zernio_down)
    results = publisher.publish_video(VIDEO, ["tiktok"], dict(COPY), "")
    assert [name for name, _ in providers] == ["zernio", "upload_post"]
    assert results["tiktok"] == {"ok": True, "url": "https://upload_post/tiktok", "provider": "upload_post"}


def test_provider_crash_is_caught(providers, monkeypatch):
    def boom(video, caps):
        raise RuntimeError("uploadLimitExceeded")
    monkeypatch.setattr(publisher, "youtube_api_upload", boom)
    monkeypatch.setattr(publisher, "routes_for", lambda p: ["youtube_api"])
    results = publisher.publish_video(VIDEO, ["youtube"], dict(COPY), "")
    assert results["youtube"]["ok"] is False and "uploadLimitExceeded" in results["youtube"]["error"]


def test_nothing_configured(monkeypatch):
    monkeypatch.setattr(publisher, "_disabled", set())
    results = publisher.publish_video(VIDEO, ["youtube", "x", "tiktok"], dict(COPY), "")
    assert {p: r["ok"] for p, r in results.items()} == {"youtube": False, "x": False, "tiktok": False}


def test_available_needs_secrets(monkeypatch):
    monkeypatch.setattr(publisher, "_disabled", set())
    assert publisher._available("youtube_api", "youtube") is False
    for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"):
        monkeypatch.setenv(k, "test-not-real")
    assert publisher._available("youtube_api", "youtube") is True
    assert publisher._available("youtube_api", "x") is False


# --------------------------------------------------------------------------- captions

def test_captions_drop_brand_hashtags_and_respect_limits(cfg):
    caps = publisher.build_captions({**COPY, "series": "GTA 6 Stunts #4", "keywords": "GTA 6 stunts"}, "SomeCreator")
    every = " ".join(v if isinstance(v, str) else " ".join(v) for v in caps.values())
    assert "RockstarGames" not in every and "#PS5" not in every, "brand hashtags flagged Instagram once"
    # search words first, the series without a second "GTA 6", no hashtags in the title (#Shorts is in the description)
    assert caps["youtube_title"] == "GTA 6: He jumped the whole bridge | Stunts #4"
    assert "#Shorts" in caps["youtube_description"]
    assert len(caps["x"]) <= 280 and "Gameplay by SomeCreator" in caps["x"]
    x_tags = [w for w in caps["x"].split() if w.startswith("#")]
    assert len(x_tags) <= cfg["publishing"]["max_hashtags"]["x"]
    ig_tags = [w for w in caps["instagram"].split() if w.startswith("#")]
    assert len(ig_tags) <= cfg["publishing"]["max_hashtags"]["instagram"]
    assert caps["first_comment"] == "Would you have tried that jump?"
    assert len(caps["tiktok"]) <= 2200


def test_clean_hashtags_never_empty():
    assert publisher.clean_hashtags(["#Rockstar", "#xbox"]) == ["#GTA6"]
    assert publisher.clean_hashtags(["#GTA6", "#gta6", "#GTA6"]) == ["#GTA6"]  # hashtags are case-insensitive


def test_add_hashtags_fits_280():
    text = "x" * 270
    assert len(publisher.add_hashtags(text)) <= 280
    # owner rule: 3-5 hashtags - a post with two gets a third safe one
    assert publisher.add_hashtags("Who wins? #GTA6 #GTAVI") == "Who wins? #GTA6 #GTAVI #GTA6Gameplay"
    assert publisher.add_hashtags("Who wins? #GTA6 #GTAVI #ViceCity") == "Who wins? #GTA6 #GTAVI #ViceCity"
