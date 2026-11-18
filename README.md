# GTA VI Top Moments — 24/7 Autopilot

Fully automated faceless GTA 6 clip channel for YouTube Shorts, Instagram Reels, TikTok and X,
running free on GitHub Actions.

## What each run does

1. **Finds footage from anywhere**: gameplay walkthroughs, missions, story cutscenes, police chases,
   heists and stunts. It searches Dailymotion and YouTube with about 20 rotating queries, plus anything
   you drop in `/clips` or paste into `config.json → sources.extra_urls`. Official trailers are
   only a last resort.
2. **Filters out junk**: GTA 5 mods posing as GTA 6, fan "concept" trailers, AI fakes, leaks,
   reaction/news/talk videos, livestreams.
3. **Downloads only a ~3 min section** of long videos, then **auto-picks the most intense seconds**
   using motion, loudness and black-frame detection (a single shot or a 2–3 cut montage).
4. **Gemini watches the clip** and rejects it if it isn't real GTA VI, has a facecam, or is boring.
   It then writes a banner, a ~50 s voiceover that matches what's on screen, a YouTube title,
   captions and an X post.
5. **Renders a 1080×1920 video**: blurred background, zoomed gameplay, wrapped hook banner, big
   word-by-word captions, AI voice with game audio ducked underneath, loudness normalised to -14 LUFS.
6. **Publishes** through the cheapest working route per platform (see below), and records history
   so nothing is posted twice.

The run fails (red ❌ in Actions, email notification) if nothing could be posted. Each run's
summary page shows the source, AI verdict, script and post links.

## Publishing routes (free first)

| Platform  | Route (in order)                    | Cost |
|-----------|-------------------------------------|------|
| YouTube   | `youtube_api` → zernio → upload_post | **Free** (official API, 100 uploads/day) |
| Instagram | `zernio` → upload_post               | **Free** (Zernio: first 2 accounts free, unlimited posts) |
| TikTok    | `zernio` → upload_post               | **Free** (the 2nd free Zernio account) |
| X         | `x_api` → zernio → upload_post       | X has **no free API tier** any more. Official API is pay-per-use, about $0.015 per post, so roughly $3/month for this schedule |

Change the order in `config.json → publishing.routes`. A provider without secrets is skipped automatically.

## GitHub secrets

| Secret | Needed for |
|--------|-----------|
| `GEMINI_API_KEY` | AI check + scripts (free at aistudio.google.com) |
| `ZERNIO_API_KEY` | Instagram + TikTok (connect exactly these 2 accounts in Zernio) |
| `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN` | YouTube. Run `python tools/youtube_auth.py client_secret.json` once on your PC |
| `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` | X (optional, paid). From console.x.com, app with Read+Write, OAuth 1.0a |
| `UPLOAD_POST_API_KEY`, `UPLOAD_POST_PROFILE` | Optional fallback (10 free uploads/month) |
| `YTDLP_COOKIES` | Optional. YouTube blocks GitHub's servers ("not a bot" check). Paste a Netscape-format `cookies.txt` exported from a **throwaway** Google account to unlock YouTube as a source. Without it, Dailymotion and the other sources still work |

### YouTube one-time setup
1. https://console.cloud.google.com → new project → *APIs & Services → Library* → enable **YouTube Data API v3**.
2. *OAuth consent screen* → External → add yourself as test user → **Publish app** (status *In production*).
   In *Testing* status the token expires every 7 days.
3. *Credentials → Create OAuth client ID → Desktop app* → download the JSON.
4. On your PC: `pip install requests` then `python tools/youtube_auth.py path/to/client_secret.json`,
   log in with the account that owns the channel, and copy the 3 printed values into GitHub secrets.

## Running manually

*Actions → GTA6 Top Moments 24/7 Autopilot → Run workflow*:
- `slot`: what to run (`vertical_clip`, `x_video_clip`, `test_all`, or a text post type).
- `dry_run` (default **on**): finds and renders the video **without posting**. Download it from the
  run's *Artifacts* to check the quality.

Locally: `python autopilot.py --slot vertical_clip --dry-run` (needs ffmpeg, plus a `.env` with the secrets).

## Files

| File | Role |
|------|------|
| `autopilot.py` | entry point: maps each cron run to its slot, runs it, records history |
| `source_fetcher.py` | footage discovery, filtering, ranking, section download |
| `highlight.py` | finds the most action-packed seconds |
| `gemini_writer.py` | AI quality gate, scripts, captions, X posts (with fallbacks) |
| `video_processor.py` | TTS, word-timed captions, vertical render |
| `publisher.py` | YouTube API / Zernio / X API / Upload-Post with automatic fallback |
| `config.json` | schedule, routes, search queries, filters, style |

State (posted history + used sources) lives in the Actions cache (`queue/`), not in git.
