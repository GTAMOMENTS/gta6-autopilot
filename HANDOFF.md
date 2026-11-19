# HANDOFF - GTA 6 autopilot (read this first)

This file is everything a new AI assistant (or developer) needs to continue the work exactly where it stopped.
Written 2026-11-19 by the previous assistant. Repo: https://github.com/GTAMOMENTS/gta6-autopilot

---------------------------------------------------------------------------------------------------------
## 1. The owner and the goals

- **Goal: the highest possible views** on a faceless GTA 6 channel, **100% free** (no paid APIs, no paid plans), **100% automated**
  (runs on GitHub Actions on a schedule, nobody touches it).
- Channels: YouTube `@GTAVI6TopMoments`, Instagram `@gtavi_topmoments`, TikTok `@gta.vi.top.moment`, X `@GTAVIMomentz`.
  Brand name in config: "GTA VI Top Moments". Owner's timezone: **Turkey (TRT, UTC+3)**. GTA 6 launched 2026-11-19.
- Owner writes short, casual messages (often typos), wants **things done, merged and published**, then a short status.
  He tests by downloading the videos and checking them himself.

### Owner's rules / decisions (do NOT undo these)
| Topic | Decision |
|---|---|
| Voice | **Human voice only**: Gemini TTS voice **"Fenrir"**. Never robotic (edge-tts/gTTS) - `voice.robot_fallback=false`. If the voice fails, post nothing (Shorts) / post without narration (long video). |
| TTS prompt | Gemini TTS gets **ONLY the script text** (style instructions were read out loud once - big bug). Never put instructions in the TTS request. |
| Script | Full narration script written by the AI (8-14 lines for Shorts), captions must match the voice exactly. |
| Spoilers | **No story spoilers ever** (AI `is_spoiler` gate + blocked title words). |
| Brands | Naming cars/brands in scripts/titles is **OK** (`quality_gate.block_brand_names=false`). Brand **hashtags** (#RockstarGames, #PS5...) stay removed (Instagram flagged "branded content" once). |
| Length | **No minimum length** for Shorts (a 1:15 minimum was tried and rejected - it dropped good short clips). |
| Streamers | **Only famous English-speaking streamers** (no Turkish/Spanish/etc.). |
| Schedule | **1 Short every 2 days + 1 long YouTube video every 3 days** (first long video published BY HAND by the owner on 2026-11-19; `not_before` 2026-11-22 16:00 UTC) (owner's choice after YouTube's upload limit, `every_days` in config) + viral stream clips (TikTok/IG/X) + 3 X text posts. |
| Quality | Must not post junk: script gate (>=4 lines/35 words, written by an AI that saw the clip), hype gate, choppy-clip filter. |
| Merging | Test (dry run in Actions) then merge. The assistant's permission classifier blocks `gh pr merge` unless the owner literally says "merge" in that message. The owner also merges PRs himself sometimes. |

---------------------------------------------------------------------------------------------------------
## 2. What runs automatically

### Workflow `.github/workflows/main.yml` ("GTA6 Top Moments 24/7 Autopilot") - one queue (concurrency group `gta6-autopilot`)
| TRT | UTC cron | Slot | What |
|---|---|---|---|
| 09:50 | 50 6 | x_text_morning | X text post |
| 16:20 | 20 13 | x_question_nostalgia | X text post |
| 20:10 | 10 17 | **long_video** (every 3 days) | 5-10 min 16:9 YouTube countdown of the day's best stream moments; a refused upload is kept in `queue/pending_long/` and retried on every run |
| 22:20 | 20 19 | x_daily_recap | X text post |
| 23:40 | 40 20 | vertical_clip (every 2 days) | Short -> YouTube Shorts, Instagram, TikTok, X |
Manual runs: Actions -> Run workflow -> `slot` (check_setup, voice_samples, vertical_clip, x_video_clip, test_all, remake,
stream_clip, long_video, text slots), `dry_run` (true = render only, artifacts), `format`, `remake_clip`, `providers`.
**Careful:** GitHub allows 1 running + 1 pending per group; dispatching while a scheduled run is pending CANCELS the scheduled one.
GitHub cron is often 30-60 min late and sometimes skips runs.

### Workflow `.github/workflows/streams.yml` ("GTA6 Viral Stream Clips") - own queue + own cache (`queue_streams/`)
Every 30 min (`7,37 * * * *`): `python stream_clips.py` -> finds the fastest-rising GTA 6 clip from famous English streamers and
posts it with the **streamer's real voice + Whisper subtitles + hook banner** (no AI voice) to TikTok/Instagram/X (not YouTube -
quota). Max 4/day, 90 min apart, never the same moment twice.

---------------------------------------------------------------------------------------------------------
## 3. Code map

| File | Role |
|---|---|
| `autopilot.py` | entry point; maps cron -> slot; Short pipeline (`run_video_slot`: sources -> AI copy -> gates -> cut -> voice -> render -> publish), `run_remake`, text slots, history, run summary, learning refresh |
| `source_fetcher.py` | footage discovery (YouTube search, Dailymotion, Twitch, X accounts, local, trailers), ranking, downloads (yt-dlp sections, YouTube client fallbacks, direct mp4 / HLS for X/Kick), blocked news/official channels |
| `gemini_writer.py` | Gemini: watches clips (video+audio upload), writes copy (prompts CLIP_PROMPT, TOUR_PROMPT), model discovery (Flash, Flash-Lite, Pro), retries, `polish_packaging` (critic) |
| `ai_fallback.py` | free backup AIs (frames only, no audio): **GitHub Models** (gpt-4.1 via workflow token, `permissions: models: read`), **Gemma** on the Gemini key, OpenRouter, NVIDIA, Groq, Mistral |
| `critic.py` | second AI rewrites titles/banners/captions/long-video title+thumbnail text for CTR; rejects invented numbers/names |
| `formats.py` | Short format rotation (top5_funny, car_chases, top5_things_to_do, secrets, stunts_crashes, stream_reactions) + **learning** (favours best-performing formats once each has 2+ videos with stats) + series numbers |
| `analytics.py` | reads YouTube views/likes/comments of recent videos with yt-dlp (free) for learning; marks deleted videos |
| `video_processor.py` | Gemini TTS (Fenrir; one request + alignment, per-line fallback, length checks), vertical render (blurred bg, zoom, ASS captions, cold open, zoom punch-in, generated whoosh/boom SFX, optional `music/`), `place_lines` |
| `highlight.py` | energy curve, best window, `cut_segments` |
| `publisher.py` | YouTube API upload (+ first comment, custom thumbnail), Zernio (IG/TikTok), X browser/API, Upload-Post; captions per platform (3-5 hashtags, keywords, series) |
| `stream_clips.py` | viral stream clips: Twitch top live streams + their clips (official API), Kick top English live channels + channel clips (public web API via curl_cffi), YouTube streamers' recordings (most-replayed peak), X clip accounts (IShowSpeedHQ); `is_famous`, `smoothness` (choppy check), AI review, render, post |
| `long_video.py` | daily long video: gather (famous English, max 2/streamer) -> download (+no-sound/choppy filter) -> AI review in batches of 6 (Pro first) -> Whisper smart cuts -> package (title/thumbnail text/intro + critic) -> 16:9 render (cold open, title cards, Fenrir lines, whoosh, outro) -> YouTube upload + thumbnail + chapters/credits |
| `transcribe.py` | **Whisper** (faster-whisper "small", local on runner, model cached via actions/cache `~/.cache/huggingface`): word timestamps, censoring, language detection, `smart_cut` (sentence/scene aligned cuts) |
| `x_source.py` | reads X accounts' media tab with Playwright + X_COOKIES (GraphQL parse, fallback: post links from the page + yt-dlp); cached |
| `x_browser.py` | posts to X through a logged-in headless browser (X_COOKIES) |
| `setup_check.py` | `check_setup` slot: tests every secret, YouTube quality check |
| `config.json` | everything tunable (schedule, formats, sources, quality_gate, voice, video.fx, publishing, learning, stream_clips, long_video, whisper, critic, gemini, ai_fallback) |

State (not in git): Actions cache `queue/` (manifest of sources/posts, history, format rotation, long_videos.json, x cache) and
`queue_streams/` (stream clip state). Dry runs don't save state.

### Secrets (GitHub -> Settings -> Secrets -> Actions)
GEMINI_API_KEY, ZERNIO_API_KEY, YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN, X_COOKIES, YTDLP_COOKIES, TWITCH_CLIENT_ID,
TWITCH_CLIENT_SECRET, OPENROUTER_API_KEY, GROQ_API_KEY, NVIDIA_API_KEY, MISTRAL_API_KEY, CEREBRAS_API_KEY, SAMBANOVA_API_KEY
(+ optional X_API_*, UPLOAD_POST_*). Text-only AI jobs use `ai_fallback.text_providers` order (Cerebras, Groq first); vision jobs `providers`.
GITHUB_MODELS_TOKEN = the workflow's own GITHUB_TOKEN (no secret needed).

---------------------------------------------------------------------------------------------------------
## 4. Limits & lessons learned (important)

- **YouTube API quota**: 10,000 units/day, upload = 1,600 -> ~6 uploads/day. **Plus a channel upload limit**
  (`uploadLimitExceeded`, rolling ~24 h) - hit on 2026-11-19 after many test uploads. Avoid live test uploads; use dry runs.
- **Gemini free tier**: Flash models run out of daily quota quickly (each Short tries several clips); Flash-Lite usually still
  works; 503/504 "high demand" is common -> one retry after 40 s. `gemini-2.5-pro` returns 404 "no longer available to new users";
  newer Pro models may work (discovered automatically). The owner's Google One / Gemini app subscription does NOT apply to the API.
- **Hugging Face / YouTube / GitHub artifact storage are blocked in the assistant's dev sandbox** - test those parts via Actions dry runs.
- Download real files from runs: GitHub artifacts (`video-<run_id>`, 5 days) - the owner can download them; the assistant can't.
- **X**: reading/posting via browser automation can trigger Cloudflare bot checks - open only one page per visit; X can lock accounts.
- **Instagram**: branded-content violation once (brand hashtags + credit lines). Account Status otherwise green.
- **TikTok**: all videos had 0 views on 2026-11-19 - unresolved; owner should check post visibility / Account status (maybe private via Zernio).
- Kick: clip pages need Cloudflare impersonation; use the clip's HLS stream (`clip_url`) - works. Kick's category-wide clip list mixes languages (removed).
- Twitch broadcaster clips can be from another game - filtered by `game_id`.
- Choppy video came from source clips (laggy streams), not from our pipeline -> `smoothness()` drops clips with >20% repeated frames.
- Audio out of sync in the long video came from concat-demuxer copy + per-clip length differences -> fixed in PR #26 (see below).

---------------------------------------------------------------------------------------------------------
## 5. Status at handoff (2026-11-19 ~12:40 UTC)

Merged to main: PRs #9-#25 (schedule, formats, voice fixes, script gate, free AI backups, X source, countdowns, brand rules,
charm pack, learning, stream clips, long video, English-only + choppy filter, Gemini Pro/Whisper/smart cuts/critic).

**Open / next steps (in order):**
1. **PR #26 - long video audio sync fix** (dry run 44525111317 PASSED: 7:50, 15 moments, Whisper cuts, critic, thumbnail;
   owner should download the run's artifact and confirm the sound is in sync) (+ excludes `deep-research-pro` from Pro discovery, + this HANDOFF.md):
   https://github.com/GTAMOMENTS/gta6-autopilot/pull/26 -> run a `long_video` **dry run** on the branch, check the log
   (`[long]`, `[cut]`, `[whisper]`, `[critic]`), ask the owner to download the artifact and confirm sound is in sync, then merge.
2. **YouTube `uploadLimitExceeded` (CONFIRMED)**: "The user has exceeded the number of videos they may upload." First seen
   2026-11-19 07:24 UTC after many test uploads, still active at 15:40 UTC (17:40 Short). Shorts still post to IG/TikTok/X;
   the long video is YouTube-only, so it is lost while the limit lasts. Don't do live test uploads. Consider: keep a failed
   long video (Actions cache/artifact) and retry the upload at the next run.
   The first live long video (run 44515332047) rendered (5:46, 11 moments) but was rejected for this reason. `run_long` does not log YouTube's error text -> add logging, and consider keeping a
   failed long video to retry at the next run. Also include the thumbnail `.jpg` in the uploaded artifacts (workflow only uploads mp4/png/mp3).
3. AI review rejected many Twitch clips as "not GTA 6" (17 of ~40) with Flash-Lite - check if those are false rejections
   (maybe ask Flash/Pro when available, or show more context in the prompt).
4. YouTube stream-recording source finds 0 (recent streams have no "most replayed" data yet) - low priority.
5. First-comment permission: owner must re-create YT_REFRESH_TOKEN with scope `youtube.force-ssl` (OAuth Playground steps were given).
6. The "secrets" Short format kept failing: "GTA 6 secrets/details" videos almost always have a YouTuber narrating (gate
   rejects narration). Queries now ask for "no commentary" footage (PR #26) - check it finds usable clips; if not, consider
   dropping "secrets" from the rotation or letting the AI find details in plain free-roam footage.
7. Watch learning data (`[learn] format scores`) after a few days; the rotation starts favouring the best formats automatically.

## 6. How to work on this repo
- Branch per change -> PR -> dry run in Actions (`gh api -X POST repos/GTAMOMENTS/gta6-autopilot/actions/workflows/main.yml/dispatches -f ref=<branch> -f 'inputs[slot]=long_video' -f 'inputs[dry_run]=true'`)
  -> read the job log (`gh api repos/.../actions/jobs/<job_id>/logs`) -> merge when the owner agrees.
- Local tests: mock the network parts (sources, Gemini, TTS) and render with real ffmpeg (ffmpeg 6.1 = same as runners).
- Commit messages end with the session attribution lines the environment asks for; never put model names in commits/PRs.
