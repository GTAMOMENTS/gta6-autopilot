import json
import os
import random

FALLBACK_HOOKS = [
    ("THE DETAIL NOBODY NOTICED", "Look closely at this GTA 6 moment that almost everyone missed on the first watch."),
    ("THIS CHANGES EVERYTHING IN GTA 6", "Rockstar packed an insane level of detail into this Leonida scene."),
    ("TOP GTA 6 MOMENT TODAY", "Watch how realistic the physics and lighting look in this standout GTA 6 clip."),
    ("DID YOU CATCH THIS IN GTA 6?", "Pause right here and look at the background detail in this GTA 6 highlight.")
]


def generate_clip_copy(video_title: str, creator: str) -> tuple[str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return random.choice(FALLBACK_HOOKS)

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = f"""
You run the faceless fan page 'GTA VI Top Moments'.
Source clip title: "{video_title}" | Creator: {creator}
Write:
1. banner: Max 6 words in ALL CAPS for the top video hook (no clickbait, no spoilers).
2. script: 1 punchy 12-18 word AI voiceover sentence explaining why this moment stands out.
Return strict JSON: {{"banner": "...", "script": "..."}}
"""
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        cleaned = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(cleaned)
        banner = str(data.get("banner", "TOP GTA 6 MOMENT")).strip()[:45]
        script = str(data.get("script", "")).strip()
        if banner and script:
            return banner, script
    except Exception as exc:
        print(f"[gemini_writer] Fallback used: {exc}")

    return random.choice(FALLBACK_HOOKS)
