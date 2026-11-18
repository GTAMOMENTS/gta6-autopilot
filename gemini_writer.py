import json
import logging
import os
import random

FALLBACK_SCRIPTS = [
    (
        "THIS GTA 6 DETAIL IS INSANE!",
        "Stop and look at the sheer level of detail Rockstar just packed into Grand Theft Auto 6! "
        "From the insane lighting reflecting off every car in Vice City to the way NPCs react across Leonida, "
        "this looks like a massive leap forward for open-world gaming. Watch the background physics in this scene—"
        "every single street feels alive, packed with traffic, wildlife, and next-gen weather effects. "
        "With GTA 6 arriving on PlayStation 5 and Xbox Series X and S, moments like this prove why fans waited over a decade. "
        "Which part of Leonida are you exploring first? Drop your answer in the comments!"
    ),
    (
        "ONLY ROCKSTAR COULD PULL THIS OFF!",
        "Look closely at this Grand Theft Auto 6 gameplay highlight because there is so much happening on screen at once! "
        "Every frame in Leonida is bursting with dense crowds, realistic vehicle handling, and cinematic atmosphere. "
        "Notice how smooth the transitions and character animations look compared to anything we saw in Los Santos. "
        "Whether you are cruising down Ocean Drive at sunset or tearing through the wetlands, the scale of this map is wild. "
        "Let me know what hyped you up the most in this clip!"
    )
]


def generate_clip_copy(video_title: str, creator: str) -> tuple[str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return random.choice(FALLBACK_SCRIPTS)

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are an energetic, enthusiastic gaming content creator running 'GTA VI Top Moments'.
Source clip: "{video_title}" | Credit: {creator}
Write:
1. banner: Max 6 words in ALL CAPS for the top video hook (exciting, punchy, zero spoilers).
2. script: An enthusiastic, high-energy 110 to 130 word voiceover script (about 48 to 52 seconds spoken aloud) breaking down the visuals, Vice City / Leonida atmosphere, physics, and excitement of this GTA 6 highlight, ending with a quick question for comments. No emojis or stage directions.
Return strict JSON: {{"banner": "...", "script": "..."}}
"""
        resp = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        cleaned = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(cleaned)
        banner = str(data.get("banner", "INSANE GTA 6 MOMENT!")).strip()[:45]
        script = str(data.get("script", "")).strip()
        if banner and len(script.split()) >= 60:
            logging.info("Gemini generated custom script: %s", banner)
            return banner, script
    except Exception as exc:
        logging.warning("Gemini fallback used: %s", exc)

    return random.choice(FALLBACK_SCRIPTS)
