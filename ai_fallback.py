"""Free backup AI providers used when Gemini is busy / down.

All of these speak the OpenAI-compatible chat API and have free tiers:
  openrouter - OPENROUTER_API_KEY  (free ":free" models, several can SEE images)
  groq       - GROQ_API_KEY        (very fast free text models)
  nvidia     - NVIDIA_API_KEY      (build.nvidia.com, free, no card, 100+ models incl. vision)
  mistral    - MISTRAL_API_KEY     (optional; its free plan may now ask for payment)

Free model names change constantly, so models are auto-discovered from each
provider's /models endpoint (pin your own in config.json -> ai_fallback.models).
For the clip check, 6 frames are pulled from the video and sent as images.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path

import requests

from common import WORK_DIR, env, load_config, parse_json_loose, probe, run

log = logging.getLogger("autopilot")

PROVIDERS = {
    # GitHub Models: free, built into GitHub Actions (the workflow's own token, permission models: read)
    "github": {"key": "GITHUB_MODELS_TOKEN", "base": "https://models.github.ai/inference", "max_images": 8,
               "detail": "low", "max_tokens": 3000, "json_mode": True, "list_models": False,
               "default_models": ["openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/gpt-4o-mini"]},
    # Gemma on the same free Gemini key (separate, much bigger free limits than Gemini Flash; sees images)
    "gemma": {"key": "GEMINI_API_KEY", "base": "https://generativelanguage.googleapis.com/v1beta/openai",
              "max_images": 8, "only": "gemma-4", "max_tokens": 8192, "default_models": []},  # gemma-3: 404 there
    # big free text models (sign-up keys): very fast, generous free tiers
    "cerebras": {"key": "CEREBRAS_API_KEY", "base": "https://api.cerebras.ai/v1", "max_images": 4},
    "sambanova": {"key": "SAMBANOVA_API_KEY", "base": "https://api.sambanova.ai/v1", "max_images": 4},
    "openrouter": {"key": "OPENROUTER_API_KEY", "base": "https://openrouter.ai/api/v1", "max_images": 12},
    "groq": {"key": "GROQ_API_KEY", "base": "https://api.groq.com/openai/v1", "max_images": 5},
    "nvidia": {"key": "NVIDIA_API_KEY", "base": "https://integrate.api.nvidia.com/v1", "max_images": 4},
    "mistral": {"key": "MISTRAL_API_KEY", "base": "https://api.mistral.ai/v1", "max_images": 8},
}
# name hints for image-capable models on providers that don't publish modalities (groq, nvidia)
VISION_HINTS = ("vision", "-vl", "vl-", "vila", "maverick", "scout", "gemma-3", "gemma-4", "multimodal",
                "omni", "llava", "pixtral", "kimi-vl")
# Families that tend to follow JSON instructions well get tried first.
PREFERRED = ("gemma", "nemotron", "qwen", "llama", "mistral-small", "mistral-medium", "pixtral",
             "gpt-oss", "deepseek", "kimi", "glm")
SKIP = ("lyria", "audio", "tts", "whisper", "guard", "embed", "moderation", "code", "ocr", "1.7b", "3b",
        "rerank", "reward", "safety", "retriever", "parse", "nvclip", "deplot", "translate", "asr", "riva",
        "cosmos", "bge", "e5-", "math", "nemoretriever")

_model_cache: dict[str, list[tuple[str, bool]]] = {}
_dead: set[str] = set()


def _cfg() -> dict:
    return load_config().get("ai_fallback", {})


def available() -> bool:
    return any(env(PROVIDERS[p]["key"]) for p in _cfg().get("providers", list(PROVIDERS)) if p in PROVIDERS)


def _headers(p: str) -> dict:
    return {"Authorization": f"Bearer {env(PROVIDERS[p]['key'])}", "Content-Type": "application/json"}


def _rank(model_id: str, vision: bool) -> tuple:
    low = model_id.lower()
    fam = next((i for i, f in enumerate(PREFERRED) if f in low), len(PREFERRED))
    sizes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b(?![a-z])", low)]
    active = re.search(r"-a(\d+(?:\.\d+)?)b", low)  # MoE "a4b" = only 4B active -> weaker
    size = float(active.group(1)) * 3 if active else (max(sizes) if sizes else 0)
    return (not vision, fam, -size)


def _models(p: str) -> list[tuple[str, bool]]:
    """[(model_id, supports_images)] best first."""
    if p in _model_cache:
        return _model_cache[p]
    pinned = _cfg().get("models", {}).get(p, []) or PROVIDERS[p].get("default_models", [])
    found: list[tuple[str, bool]] = [(m, True) for m in pinned]
    try:
        if PROVIDERS[p].get("list_models", True) is False:
            raise LookupError("pinned models only")
        r = requests.get(f"{PROVIDERS[p]['base']}/models", headers=_headers(p), timeout=20)
        r.raise_for_status()
        for m in r.json().get("data", []):
            mid = m.get("id", "").removeprefix("models/")
            low = mid.lower()
            if PROVIDERS[p].get("only") and PROVIDERS[p]["only"] not in low:
                continue
            if not mid or any(s in low for s in SKIP) or m.get("active") is False:
                continue
            if p == "openrouter":
                pr = m.get("pricing", {})
                free = mid.endswith(":free") or (pr.get("prompt") == "0" and pr.get("completion") == "0")
                arch = m.get("architecture", {})
                if not free or "text" not in arch.get("output_modalities", ["text"]):
                    continue
                vision = "image" in arch.get("input_modalities", [])
            elif p == "mistral":
                caps = m.get("capabilities", {})
                if not caps.get("completion_chat", True):
                    continue
                vision = bool(caps.get("vision"))
            else:  # groq / nvidia expose no modality info -> guess from the name
                vision = any(k in low for k in VISION_HINTS)
            found.append((mid, vision))
    except LookupError:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("[ai-fallback] %s model list failed: %.150s", p, exc)
    uniq = list(dict(found).items())
    pinned_set = set(pinned)
    uniq.sort(key=lambda mv: (mv[0] not in pinned_set,) + _rank(*mv))
    chosen = [mv for mv in uniq if mv[0] != "openrouter/free"][: int(_cfg().get("max_models_per_provider", 4))]
    if p == "openrouter" and any(m == "openrouter/free" for m, _ in uniq):
        chosen.append(("openrouter/free", True))  # OpenRouter's auto-router to whichever free model is up
    _model_cache[p] = chosen
    log.info("[ai-fallback] %s models: %s", p, [m for m, _ in _model_cache[p]])
    return _model_cache[p]


def extract_frames(clip: Path, count: int = 6) -> list[str]:
    """Return base64 JPEGs sampled evenly across the clip."""
    dur = probe(clip)["duration"] or 30
    out_dir = WORK_DIR / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(count):
        t = dur * (i + 0.5) / count
        f = out_dir / f"f{i}.jpg"
        run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(clip), "-frames:v", "1",
             "-vf", "scale=640:-2", "-q:v", "5", str(f)], timeout=60)
        if f.exists():
            frames.append(base64.b64encode(f.read_bytes()).decode())
    return frames


def _parse(text: str) -> dict | None:
    """Thinking models write their reasoning first (<think>, <thought>, "The user wants...") and the JSON after it."""
    if not text:
        return None
    text = re.sub(r"<(think|thought|thinking|reasoning)>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"^.*?</(think|thought|thinking|reasoning)>", " ", text, flags=re.S | re.I)  # unclosed opening
    data = parse_json_loose(text)
    if data:
        return data
    # last balanced {...} block in the text (the answer usually comes at the end)
    depth, end = 0, None
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == "}":
            if depth == 0:
                end = i
            depth += 1
        elif ch == "{" and depth:
            depth -= 1
            if depth == 0 and end is not None:
                data = parse_json_loose(text[i:end + 1])
                if data:
                    return data
                end = None
    return None


_down_until = 0.0


def _message_text(msg: dict) -> str:
    """Thinking models sometimes put the answer in 'reasoning' and leave 'content' empty."""
    for key in ("content", "reasoning_content", "reasoning"):
        val = msg.get(key)
        if isinstance(val, list):  # some providers return content parts
            val = " ".join(str(x.get("text", "")) for x in val if isinstance(x, dict))
        if val and _parse(val):
            return val
    return msg.get("content") or msg.get("reasoning") or ""


STRICT = ("\n\nIMPORTANT: reply with ONLY the JSON object, starting with { and ending with }. "
          "Put every text value in double quotes. Do NOT describe the frames or explain anything outside the JSON.")


def _call(p: str, model: str, prompt: str, frames: list[str] | None, temperature: float,
          timeout: float = 90) -> tuple[dict | None, int]:
    prompt = prompt + STRICT
    content: list | str = prompt
    if frames:
        imgs = frames[: PROVIDERS[p]["max_images"]]
        img = {"detail": PROVIDERS[p]["detail"]} if PROVIDERS[p].get("detail") else {}
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}", **img}} for b in imgs]
    body = {"model": model, "temperature": temperature, "max_tokens": PROVIDERS[p].get("max_tokens", 4096),
            "messages": [{"role": "user", "content": content}]}
    if PROVIDERS[p].get("json_mode"):
        body["response_format"] = {"type": "json_object"}
    if p == "openrouter":
        body["reasoning"] = {"effort": "low"}
    r = requests.post(f"{PROVIDERS[p]['base']}/chat/completions", headers=_headers(p), timeout=timeout, json=body)
    if r.status_code != 200:
        log.warning("[ai-fallback] %s/%s HTTP %s: %.140s", p, model, r.status_code, r.text)
        return None, r.status_code
    try:
        msg = r.json()["choices"][0]["message"]
    except (KeyError, IndexError, ValueError):
        log.warning("[ai-fallback] %s/%s odd response (HTTP %s, %s): %.300s", p, model, r.status_code,
                    r.headers.get("content-type", "?"), r.text)
        return None, 200
    text = _message_text(msg)
    data = _parse(text)
    if not data:
        log.warning("[ai-fallback] %s/%s returned no JSON: %.160s", p, model, text.replace("\n", " "))
    return data, 200


def chat_json(prompt: str, frames: list[str] | None = None, temperature: float = 0.9) -> dict | None:
    """Try every configured free provider/model ONCE until one returns valid JSON.
    Rate-limited / busy / broken model -> next model immediately, no waiting."""
    global _down_until
    if time.time() < _down_until:
        return None
    deadline = time.time() + int(_cfg().get("max_total_sec", 240))
    order = _cfg().get("providers", list(PROVIDERS)) if frames else \
        _cfg().get("text_providers", ["cerebras", "groq", "sambanova", "github", "nvidia", "mistral", "gemma", "openrouter"])
    for p in order:
        if p not in PROVIDERS or not env(PROVIDERS[p]["key"]):
            continue
        for model, vision in _models(p):
            if (p, model) in _dead or (frames and not vision):
                continue
            left = deadline - time.time()
            if left < 15:
                log.warning("[ai-fallback] time cap reached")
                break
            try:
                data, status = _call(p, model, prompt, frames, temperature, timeout=min(90, left))
            except Exception as exc:  # noqa: BLE001
                log.warning("[ai-fallback] %s/%s failed: %.150s - next model", p, model, exc)
                continue
            if data:
                log.info("[ai-fallback] response from %s/%s%s", p, model, " (with frames)" if frames else "")
                data["_ai_model"] = f"{p}/{model}"
                return data
            if status in (400, 401, 402, 403, 404):
                _dead.add((p, model))
    _down_until = time.time() + 1800  # everything busy: don't keep retrying this run
    return None
