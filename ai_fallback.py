"""Free backup AI providers used when Gemini is busy / down.

All of these speak the OpenAI-compatible chat API and have free tiers:
  openrouter - OPENROUTER_API_KEY  (free ":free" models, several can SEE images)
  groq       - GROQ_API_KEY        (very fast free text models)
  nvidia     - NVIDIA_API_KEY      (build.nvidia.com, free, no card, 100+ models incl. vision)
  mistral    - MISTRAL_API_KEY     (optional; its free plan may now ask for payment)

Free model names change constantly, so models are auto-discovered from each
provider's /models endpoint (pin your own in config.json -> ai_fallback.models).
For the clip check, 6 frames are pulled from the video and sent as images.
(GitHub Models was here too - GitHub retired it on 2026-07-30; its endpoint only answers "OK" now.)
"""
from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path

import requests

from common import WORK_DIR, env, load_config, parse_json_loose, probe, run

log = logging.getLogger("autopilot")

PROVIDERS = {
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
_dead: set[tuple[str, str]] = set()
# (provider, model) -> answers without usable JSON this run. Thinking models that write prose (~45 s each) and
# endpoints that answer with something else entirely are skipped after MAX_STRIKES - a failed chain used to spend
# its whole 4-minute budget asking the same ones again on every call.
_strikes: dict[tuple[str, str], int] = {}
MAX_STRIKES = 2
# (provider, model) that refused IMAGES (text-only models): still asked for text jobs (judge, critic) - a vision
# 400 used to put the best text model (cerebras/gpt-oss-120b, pinned = assumed to see images) out for the run.
_dead_vision: set[tuple[str, str]] = set()


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
    # pinned models see images only if the name says so (gpt-oss / qwen text models were sent frames)
    found: list[tuple[str, bool]] = [(m, any(k in m.lower() for k in VISION_HINTS)) for m in pinned]
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
    order = {m: i for i, m in enumerate(pinned)}  # pinned models keep the order set in config.json
    uniq.sort(key=lambda mv: (order.get(mv[0], len(order)),) + _rank(*mv))
    chosen = [mv for mv in uniq if mv[0] != "openrouter/free"][: int(_cfg().get("max_models_per_provider", 4))]
    if p == "openrouter" and any(m == "openrouter/free" for m, _ in uniq):
        chosen.append(("openrouter/free", True))  # OpenRouter's auto-router to whichever free model is up
    _model_cache[p] = chosen
    log.info("[ai-fallback] %s models: %s", p, [m for m, _ in _model_cache[p]])
    return _model_cache[p]


def extract_frames(clip: Path, count: int = 6, times: list[float] | None = None) -> list[str]:
    """Return base64 JPEGs sampled evenly across the clip (or at these seconds)."""
    if times is None:
        dur = probe(clip)["duration"] or 30
        times = [dur * (i + 0.5) / count for i in range(count)]
    out_dir = WORK_DIR / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, t in enumerate(times):
        f = out_dir / f"f{i}.jpg"
        run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(clip), "-frames:v", "1",
             "-vf", "scale=640:-2", "-q:v", "5", str(f)], timeout=60)
        if f.exists():
            frames.append(base64.b64encode(f.read_bytes()).decode())
            f.unlink(missing_ok=True)  # the next request reuses the names
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


# when every backup failed: "vision" (frames) and "text" rest separately - a failed frames chain (no model that sees
# images answered) used to switch off the text-only judge and critic too, for 30 minutes
_down_until: dict[str, float] = {"vision": 0.0, "text": 0.0}


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
    except (KeyError, IndexError, ValueError, TypeError):
        log.warning("[ai-fallback] %s/%s odd response (HTTP %s, %s): %.300s", p, model, r.status_code,
                    r.headers.get("content-type", "?"), r.text)
        return None, 200
    text = _message_text(msg)
    data = _parse(text)
    if not data:
        log.warning("[ai-fallback] %s/%s returned no JSON: %.160s", p, model, text.replace("\n", " "))
    return data, 200


def chat_json(prompt: str, frames: list[str] | None = None, temperature: float = 0.9,
              all_frames: bool = False) -> dict | None:
    """Try every configured free provider/model ONCE until one returns valid JSON.
    Rate-limited / busy / broken model -> next model immediately, no waiting.
    all_frames: skip providers that would get only some of the frames (the prompt numbers every one)."""
    kind = "vision" if frames else "text"
    if time.time() < _down_until.get(kind, 0.0):
        return None
    deadline = time.time() + int(_cfg().get("max_total_sec", 240))
    per_call = float(_cfg().get("call_timeout_sec", 60))
    order = _cfg().get("providers", list(PROVIDERS)) if frames else \
        _cfg().get("text_providers", ["cerebras", "groq", "sambanova", "nvidia", "mistral", "gemma", "openrouter"])
    for p in order:
        if p not in PROVIDERS or not env(PROVIDERS[p]["key"]):
            continue
        if frames and all_frames and len(frames) > PROVIDERS[p]["max_images"]:
            continue
        for model, vision in _models(p):
            if (p, model) in _dead or (frames and (not vision or (p, model) in _dead_vision)):
                continue
            left = deadline - time.time()
            if left < 15:
                log.warning("[ai-fallback] time cap reached")
                break
            try:
                data, status = _call(p, model, prompt, frames, temperature, timeout=min(per_call, left))
            except Exception as exc:  # noqa: BLE001
                # a timeout is a strike too: nvidia's gemma-4 timed out after 90 s seven times in a row, and the
                # chain ran out of time before the models that answer
                log.warning("[ai-fallback] %s/%s failed: %.150s - next model", p, model, exc)
                _strike(p, model)
                continue
            if data:
                log.info("[ai-fallback] response from %s/%s%s", p, model, " (with frames)" if frames else "")
                data["_ai_model"] = f"{p}/{model}"
                return data
            if frames and status in (400, 413, 415, 422):  # can't take these images - text jobs still can
                _dead_vision.add((p, model))
            elif status in (400, 401, 402, 403, 404):
                _dead.add((p, model))
            elif status == 200:  # it answered, but nothing usable
                _strike(p, model)
    _down_until[kind] = time.time() + float(_cfg().get("rest_after_fail_sec", 600))  # all busy: rest a while
    return None


def _strike(p: str, model: str) -> None:
    _strikes[(p, model)] = _strikes.get((p, model), 0) + 1
    if _strikes[(p, model)] >= MAX_STRIKES:
        log.info("[ai-fallback] %s/%s failed / gave no usable answer %d times - skipping it this run", p, model,
                 MAX_STRIKES)
        _dead.add((p, model))
