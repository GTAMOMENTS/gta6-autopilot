"""The critic + the packaging rules.

polish(): a second, strong free text model (GPT-4.1 via GitHub Models first, then the other free backups)
reviews the packaging the video AI wrote - title, banner/thumbnail text, caption, X text, hook - and rewrites it
for more clicks without inventing anything. A rewrite that breaks a rule keeps the original:
  - empty / too long / too short / says "GTA VI" / has a link
  - adds numbers or names that aren't in the video (_invents)
  - adds words that sound like cheats (hack, glitch, exploit, mod...) - money content may not use them at all
  - drops a search keyword the original had ("GTA 6", "make money", "funny moments", "no spoilers"...)
  - adds a story-spoiler word, emojis, hashtags or too many ALL-CAPS words

The rules themselves (apply_rules / fix_title / sanitize / fix_count) also run when no critic AI answers:
every title says "GTA 6" near the front, countdown numbers match the real count, and money packaging never
sounds like cheats or glitches (the channel only shows legit methods).
"""
from __future__ import annotations

import json
import logging
import re

from common import load_config, load_json

log = logging.getLogger("autopilot")

PROMPT = """You are the packaging expert of a fast-growing GTA 6 channel (YouTube Shorts, TikTok, X, long YouTube videos).
Another AI watched the video and wrote the packaging below. Make it get MORE CLICKS and MORE WATCH TIME, honestly:
- curiosity gap, strong verbs, specific details from the video; at most {caps} CAPITALISED power words per title
- the hook must make people stay; titles must make people click; NEVER promise anything the video doesn't deliver
- keep every fact from the description - DO NOT invent events, names or numbers that aren't in it
- SEARCH: people find these videos by searching. Titles and captions start with the words people type ("GTA 6" +
  the topic, e.g. "GTA 6 Money Tip", "GTA 6 Police Chase"). KEEP every search keyword the original has{keep}.{searches}
- {honesty}
- respect each field's limit exactly; say "GTA 6", never "GTA VI"; no story spoilers; no hashtags, emojis or links
  unless the field already had them
Fields:
{guide}
What the video shows: {context}

Rewrite each field below, or return it unchanged if it is already excellent:
{fields}

Return ONLY JSON with exactly these keys: {keys}"""

FIELD_GUIDE = {
    "youtube_title": '"GTA 6" + the search phrase first (phones show only the first ~40 characters), then the hook',
    "title": '"GTA 6" + the search phrase first (phones show only the first ~40 characters), then the hook; '
             "keep the countdown number exactly",
    "banner": "big text over the first seconds of the video: ALL CAPS, max 6 words, makes people NEED to see the end",
    "thumbnail_text": "2-4 huge ALL CAPS words that add curiosity to the title instead of repeating it",
    "caption": "TikTok/Instagram caption: the first sentence carries the search words (TikTok search reads it), "
               "the last one is a question that makes people comment",
    "x_text": "X post: one short punchy line (a reaction or hot take about the clip), no links, no hashtags",
    "intro_line": "spoken over the cold open: makes people stay until number one",
    "outro_line": "spoken at the end: asks a question and to subscribe",
}

SENTENCE_FIELDS = {"caption", "intro_line", "x_text", "outro_line", "description", "summary"}
TITLE_FIELDS = {"youtube_title", "title"}
CAPS_FIELDS = {"banner", "thumbnail_text"}           # written in ALL CAPS on purpose
SEARCH_FIELDS = {"caption", "description", "summary"}  # indexed by search: must keep "GTA 6" + the topic words

# Words that make a video sound like cheats, glitches or mods. The channel shows legit gameplay only, and these
# words also make YouTube / TikTok treat a video as "exploit" content. config: critic.cheat_words replaces the list.
CHEAT_WORDS = ["hack", "hacks", "hacked", "hacking", "hacker", "cheat", "cheats", "cheating", "cheat code",
               "cheat codes", "glitch", "glitches", "glitched", "glitching", "glitchy", "exploit", "exploits",
               "exploiting", "mod", "mods", "modded", "modding", "modder", "mod menu", "trainer", "dupe", "duping",
               "duplication", "unlimited money", "unlimited cash", "infinite money", "infinite cash", "money drop",
               "god mode", "aimbot"]
MONEY_ONLY_WORDS = ["trick", "tricks"]  # "money trick" = glitch in GTA YouTube slang
# money content: the AI's own words are fixed instead of thrown away ("8 Money Hacks" -> "8 Money Tips")
CHEAT_FIXES = [(r"(?:unlimited|infinite)\s+(money|cash)", r"fast \1"), (r"hacks", "tips"), (r"hack", "tip"),
               (r"cheats", "tips"), (r"cheat", "tip"), (r"tricks", "tips"), (r"trick", "tip"),
               (r"glitches", "methods"), (r"glitch", "method"), (r"exploits", "methods"), (r"exploit", "method")]

SPOILER_WORDS = ["ending", "endings", "dies", "died", "death", "killed", "betray", "betrays", "betrayed", "betrayal",
                 "twist", "final mission", "last mission", "boss fight", "plot", "story mission", "cutscene", "finale"]
_GAMEPLAY_OK = {"dies", "died", "death", "killed", "twist", "plot", "cutscene", "story mission"}

# search keywords a rewrite must keep when the original has them (config: critic.keep_terms adds more)
KEEP_TERMS = ["gta 6", "make money", "money", "cash", "how to", "best", "fast", "easy", "funny", "moment", "chase",
              "stunt", "crash", "secret", "hidden", "easter egg", "tip", "guide", "things to do", "no spoiler"]
ACRONYMS = {"GTA", "NPC", "NPCS", "HUD", "USA", "ATM", "ATMS", "SUV", "DLC", "FPS", "POV", "RPG", "XP", "VS", "AI",
            "OMG", "LOL", "WTF", "IRL", "TV", "UFO", "RIP"}
_STOP = {"a", "an", "the", "to", "in", "of", "on", "for", "and", "or", "is", "are", "you", "your", "this", "that",
         "with", "how", "what", "why", "it", "its", "my", "i", "do", "does", "can", "where", "gta", "6", "vs"}
# trend phrases we never use as keywords: news / leaks / platforms / downloads / story (spoilers)
_BAD_SEARCH = re.compile(r"\b(leak\w*|release|price|cost|pre ?order|trailer|download|apk|mobile|free|torrent|crack|"
                         r"ending|story|mission|dies|death|spoiler\w*|ps4|ps5|xbox|pc|steam|reddit|wiki|delay\w*|"
                         r"news|rumou?r\w*|reaction|map size|system requirements)\b", re.I)

_GTA_RE = re.compile(r"(?<![#\w])(?:gta|grand\s+theft\s+auto)[\s-]?(?:6|vi)(?!\w)", re.I)
_GTA_VI_RE = re.compile(r"\b(?:gta|grand\s+theft\s+auto)\s*vi\b", re.I)
_LINK_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
_MONEY_RE = re.compile(r"(?:\b(?:money|cash|rich|richer|earn|earning|earned|payout|profit|income|paid)\b|\$\d)", re.I)


def _cfg() -> dict:
    return load_config().get("critic", {})


def _words_re(words: list[str]) -> re.Pattern:
    alts = sorted({w.strip().lower() for w in words if w.strip()}, key=len, reverse=True)
    body = "|".join(r"\s+".join(re.escape(p) for p in w.split()) for w in alts) or r"(?!x)x"
    return re.compile(r"(?<![a-z0-9])(" + body + r")(?![a-z0-9])", re.I)


def _cheat_re(money: bool) -> re.Pattern:
    words = list(_cfg().get("cheat_words") or CHEAT_WORDS)
    return _words_re(words + (MONEY_ONLY_WORDS if money else []))


def cheat_words(text: str, money: bool = False) -> list[str]:
    """The cheat / glitch / mod words in a text (money=True also counts "trick")."""
    return [m.group(1).lower() for m in _cheat_re(money).finditer(text or "")]


def spoiler_words(text: str, story_only: bool = False) -> list[str]:
    """Story-spoiler words. story_only: only the clear ones ("my car died" is fine in an X post)."""
    words = [w for w in SPOILER_WORDS if w not in _GAMEPLAY_OK] if story_only else SPOILER_WORDS
    return [m.group(1).lower() for m in _words_re(words).finditer(text or "")]


def is_money(*texts: str) -> bool:
    return bool(_MONEY_RE.search(" ".join(str(t) for t in texts if t)))


def norm_gta(text: str) -> str:
    """'GTA VI' / 'GTA6' / 'Grand Theft Auto VI' -> 'GTA 6' (what people type into search). Hashtags stay."""
    return _GTA_RE.sub("GTA 6", text or "")


def _case_like(src: str, rep: str) -> str:
    if src.isupper():
        return rep.upper()
    if all(w[:1].isupper() for w in src.split()):
        return " ".join(w[:1].upper() + w[1:] for w in rep.split())
    return rep[:1].upper() + rep[1:] if src[:1].isupper() else rep


def sanitize(text: str, money: bool = True) -> str | None:
    """Money packaging must never sound like cheats: "Money Hacks" -> "Money Tips", "Glitch" -> "Method".
    Returns None when a word can't be fixed sensibly (mod menu, modded, dupe...)."""
    if not text or not cheat_words(text, money):
        return text
    out = text
    for pat, rep in CHEAT_FIXES:
        out = re.sub(r"(?<![a-z0-9])" + pat + r"(?![a-z0-9])",
                     lambda m, rep=rep: _case_like(m.group(0), m.expand(rep)), out, flags=re.I)
    return None if cheat_words(out, money) else out


def _strip_words(text: str, money: bool) -> str:
    """Last resort for sanitize(): delete the cheat words and tidy the gaps."""
    out = _cheat_re(money).sub("", text)
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+([,.!?:;)])", r"\1", out)).strip(" -–—|:,")


# --------------------------------------------------------------------------- search keywords

def _stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("iest"):
        return w[:-4] + "y"
    if len(w) > 4 and w.endswith("ier"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("est"):
        return w[:-3]
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses", "zzes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _stems(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", norm_gta(text or "").lower())]


def has_term(text: str, term: str) -> bool:
    """Phrase match on word stems: "Fastest Ways" has "fast way", "Funniest Moments" has "funny moment"."""
    t, p = _stems(text), _stems(term)
    return bool(p) and any(t[i:i + len(p)] == p for i in range(len(t) - len(p) + 1))


def _meaningful(phrase: str) -> list[str]:
    return [w for w in _stems(phrase) if w not in _STOP]


def keyword_list(keywords: str) -> list[str]:
    """A format's search phrases, comma-separated in config ("GTA 6 police chase, car chase" ->
    ["GTA 6 police chase", "GTA 6 car chase"]) - each one usable on its own (a YouTube tag, a title phrase)."""
    out = []
    for p in (keywords or "").split(","):
        p = re.sub(r"\s+", " ", norm_gta(p)).strip()
        if p:
            out.append(p if has_term(p, "gta 6") else f"GTA 6 {p}")
    return list(dict.fromkeys(out))


def keyword_phrases(keywords: str) -> list[str]:
    """The search phrases a rewrite must keep when the original title has them: each comma-separated phrase of a
    format's keywords without "GTA 6" (a long one as its two-word parts: "how to make money fast" -> "make money",
    "money fast", ...). Never across two phrases: "police chase car chase" made "chase car" a protected keyword,
    and the critic kept the stuffed title "GTA 6 police chase car chase at Vice City airport goes crazy"."""
    out = []
    for phrase in (keywords or "").split(","):
        words = [w for w in re.findall(r"[a-z0-9']+", norm_gta(phrase).lower()) if w not in ("gta", "6")]
        out += [" ".join(words)] if len(words) <= 3 else [f"{a} {b}" for a, b in zip(words, words[1:])]
    return list(dict.fromkeys(p for p in out if p))


def top_searches(limit: int = 12) -> list[str]:
    """What GTA 6 players type into YouTube/Google right now (trend radar, queue/trends.json -> top_searches),
    without cheat / leak / news / story phrases. Best first."""
    try:
        import trends
        data = load_json(trends.TRENDS_FILE, {})
    except Exception:  # noqa: BLE001 - trends must never break packaging
        return []
    out = []
    for s in data.get("top_searches") or []:
        s = re.sub(r"\s+", " ", str(s)).strip().lower()
        if not s or len(s) > 60 or _BAD_SEARCH.search(s) or cheat_words(s, money=True) or spoiler_words(s):
            continue
        s = norm_gta(s).replace("GTA 6", "gta 6")
        if s not in out and _meaningful(s):
            out.append(s)
    return out[:limit]


def relevant_searches(about: str, limit: int = 3, min_cover: float = 0.67) -> list[str]:
    """Trending search phrases that match this video: most of the phrase's words are in `about` (title,
    keywords, caption...). Only those may become tags / keywords - irrelevant tags are misleading."""
    have = set(_stems(about))
    out = []
    for p in top_searches(50):
        words = _meaningful(p)
        if words and sum(w in have for w in words) / len(words) >= min_cover:
            out.append(p)
    return out[:limit]


# --------------------------------------------------------------------------- title fixes

_COUNT_WORDS = (r"(?:best|top|craziest|crazy|funniest|funny|insane|wildest|wild|easiest|easy|fastest|fast|biggest|"
                r"big|greatest|coolest|cool|hidden|secret|secrets|legit|new|epic|brutal|simple|quick|smart|must|"
                r"ways?|things?|moments?|methods?|tips?|money|cash|cars?|places?|spots?|jobs?|activities|details?|"
                r"features?|reasons?|mistakes?|clips?|streamers?|fails?|stunts?|crash(?:es)?|chases?)")


def fix_count(text: str, n: int | None) -> str:
    """The countdown number in a title must be the real one: "Top 10" / "10 INSANE Ways" -> n.
    Never touches "GTA 6", "$4,800", "#3" or numbers that aren't a list count ("in 10 minutes")."""
    if not n or not text:
        return text
    text = re.sub(r"(?i)\b(top\s+)\d+\b", lambda m: f"{m.group(1)}{n}", text)
    return re.sub(r"(?<![\w$#,.])(?<!gta )(\d+)(?=\s+" + _COUNT_WORDS + r"\b)", str(n), text, flags=re.I)


def seo_front(title: str, within: int | None = None) -> str:
    """Search words first: "GTA 6" must be in the first ~40 characters (all phones show). A trailing
    "... In GTA 6" moves to the front ("GTA 6: ..."); a title without "GTA 6" gets it."""
    within = int(load_config().get("publishing", {}).get("title_keyword_within", 40) if within is None else within)
    t = re.sub(r"\s+", " ", norm_gta(title or "")).strip().strip('"')
    if not t:
        return t
    m = re.search(r"GTA 6", t)
    if not m:
        return f"GTA 6: {t}"
    if m.start() <= within:
        return t
    rest = re.sub(r"\s*(?:[-–—|:,]\s*|\b(?:in|on|of|from|for)\s+)GTA 6\s*([!?.]*)\s*$", r"\1", t, flags=re.I)
    if rest != t and "GTA 6" not in rest and rest.strip():
        return f"GTA 6: {rest.strip()}"
    return t


def trim(text: str, n: int) -> str:
    """Cut at a word boundary, never leaving a "(" open."""
    if len(text) <= n:
        return text
    cut = text[:n + 1].rsplit(" ", 1)[0].rstrip(" ,;:-–—|(")
    if cut.count("(") > cut.count(")"):
        cut = cut[:cut.rfind("(")].rstrip(" ,;:-–—|")
    return cut or text[:n]


def fix_title(title: str, n: int | None = None, money: bool = False, max_len: int = 100,
              fallback: str = "") -> str:
    """Every title the channel posts: GTA 6 first, real countdown number, no cheat words for money, no hashtags."""
    t = re.sub(r"#[A-Za-z]\w*", "", _LINK_RE.sub("", norm_gta(title or "")))
    t = re.sub(r"\s+", " ", t).strip(" -–—|:")
    if money and cheat_words(t, True):
        t = sanitize(t, True) or (fallback and norm_gta(fallback)) or _strip_words(t, True)
    return trim(seo_front(fix_count(t, n)), max_len) if t else t


def apply_rules(fields: dict, money: bool = False, count: int | None = None, limits: dict | None = None,
                fallbacks: dict | None = None) -> dict:
    """The packaging rules every field goes through, with or without the critic AI."""
    out = {}
    for k, v in fields.items():
        v = re.sub(r"\s+", " ", str(v or "")).strip()
        if not v:
            out[k] = v
            continue
        v = norm_gta(_LINK_RE.sub("", v)).strip()
        if money and cheat_words(v, True):
            fixed = sanitize(v, True)
            if fixed is None:
                log.info("[critic] %s sounded like cheats/glitches for a money video: %r", k, v)
                fixed = (fallbacks or {}).get(k) or _strip_words(v, True)
            v = fixed
        if k in TITLE_FIELDS:
            v = fix_title(v, count, money, (limits or {}).get(k, 100))
        elif k in CAPS_FIELDS:
            v = fix_count(v, count)
        out[k] = v
    return out


# --------------------------------------------------------------------------- critic guards

def _invents(field: str, new: str, known: str) -> bool:
    """True if the rewrite adds facts: numbers that aren't in the original/description, or (in sentence-style
    fields, where it shows) capitalised names mid-sentence that the video description never mentions."""
    known_low = known.lower()
    if {n for n in re.findall(r"\d+", new)} - {n for n in re.findall(r"\d+", known)} - {"6"}:
        return True
    if field in SENTENCE_FIELDS:
        for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b", new):
            if m.group(1).lower() not in known_low and m.group(1) not in ("GTA",):
                return True
    return False


def _caps_words(text: str) -> int:
    return sum(1 for w in re.findall(r"\b[A-Z][A-Z']{2,}\b", text) if w.replace("'", "") not in ACRONYMS)


def _keep_terms(field: str, original: str, keep: list[str] | None) -> tuple[list[str], list[str]]:
    """The search keywords the original has that a rewrite must keep: (every one of these, at least one of these).
    The format's own phrases (keep) are alternatives - "police chase" / "car chase": one is enough. Requiring all
    of them protected a stuffed title ("GTA 6 police chase car chase at Vice City airport goes crazy")."""
    if field in TITLE_FIELDS:
        must = KEEP_TERMS + list(_cfg().get("keep_terms") or []) + top_searches(25)
    elif field in SEARCH_FIELDS:
        must = ["gta 6"]
    else:
        return [], []

    def present(terms):
        return [t for t in dict.fromkeys(x.lower() for x in terms if x) if has_term(original, t)]
    return present(must), present(keep or [])


def reject_reason(field: str, new: str, old: str, context: str, limit: int, money: bool,
                  keep: list[str] | None = None) -> str:
    """Why a rewrite must not replace the original ("" = it may)."""
    if not new:
        return "empty"
    if len(new) > limit:
        return f"too long ({len(new)} > {limit})"
    if field in TITLE_FIELDS and len(new) < min(20, len(old)):
        return "too short"
    if _GTA_VI_RE.search(new):
        return 'says "GTA VI"'
    if _LINK_RE.search(new):
        return "has a link"
    if _invents(field, new, f"{old} {context}"):
        return "added names/numbers not in the video"
    added = set(cheat_words(new, money)) - (set() if money else set(cheat_words(old)))
    if added:
        return f"sounds like cheats/glitches ({', '.join(sorted(added))})"
    added = set(spoiler_words(new)) - set(spoiler_words(old))
    if added:
        return f"spoiler words ({', '.join(sorted(added))})"
    if _EMOJI_RE.search(new) and not _EMOJI_RE.search(old):
        return "added emojis"
    if re.search(r"#[A-Za-z]", new) and not re.search(r"#[A-Za-z]", old):
        return "added hashtags"
    if field in TITLE_FIELDS and _caps_words(new) > max(int(_cfg().get("max_caps_words", 2)), _caps_words(old)):
        return "too many CAPS words"
    must, any_of = _keep_terms(field, old, keep)
    lost = [t for t in must if not has_term(new, t)]
    if any_of and not any(has_term(new, t) for t in any_of):
        lost.append(" / ".join(any_of))
    if lost:
        return f"dropped search keywords ({', '.join(lost)})"
    return ""


def polish(fields: dict, limits: dict, context: str, money: bool | None = None, keep: list[str] | None = None,
           count: int | None = None, fallbacks: dict | None = None) -> dict:
    """fields: {name: text}; limits: {name: max chars}. Returns the improved fields (originals where a rewrite
    was rejected) - always passed through the packaging rules (apply_rules).
    money: money content (None = detect from the text); keep: extra search keywords to keep (the format's
    keywords); count: the real countdown number for titles / thumbnail text."""
    if money is None:
        money = is_money(context, *fields.values())
    fields = apply_rules(fields, money, count, limits, fallbacks)
    cfg = _cfg()
    if not cfg.get("enabled", True) or not fields:
        return fields
    import ai_fallback
    if not ai_fallback.available():
        return fields
    listing = "\n".join(f'- {k} (max {limits.get(k, 100)} characters): "{v}"' for k, v in fields.items())
    guide = "\n".join(f"- {k}: {FIELD_GUIDE[k]}" for k in fields if k in FIELD_GUIDE) or "-"
    searches = top_searches(10)
    keep_now = sorted({t for k, v in fields.items() for t in _keep_terms(k, v, keep)[0]})
    alts = sorted({t for k, v in fields.items() for t in _keep_terms(k, v, keep)[1]} - set(keep_now))
    if len(alts) == 1:
        keep_now.append(alts[0])
    elif alts:
        keep_now.append("one of: " + " / ".join(alts))
    honesty = ("MONEY VIDEO: the methods are legit gameplay - NEVER words that suggest cheats, glitches, hacks, "
               "exploits, mods or tricks (hack, glitch, exploit, cheat, mod, trick, unlimited, infinite); say tips / "
               "ways / methods / guide instead" if money else
               "never add words that suggest cheats, glitches, hacks, exploits or mods")
    prompt = PROMPT.format(
        caps=int(cfg.get("max_caps_words", 2)), context=context[:1500], fields=listing, keys=json.dumps(list(fields)),
        guide=guide, honesty=honesty, keep=f" ({', '.join(keep_now)})" if keep_now else "",
        searches=("\n  What GTA 6 players search right now (use one word for word ONLY if it honestly matches the "
                  "video): " + "; ".join(searches)) if searches else "")
    try:  # what got the most / fewest views on our channel - learn from it
        import analytics
        views = analytics.view_examples()
        if views:
            prompt += "\n\n" + views
    except Exception:  # noqa: BLE001
        pass
    data = ai_fallback.chat_json(prompt, temperature=0.7)
    if not isinstance(data, dict):
        return fields
    out, changed = dict(fields), []
    for k, old in fields.items():
        new = re.sub(r"\s+", " ", str(data.get(k) or "")).strip().strip('"')
        why = reject_reason(k, new, old, context, limits.get(k, 100), money, keep)
        if why:
            if new and new != old:
                log.info("[critic] kept the original %s: %r (rewrite %r %s)", k, old, new, why)
            continue
        new = apply_rules({k: new}, money, count, limits)[k]
        if new and new != old:
            out[k] = new
            changed.append(k)
    if changed:
        log.info("[critic] %s improved: %s", data.get("_ai_model", "AI"),
                 {k: f"{fields[k]!r} -> {out[k]!r}" for k in changed})
    return out
