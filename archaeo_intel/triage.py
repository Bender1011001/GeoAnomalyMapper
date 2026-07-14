"""VLM triage of detector candidates (ancient / modern / natural).

The prompt is A/B-validated on labeled anchor chips (2026-07-13, 5 repeats
each): v3 scores 19/20 where the previous prompt scored 10/20 — the old wording
was SYSTEMATICALLY wrong (0/5) on the two hardest true-tell classes:
- a tell with a modern village on its summit (Tell Brak) -> called MODERN
- a bare excavated tell (Tell Chuera) -> called NATURAL
v3 keeps the negatives perfect (flat village = MODERN 5/5, Konya volcanic
knoll = NATURAL 5/5). Lesson: prompt changes need a labeled anchor set and
repeats — single-call validation cannot distinguish stochastic flips from
systematic misclassification.

Chips are (true-color | DEM hillshade) side-by-side panels ~2.6 km across.
The endpoint is any Anthropic-Messages-compatible API (default: the local
antigravity proxy).
"""
import base64
import json
import time
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "http://localhost:8080/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-6"

PROMPT_V3 = (
    "You are a Near-East survey archaeologist. LEFT=Sentinel-2 true color, "
    "RIGHT=DEM hillshade; red+ = site (~2.6 km across). Classify the flagged feature. "
    "RULE 1 - ANCIENT (tell): a SINGLE smooth rounded or flat-topped EARTHEN mound "
    "rising in ISOLATION from an otherwise FLAT agricultural plain. Judge the MOUND, "
    "not what sits on it: most tells carry a modern village on their summit (still "
    "ANCIENT), and bare tells often show excavation trenches or a stepped profile "
    "(still ANCIENT). Buff/pale soil, an outer town halo, or radiating hollow-ways "
    "strengthen the call. "
    "RULE 2 - MODERN: buildings/roads/fields/industry on FLAT ground with NO "
    "underlying mound in the hillshade. "
    "RULE 3 - NATURAL: dark rocky/volcanic cones, rough boulder texture, dendritic "
    "erosion gullies, or one of a CLUSTER/trend of similar hills - natural relief is "
    "regional, tells are isolated anomalies. "
    "The hillshade is the primary evidence for mound vs no-mound; the optical color "
    "decides earthen-buff vs dark-rocky. "
    'Reply strict JSON: {"class":"ANCIENT|MODERN|NATURAL","confidence":0-1,"reason":"one sentence"}'
)


def classify_chip(png_path, model: str = DEFAULT_MODEL,
                  endpoint: str = DEFAULT_ENDPOINT,
                  prompt: str = PROMPT_V3, retries: int = 3) -> dict:
    """Classify one candidate chip. Returns {'class','confidence','reason'}."""
    img = base64.b64encode(Path(png_path).read_bytes()).decode()
    body = json.dumps({"model": model, "max_tokens": 400, "messages": [
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/png", "data": img}},
            {"type": "text", "text": prompt}]}]}).encode()
    req = urllib.request.Request(endpoint, data=body, headers={
        "content-type": "application/json",
        "anthropic-version": "2023-06-01", "x-api-key": "dummy"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=150) as r:
                d = json.loads(r.read())
            txt = "".join(b.get("text", "") for b in d.get("content", [])
                          if b.get("type") == "text")
            return json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
        except Exception:
            time.sleep(3 * (attempt + 1))
    return {"class": "ERROR", "confidence": 0, "reason": "vlm call failed"}
