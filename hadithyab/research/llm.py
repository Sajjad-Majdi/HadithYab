"""Plain text completion over a chain of free LLM endpoints.

Each entry of LLM_CHAIN is tried in turn until one answers:
  gemini:<model>                  Google AI Studio, keys from GEMINI_API_KEYS
  openai:<name>:<model>           any OpenAI-compatible API; <name> picks
                                  <NAME>_BASE_URL and <NAME>_API_KEYS
"""
import itertools
import os
import threading

import requests

from ..core.embed import GEMINI_BASE

# Each Gemini model has its own free quota, so the chain adds capacity as it falls through.
DEFAULT_CHAIN = "gemini:gemini-3.5-flash-lite,gemini:gemini-3.1-flash-lite,gemini:gemma-4-31b-it,gemini:gemma-4-26b-a4b-it"
CHAIN = [c.strip() for c in os.environ.get("LLM_CHAIN", DEFAULT_CHAIN).split(",") if c.strip()]

_session = requests.Session()
_rings = {}
_lock = threading.Lock()


def keys_for(name):
    raw = os.environ.get(f"{name.upper()}_API_KEYS") or os.environ.get(f"{name.upper()}_API_KEY", "")
    return [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]


def next_key(name):
    with _lock:
        if name not in _rings:
            _rings[name] = itertools.cycle(keys_for(name) or [""])
        return next(_rings[name])


def _gemini(model, prompt, temperature, max_tokens):
    config = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if model.startswith("gemini"):
        config["thinkingConfig"] = {"thinkingLevel": "minimal"}
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config}
    r = _session.post(f"{GEMINI_BASE}/v1beta/models/{model}:generateContent",
                      headers={"x-goog-api-key": next_key("gemini")}, json=body, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"{model} {r.status_code}: {r.text[:160]}")
    parts = r.json()["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _openai(name, model, prompt, temperature, max_tokens):
    base = os.environ[f"{name.upper()}_BASE_URL"].rstrip("/")
    r = _session.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {next_key(name)}"}, json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": max_tokens}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"{name}/{model} {r.status_code}: {r.text[:160]}")
    return r.json()["choices"][0]["message"]["content"] or ""


def complete(prompt, temperature=0.2, max_tokens=1024):
    error = None
    for entry in CHAIN:
        kind, _, rest = entry.partition(":")
        for _ in range(2):  # a second key for the same model before moving on
            try:
                if kind == "gemini":
                    return _gemini(rest, prompt, temperature, max_tokens)
                name, _, model = rest.partition(":")
                return _openai(name, model, prompt, temperature, max_tokens)
            except (requests.RequestException, RuntimeError, KeyError) as e:
                error = e
    raise error or RuntimeError("no LLM configured")
