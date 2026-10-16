"""Model providers the agent can talk to, and how to check a connection to each.

Two roles need different APIs:

- decision providers serve a System One style decisions endpoint (choice / noul
  / score questions answered with probabilities);
- orchestrator providers serve a normal chat API that accepts images. Most are
  OpenAI-compatible (`/chat/completions`); Anthropic uses its Messages API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    api: str                  # "decisions" | "openai" | "anthropic"
    base_url: str             # decisions: the full endpoint; chat APIs: the base the paths hang off
    key_env: str = ""         # "" means no key needed (local servers)
    image_mode: str = "parts" # decisions only: how screenshots are attached
    headers: dict[str, str] = field(default_factory=dict)
    default_model: str = ""
    models: tuple[str, ...] = ()
    key_url: str = ""         # where to get a key

    @property
    def needs_key(self) -> bool:
        return bool(self.key_env)

    def key(self) -> str:
        return os.environ.get(self.key_env, "") if self.key_env else ""


DECISION_PROVIDERS: dict[str, Provider] = {p.id: p for p in [
    Provider("openrouter", "OpenRouter", "decisions", "https://openrouter.ai/api/alpha/decisions",
             "OPENROUTER_API_KEY", "parts", default_model="inception/mercury-decide:free",
             models=("inception/mercury-decide:free", "inception/mercury-decide", "typesafe/jev-1.13"),
             key_url="https://openrouter.ai/settings/keys"),
    Provider("typesafe", "TypeSafe (Jev)", "decisions", "https://api.typesafe.ai/v1/systemone",
             "TYPESAFE_API_KEY", "assets", headers={"TypeSafe-Danger-Zone-Images": "true"},
             default_model="jev-latest", models=("jev-latest",), key_url="https://typesafe.ai"),
]}

ORCHESTRATOR_PROVIDERS: dict[str, Provider] = {p.id: p for p in [
    Provider("openrouter", "OpenRouter", "openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
             default_model="anthropic/claude-sonnet-5.5", key_url="https://openrouter.ai/settings/keys"),
    Provider("anthropic", "Anthropic", "anthropic", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY",
             default_model="claude-sonnet-5-5", key_url="https://console.anthropic.com/settings/keys"),
    Provider("openai", "OpenAI", "openai", "https://api.openai.com/v1", "OPENAI_API_KEY",
             key_url="https://platform.openai.com/api-keys"),
    Provider("google", "Google Gemini", "openai", "https://generativelanguage.googleapis.com/v1beta/openai",
             "GEMINI_API_KEY", key_url="https://aistudio.google.com/apikey"),
    Provider("xai", "xAI", "openai", "https://api.x.ai/v1", "XAI_API_KEY", key_url="https://console.x.ai"),
    Provider("groq", "Groq", "openai", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
             key_url="https://console.groq.com/keys"),
    Provider("ollama", "Ollama (local)", "openai", "http://localhost:11434/v1"),
    Provider("lmstudio", "LM Studio (local)", "openai", "http://localhost:1234/v1"),
]}

ANTHROPIC_VERSION = "2023-06-01"


def auth_headers(p: Provider, key: str) -> dict[str, str]:
    if p.api == "anthropic":
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
    return {"Authorization": f"Bearer {key}"} if key else {}


class ConnectionFailed(RuntimeError):
    pass


def _short_error(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", body)
        msg = err.get("message") if isinstance(err, dict) else str(err)
    except ValueError:
        msg = resp.text
    return f"HTTP {resp.status_code}: {str(msg)[:160]}"


def list_models(p: Provider, key: str | None = None, timeout: float = 15.0) -> list[str]:
    """Model ids the provider offers. Raises ConnectionFailed on a bad key or no connection."""
    key = p.key() if key is None else key
    if p.needs_key and not key:
        raise ConnectionFailed("no key")
    try:
        if p.api == "decisions":
            return list(p.models)
        resp = httpx.get(f"{p.base_url}/models", headers=auth_headers(p, key), timeout=timeout)
    except httpx.HTTPError as exc:
        raise ConnectionFailed(f"can't reach {p.base_url}: {exc}") from exc
    if resp.status_code >= 400:
        raise ConnectionFailed(_short_error(resp))
    data = resp.json().get("data", [])
    return sorted(m["id"] for m in data if isinstance(m, dict) and "id" in m)


def check(p: Provider, key: str | None = None, model: str = "", timeout: float = 15.0) -> str:
    """Prove the key works. Returns a short success note; raises ConnectionFailed otherwise."""
    key = p.key() if key is None else key
    if p.needs_key and not key:
        raise ConnectionFailed("no key")
    try:
        if p.id == "openrouter":
            resp = httpx.get("https://openrouter.ai/api/v1/key", headers=auth_headers(p, key), timeout=timeout)
            if resp.status_code >= 400:
                raise ConnectionFailed(_short_error(resp))
            label = (resp.json().get("data") or {}).get("label") or "key"
            return f"OpenRouter {label} OK"
        if p.api == "decisions":
            # No key-check endpoint: ask one tiny yes/no question.
            payload = {"model": model or p.default_model, "state": {"x": 1},
                       "questions": {"ok": {"type": "noul", "instructions": "Is x equal to 1?"}}}
            resp = httpx.post(p.base_url, json=payload, timeout=timeout,
                              headers={**auth_headers(p, key), "Content-Type": "application/json"})
            if resp.status_code >= 400:
                raise ConnectionFailed(_short_error(resp))
            return f"{p.name} OK"
    except httpx.HTTPError as exc:
        raise ConnectionFailed(f"can't reach {p.name}: {exc}") from exc
    n = len(list_models(p, key, timeout))
    return f"{p.name} OK, {n} models"
