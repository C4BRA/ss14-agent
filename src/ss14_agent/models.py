"""Model lists for the pickers, per provider and role."""

from __future__ import annotations

from typing import Any

import httpx

CATALOG_URL = "https://openrouter.ai/api/v1/models"
# OpenRouter doesn't flag Decisions API models in the catalog, so match by name.
DECISION_HINTS = ("decide", "decision", "jev", "systemone")


def fetch_catalog(timeout: float = 15.0) -> list[dict[str, Any]]:
    resp = httpx.get(CATALOG_URL, timeout=timeout)
    resp.raise_for_status()
    return list(resp.json().get("data", []))


def _merge(preferred: list[str], found: list[str]) -> list[str]:
    out = list(dict.fromkeys(preferred))
    out += sorted(m for m in set(found) if m not in out)
    return out


def decision_models(catalog: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    found = [m["id"] for m in catalog if any(h in m.get("id", "").lower() for h in DECISION_HINTS)]
    return _merge(preferred, found)


def vision_models(catalog: list[dict[str, Any]], preferred: list[str]) -> list[str]:
    """Chat models that accept images and write text: usable as the orchestrator."""
    found = []
    for m in catalog:
        arch = m.get("architecture") or {}
        mid = m.get("id", "")
        if ("image" in arch.get("input_modalities", []) and "text" in arch.get("output_modalities", ["text"])
                and not any(h in mid.lower() for h in DECISION_HINTS)):
            found.append(mid)
    return _merge(preferred, found)


def models_for(role: str, cfg) -> list[str]:
    """Model ids for the picker: role is "decision" or "orchestrator", cfg its config section.

    OpenRouter's public catalog is filtered by role; other providers list their own
    models (which needs a working key). Raises providers.ConnectionFailed on failure.
    """
    from . import providers

    prov = cfg.provider_info
    preferred = [m for m in [cfg.model, prov.default_model, *prov.models, *cfg.models] if m]
    if prov.id == "openrouter":
        try:
            catalog = fetch_catalog()
        except httpx.HTTPError as exc:
            raise providers.ConnectionFailed(f"can't load OpenRouter's model list: {exc}") from exc
        return decision_models(catalog, preferred) if role == "decision" else vision_models(catalog, preferred)
    if role == "decision":
        return _merge(preferred, [])
    return _merge(preferred, providers.list_models(prov, cfg.api_key))
