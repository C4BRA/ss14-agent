"""Model lists for the pickers: OpenRouter's public catalog, filtered per role."""

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
