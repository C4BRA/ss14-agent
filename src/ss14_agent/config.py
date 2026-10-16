"""Settings loading: config/default.toml, overlaid by config/local.toml or --config."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.toml"
LOCAL_CONFIG = REPO_ROOT / "config" / "local.toml"
PLAYBOOK_DIR = REPO_ROOT / "playbooks"

Rect = tuple[float, float, float, float]


@dataclass
class GameCfg:
    window_match: list[str] = field(default_factory=lambda: ["Space Station 14"])


@dataclass
class LayoutCfg:
    viewport: Rect = (0.0, 0.0, 0.70, 1.0)
    chat: Rect = (0.705, 0.40, 0.29, 0.52)
    viewport_tiles_high: int = 15
    viewport_tiles_wide: int = 21
    tile_px: float = 0
    titlebar_pt: float = -1


@dataclass
class CaptureCfg:
    backend: str = "auto"


@dataclass
class DecisionCfg:
    endpoint: str = "https://openrouter.ai/api/alpha/decisions"
    model: str = "inception/mercury-decide:free"
    api_key_env: str = "OPENROUTER_API_KEY"
    image_mode: str = "parts"
    image_width: int = 1024
    timeout_s: float = 20
    max_calls_per_s: float = 2.0
    min_confidence: float = 0.30
    # Shown in the model picker alongside any decision models OpenRouter lists.
    models: list[str] = field(default_factory=lambda: ["inception/mercury-decide:free", "inception/mercury-decide"])

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass
class OrchestratorCfg:
    """A general vision LLM that plans; the decision model carries out the plan."""

    enabled: bool = True
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "anthropic/claude-sonnet-5.5"
    api_key_env: str = "OPENROUTER_API_KEY"
    interval_s: float = 20.0
    image_width: int = 1024
    timeout_s: float = 60.0
    max_tokens: int = 700
    models: list[str] = field(default_factory=lambda: ["anthropic/claude-sonnet-5.5"])

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


@dataclass
class AgentCfg:
    tick_hz: float = 15
    playbook: str = "janitor"
    heartbeat_s: float = 4.0
    chat_cooldown_s: float = 15.0
    max_markers: int = 12
    reach_tiles: float = 1.4
    takeover_on_input: bool = True
    save_frames: bool = True


@dataclass
class OcrCfg:
    chat_interval_s: float = 1.0


@dataclass
class Config:
    game: GameCfg = field(default_factory=GameCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    capture: CaptureCfg = field(default_factory=CaptureCfg)
    decision: DecisionCfg = field(default_factory=DecisionCfg)
    orchestrator: OrchestratorCfg = field(default_factory=OrchestratorCfg)
    agent: AgentCfg = field(default_factory=AgentCfg)
    ocr: OcrCfg = field(default_factory=OcrCfg)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _section(cls, data: dict[str, Any]):
    known = {f for f in cls.__dataclass_fields__}
    kwargs = {}
    for k, v in data.items():
        if k not in known:
            raise ValueError(f"Unknown setting '{k}' in [{cls.__name__.removesuffix('Cfg').lower()}]")
        kwargs[k] = tuple(v) if isinstance(v, list) and k in ("viewport", "chat") else v
    return cls(**kwargs)


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader so users don't need python-dotenv. Existing env vars win."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | Path | None = None) -> Config:
    data: dict[str, Any] = {}
    if DEFAULT_CONFIG.exists():
        data = tomllib.loads(DEFAULT_CONFIG.read_text())
    override = Path(path) if path else (LOCAL_CONFIG if LOCAL_CONFIG.exists() else None)
    if override:
        data = _merge(data, tomllib.loads(Path(override).read_text()))
    sections = {
        "game": GameCfg, "layout": LayoutCfg, "capture": CaptureCfg,
        "decision": DecisionCfg, "orchestrator": OrchestratorCfg, "agent": AgentCfg, "ocr": OcrCfg,
    }
    unknown = set(data) - set(sections)
    if unknown:
        raise ValueError(f"Unknown config section(s): {', '.join(sorted(unknown))}")
    return Config(**{name: _section(cls, data.get(name, {})) for name, cls in sections.items()})
