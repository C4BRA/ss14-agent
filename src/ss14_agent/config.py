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
    masks: list[Rect] = field(default_factory=list)


@dataclass
class CaptureCfg:
    backend: str = "auto"


@dataclass
class DecisionCfg:
    provider: str = "openrouter"
    model: str = "inception/mercury-decide:free"
    endpoint: str = ""        # "" = the provider's
    api_key_env: str = ""     # "" = the provider's
    image_mode: str = ""      # "" = the provider's
    image_width: int = 1024
    timeout_s: float = 20
    max_calls_per_s: float = 2.0
    min_confidence: float = 0.30
    # Extra entries for the model picker (the provider's own list is added).
    models: list[str] = field(default_factory=list)

    @property
    def provider_info(self):
        from .providers import DECISION_PROVIDERS

        if self.provider not in DECISION_PROVIDERS:
            raise ValueError(f"Unknown decision provider '{self.provider}' (choose from {', '.join(DECISION_PROVIDERS)})")
        return DECISION_PROVIDERS[self.provider]

    @property
    def url(self) -> str:
        return self.endpoint or self.provider_info.base_url

    @property
    def key_var(self) -> str:
        return self.api_key_env or self.provider_info.key_env

    @property
    def api_key(self) -> str:
        return os.environ.get(self.key_var, "") if self.key_var else ""

    @property
    def layout(self) -> str:
        return self.image_mode or self.provider_info.image_mode


@dataclass
class OrchestratorCfg:
    """A general vision LLM that plans; the decision model carries out the plan."""

    enabled: bool = True
    provider: str = "openrouter"
    model: str = "anthropic/claude-sonnet-5.5"
    endpoint: str = ""        # "" = the provider's base URL
    api_key_env: str = ""     # "" = the provider's
    interval_s: float = 20.0
    image_width: int = 1024
    timeout_s: float = 60.0
    max_tokens: int = 700
    models: list[str] = field(default_factory=list)

    @property
    def provider_info(self):
        from .providers import ORCHESTRATOR_PROVIDERS

        if self.provider not in ORCHESTRATOR_PROVIDERS:
            raise ValueError(
                f"Unknown orchestrator provider '{self.provider}' (choose from {', '.join(ORCHESTRATOR_PROVIDERS)})"
            )
        return ORCHESTRATOR_PROVIDERS[self.provider]

    @property
    def base_url(self) -> str:
        return (self.endpoint or self.provider_info.base_url).rstrip("/")

    @property
    def key_var(self) -> str:
        return self.api_key_env or self.provider_info.key_env

    @property
    def api_key(self) -> str:
        return os.environ.get(self.key_var, "") if self.key_var else ""


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
        if k in ("viewport", "chat") and isinstance(v, list):
            v = tuple(v)
        elif k == "masks":
            v = [tuple(r) for r in v]
        kwargs[k] = v
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
