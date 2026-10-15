"""Job playbooks: the standing duties a role has, and how to do them.

A decision model only chooses between options. The playbook gives it the
structure a vague goal like "do your job" lacks: a short list of duties to
prioritise, simple steps for each, and phrases it can say.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .config import PLAYBOOK_DIR


@dataclass
class Duty:
    id: str
    description: str
    cue: str = ""
    steps: list[str] = field(default_factory=list)


@dataclass
class Playbook:
    name: str
    summary: str
    duties: list[Duty]
    phrases: list[str]

    def duty(self, duty_id: str | None) -> Duty | None:
        return next((d for d in self.duties if d.id == duty_id), None)


def available() -> list[str]:
    return sorted(p.stem for p in PLAYBOOK_DIR.glob("*.toml"))


def load(name_or_path: str) -> Playbook:
    path = Path(name_or_path)
    if not path.suffix:
        path = PLAYBOOK_DIR / f"{name_or_path}.toml"
    data = tomllib.loads(path.read_text())
    duties = [Duty(**d) for d in data.get("duties", [])]
    if not duties:
        raise ValueError(f"Playbook {path} has no [[duties]]")
    ids = [d.id for d in duties]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Playbook {path} has duplicate duty ids")
    return Playbook(
        name=data.get("name", path.stem),
        summary=data.get("summary", "").strip(),
        duties=duties,
        phrases=list(data.get("phrases", [])),
    )
