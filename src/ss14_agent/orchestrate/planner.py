"""Orchestrator client: a chat model (OpenAI-compatible or Anthropic) given a screenshot, returning a JSON plan.

The decision model is fast and good at picking from a menu, but it only sees
one moment and can't reason about goals. The orchestrator runs every few
decisions instead of every tick: it reads the screen, the chat, the memory and
what has happened, and writes the goal, a short plan, which step is current,
and optionally a line to say. That plan goes into every decision request.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import OrchestratorCfg
from ..providers import auth_headers

SYSTEM_PROMPT = """You are the strategic brain of an AI agent playing Space Station 14, a \
multiplayer space-station role-playing game, as a crew member. You see a screenshot of the game \
(top-down, north is up, your character is in the cyan box at the centre, numbered yellow boxes \
are things the agent can target) and a JSON description of the agent's state, memory and recent \
actions.

A fast decision model controls the character. Several times a second it picks one action from a \
fixed menu (walk to a marker, click or activate a marker, inspect, explore a direction, walk back \
to something found earlier, use/drop/swap held items, say a phrase, wait). It can't plan or \
remember. Your job is to give it direction: decide what the character should be trying to do \
right now and break it into a few concrete steps it can carry out with that menu.

Guidelines:
- Stay in role and do the job described in your_role. Follow instruction_from_your_operator if \
present.
- Steps must be things visible on screen or reachable with the menu, e.g. "walk to marker 3 (the \
mop)" or "explore east, toward the unexplored area". Refer to markers by number when you can.
- Look at your_recent_actions: if a step keeps failing, change approach.
- Use the memory: go back to things already identified instead of searching again; explore \
directions marked unexplored.
- Only set "say" when someone spoke to the character or speaking clearly helps. Keep it short, \
in character, plain text, and never more than one sentence.

Reply with one JSON object and nothing else:
{"goal": "<what you're trying to achieve, one sentence>",
 "duty": "<one duty id from the list, or null>",
 "steps": ["<step 1>", "<step 2>", "..."],
 "current_step": <index of the step to do now, starting at 0>,
 "say": "<a line to say in chat, or null>",
 "note": "<anything worth remembering for later, or null>"}"""


class OrchestratorError(RuntimeError):
    pass


@dataclass
class Plan:
    goal: str
    steps: list[str]
    current_step: int = 0
    duty: str | None = None
    say: str | None = None
    note: str | None = None
    model: str = ""
    latency_ms: float = 0.0
    t: float = field(default_factory=time.time)

    @property
    def step(self) -> str:
        if not self.steps:
            return ""
        return self.steps[max(0, min(self.current_step, len(self.steps) - 1))]

    def for_state(self) -> dict[str, Any]:
        return {"goal": self.goal, "steps": self.steps, "do_now": self.step}


def _clean_line(text: Any, limit: int) -> str | None:
    if not isinstance(text, str):
        return None
    text = " ".join(text.split()).strip().strip('"')
    if not text or text.lower() in ("null", "none"):
        return None
    return text[:limit]


def parse_plan(text: str, duties: list[str]) -> Plan:
    """Pull the JSON object out of a model reply (tolerating code fences and chatter)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise OrchestratorError(f"No JSON object in orchestrator reply: {text[:200]!r}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise OrchestratorError(f"Orchestrator reply isn't valid JSON ({exc}): {text[:200]!r}") from exc
    steps = [s for s in (_clean_line(x, 200) for x in data.get("steps") or []) if s][:6]
    goal = _clean_line(data.get("goal"), 240) or (steps[0] if steps else "")
    if not goal:
        raise OrchestratorError(f"Orchestrator plan has no goal: {text[:200]!r}")
    try:
        cur = int(data.get("current_step") or 0)
    except (TypeError, ValueError):
        cur = 0
    duty = data.get("duty")
    return Plan(
        goal=goal,
        steps=steps or [goal],
        current_step=max(0, min(cur, max(len(steps) - 1, 0))),
        duty=duty if duty in duties else None,
        say=_clean_line(data.get("say"), 120),
        note=_clean_line(data.get("note"), 200),
    )


class Orchestrator:
    def __init__(self, cfg: OrchestratorCfg, http: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self._http = http or httpx.Client(timeout=cfg.timeout_s)

    def plan(self, state: dict[str, Any], duties: list[str], image_b64: str | None = None,
             previous: Plan | None = None, media_type: str = "image/jpeg") -> Plan:
        prov = self.cfg.provider_info
        key = self.cfg.api_key
        if self.cfg.key_var and not key:
            raise OrchestratorError(f"No API key for {prov.name}: paste it in the control window or set {self.cfg.key_var}")
        context = dict(state)
        context["duty_ids"] = duties
        if previous is not None:
            context["your_previous_plan"] = {**previous.for_state(), "made_seconds_ago": round(time.time() - previous.t)}
        text_part = json.dumps(context, ensure_ascii=False)
        headers = {"Content-Type": "application/json", **auth_headers(prov, key)}
        if prov.api == "anthropic":
            content: list[dict[str, Any]] = [{"type": "text", "text": text_part}]
            if image_b64:
                content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}})
            url = f"{self.cfg.base_url}/messages"
            payload = {"model": self.cfg.model, "system": SYSTEM_PROMPT, "max_tokens": self.cfg.max_tokens,
                       "temperature": 0.3, "messages": [{"role": "user", "content": content}]}
        else:
            content = [{"type": "text", "text": text_part}]
            if image_b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}})
            url = f"{self.cfg.base_url}/chat/completions"
            headers["X-Title"] = "ss14-agent"
            payload = {
                "model": self.cfg.model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
                "max_tokens": self.cfg.max_tokens,
                "temperature": 0.3,
            }
        started = time.perf_counter()
        try:
            resp = self._http.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise OrchestratorError(f"Request to {prov.name} failed: {exc}") from exc
        latency = (time.perf_counter() - started) * 1000
        if resp.status_code >= 400:
            raise OrchestratorError(f"HTTP {resp.status_code} from {prov.name}: {resp.text[:300]}")
        try:
            body = resp.json()
            if prov.api == "anthropic":
                text = "".join(b.get("text", "") for b in body["content"] if b.get("type") == "text")
            else:
                text = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise OrchestratorError(f"Unexpected response from {prov.name}: {resp.text[:300]}") from exc
        if isinstance(text, list):  # some providers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        plan = parse_plan(text or "", duties)
        plan.model, plan.latency_ms = self.cfg.model, latency
        return plan
