"""Client for System One style decision endpoints (OpenRouter Decisions API, TypeSafe Jev).

Request shape (shared by Jev and Mercury Decide):

    {"model": "...", "state": {...}, "questions": {"<id>": {"type": "choice" | "noul" | "score",
                                                         "instructions": "...", "criteria": ...}}}

- choice: criteria is {option_id: description}; answer has "choice", "confidence", "probabilities".
- noul (yes/no): criteria is {"true": ..., "false": ...}; answer has "noul" (probability of true).
- score: criteria is a list of level descriptions; answer has "score".

Image attachment isn't documented for the OpenRouter endpoint yet, so the
client supports several layouts, and `ss14-agent probe` finds the one that works.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import DecisionCfg

log = logging.getLogger(__name__)

IMAGE_MODES = ("assets", "assets_dataurl", "state_image", "none")


@dataclass
class Answer:
    kind: str
    choice: str | None = None
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    value: float | None = None   # noul probability-of-true, or score

    @property
    def yes(self) -> float:
        """Probability of 'true' for a noul question."""
        if self.value is not None:
            return self.value
        return float(self.probabilities.get("true", 0.0))


@dataclass
class DecisionResult:
    answers: dict[str, Answer]
    latency_ms: float
    raw: dict[str, Any]


class DecisionError(RuntimeError):
    pass


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    return None


def parse_answer(qtype: str, raw: Any) -> Answer:
    if not isinstance(raw, dict):
        return Answer(kind=qtype)
    probs = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items() if _as_float(v) is not None}
    ans = Answer(kind=qtype, probabilities=probs)
    if qtype == "choice":
        c = raw.get("choice")
        ans.choice = c if isinstance(c, str) and c else (max(probs, key=probs.get) if probs else None)
        conf = _as_float(raw.get("confidence"))
        ans.confidence = conf if conf is not None else (probs.get(ans.choice, 0.0) if ans.choice else 0.0)
    elif qtype == "noul":
        v = _as_float(raw.get("noul"))
        if v is None and "true" in probs:
            v = probs["true"]
        ans.value = v
        ans.confidence = max(v, 1 - v) if v is not None else 0.0
    else:
        ans.value = _as_float(raw.get("score"))
        conf = _as_float(raw.get("confidence"))
        ans.confidence = conf or 0.0
    return ans


def parse_response(questions: dict[str, dict[str, Any]], body: dict[str, Any]) -> dict[str, Answer]:
    answers = body.get("answers")
    if answers is None and isinstance(body.get("data"), dict):
        answers = body["data"].get("answers")
    if not isinstance(answers, dict):
        raise DecisionError(f"Response has no 'answers' object: {str(body)[:300]}")
    return {qid: parse_answer(q.get("type", "choice"), answers.get(qid)) for qid, q in questions.items()}


def build_payload(
    model: str,
    state: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    image_b64: str | None,
    image_mode: str,
    media_type: str = "image/jpeg",
) -> dict[str, Any]:
    state = dict(state)
    payload: dict[str, Any] = {"model": model, "state": state, "questions": questions}
    if image_b64 and image_mode != "none":
        data_url = f"data:{media_type};base64,{image_b64}"
        if image_mode == "assets":
            payload["assets"] = {"screen": {"type": "image", "media_type": media_type, "data": image_b64}}
            state["screen"] = "asset:screen"
        elif image_mode == "assets_dataurl":
            payload["assets"] = {"screen": data_url}
            state["screen"] = "asset:screen"
        elif image_mode == "state_image":
            state["screen"] = data_url
        else:
            raise ValueError(f"Unknown image_mode '{image_mode}' (expected one of {IMAGE_MODES})")
    return payload


class DecisionClient:
    def __init__(self, cfg: DecisionCfg, http: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self._http = http or httpx.Client(timeout=cfg.timeout_s)
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._backoff = 0.0

    def _wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + 1.0 / max(self.cfg.max_calls_per_s, 0.01)
        if wait > 0:
            time.sleep(wait)

    def decide(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
        image_b64: str | None = None,
        image_mode: str | None = None,
        media_type: str = "image/jpeg",
    ) -> DecisionResult:
        key = self.cfg.api_key
        if not key:
            raise DecisionError(f"No API key: set {self.cfg.api_key_env} in your environment or .env file")
        payload = build_payload(self.cfg.model, state, questions, image_b64, image_mode or self.cfg.image_mode, media_type)
        for _attempt in range(4):
            self._wait_turn()
            started = time.perf_counter()
            try:
                resp = self._http.post(
                    self.cfg.endpoint,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                             "X-Title": "ss14-agent"},
                    json=payload,
                )
            except httpx.HTTPError as exc:
                raise DecisionError(f"Request failed: {exc}") from exc
            latency = (time.perf_counter() - started) * 1000
            if resp.status_code == 429:
                self._backoff = min(30.0, max(2.0, self._backoff * 2))
                try:
                    retry_after = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    retry_after = self._backoff
                log.warning("Rate limited by the decision endpoint; waiting %.1fs", retry_after)
                time.sleep(retry_after)
                continue
            if resp.status_code >= 400:
                raise DecisionError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            self._backoff = 0.0
            body = resp.json()
            return DecisionResult(parse_response(questions, body), latency, body)
        raise DecisionError("Still rate limited after 4 attempts")
