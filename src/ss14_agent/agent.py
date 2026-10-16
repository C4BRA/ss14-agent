"""The agent: a fast control loop (capture -> perceive -> run skill), a decision loop, and a planner.

    control thread, tick_hz:     grab frame, update odometry/markers/chat, tick the current skill
    decision thread, on demand:  annotated screenshot + state + plan -> decision model -> next skill
    planner thread, every ~20 s: screenshot + state + memory -> orchestrator LLM -> goal and steps

Decisions are requested when the current skill finishes, when new chat
appears, or on a heartbeat. The character keeps moving while a decision is
in flight.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from . import playbook as playbooks
from .annotate import annotate, to_jpeg_b64
from .config import REPO_ROOT, Config
from .decide.actions import ActionSpec, build_actions, build_questions, build_state
from .decide.client import DecisionClient, DecisionError, DecisionResult
from .memory import WorldMemory
from .orchestrate import Orchestrator, OrchestratorError, Plan
from .perception.observer import Observation, Observer
from .platform import Platform, WindowInfo
from .skills import Skill, SkillContext, make_skill

log = logging.getLogger("ss14_agent")


@dataclass
class DecisionView:
    """What the control window shows about the latest decision."""

    t: float
    reason: str
    duty: str | None
    action: str
    confidence: float
    top: list[tuple[str, float]]
    spoken_to: float
    danger: float
    latency_ms: float
    error: str = ""


@dataclass
class AgentStatus:
    mode: str = "stopped"               # stopped | waiting | running | paused
    window: str = ""
    skill: str = ""
    duty: str | None = None
    decision: DecisionView | None = None
    fps: float = 0.0
    pos: tuple[float, float] = (0.0, 0.0)
    log: deque[str] = field(default_factory=lambda: deque(maxlen=300))
    log_count: int = 0
    plan: Plan | None = None
    plan_error: str = ""


class Agent:
    def __init__(self, cfg: Config, platform: Platform, playbook_name: str | None = None,
                 client: DecisionClient | None = None, orchestrator: Orchestrator | None = None) -> None:
        self.cfg = cfg
        self.p = platform
        self.playbook = playbooks.load(playbook_name or cfg.agent.playbook)
        self.client = client or DecisionClient(cfg.decision)
        self.orchestrator = orchestrator or (Orchestrator(cfg.orchestrator) if cfg.orchestrator.enabled else None)
        self._want_plan = threading.Event()
        self._plan_reason = "start"
        self._last_plan_at = 0.0
        self._fails_in_row = 0
        self.observer = Observer(cfg, platform.ocr, platform.windows.screen_size())
        self.status = AgentStatus()
        self.operator_note = ""
        self.preview: Image.Image | None = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._want_decision = threading.Event()
        self._decision_reason = "start"
        self._decision_busy = False
        self._results: deque[tuple[str, DecisionResult | None, list[ActionSpec], str]] = deque()
        self._latest_obs: Observation | None = None
        self._win: WindowInfo | None = None

        self._skill: Skill | None = None
        self._skill_spec: ActionSpec | None = None
        self._history: deque[dict[str, str]] = deque(maxlen=20)
        self._notes: deque[str] = deque(maxlen=30)
        self.memory = WorldMemory()
        self._duty_since = time.monotonic()
        self._duty_actions = 0
        self._last_decision_at = 0.0
        self._last_say_at = -1e9
        self.ctx = SkillContext(input=platform.input, ocr=platform.ocr, on_label=self._on_label)

        self._threads: list[threading.Thread] = []
        self._trace_dir: Path | None = None
        self._trace_n = 0

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._paused.clear()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._trace_dir = REPO_ROOT / "logs" / stamp
        (self._trace_dir / "frames").mkdir(parents=True, exist_ok=True)
        if self.cfg.agent.takeover_on_input:
            self.p.watcher.start(self._on_user_input, self._on_hotkey)
        self._threads = [
            threading.Thread(target=self._control_loop, name="control", daemon=True),
            threading.Thread(target=self._decision_loop, name="decision", daemon=True),
        ]
        if self.orchestrator is not None:
            self._threads.append(threading.Thread(target=self._plan_loop, name="planner", daemon=True))
            self._request_plan("start")
        for t in self._threads:
            t.start()
        self._say(f"Agent started as {self.playbook.name}. Logs: {self._trace_dir}")

    def stop(self) -> None:
        self._stop.set()
        self._want_decision.set()
        self._want_plan.set()
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []
        self._end_skill(None)
        self.p.input.release_all()
        self.p.watcher.stop()
        self.status.mode = "stopped"
        self._say("Agent stopped.")

    def pause(self, reason: str = "paused") -> None:
        if not self._paused.is_set():
            self._paused.set()
            self._results.clear()
            self.p.input.release_all()
            self._say(f"Paused ({reason}). Press F8 or Resume to continue.")

    def resume(self) -> None:
        if self._paused.is_set():
            self._paused.clear()
            if self._win:
                self.p.windows.activate(self._win)
            self._request_decision("resume")
            self._say("Resumed.")

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    # ---- callbacks -------------------------------------------------------

    def _on_user_input(self) -> None:
        if self._threads and not self._paused.is_set():
            self.pause("you took over")

    def _on_hotkey(self) -> None:
        if self._paused.is_set():
            self.resume()
        else:
            self.pause("F8")

    def _on_label(self, x: float, y: float, name: str) -> None:
        self.memory.add_landmark(x, y, name)
        self._notes.append(f"Inspected something at ({x:.0f}, {y:.0f}): {name}")

    def _say(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        self.status.log.append(line)
        self.status.log_count += 1
        log.info(msg)

    # ---- control loop ----------------------------------------------------

    def _known_labels(self, obs: Observation) -> dict[int, str]:
        known: dict[int, str] = {}
        for m in obs.markers:
            name = self.memory.label_at(m.world)
            if name:
                known[m.id] = name
        return known

    def _request_decision(self, reason: str) -> None:
        self._decision_reason = reason
        self._want_decision.set()

    def _end_skill(self, outcome: str | None) -> None:
        if self._skill is not None:
            self._skill.stop(self.ctx)
            if outcome is not None and self._skill_spec is not None:
                spec = self._skill_spec
                self._history.append({"duty": self.status.duty or "-", "action": spec.id,
                                      "what": spec.description, "result": outcome})
                self.memory.record(spec.skill, spec.args.get("world"), outcome)
                self._duty_actions += 1
                self._fails_in_row = self._fails_in_row + 1 if outcome.startswith("failed") else 0
                if self._fails_in_row >= 2:
                    self._request_plan("repeated failures")
        self._skill = None
        self._skill_spec = None
        self.status.skill = ""

    def _start_skill(self, spec: ActionSpec, obs: Observation) -> None:
        self._end_skill("interrupted by a new decision" if self._skill else None)
        try:
            skill = make_skill(spec.skill, spec.args, self.cfg.agent.reach_tiles)
        except Exception as exc:
            self._say(f"Couldn't start {spec.id}: {exc}")
            self._request_decision("error")
            return
        if spec.skill == "say":
            self._last_say_at = time.monotonic()
            plan = self.status.plan
            if spec.id == "say:plan" and plan is not None:
                plan.say = None  # said it; don't offer the same line again
        self._skill, self._skill_spec = skill, spec
        skill.start(self.ctx, obs)
        self.status.skill = skill.describe()
        self._say(f"→ {spec.id}: {skill.describe()}")

    def _same_target(self, a: ActionSpec, b: ActionSpec) -> bool:
        if a.skill != b.skill:
            return False
        wa, wb = a.args.get("world"), b.args.get("world")
        if wa and wb:
            return math.hypot(wa[0] - wb[0], wa[1] - wb[1]) < 1.0
        return a.args == b.args

    def _apply_results(self, obs: Observation) -> None:
        while self._results:
            reason, result, actions, error = self._results.popleft()
            if result is None:
                continue
            spec = self._pick(result, actions)
            if spec is None:
                continue
            if self._skill is None:
                self._start_skill(spec, obs)
            elif self._should_interrupt(reason, result, spec):
                self._start_skill(spec, obs)

    def _should_interrupt(self, reason: str, result: DecisionResult, spec: ActionSpec) -> bool:
        """Whether a fresh decision may cut short the skill that's running.

        Letting every heartbeat swap the plan made the agent dither: two near-equal
        options would alternate and it never got anywhere. Now a running skill is only
        replaced for chat, danger, a new plan, or a heartbeat pick the model is confident about.
        """
        if reason not in ("chat", "heartbeat", "plan") or spec.skill == "wait" or self._same_target(spec, self._skill_spec):
            return False
        a = result.answers
        if "danger" in a and a["danger"].yes > 0.6:
            return True
        if reason == "chat":
            return "spoken_to" in a and a["spoken_to"].yes > 0.5
        act = a.get("action")
        # A fresh plan is a deliberate change of direction, so it needs less certainty.
        return act is not None and act.confidence >= (0.4 if reason == "plan" else 0.6)

    def _pick(self, result: DecisionResult, actions: list[ActionSpec]) -> ActionSpec | None:
        by_id = {a.id: a for a in actions}
        ans = result.answers.get("action")
        if ans is None or ans.choice not in by_id:
            return by_id.get("wait")
        if ans.confidence < self.cfg.decision.min_confidence:
            return by_id.get("wait")
        return by_id[ans.choice]

    def _grab(self) -> Observation | None:
        if self._win is None or self.observer._n % 30 == 0:
            win = self.p.windows.find(self.cfg.game.window_match)
            if win is None:
                self._win = None
                return None
            if self._win is None:
                self._say(f"Attached to window '{win.title or win.owner}' ({win.width:.0f}x{win.height:.0f})")
                self.p.windows.activate(win)
            self._win = win
        frame = self.p.capturer.grab(self._win)
        return self.observer.observe(frame, self._win)

    def _control_loop(self) -> None:
        period = 1.0 / self.cfg.agent.tick_hz
        last = time.monotonic()
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                obs = self._grab()
            except Exception as exc:
                self._say(f"Capture failed: {exc}")
                obs = None
            if obs is None:
                self.status.mode = "waiting"
                self.status.window = "SS14 window not found"
                self._end_skill(None)
                time.sleep(1.0)
                continue
            self.status.window = self._win.title or self._win.owner
            with self._lock:
                self._latest_obs = obs
            self.status.pos = obs.pos
            self.memory.update(obs)

            if self._paused.is_set():
                self.status.mode = "paused"
                if self._skill:
                    self._end_skill("paused by operator")
            else:
                self.status.mode = "running"
                self._apply_results(obs)
                if self._skill is not None:
                    st = self._skill.tick(self.ctx, obs)
                    if st.done:
                        outcome = st.state + (f": {st.detail}" if st.detail else "")
                        self._say(f"   {self.status.skill} → {outcome}")
                        self._end_skill(outcome)
                        self._request_decision("idle")
                elif not self._decision_busy:
                    self._request_decision("idle")
                if obs.new_chat:
                    self._request_decision("chat")
                    self._request_plan("chat")
                elif time.monotonic() - self._last_decision_at > self.cfg.agent.heartbeat_s:
                    self._request_decision("heartbeat")

            if self.observer._n % 3 == 0:
                img = annotate(obs.frame, obs.geo, obs.markers, width=900)
                self.preview = img
            now = time.monotonic()
            self.status.fps = 0.9 * self.status.fps + 0.1 * (1.0 / max(now - last, 1e-3))
            last = now
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
        self.p.input.release_all()

    # ---- decision loop ---------------------------------------------------

    def _decision_loop(self) -> None:
        while not self._stop.is_set():
            self._want_decision.wait(timeout=0.5)
            if self._stop.is_set():
                break
            if self._paused.is_set():
                time.sleep(0.2)
                continue
            if not self._want_decision.is_set():
                continue
            self._want_decision.clear()
            with self._lock:
                obs = self._latest_obs
            if obs is None:
                continue
            reason = self._decision_reason
            self._decision_busy = True
            try:
                self._decide(obs, reason)
            finally:
                self._decision_busy = False
                self._last_decision_at = time.monotonic()

    def _state(self, obs: Observation, known: dict[int, str]) -> dict[str, Any]:
        duty_info = None
        if self.status.duty:
            duty_info = {
                "minutes_on_this_duty": round((time.monotonic() - self._duty_since) / 60, 1),
                "actions_taken_on_this_duty": self._duty_actions,
            }
        return build_state(obs, self.playbook, self.status.duty, list(self._history), list(self._notes),
                           self.operator_note, known, self.memory, duty_info)

    def _set_duty(self, duty: str, why: str) -> None:
        if duty != self.status.duty:
            self._say(f"Duty: {duty} ({why})")
            self._duty_since = time.monotonic()
            self._duty_actions = 0
        self.status.duty = duty

    def _decide(self, obs: Observation, reason: str) -> None:
        known = self._known_labels(obs)
        can_chat = time.monotonic() - self._last_say_at > self.cfg.agent.chat_cooldown_s
        actions = build_actions(obs, self.playbook, self.cfg.agent.reach_tiles, can_chat, known, self.memory)
        plan = self.status.plan
        if plan is not None and plan.say and can_chat:
            actions.append(ActionSpec("say:plan", "say", f'Say in local chat: "{plan.say}"', {"text": plan.say}))
        state = self._state(obs, known)
        if plan is not None:
            state["plan_from_your_orchestrator"] = plan.for_state()
        questions = build_questions(self.playbook, actions)
        img = annotate(obs.frame, obs.geo, obs.markers, width=self.cfg.decision.image_width)
        b64 = to_jpeg_b64(img)
        try:
            result = self.client.decide(state, questions, b64)
        except DecisionError as exc:
            self._say(f"Decision failed: {exc}")
            self.status.decision = DecisionView(time.time(), reason, self.status.duty, "", 0, [], 0, 0, 0, str(exc))
            self._results.append((reason, None, actions, str(exc)))
            time.sleep(1.0)
            return
        a = result.answers
        duty = a.get("duty")
        # The orchestrator's duty wins; the decision model only picks one when there's no plan.
        if duty and duty.choice and not (plan is not None and plan.duty):
            self._set_duty(duty.choice, f"{duty.confidence:.0%}")
        act = a.get("action")
        top = sorted((act.probabilities if act else {}).items(), key=lambda kv: -kv[1])[:8]
        view = DecisionView(
            t=time.time(), reason=reason, duty=self.status.duty,
            action=(act.choice or "") if act else "", confidence=act.confidence if act else 0.0,
            top=top, spoken_to=a["spoken_to"].yes if "spoken_to" in a else 0.0,
            danger=a["danger"].yes if "danger" in a else 0.0, latency_ms=result.latency_ms,
        )
        self.status.decision = view
        self._results.append((reason, result, actions, ""))
        self._trace(state, questions, result, img, reason)

    # ---- planner loop ----------------------------------------------------

    def _request_plan(self, reason: str) -> None:
        if self.orchestrator is not None:
            self._plan_reason = reason
            self._want_plan.set()

    def _plan_loop(self) -> None:
        while not self._stop.is_set():
            due = time.monotonic() - self._last_plan_at > self.cfg.orchestrator.interval_s
            if not self._want_plan.wait(timeout=0.5) and not due:
                continue
            if self._stop.is_set():
                break
            if self._paused.is_set():
                time.sleep(0.2)
                continue
            if not self.cfg.orchestrator.enabled:  # switched off from the control window
                self.status.plan = None
                self._want_plan.clear()
                time.sleep(0.5)
                continue
            # Chat and failures can fire in bursts; don't re-plan more than every few seconds.
            if time.monotonic() - self._last_plan_at < 4.0:
                time.sleep(0.2)
                continue
            reason = self._plan_reason if self._want_plan.is_set() else "interval"
            self._want_plan.clear()
            with self._lock:
                obs = self._latest_obs
            if obs is None:
                time.sleep(0.5)
                continue
            self._plan(obs, reason)
            self._last_plan_at = time.monotonic()

    def _plan(self, obs: Observation, reason: str) -> None:
        state = self._state(obs, self._known_labels(obs))
        img = annotate(obs.frame, obs.geo, obs.markers, width=self.cfg.orchestrator.image_width)
        duties = [d.id for d in self.playbook.duties]
        try:
            plan = self.orchestrator.plan(state, duties, to_jpeg_b64(img), self.status.plan)
        except OrchestratorError as exc:
            self.status.plan_error = str(exc)
            self._say(f"Planning failed: {exc}")
            return
        self.status.plan, self.status.plan_error = plan, ""
        self._fails_in_row = 0
        if plan.duty:
            self._set_duty(plan.duty, "orchestrator")
        if plan.note:
            self._notes.append(f"Orchestrator note: {plan.note}")
        self._say(f"Plan ({reason}, {plan.latency_ms / 1000:.1f} s): {plan.goal} → now: {plan.step}")
        self._request_decision("plan")
        if self._trace_dir is not None:
            rec = {"t": time.time(), "reason": reason, "model": plan.model, "latency_ms": round(plan.latency_ms),
                   "goal": plan.goal, "steps": plan.steps, "current_step": plan.current_step,
                   "duty": plan.duty, "say": plan.say, "note": plan.note}
            with (self._trace_dir / "plans.jsonl").open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def _trace(self, state: dict[str, Any], questions: dict[str, Any], result: DecisionResult,
               img: Image.Image, reason: str) -> None:
        if self._trace_dir is None:
            return
        self._trace_n += 1
        rec = {
            "n": self._trace_n, "t": time.time(), "reason": reason, "latency_ms": round(result.latency_ms, 1),
            "state": state, "options": list(questions["action"]["criteria"].keys()),
            "answers": {k: vars(v) for k, v in result.answers.items()},
        }
        with (self._trace_dir / "decisions.jsonl").open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        if self.cfg.agent.save_frames:
            img.convert("RGB").save(self._trace_dir / "frames" / f"{self._trace_n:05d}.jpg", quality=80)
