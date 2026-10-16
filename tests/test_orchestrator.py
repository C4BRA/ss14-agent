import json
import time

import httpx
import pytest

from fakes import FakeWorld, fake_layout, fake_platform
from ss14_agent import models
from ss14_agent.agent import Agent
from ss14_agent.config import load_config
from ss14_agent.decide.client import Answer, DecisionResult
from ss14_agent.orchestrate import Orchestrator, OrchestratorError, Plan, parse_plan

DUTIES = ["get_equipment", "clean_spills", "patrol"]


def test_parse_plan_tolerates_fences_and_bad_fields():
    reply = """Sure! ```json
{"goal": "Get the mop", "duty": "get_equipment", "steps": ["walk to marker 2", "click marker 2"],
 "current_step": 7, "say": "  On my way!  ", "note": null}
```"""
    p = parse_plan(reply, DUTIES)
    assert p.goal == "Get the mop" and p.duty == "get_equipment"
    assert p.current_step == 1 and p.step == "click marker 2"
    assert p.say == "On my way!" and p.note is None
    assert parse_plan('{"goal": "x", "duty": "fly", "steps": []}', DUTIES).duty is None
    with pytest.raises(OrchestratorError):
        parse_plan("no json here", DUTIES)


def test_orchestrator_sends_image_and_parses(monkeypatch):
    seen = {}

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen["body"] = body
        content = {"goal": "Patrol the hall", "duty": "patrol", "steps": ["explore east"], "current_step": 0}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})

    cfg = load_config().orchestrator
    monkeypatch.setenv(cfg.key_var, "k")
    orch = Orchestrator(cfg, httpx.Client(transport=httpx.MockTransport(handler)))
    plan = orch.plan({"your_role": "Janitor"}, DUTIES, "QUJD")
    assert plan.goal == "Patrol the hall" and plan.model == cfg.model
    user = seen["body"]["messages"][1]["content"]
    assert user[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert json.loads(user[0]["text"])["duty_ids"] == DUTIES


def test_model_lists():
    catalog = [
        {"id": "inception/mercury-decide:free", "architecture": {"input_modalities": ["text", "image"]}},
        {"id": "typesafe/jev-1.13", "architecture": {"input_modalities": ["text", "image"]}},
        {"id": "anthropic/claude-sonnet-5.5", "architecture": {"input_modalities": ["text", "image"],
                                                                "output_modalities": ["text"]}},
        {"id": "some/text-only", "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}},
    ]
    dec = models.decision_models(catalog, ["inception/mercury-decide:free"])
    assert dec == ["inception/mercury-decide:free", "typesafe/jev-1.13"]
    assert models.vision_models(catalog, []) == ["anthropic/claude-sonnet-5.5"]


class FakeOrchestrator:
    def __init__(self):
        self.n = 0

    def plan(self, state, duties, image_b64=None, previous=None, media_type="image/jpeg"):
        self.n += 1
        return Plan(goal="Find a spill", steps=["explore north", "mop it"], duty="clean_spills", say="Wet floor!")


class RecordingClient:
    def __init__(self):
        self.calls = []

    def decide(self, state, questions, image_b64=None, image_mode=None, media_type="image/jpeg"):
        self.calls.append((state, questions))
        ans = {
            "duty": Answer("choice", choice="patrol", confidence=0.9, probabilities={"patrol": 0.9}),
            "action": Answer("choice", choice="wait", confidence=0.9, probabilities={"wait": 0.9}),
            "spoken_to": Answer("noul", value=0.0),
            "danger": Answer("noul", value=0.0),
        }
        return DecisionResult(ans, 5.0, {})


def test_agent_follows_the_plan(tmp_path, monkeypatch):
    cfg = fake_layout(load_config())
    cfg.agent.save_frames = False
    cfg.agent.tick_hz = 60
    monkeypatch.setattr("ss14_agent.agent.REPO_ROOT", tmp_path)
    client, orch = RecordingClient(), FakeOrchestrator()
    agent = Agent(cfg, fake_platform(FakeWorld()), "janitor", client=client, orchestrator=orch)
    agent.start()
    try:
        deadline = time.time() + 6
        while time.time() < deadline and not any("plan_from_your_orchestrator" in s for s, _ in client.calls):
            time.sleep(0.05)
    finally:
        agent.stop()
    assert orch.n >= 1
    state, questions = next((s, q) for s, q in client.calls if "plan_from_your_orchestrator" in s)
    assert state["plan_from_your_orchestrator"]["do_now"] == "explore north"
    assert "say:plan" in questions["action"]["criteria"]
    # The orchestrator's duty wins over the decision model's.
    assert agent.status.duty == "clean_spills"
    assert (tmp_path / "logs").exists() and list((tmp_path / "logs").glob("*/plans.jsonl"))
