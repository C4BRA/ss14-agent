import json

import httpx
import pytest

from fakes import FakeWorld, fake_layout, WIN, FakeOcr
from ss14_agent import playbook as playbooks
from ss14_agent.config import load_config
from ss14_agent.decide.actions import build_actions, build_questions, build_state
from ss14_agent.decide.client import DecisionClient, DecisionError, build_payload, parse_response
from ss14_agent.perception.observer import Observer


def _obs():
    cfg = fake_layout(load_config())
    return Observer(cfg, FakeOcr()).observe(FakeWorld().frame(), WIN, read_chat=False)


def test_playbooks_load():
    names = playbooks.available()
    assert {"janitor", "engineer", "botanist"} <= set(names)
    for n in names:
        pb = playbooks.load(n)
        assert pb.duties and pb.phrases


def test_actions_and_questions_are_valid_for_the_api():
    obs = _obs()
    pb = playbooks.load("janitor")
    acts = build_actions(obs, pb, reach=1.4, can_chat=True, known={1: "mop"})
    ids = [a.id for a in acts]
    assert len(ids) == len(set(ids)) and len(ids) <= 255
    assert "wait" in ids and "say:0" in ids and "inspect:1" not in ids
    q = build_questions(pb, acts)
    assert set(q) == {"duty", "action", "spoken_to", "danger"}
    assert set(q["duty"]["criteria"]) == {d.id for d in pb.duties}
    assert q["spoken_to"]["type"] == "noul"
    state = build_state(obs, pb, "clean_spills", [{"action": "x", "result": "ok"}], [], "clean medbay", {1: "mop"})
    json.dumps(state)
    assert state["current_duty"]["id"] == "clean_spills"
    assert state["instruction_from_your_operator"] == "clean medbay"
    no_chat = build_actions(obs, pb, 1.4, can_chat=False)
    assert not any(a.id.startswith("say:") for a in no_chat)


def test_parse_response_variants():
    qs = {"a": {"type": "choice"}, "b": {"type": "noul"}, "c": {"type": "score"}, "d": {"type": "choice"}}
    body = {"answers": {
        "a": {"choice": "x", "confidence": 0.7, "probabilities": {"x": 0.7, "y": 0.3}},
        "b": {"noul": 0.2},
        "c": {"score": 3},
        "d": {"probabilities": {"p": 0.1, "q": 0.9}},
    }}
    ans = parse_response(qs, body)
    assert ans["a"].choice == "x" and ans["a"].confidence == 0.7
    assert ans["b"].yes == 0.2
    assert ans["c"].value == 3
    assert ans["d"].choice == "q" and ans["d"].confidence == 0.9
    nested = parse_response({"b": {"type": "noul"}}, {"data": {"answers": {"b": {"noul": True}}}})
    assert nested["b"].yes == 1.0
    with pytest.raises(DecisionError):
        parse_response(qs, {"error": "nope"})


def test_payload_image_modes():
    q = {"x": {"type": "noul"}}
    p = build_payload("m", {"a": 1}, q, "QUJD", "assets")
    assert p["assets"]["screen"]["data"] == "QUJD" and p["state"]["screen"] == "asset:screen"
    p = build_payload("m", {"a": 1}, q, "QUJD", "state_image")
    assert p["state"]["screen"].startswith("data:image/jpeg;base64,") and "assets" not in p
    p = build_payload("m", {"a": 1}, q, "QUJD", "none")
    assert "screen" not in p["state"]


def test_client_posts_and_retries_on_429(monkeypatch):
    calls = []

    def handler(request: httpx.Request):
        calls.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer k"
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"answers": {"x": {"noul": 0.9}}})

    cfg = load_config().decision
    cfg.max_calls_per_s = 1000
    monkeypatch.setenv(cfg.api_key_env, "k")
    client = DecisionClient(cfg, httpx.Client(transport=httpx.MockTransport(handler)))
    res = client.decide({"s": 1}, {"x": {"type": "noul"}})
    assert res.answers["x"].yes == 0.9
    assert len(calls) == 2 and calls[1]["model"] == cfg.model


def test_client_requires_key(monkeypatch):
    cfg = load_config().decision
    monkeypatch.delenv(cfg.api_key_env, raising=False)
    with pytest.raises(DecisionError):
        DecisionClient(cfg).decide({}, {})
