from fakes import WIN, FakeOcr, FakeWorld, fake_layout, fake_platform
from ss14_agent import playbook as playbooks
from ss14_agent.agent import Agent
from ss14_agent.config import load_config
from ss14_agent.decide.actions import ActionSpec, build_actions, build_state
from ss14_agent.decide.client import Answer, DecisionResult
from ss14_agent.memory import DIRECTIONS, WorldMemory, compass
from ss14_agent.perception.observer import Observer


def _obs():
    cfg = fake_layout(load_config())
    return Observer(cfg, FakeOcr()).observe(FakeWorld().frame(), WIN, read_chat=False)


def test_compass():
    assert compass(0, -12) == "12 tiles north"
    assert compass(5, 5) == "7 tiles south-east"
    assert compass(0.5, 0) == "right here"


def test_exploring_prefers_unseen_ground():
    obs = _obs()
    mem = WorldMemory()
    mem.update(obs)
    hx, hy = obs.geo.half_tiles
    # Straight ahead the first unseen tile is just past the edge of the screen.
    tx, ty = mem.explore_target(obs.pos, DIRECTIONS["north"])
    assert ty < -hy and abs(tx) <= mem.explore_reach
    # Pretend everything to the north has been seen: north stops being worth exploring.
    mem.seen |= {(x, y) for x in range(-40, 41) for y in range(-40, 1)}
    assert mem.unexplored(obs.pos, DIRECTIONS["north"]) == 0.0
    assert mem.explore_target(obs.pos, DIRECTIONS["north"]) is None
    assert mem.unexplored(obs.pos, DIRECTIONS["south"]) > 0.5
    assert mem.summary(obs.pos)["exploration_by_direction"]["north"] == "already explored"


def test_actions_carry_memory():
    obs = _obs()
    mem = WorldMemory()
    mem.update(obs)
    mem.seen |= {(x, y) for x in range(-40, 41) for y in range(-40, 1)}
    mem.add_landmark(0, -20, "mop bucket")
    m = obs.markers[0]
    mem.record("click", m.world, "failed: couldn't get in reach")
    acts = {a.id: a for a in build_actions(obs, playbooks.load("janitor"), 1.4, True, {}, mem)}
    assert "already explored" in acts["explore:north"].description
    assert acts["explore:south"].args["target"] is not None
    assert "mop bucket" in acts["go_back:0"].description and "20 tiles north" in acts["go_back:0"].description
    assert "already tried this here 1 time" in acts[f"click:{m.id}"].description
    state = build_state(obs, playbooks.load("janitor"), None, [], [], memory=mem)
    assert state["memory"]["things_you_have_identified"] == ["mop bucket: 20 tiles north"]


def _result(choice, conf, spoken=0.0, danger=0.0):
    return DecisionResult({
        "action": Answer("choice", choice=choice, confidence=conf, probabilities={choice: conf}),
        "spoken_to": Answer("noul", value=spoken),
        "danger": Answer("noul", value=danger),
    }, 1.0, {})


def test_heartbeat_does_not_flip_flop_a_running_skill():
    cfg = fake_layout(load_config())
    agent = Agent(cfg, fake_platform(FakeWorld()), "janitor", client=object())
    agent._skill_spec = ActionSpec("explore:north", "explore", "", {"dir": (0, -1)})
    south = ActionSpec("explore:south", "explore", "", {"dir": (0, 1)})
    assert not agent._should_interrupt("heartbeat", _result("explore:south", 0.22), south)
    assert agent._should_interrupt("heartbeat", _result("explore:south", 0.8), south)
    assert agent._should_interrupt("heartbeat", _result("explore:south", 0.2, danger=0.9), south)
    assert not agent._should_interrupt("chat", _result("explore:south", 0.9, spoken=0.1), south)
    assert agent._should_interrupt("chat", _result("explore:south", 0.3, spoken=0.9), south)
