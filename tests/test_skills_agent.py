import time

from fakes import FakeInput, FakeOcr, FakeWorld, WIN, fake_layout, fake_platform
from ss14_agent.agent import Agent
from ss14_agent.config import load_config
from ss14_agent.decide.client import Answer, DecisionResult
from ss14_agent.perception.observer import Observer
from ss14_agent.skills import SkillContext, WalkTo, astar
from ss14_agent.skills.basic import Click, Say
from ss14_agent.skills.walk import keys_for


def test_astar_routes_around_walls():
    wall = {(2, y) for y in range(-3, 4)}
    path = astar((0, 0), (4, 0), wall)
    assert path[0] == (0, 0) and path[-1] == (4, 0)
    assert not set(path) & wall
    assert astar((0, 0), (4, 0), {(2, y) for y in range(-20, 21)}, margin=3) is None


def test_keys_for_directions():
    assert keys_for(1, 0) == {"d"}
    assert keys_for(0, -1) == {"w"}
    assert keys_for(1, 1) == {"d", "s"}
    assert keys_for(0, 0) == set()


def _setup():
    cfg = fake_layout(load_config())
    world = FakeWorld(speed_tiles_per_frame=0.2)
    inp = FakeInput(world)
    obs_er = Observer(cfg, FakeOcr())
    ctx = SkillContext(input=inp, ocr=FakeOcr())
    return cfg, world, inp, obs_er, ctx


def test_walk_to_reaches_target_in_fake_world():
    cfg, world, inp, observer, ctx = _setup()
    obs = observer.observe(world.frame(), WIN)
    skill = WalkTo((3, -2))
    skill.start(ctx, obs)
    st = None
    for _ in range(200):
        world.step()
        obs = observer.observe(world.frame(), WIN)
        st = skill.tick(ctx, obs)
        if st.done:
            break
    assert st is not None and st.state == "ok", st
    assert abs(world.px / 32 - 43) < 0.6 and abs(world.py / 32 - 38) < 0.6


def test_click_walks_into_reach_then_clicks():
    cfg, world, inp, observer, ctx = _setup()
    obs = observer.observe(world.frame(), WIN)
    skill = Click((4, 0), reach=1.4)
    skill.start(ctx, obs)
    for _ in range(200):
        world.step()
        obs = observer.observe(world.frame(), WIN)
        st = skill.tick(ctx, obs)
        if st.done:
            break
        time.sleep(0.005)
    assert st.state == "ok"
    clicks = [e for e in inp.events if e[0] == "click"]
    assert len(clicks) == 1


def test_say_types_into_chat():
    cfg, world, inp, observer, ctx = _setup()
    obs = observer.observe(world.frame(), WIN)
    Say("Hello!").start(ctx, obs)
    assert ("tap", "t", ()) in inp.events and ("type", "Hello!") in inp.events
    assert ("tap", "return", ()) in inp.events


class ScriptedClient:
    """Always picks the first marker to walk to, then waits."""

    def __init__(self):
        self.n = 0

    def decide(self, state, questions, image_b64=None, image_mode=None, media_type="image/jpeg"):
        self.n += 1
        opts = list(questions["action"]["criteria"])
        choice = next((o for o in opts if o.startswith("walk_to:")), "wait") if self.n == 1 else "wait"
        ans = {
            "duty": Answer("choice", choice="patrol", confidence=0.8, probabilities={"patrol": 0.8}),
            "action": Answer("choice", choice=choice, confidence=0.9, probabilities={choice: 0.9}),
            "spoken_to": Answer("noul", value=0.1),
            "danger": Answer("noul", value=0.0),
        }
        return DecisionResult(ans, 5.0, {})


def test_agent_end_to_end_with_fakes(tmp_path, monkeypatch):
    cfg = fake_layout(load_config())
    cfg.agent.save_frames = False
    cfg.agent.tick_hz = 60
    world = FakeWorld(speed_tiles_per_frame=0.2)
    platform = fake_platform(world)
    client = ScriptedClient()
    monkeypatch.setattr("ss14_agent.agent.REPO_ROOT", tmp_path)
    agent = Agent(cfg, platform, "janitor", client=client)
    agent.start()
    try:
        deadline = time.time() + 6
        while time.time() < deadline and client.n < 3:
            time.sleep(0.05)
    finally:
        agent.stop()
    assert client.n >= 2
    assert agent.status.duty == "patrol"
    assert any("walk_to:" in line for line in agent.status.log)
    assert (tmp_path / "logs").exists()
    assert platform.input.held() == set()
