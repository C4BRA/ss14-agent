import numpy as np
from fakes import WIN, FakeOcr, FakeWorld, fake_layout, fake_platform

from ss14_agent.agent import Agent
from ss14_agent.config import load_config
from ss14_agent.decide.client import Answer, DecisionResult
from ss14_agent.memory import WorldMemory
from ss14_agent.perception.library import TileLibrary, tile_crop
from ss14_agent.perception.observer import Observer
from ss14_agent.perception.scene import build_scene
from ss14_agent.skills import SkillContext
from ss14_agent.skills.basic import Examine


def _obs():
    cfg = fake_layout(load_config())
    return Observer(cfg, FakeOcr()).observe(FakeWorld().frame(), WIN, read_chat=False)


def _sprite(seed, size=24):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_library_learns_matches_and_persists(tmp_path):
    lib = TileLibrary(tmp_path / "lib", threshold=0.9)
    tray, bucket = _sprite(1), _sprite(2)
    assert lib.learn(tray, "Hydroponics Tray")
    assert not lib.learn(tray, "hydroponics tray")       # same picture, nothing new
    assert lib.match(tray)[0] == "hydroponics tray"
    # Brighter lighting still matches; a different sprite doesn't.
    lit = np.clip(tray.astype(int) * 1.2 + 10, 0, 255).astype(np.uint8)
    assert lib.match(lit)[0] == "hydroponics tray"
    assert lib.match(bucket) is None
    # A re-probe that names the same picture differently replaces the old name.
    lib.learn(tray, "seed extractor")
    assert lib.match(tray)[0] == "seed extractor" and len(lib) == 1
    lib.save()
    again = TileLibrary(tmp_path / "lib")
    assert again.match(tray)[0] == "seed extractor"


def test_scene_names_things_and_draws_a_map(tmp_path):
    obs = _obs()
    m1, m2 = obs.markers[0], obs.markers[1]
    mem = WorldMemory()
    mem.add_landmark(*m1.world, "mop bucket")
    lib = TileLibrary(tmp_path / "lib")
    lib.learn(tile_crop(obs.frame, obs.geo, m2.dx, m2.dy), "plant")
    scene = build_scene(obs, mem, lib, read_text=["Examined tray: needs water"])
    things = {t.id: t for t in scene.things}
    assert (things[m1.id].label, things[m1.id].source) == ("mop bucket", "probed")
    assert (things[m2.id].label, things[m2.id].source) == ("plant", "learned")
    hx, hy = obs.geo.half_tiles
    assert len(scene.rows) == 2 * hy + 1 and all(len(r) == 2 * hx + 1 for r in scene.rows)
    assert scene.rows[hy][hx] == "@"
    assert scene.rows[hy + m1.dy][hx + m1.dx] == things[m1.id].char
    st = scene.for_state()
    assert st["things"][0]["what"] == "mop bucket" and st["text_read_from_the_game"]
    assert "MAP" in scene.as_text() and "mop bucket" in scene.as_text()


class RecordingClient:
    def __init__(self):
        self.calls = []

    def decide(self, state, questions, image_b64=None, image_mode=None, media_type="image/jpeg"):
        self.calls.append((state, questions, image_b64))
        ans = {"action": Answer("choice", choice="wait", confidence=0.9, probabilities={"wait": 0.9}),
               "spoken_to": Answer("noul", value=0.0), "danger": Answer("noul", value=0.0)}
        return DecisionResult(ans, 1.0, {})


def test_text_mode_sends_no_image(tmp_path, monkeypatch):
    monkeypatch.setattr("ss14_agent.agent.REPO_ROOT", tmp_path)
    cfg = fake_layout(load_config())
    cfg.orchestrator.enabled = False
    client = RecordingClient()
    agent = Agent(cfg, fake_platform(FakeWorld()), "janitor", client=client)
    obs = _obs()
    cfg.perception.mode = "text"
    agent._decide(obs, "manual")
    state, questions, img = client.calls[-1]
    assert img is None and "what_you_see" in state and "markers_on_screen" not in state
    assert any(k.startswith("examine:") for k in questions["action"]["criteria"])
    assert '"image": "none"' in agent.status.request_text
    cfg.perception.mode = "image"
    agent._decide(obs, "manual")
    state, _, img = client.calls[-1]
    assert img and "markers_on_screen" in state and "what_you_see" not in state


def test_idle_agent_probes_unknown_things_and_learns_them(tmp_path, monkeypatch):
    monkeypatch.setattr("ss14_agent.agent.REPO_ROOT", tmp_path)
    cfg = fake_layout(load_config())
    cfg.orchestrator.enabled = False
    world = FakeWorld()
    platform = fake_platform(world, ocr_lines=["Mop Bucket", "Examine"])
    agent = Agent(cfg, platform, "janitor", client=RecordingClient())
    obs = Observer(cfg, platform.ocr).observe(world.frame(), WIN, read_chat=False)
    assert agent._maybe_probe(obs)
    assert agent._skill_spec.skill == "inspect" and agent._skill_spec.id.startswith("probe:")
    assert not agent._maybe_probe(obs)          # rate limited
    target = agent._skill_spec.args["world"]
    skill = agent._skill
    skill.tick(agent.ctx, obs)
    skill._t0 -= 1                               # menu has had time to open
    st = skill.tick(agent.ctx, obs)
    assert st.state == "ok" and "mop bucket" in st.detail.lower()
    assert agent.memory.label_at(target) == "Mop Bucket"
    assert agent.library.names == ["mop bucket"]
    assert (tmp_path / "data" / "tiles" / "default.npz").exists()


def test_examine_reads_text_with_shift_click():
    obs = _obs()
    inp = fake_platform().input
    got = []
    ctx = SkillContext(input=inp, ocr=FakeOcr(["Hydroponics tray", "It needs water."]),
                       on_text=lambda x, y, lines: got.append(lines))
    m = obs.markers[0]
    sk = Examine(m.world)
    sk.tick(ctx, obs)
    assert any(e[0] == "click" for e in inp.events)
    sk._t0 -= 1
    st = sk.tick(ctx, obs)
    assert st.state == "ok" and got == [["Hydroponics tray", "It needs water."]]


def test_walk_never_gives_up_on_a_bad_map():
    from ss14_agent.skills import WalkTo
    obs = _obs()
    inp = fake_platform().input
    ctx = SkillContext(input=inp, ocr=FakeOcr())
    # Walled in by old bumps and darkness on every side: still heads for the goal.
    ctx.blocked |= {(x, y) for x in range(-1, 2) for y in range(-1, 2)} - {(0, 0)}
    obs.space |= {(x, y) for x in range(-3, 4) for y in range(-3, 4)} - {(0, 0)}
    sk = WalkTo((5, 0))
    sk.start(ctx, obs)
    st = sk.tick(ctx, obs)
    assert st.state == "running" and "d" in inp.held()
    assert not ctx.blocked & {(1, 0), (-1, 0)}       # the nearby bumps were forgotten


def test_close_window_only_when_something_is_open():
    from ss14_agent import playbook as playbooks
    from ss14_agent.decide.actions import build_actions
    obs = _obs()
    pb = playbooks.load("janitor")

    def ids(window_open):
        return {a.id for a in build_actions(obs, pb, 1.4, False, {}, None, True, window_open)}

    assert "close_window" not in ids(False)
    assert "close_window" in ids(True) and "close_window" in ids(None)


def test_actions_that_keep_failing_leave_the_menu(tmp_path, monkeypatch):
    import time as _t
    monkeypatch.setattr("ss14_agent.agent.REPO_ROOT", tmp_path)
    cfg = fake_layout(load_config())
    cfg.orchestrator.enabled = False
    client = RecordingClient()
    agent = Agent(cfg, fake_platform(FakeWorld()), "janitor", client=client)
    obs = _obs()
    agent._failures[("explore:west", (0, 0))] = [_t.monotonic(), _t.monotonic()]
    agent._decide(obs, "manual")
    opts = client.calls[-1][1]["action"]["criteria"]
    assert "explore:west" not in opts and "explore:east" in opts
