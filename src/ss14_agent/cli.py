"""Command line entry point: `ss14-agent [gui|run|doctor|snap|probe|decide-once|windows|models]`."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import REPO_ROOT, Config, load_config, load_dotenv


def _platform(cfg: Config):
    from .platform import get_platform

    return get_platform(cfg.capture.backend)


def cmd_windows(cfg: Config, args) -> int:
    p = _platform(cfg)
    for w in p.windows.list_windows():
        print(f"{w.window_id:>7}  {w.owner[:24]:<24}  {w.title[:40]:<40}  {w.width:.0f}x{w.height:.0f}")
    return 0


def cmd_doctor(cfg: Config, args) -> int:
    ok_all = True
    if sys.platform == "darwin":
        from .mac import permissions

        print("macOS permissions (granted to the app running this command, e.g. Terminal):")
        for name, ok in permissions.check(prompt=True).items():
            print(f"  [{'x' if ok else ' '}] {name}")
            ok_all &= ok
        print("  If you just granted one, quit and reopen your terminal app.\n")
    from . import providers

    for role, c in (("Decision", cfg.decision), ("Orchestrator", cfg.orchestrator)):
        if role == "Orchestrator" and not cfg.orchestrator.enabled:
            print("Orchestrator: off")
            continue
        prov = c.provider_info
        try:
            note = providers.check(prov, c.api_key, c.model)
            print(f"{role}: {prov.name}, {c.model}: connected ({note})")
        except providers.ConnectionFailed as exc:
            print(f"{role}: {prov.name}, {c.model}: NOT connected ({exc})")
            ok_all = False
    p = _platform(cfg)
    win = p.windows.find(cfg.game.window_match)
    if win is None:
        print("SS14 window: not found. Start the game and join a server (titles need Screen Recording permission).")
        return 1
    print(f"SS14 window: '{win.title or win.owner}' {win.width:.0f}x{win.height:.0f} pt at ({win.x:.0f}, {win.y:.0f})")
    t0 = time.perf_counter()
    frame = p.capturer.grab(win)
    print(f"Capture ({p.capturer.name}): {frame.shape[1]}x{frame.shape[0]} px in {1000 * (time.perf_counter() - t0):.0f} ms")
    return 0 if ok_all else 1


def cmd_snap(cfg: Config, args) -> int:
    from .annotate import annotate
    from .perception.observer import Observer

    p = _platform(cfg)
    win = p.windows.find(cfg.game.window_match)
    if win is None:
        print("SS14 window not found.")
        return 1
    frame = p.capturer.grab(win)
    obs = Observer(cfg, p.ocr, p.windows.screen_size()).observe(frame, win, read_chat=False)
    img = annotate(frame, obs.geo, obs.markers, draw_layout=True)
    out_dir = REPO_ROOT / "snaps"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"snap-{time.strftime('%Y%m%d-%H%M%S')}.png"
    img.save(out)
    g = obs.geo
    print(f"Saved {out}")
    print(f"  scale {g.scale:.2f} px/pt, title bar {g.content_top:.0f} px, tile {g.tile_px:.1f} px")
    print(f"  game view {g.view_w:.0f}x{g.view_h:.0f} px at ({g.view_x:.0f}, {g.view_y:.0f}); {len(obs.markers)} markers")
    print("  Green box = game viewport, magenta = chat panel, cyan = you, yellow = markers.")
    print("  If they don't line up, adjust [layout] in config/local.toml and run snap again.")
    if args.ocr:
        x, y, w, h = (int(v) for v in g.chat_rect)
        for line in p.ocr.read(frame[y:y + h, x:x + w]):
            print(f"  chat OCR: {line.text}  ({line.confidence:.2f})")
    return 0


def cmd_probe(cfg: Config, args) -> int:
    """Check the decision endpoint works, and which image layout (if any) it actually reads."""
    import numpy as np
    from PIL import Image

    from .annotate import to_png_b64
    from .decide.client import IMAGE_MODES, DecisionClient, DecisionError

    client = DecisionClient(cfg.decision)
    print(f"Endpoint {cfg.decision.url}, model {cfg.decision.model}\n")
    q_text = {"is_morning": {"type": "noul", "instructions": "Is the time in the state in the morning?",
                             "criteria": {"true": "Morning (before noon).", "false": "Not morning."}}}
    try:
        r1 = client.decide({"time": "08:15"}, q_text, image_mode="none")
        r2 = client.decide({"time": "21:40"}, q_text, image_mode="none")
    except DecisionError as exc:
        print(f"Text-only call failed: {exc}")
        return 1
    print(f"Text-only: p(morning | 08:15) = {r1.answers['is_morning'].yes:.2f}, "
          f"p(morning | 21:40) = {r2.answers['is_morning'].yes:.2f}  ({r1.latency_ms:.0f} ms)")

    def solid(rgb):
        return to_png_b64(Image.fromarray(np.full((256, 256, 3), rgb, dtype=np.uint8)))

    red, blue = solid((220, 20, 20)), solid((20, 40, 220))
    q_img = {"is_red": {"type": "noul", "instructions": "Look at the attached screen image. Is it a solid red image?",
                        "criteria": {"true": "The image is red.", "false": "The image is not red."}}}
    state = {"task": "Answer about the attached image in `screen`."}
    working = []
    print("\nImage layouts (a working layout gives a high p(red) for the red image and low for the blue):")
    for mode in [m for m in IMAGE_MODES if m != "none"]:
        try:
            pr = client.decide(state, q_img, red, image_mode=mode, media_type="image/png")
            pb = client.decide(state, q_img, blue, image_mode=mode, media_type="image/png")
        except DecisionError as exc:
            print(f"  {mode:<15} error: {str(exc)[:160]}")
            continue
        a, b = pr.answers["is_red"].yes, pb.answers["is_red"].yes
        works = a - b > 0.4
        if works:
            working.append(mode)
        verdict = "WORKS" if works else "ignored"
        print(f"  {mode:<15} p(red|red)={a:.2f}  p(red|blue)={b:.2f}  {verdict}  ({pr.latency_ms:.0f} ms)")
    if working:
        print(f"\nSet image_mode = \"{working[0]}\" under [decision] in config/local.toml.")
    else:
        print("\nNo layout made the model see the image. The agent will still run on text (markers, chat),"
              "\nbut set image_mode = \"none\" to save bandwidth, and check the provider's docs for image input.")
    return 0


def cmd_decide_once(cfg: Config, args) -> int:
    from .agent import Agent

    p = _platform(cfg)
    agent = Agent(cfg, p, args.playbook)
    win = p.windows.find(cfg.game.window_match)
    if win is None:
        print("SS14 window not found.")
        return 1
    frame = p.capturer.grab(win)
    obs = agent.observer.observe(frame, win)
    agent.operator_note = args.note or ""
    if agent.orchestrator is not None:
        print(f"Planning with {cfg.orchestrator.model}...")
        agent._plan(obs, "manual")
        plan = agent.status.plan
        if plan is None:
            print(f"Planning failed: {agent.status.plan_error}\n(continuing without a plan)")
        else:
            print(f"Goal: {plan.goal}   ({plan.latency_ms:.0f} ms)")
            for i, step in enumerate(plan.steps):
                print(f"  {'→' if i == plan.current_step else ' '} {i + 1}. {step}")
            if plan.say:
                print(f"  Would say: {plan.say!r}")
        agent._results.clear()
    print(f"Deciding with {cfg.decision.model}...")
    agent._decide(obs, "manual")
    d = agent.status.decision
    if d is None or d.error:
        print(f"Decision failed: {d.error if d else 'unknown'}")
        return 1
    print(f"Duty: {d.duty}   spoken_to={d.spoken_to:.2f}  danger={d.danger:.2f}   {d.latency_ms:.0f} ms")
    print(f"Chosen action: {d.action} ({d.confidence:.0%})")
    for opt, prob in d.top:
        print(f"  {prob:6.1%}  {opt}")
    return 0


def cmd_run(cfg: Config, args) -> int:
    from .agent import Agent

    agent = Agent(cfg, _platform(cfg), args.playbook)
    agent.operator_note = args.note or ""
    agent.start()
    print("Running. Press F8 in game to pause/resume, Ctrl-C here to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        agent.stop()
    return 0


def cmd_models(cfg: Config, args) -> int:
    from . import models, providers

    for role, c in (("Decision", cfg.decision), ("Orchestrator", cfg.orchestrator)):
        prov = c.provider_info
        try:
            ids = models.models_for(role.lower(), c)
        except (providers.ConnectionFailed, OSError) as exc:
            print(f"{role} models from {prov.name}: couldn't list ({exc})\n")
            continue
        print(f"{role} models from {prov.name}:")
        for m in ids:
            print(f"  {m}{'   <- current' if m == c.model else ''}")
        print()
    return 0


def cmd_gui(cfg: Config, args) -> int:
    from .gui.app import run_gui

    return run_gui(cfg, args.playbook)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ss14-agent", description=__doc__)
    parser.add_argument("--config", type=Path, help="TOML file overriding config/default.toml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd")
    gui = sub.add_parser("gui", help="open the control window (default)")
    run = sub.add_parser("run", help="run without the control window")
    run.add_argument("--note", help="instruction for the agent, e.g. 'clean the bar'")
    sub.add_parser("doctor", help="check permissions, API key, and that the game window can be captured")
    snap = sub.add_parser("snap", help="save an annotated screenshot to check the layout settings")
    snap.add_argument("--ocr", action="store_true", help="also print what OCR reads in the chat panel")
    sub.add_parser("probe", help="test the decision API and find which image layout it reads")
    once = sub.add_parser("decide-once", help="make one decision on the current screen and print it (no input sent)")
    once.add_argument("--note", help="instruction for the agent")
    sub.add_parser("windows", help="list on-screen windows")
    sub.add_parser("models", help="list the models the chosen decision and orchestrator providers offer")
    for p in (gui, run, once):
        p.add_argument("--playbook", default=None, help="playbook name in playbooks/ (default from config)")
        p.add_argument("--decision-model", help="decision model id, e.g. inception/mercury-decide:free")
        p.add_argument("--orchestrator-model", help="orchestrator model id, e.g. anthropic/claude-sonnet-5.5")
        p.add_argument("--no-orchestrator", action="store_true", help="run on the decision model alone")
    args = parser.parse_args(argv)
    if not hasattr(args, "playbook"):
        args.playbook = None

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    from .keystore import load_saved_keys
    from .providers import DECISION_PROVIDERS, ORCHESTRATOR_PROVIDERS

    load_saved_keys([p.key_env for p in (*DECISION_PROVIDERS.values(), *ORCHESTRATOR_PROVIDERS.values())])
    cfg = load_config(args.config)
    if getattr(args, "decision_model", None):
        cfg.decision.model = args.decision_model
    if getattr(args, "orchestrator_model", None):
        cfg.orchestrator.model = args.orchestrator_model
        cfg.orchestrator.enabled = True
    if getattr(args, "no_orchestrator", False):
        cfg.orchestrator.enabled = False
    handlers = {
        None: cmd_gui, "gui": cmd_gui, "run": cmd_run, "doctor": cmd_doctor, "snap": cmd_snap,
        "probe": cmd_probe, "decide-once": cmd_decide_once, "windows": cmd_windows,
        "models": cmd_models,
    }
    return handlers[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
