# ss14-agent

An agent that plays [Space Station 14](https://spacestation14.com/) the way a person does: it **looks at the game window** and **presses keys and clicks the mouse**. A decision model ([Inception's Mercury Decide](https://openrouter.ai/inception/mercury-decide:free) on OpenRouter, by default) chooses what to do several times a second.

The goal is a proof of concept: a decision model can play a real-time multiplayer game from pixels, with no game modding, memory reading or network tricks.

> **Status: first version, not yet run against a live game.** It's macOS-only for now. The core loop is tested against a simulated game (`pytest`), but the macOS capture/input code and the OpenRouter image format still need checking on a real Mac. Start with the [first run](#first-run) steps below.

> **Use it only on a server you run.** Automation is against the rules on public SS14 servers.

## How it works

```
 SS14 client (unmodified)                         ss14-agent
┌───────────────────────┐   window pixels   ┌──────────────────────────────────────────────┐
│                       │ ────────────────▶ │ Fast loop, ~15 Hz                            │
│   your character      │                   │   capture → tile grid → odometry → markers   │
│   in a game round     │                   │   OCR new chat lines (Apple Vision)          │
│                       │ ◀──────────────── │   run the current skill (walk, click, E...)  │
└───────────────────────┘   keys + mouse    │                                              │
                                            │ Decision loop, ~1-2 Hz                       │
                                            │   numbered-marker screenshot + text state    │
                                            │   → Mercury Decide: duty? action? danger?    │
                                            │   → next skill                               │
                                            └──────────────────────────────────────────────┘
```

- **Seeing.** Each frame is cut into the game's 32 px tile grid. Your character is always at the centre (SS14's camera follows you), so positions are just tile offsets. Interesting tiles get numbered yellow markers. Walking is tracked by measuring how the floor slides across the screen (phase correlation), and chat is read with Apple's built-in OCR.
- **Deciding.** Mercury Decide doesn't write text; it picks from options and returns a probability for each. Each decision asks four questions in one call: which **duty** from the job playbook to work on, which **action** to take next (`walk_to:3`, `click:5`, `activate:2`, `inspect:4`, `explore:north`, `say:1`, `wait`...), whether someone **spoke to you**, and whether you're **in danger**.
- **Acting.** Actions are closed-loop skills: `walk_to` runs A* over the tiles and holds W/A/S/D, remembers walls it bumps into, and gives up if stuck. `click` walks into reach first, then clicks. `inspect` right-clicks and reads the entity menu, so the agent learns names like "mop bucket".
- **Planning.** A decision model on its own only reacts to the moment. An **orchestrator** (any vision chat model on OpenRouter, Claude Sonnet 5.5 by default) looks at the screen, chat, memory and recent results every ~20 seconds, or sooner after chat or repeated failures. It writes a goal, a few concrete steps, which step is current, which duty applies, and optionally a line to say. Every decision request carries that plan, and the decision model picks the action that carries out the current step. Turn it off to run on the decision model alone.
- **Remembering.** The model sees one screenshot per decision, so the agent keeps a memory for it (`memory.py`): a map of every tile that has been on screen, things it has identified and where they are, and what it already tried at each spot. Exploring heads for the nearest area it hasn't seen, each option says how explored that direction is, `go_back:N` walks back to things found earlier, and options that already failed say so. A running skill is only cut short by chat, danger, or a confident new decision, so near-ties don't make it dither.
- **Job playbooks** (`playbooks/*.toml`) give the model structure that "do your job" lacks: a handful of duties, simple steps for each, and phrases it can say. Janitor and Station Engineer are included.

## Requirements

- macOS 13 or newer (macOS 14+ uses ScreenCaptureKit; older versions fall back to the classic capture API)
- Python 3.11+
- An [OpenRouter API key](https://openrouter.ai/keys). `inception/mercury-decide:free` costs nothing during early access.
- SS14 and a server you control. The easiest option is a local server on the same Mac (a downloaded server build, or `dotnet run` in a space-station-14 checkout), then connect to `localhost` from the launcher.

## Install

```bash
git clone https://github.com/C4BRA/ss14-agent.git
cd ss14-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then paste your OpenRouter key into .env
```

## First run

1. **Set up SS14.** In Options, set the UI layout to **Separated** (game on the left, chat panel on the right). Run the game in a window and keep its size fixed. Join your server and spawn as **Janitor**.
2. **Grant permissions.** Run `ss14-agent doctor`. macOS will ask to give your terminal app **Screen Recording**, **Accessibility** and **Input Monitoring**. Approve them in System Settings > Privacy & Security, then quit and reopen the terminal. Run `doctor` again until it can see and capture the SS14 window.
3. **Check the layout.** Run `ss14-agent snap --ocr` and open the PNG it saves under `snaps/`. The green box should be the game view, the cyan box your character, the grid lines should match the floor tiles, and the magenta box should cover chat. If anything is off, copy the `[layout]` section from `config/default.toml` into `config/local.toml` and adjust it.
4. **Check the model can see.** Run `ss14-agent probe`. It makes a few tiny calls to confirm your key works and tests which image format the endpoint actually reads. Put the format it reports into `config/local.toml`:
   ```toml
   [decision]
   image_mode = "assets"   # whatever probe printed
   ```
5. **Dry run.** `ss14-agent decide-once` makes one decision on the current screen and prints the top options with their probabilities, without touching your keyboard or mouse.
6. **Let it play.** `ss14-agent` opens the control window. Press **Start**. Press **F8** at any time to pause or resume, and pressing any key or clicking in the game also pauses it, so you can take over instantly.

## Control window

- **Job** picks the playbook. **Instruction** is an optional note the model sees, like "clean the bar".
- The **live view** shows exactly what the model sees: the game with numbered markers, and your character in cyan.
- **Status** shows the current duty, the skill running, the last decision with its confidence, the "spoken to" and "danger" readings, and API latency.
- **Top options** lists the probabilities the model gave its best choices.
- **Decision model** and **Orchestrator** pick the two models from lists loaded from OpenRouter's catalog; you can also type any model id. Pick "(off)" to run without an orchestrator. `ss14-agent models` prints the same lists, and `--decision-model`, `--orchestrator-model` and `--no-orchestrator` work on `gui`, `run` and `decide-once`.
- **Overlay** floats the window above the game without taking focus, so you can watch while the game keeps keyboard focus. The agent captures the game window directly, so covering it doesn't affect what the model sees. While the agent runs, clicks pass through the window to the game; press **F8** to pause and use the buttons.
- Every decision is logged to `logs/<time>/decisions.jsonl`, with the image it saw in `logs/<time>/frames/`.

## Commands

| Command | What it does |
|---|---|
| `ss14-agent` / `ss14-agent gui` | Open the control window |
| `ss14-agent run [--playbook engineer] [--note "..."]` | Run without the window; Ctrl-C to stop |
| `ss14-agent doctor` | Check permissions, API key, window capture |
| `ss14-agent snap [--ocr]` | Save an annotated screenshot to check the layout |
| `ss14-agent probe` | Test the decision API and which image format it reads |
| `ss14-agent decide-once` | One decision on the current screen, no input sent |
| `ss14-agent windows` | List on-screen windows (if the game isn't found) |

## Configuration

Defaults are in `config/default.toml`. Override any of them in `config/local.toml` (git-ignored). The settings you're most likely to change:

- `[layout]` `viewport`, `chat`, `tile_px`, `titlebar_pt`: match your window. Use `snap` to check.
- `[decision]` `image_mode`: set from `probe`. `max_calls_per_s`: lower it if the free tier rate-limits you. `min_confidence`: below this the agent waits instead of acting.
- `[agent]` `playbook`, `heartbeat_s` (re-decide at least this often), `chat_cooldown_s`, `reach_tiles`.

## Writing a playbook

Copy `playbooks/janitor.toml`. Each `[[duties]]` entry has an `id`, a `description`, a `cue` (when it applies) and a few `steps`. `phrases` are the only things the agent can say, because the decision model picks rather than writes. Keep duties to about five; fewer, clearer options get better choices.

## Known limits

- Markers come from a simple "this tile stands out from the floor" heuristic, so walls and decals sometimes get marked and small items sometimes don't. The model also sees the image, and `inspect` helps. A trained detector is the planned next step.
- The map is learned by bumping into things, and positions are relative to where the agent started. Walking through doors needs the door to open (`activate`).
- It assumes the default key bindings (WASD, E, Q, X, Z, T) and north-up camera.
- Real-time combat isn't a goal of this version.

## Development

```bash
pytest           # runs everywhere; uses a simulated game world
ruff check src tests
```

The macOS-specific code lives in `src/ss14_agent/mac/`, behind the interfaces in `src/ss14_agent/platform.py`. The design write-up is in [`docs/architecture.md`](docs/architecture.md).

## License

MIT
