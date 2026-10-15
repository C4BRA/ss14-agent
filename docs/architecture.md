# SS14 Vision Agent: Pixels In, Keyboard and Mouse Out

_Status: proposal, 2026-10-15. Replaces the in-client bridge recommendation in the earlier in-client bridge design for the proof-of-concept goal: a multimodal model plays Space Station 14 in real time using only what a human sees and the inputs a human has._

## 1. The recommendation

Build a **hierarchical vision agent** that runs beside an **unmodified SS14 client**:

1. A **fast perception and control loop** (10 to 30 Hz) captures the game window, extracts cheap visual facts (where things are on screen, health and hand slots, new chat text), and executes motor skills with closed-loop feedback: walk toward a screen point, click a sprite, open the right-click menu and pick an entry.
2. A **slow multimodal planner** (one decision every one to three seconds) looks at an annotated screenshot plus extracted text and chooses the next skill, referring to things on screen by numbered markers.
3. A **thin input layer** turns skills into OS-level keyboard and mouse events sent to the game window.

A **multimodal decision model** (d3, JEV-27B-VL, Jev 1.14, Microsoft-Decision-1 and similar) sits between the two as a **tactical layer** running several times a second: it scores a fixed set of options against the current frame, for example which marker to act on next, whether the agent is under attack, or whether the current skill succeeded. Section 7.4 covers it.

The planner never presses individual keys and the fast loop never makes judgement calls. That split is what makes "real time" achievable with today's multimodal models, which take roughly one to several seconds per call. It's the same idea behind the credible "AI plays Minecraft" results: DeepMind's SIMA agents act from pixels with keyboard and mouse, and OpenAI's VPT ran a fast low-level policy. Agents that drive the game through an API (like Voyager with Mineflayer) are impressive but aren't pixel-based, which is exactly the distinction this project is making.

Everything in the runtime path uses only the screen and the input devices. No game code changes, no memory reading, no network inspection. That's the claim the demo proves, so keep it clean.

**Where it runs:** still a local or private server you control. Automation is against the rules on public servers regardless of how it's built, and a private server also lets you fix the map, job and round settings so demos are repeatable. The nice change from the earlier design is that a stock client works: no fork, no sandbox changes.

## 2. Why SS14 is a good vision target

Some properties of the game make pixel play much more tractable than it first looks:

- **The camera is locked to your character.** The player is always at the centre of the game viewport, so "where am I" is free and "move toward X" is just the vector from the centre to X.
- **Top-down, fixed tile grid.** The engine renders 32 pixels per tile at default zoom (`Robust.Client/Graphics/ClientEye/EyeManager.cs:23`). With a fixed window size and zoom, you can overlay an exact tile grid on every frame, which makes positions, distances and pathing simple.
- **The game will tell you names if you ask.** Right-click opens an entity menu listing everything under the cursor by name (`Content.Client/ContextMenu/UI/EntityMenuUIController.cs:75`), and Shift+left-click examines an entity and prints a description to chat. These are human-available "probes" that turn a sprite the detector can't identify into text.
- **Lots of the important state is text.** Chat, examine results, popups, and most machine UIs are readable text, which OCR handles well.

The hard parts are real too: small pixel-art sprites, cluttered rooms, overlapping entities, lighting and darkness, and lots of UI windows. The design leans on the properties above to work around them.

## 3. System overview

```
┌──────────────────────────── user's machine ─────────────────────────────┐
│                                                                         │
│   SS14 client (stock, windowed, fixed size and zoom)                    │
│        ▲ OS input events                     │ window frames            │
│        │                                     ▼                          │
│   ┌────┴─────────┐   skills    ┌──────────────────────────────────┐     │
│   │ Input layer  │◀────────────│ Fast loop (10-30 Hz)             │     │
│   │ keys, mouse, │             │  Capture -> Perception -> State  │     │
│   │ holds        │             │  Skill executor (closed loop)    │     │
│   └──────────────┘             │  Interrupt detector              │     │
│                                └───────┬──────────────▲───────────┘     │
│                    annotated frame,    │              │ skill calls     │
│                    text, events        ▼              │                 │
│                                ┌──────────────────────┴───────────┐     │
│                                │ Planner (0.3-1 Hz)               │     │
│                                │  multimodal model + tool calls   │     │
│                                │  memory: role, goals, summary    │     │
│                                └──────────────────────────────────┘     │
│                                                                         │
│   Demo overlay (OBS / separate window): markers, current skill,         │
│   planner's last reasoning line                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

One Python process is enough for all of it. Run capture and the fast loop on one thread or async task, the planner call on another, so the character keeps moving while the model thinks.

## 4. Capture

- Run the client **windowed at a fixed size**, e.g. 1280x720, with fixed zoom and UI scale. Every pixel offset in the perception code depends on this.
- Capture just the game window, not the screen:
  - Windows: Windows Graphics Capture (e.g. via `dxcam` or `windows-capture` in Python). Low latency, works with the window partly covered.
  - Linux: `mss` on X11, or PipeWire screencast on Wayland.
  - macOS: ScreenCaptureKit.
- Target 30 fps capture for the fast loop. The planner gets one frame per decision.
- Game settings that help vision: turn screen shake intensity down and enable reduced motion in Accessibility (`Content.Shared/CCVar/CCVars.Accessibility.cs`, `CCVars.Accessibility.Motion.cs`), use a larger chat font, and dock the chat panel in a fixed position.

## 5. Perception (fast, local, no model calls)

The fast loop maintains a small **screen state** from each frame:

| Fact | How | Notes |
|---|---|---|
| Tile grid | Fixed math from window size and zoom; player tile is the centre | Lets every detection snap to a tile coordinate relative to the player |
| Walkable vs blocked tiles | Colour/texture classifier per tile (floor, wall, window, door, space) | A tiny CNN on 32x32 tile crops is enough. Feeds the walking skill |
| Objects and mobs | A small object detector (e.g. a YOLO-family model) fine-tuned on SS14 screenshots | Classes: mob, item, door, machine, container, table, spill, etc. Generic classes, not every prototype |
| HUD | Fixed screen regions: health/damage indicator, hand slots, active hand, alerts column | Template match or crop-and-classify |
| Chat and popups | OCR on the chat panel region, diffed against last frame to find new lines | Also catches examine results |
| Open UI windows | Detect window frames; OCR their contents when the planner needs them | |
| Motion check | Frame difference around the centre | Confirms the character actually moved (stuck detection) |

**Training the detector.** You need a few thousand labelled screenshots. Two options, both offline only:

- Hand-label a few hundred frames, train, then use the model to pre-label more and correct them.
- Faster: generate labels automatically by running a dev build that dumps entity screen boxes alongside screenshots. This uses game internals only to produce training labels; the agent at runtime still sees only pixels. Say so in the write-up of the demo so the claim stays honest.

Start with OCR plus the tile classifier and skip the detector in the first milestone; the planner model can see sprites itself and the right-click menu can name them.

## 6. Actions: skills on top of raw input

### 6.1 Input layer

Raw primitives, sent to the focused game window:

- `key_down(k)`, `key_up(k)`, `tap(k)`, `type_text(s)`
- `mouse_move(x, y)`, `click(button, mods)`, `drag`
- Windows: `SendInput`. Linux: `uinput` or `xdotool`. macOS: Quartz events. General computer-use drivers such as Cua Driver aren't a good fit for the in-game layer: they target accessibility trees and background delivery, don't document key-down/key-up holds (needed for WASD movement), and their own action-support doc says background clicks aren't proven to reach games or canvases. They're fine for the out-of-game steps (opening the launcher, connecting to the server). The game must have focus; give the agent a dedicated machine or VM for demos so a stray click doesn't send keys into another app.

Default SS14 binds worth knowing (from `Resources/keybinds.yml`; read the user's actual binds at startup since they're rebindable): WASD move, `E` activate in world, `Q` drop, `Ctrl+Q` throw, `X` swap hands, `I` inventory, `Shift+Left click` examine, `Ctrl+Left click` pull, `Right click` entity menu.

### 6.2 Skills (what the planner calls)

Each skill is a small closed-loop controller in the fast loop that runs until success, failure or interrupt, and reports back.

| Skill | Behaviour |
|---|---|
| `walk_to(marker or tile)` | A* over the classified tile grid on screen; holds WASD along the path, re-plans every frame, stops on arrival, fails if stuck (motion check) |
| `walk_dir(dir, tiles)` | Explore toward the screen edge when the target is off screen |
| `click(marker)` | Move cursor to the marker's sprite centre, left-click; verifies something changed (popup, hand slot, window) |
| `activate(marker)` | Walk into reach if needed, hover, press `E` |
| `pickup(marker)` / `drop()` / `swap_hands()` | Click with empty hand / `Q` / `X`, verified via hand slot HUD |
| `use_held_on(marker)` | Click target with the active hand's item |
| `inspect(marker)` | Right-click, OCR the entity menu, return the names; or Shift+click and return the examine text |
| `menu_select(text)` | Find an entry in an open context menu or UI window by OCR and click it |
| `say(text, channel)` | Open chat, type, send (channel prefix for radio) |
| `close_windows()` | Escape until no UI windows remain |
| `wait(s)` | Keep observing; interrupts still fire |

This is about 15 skills. Keep the planner's tool list that short; a long list hurts decision quality more than it adds capability.

## 7. The planner (multimodal model)

### 7.1 What it sees each decision

- **An annotated screenshot**: the game frame downscaled (e.g. to 960x540) with numbered markers drawn on detected objects and doors ("set-of-marks"), plus a faint tile grid. Markers let the model say `click(7)` instead of guessing pixel coordinates, which is the single biggest reliability win for vision agents.
- **A text sidebar**: new chat lines since last decision, current HUD reading (health, hands), the last skill results, any interrupt that triggered this call.
- **Memory**: job and goals, a rolling round summary, a short notes list it can append to.

### 7.2 The loop

```
while running:
    trigger = wait_for(skill_finished | interrupt | heartbeat(4s))
    frame, text = fast_loop.snapshot_annotated()
    calls = planner.decide(static_context, memory, frame, text, tools)
    fast_loop.run(calls)          # returns immediately; skills execute in the fast loop
    memory.update(text, calls)
```

The planner is called when the current skill finishes, when an interrupt fires, or on a heartbeat so the agent never freezes. While a call is in flight the current skill keeps running, so movement doesn't stop to think. Interrupts the fast loop can detect from pixels alone: health indicator changed, red damage flash, the character's name appearing in new chat text, a UI window opening, the character not moving when it should.

### 7.3 Model choice and budget

- **API:** a current Claude model with vision and tool use. Claude Sonnet 5.5 (`claude-sonnet-5-5`) for the planner is a good default; Claude Haiku 5.5 (`claude-haiku-5-5`) if you want faster, cheaper decisions. Put the static context (game primer, job, rules, tool definitions) behind a prompt-cache breakpoint. An image costs roughly `width × height / 750` tokens, so a 960x540 frame is about 700 tokens. At one decision every two seconds that's on the order of 30 calls a minute; check the current pricing page before long runs.
- **Local:** any open-weight vision-language model behind an OpenAI-compatible server (vLLM, llama.cpp). It will be slower or weaker; the skill layer is what keeps it usable.
- I'd use your own tool schema rather than Anthropic's generic computer-use tool. Computer use is built for desktop apps with one action per screenshot; a game needs held keys, closed-loop skills and interrupts, which your skill layer provides.

### 7.4 Decision models as the tactical layer

Multimodal decision models are a newer class of model built for exactly the gap between the fast loop and the planner. Instead of generating text, they take a state (text or JSON plus images) and a set of typed questions (yes/no, choose one of N options, rating, and for Jev a bounding box), and return a calibrated probability for every option in one forward pass. Published figures (vendor-reported, check on your own hardware):

| Model | Vision | Where it runs | Reported speed |
|---|---|---|---|
| d3 family (vLLM Semantic Router "Decision 3.0"), 0.85B to 26B | Yes, every size | Self-hosted, Apache-2.0 | ~60-170 ms per decision with an image for the small sizes, ~340 ms for full d3, on one AMD MI325X |
| JEV-27B-VL (autotrust) | Yes | Self-hosted (Transformers / vLLM `POST /v1/decide`) | ~120-300 ms per decision in its Tetris, Mario and Snake demos |
| Jev 1.14 (TypeSafe) | Experimental, opt-in; adds a Box question | Hosted API | Not stated; vendor calls images not production-ready yet |
| Microsoft-Decision-1 | Not stated (text options) | Foundry / OpenRouter | "Fastest measured" in Microsoft's benchmark; no ms figure |
| decider-4b/12b (Mapika), GLiNER2.5-Decide | Text only | Local | 50-130 ms |
| GLiDE (Fastino) | Text only | Hosted | Too slow for this loop (~16 s) |

Sources: [Decision 3.0 overview](https://www.marktechpost.com/2026/10/11/what-is-decision-3-0-vllm-semantic-routers-new-open-decision-models/), [d3 on Hugging Face](https://huggingface.co/vllm-sr/d3), [JEV-27B-VL](https://hackernoon.com/jev-27b-vl-can-ai-make-better-decisions-from-images), [Jev 1.14](https://ai-tldr.dev/releases/typesafe-jev-1-14/), [Microsoft-Decision-1](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/), [AIMultiple decision-model comparison](https://aimultiple.com/decision-models).

**What they're good for here.** Every SS14 moment can be framed as "pick one of these on-screen options", and the skill layer already produces the options (numbered markers × available skills). So the decision model becomes the agent's System 1, at 3 to 10 decisions per second:

- **Next action selection.** State: the annotated frame, the current goal from the planner ("clean the spills in the hallway"), HUD text. Question: choose one of `walk_to(3)`, `pickup(5)`, `use_held_on(8)`, `inspect(2)`, `wait`. The planner only gets called when the goal changes or the top option's probability is low.
- **Interrupts and checks.** Yes/no questions on each frame or every few frames: "Is the character being attacked?", "Is a UI window open?", "Did the last action succeed?", "Is someone talking to me?". These replace hand-written pixel heuristics and give calibrated confidences you can threshold.
- **Target finding before the detector exists.** Jev's Box question, or a choice over tile-grid cells, can locate "the mop" or "the nearest spill" without training your own detector. That shortens the build plan.
- **Combat.** The reflex skill can ask "which of these enemies to attack / move which way to dodge" at ~5 Hz, which is far closer to playable than a 2-second planner.

**What they can't do.** They don't write free text, so chat replies, long-horizon goals, and anything open-ended still go to the generative planner (Claude Sonnet 5.5 or a local VLM). They only choose among options you give them, so the skill layer's option list is now the most important thing to get right.

**Recommended pick to start:** a small d3 size (d3-mini or d3-flash) self-hosted next to the game for action selection and interrupts, since it's open, supports images at every size, and has published per-image latency in the right range. Try JEV-27B-VL if you have the GPU memory and want a stronger model; use Jev 1.14's API for Box queries if self-hosting isn't practical. The decision layer should sit behind one interface (`decide(state, images, questions) -> probabilities`) so you can benchmark several on the same recorded frames.

The loop from 7.2 becomes:

```
fast loop (every frame):      capture, perceive, run current skill
tactical (every 100-300 ms):  decision model picks next skill / checks interrupts
                              if low confidence or goal done -> wake planner
planner (every 2-10 s):       generative VLM sets goal, writes chat, updates memory
```

## 8. What "real time" means here, honestly

| Layer | Rate | Typical latency |
|---|---|---|
| Capture + perception | 10-30 Hz | 10-50 ms per frame on a GPU |
| Skill execution | every frame | reacts within one frame |
| Tactical decision model | 3-10 Hz | ~60-350 ms per image decision (vendor figures) |
| Planner decision | 0.1-0.5 Hz | 1-3 s per call (API), more for local |

So the agent moves, paths, and reacts to "I'm stuck" instantly, while choices like "go to Medbay" or "answer the Captain" happen at human-conversation speed. That's playable for most crew jobs (janitor, botanist, cargo, engineer doing repairs, chatting). Fast combat is still the weakest area. With the tactical decision model choosing targets and dodge directions several times a second inside a `fight(marker)` skill, it becomes plausible rather than hopeless; the planner should never aim.

## 9. Build plan

1. **Setup.** Local SS14 server (stock build), client windowed at fixed size and zoom. Python project with capture working at 30 fps and the frame shown in a debug window with the tile grid overlaid.
2. **Hands on the controls.** Input layer plus `walk_dir`, `walk_to(tile)` using a tile classifier, verified by motion check. Script it by hand: walk a lap of the arrivals hallway.
3. **Read the screen.** OCR on chat and the right-click menu, HUD hand-slot reading. Script: `inspect` everything near spawn.
4. **First planner.** Annotated screenshots (start with markers from a simple heuristic or from the tile classifier), the skill tools, Claude via API, the async loop and interrupts. First task: "You're the Janitor. Find your mop and bucket and clean the spills in the hallway; reply to anyone who talks to you."
5. **Tactical decision model.** Self-host a d3 size, record frames from step 4, and benchmark it on next-action choice and the interrupt questions. Move per-step choices from the planner to it, and measure decisions per second and task success before and after.
6. **Detector (optional now).** If decision-model target finding isn't good enough: Collect frames during step 4 runs, label, train, replace the heuristic markers. Measure the improvement.
7. **Demo polish.** OBS overlay showing markers, current skill and the planner's one-line intention; record a full shift.

**How to judge the proof of concept:** pick three or four scripted tasks with clear success (clean N spills, fetch an item from another department and hand it to a named player, open a specific locker using an ID). Report success rate, time to complete, and number of planner calls, over repeated rounds on the same map.

## 10. Risks and mitigations

- **Detector misses things in dark or cluttered rooms.** Fallback to `inspect` probes; the planner can also see the raw frame.
- **UI windows vary a lot.** Start with tasks that need few UIs; add per-window OCR handling as needed.
- **Input focus.** Run in a VM or dedicated machine for demos.
- **Game updates change sprites and layout.** Pin the server and client version for the duration of the project.
