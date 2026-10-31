# Generative-AI Embodied Robot Assistant (CSCI444/944, Group 10)

A spoken-language robot assistant for Webots R2025a. The user speaks a goal
("Find the red ball", "Go to the lab, then come back to reception"). The robot
asks for missing information, a local language model (Qwen2.5-3B) turns the goal
into a symbolic plan, a deterministic validator accepts or rejects the plan, and
the E-puck carries it out with its motors, GPS, compass, proximity sensors and
camera. If a step fails, the robot revises the rest of the plan (at most twice)
or stops and explains why.

```text
microphone -> SpeechIO (Vosk) -> DialogueManager -> LLMInterface (Qwen) -> plan
           -> PlanValidator -> TaskExecutive -> RobotSkills -> Navigator / ObjectPerception
           -> Webots devices;  WorldState holds goal, step, evidence and outcomes
```

## 1. Software versions

| Component | Version |
|---|---|
| Webots | R2025a |
| Python (the one Webots runs) | 3.12 |
| Language model | Qwen2.5-3B-Instruct, GGUF Q4_K_M (`qwen2.5-3b-instruct-q4_k_m.gguf`, ~2.1 GB) |
| LLM runtime | llama-cpp-python 0.3.36 |
| Speech recognition | Vosk 0.3.44 with `vosk-model-small-en-us-0.15` |
| Speech output | macOS `say`; pyttsx3 2.99 on Windows/Linux |
| Audio input | sounddevice 0.5.5 |
| Vision | OpenCV 4.13, NumPy 2.5 |

The model is **not** the Lab 5 model (Qwen2.5-0.5B), so it is downloaded
separately (section 2). Exact package versions and download links are in
`requirements.txt`.

## 2. Installation

Run every command from the project folder, with the Python that Webots uses
(Webots: *Tools > Preferences > General > Python command*).

1. **Packages**
   ```bash
   python -m pip install -r requirements.txt
   ```
   *macOS:* the pre-built llama-cpp-python wheels for macOS were broken when we
   tested them. Build it from source instead (takes a few minutes):
   ```bash
   CMAKE_ARGS="-DGGML_METAL=OFF" python -m pip install --no-binary=llama-cpp-python llama-cpp-python==0.3.36
   ```
2. **Language model** (no account needed):
   ```bash
   curl -L -o models/qwen2.5-3b-instruct-q4_k_m.gguf https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
   ```
3. **Speech model:** download
   <https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip> and unzip
   it into `models/` (or `~/Documents/vosk-models/`, or point `VOSK_MODEL_PATH`
   at the folder).
4. **Model configuration:** `config/llm_config.json` is ready to use and holds no
   credentials:

   | Setting | Value | Why |
   |---|---|---|
   | `temperature` | 0.0 | the same goal gives the same plan |
   | `seed` | 0 | reproducible runs |
   | `json_mode` | true | the runtime only produces valid JSON |
   | `constrained_output` | true | the model can only write plans that fit a JSON schema (section 6) |
   | `n_ctx` / `max_tokens` | 4096 / 1000 | the longest prompt (a correction) fits |
   | `timeout_seconds` | 180 | a model call takes 5–60 s on our laptop CPU; slower machines need the margin |
   | `max_retries` | 1 | one corrected plan after a rejection |

   No online model or API key is used.

## 3. Microphone

- **macOS:** Webots has no microphone permission of its own, so the
  microphone only delivers silence when Webots is started from the Dock
  (the console then prints a warning). Start Webots from the Terminal app,
  which has the permission:
  ```bash
  /Applications/Webots.app/Contents/MacOS/webots ~/path/to/project/worlds/project_genai_embodied_assistant.wbt
  ```
  The first time, allow Terminal under *System Settings > Privacy & Security >
  Microphone*.
- **Windows:** allow desktop apps to use the microphone under *Settings >
  Privacy > Microphone*.
- Choose another input with `SPEECH_INPUT_DEVICE=<index or part of the name>`.
- `SPEECH_INPUT=console` lets you type requests instead (debugging only; the
  assessed input is speech).
- Use a quiet room. Background speech is filtered (see section 6), but a TV or
  video can still be heard as a request.

## 4. Running

1. Open `worlds/project_genai_embodied_assistant.wbt` (see section 3 on macOS).
   The robot controller is `project3_student`, and the setup Supervisor
   `scene_randomizer` places the red ball at a random approved spawn point on
   every start or reset.
2. Wait for `Robot Assistant ready.` and `[SPEECH] Listening on the microphone.`
   The model loads on the first request, so the first plan takes longest.
3. Speak a request, for example:
   - "Go to the laboratory." / "Go to the lab, then come back to reception."
   - "Find the red ball." / "Find the ball" (the robot asks which colour)
   - "Look in the lab. If the red ball is not there, check the meeting room."
   - "Check the lab first; only search Office A if the ball is absent."
   - "Fly through the wall to Office B." (the robot offers to drive instead)
   - "What are you doing and why?" at any time
   - "Stop" at any time

The console shows what was heard (`[HEARD:GOAL]`, `[HEARD:UNKNOWN]` ...), the
validated plan, navigation events (`[NAV]`) and one `[TRIAL]` record per task.

## 5. Tests and evaluation

```bash
python -m unittest discover -s tests -v          # all tests, no Webots needed
python evaluation/run_planner_eval.py            # real model on test goals
python evaluation/run_planner_eval.py --no-schema  # the same without the output schema
python evaluation/evaluate_trials.py console.txt # results of Webots trials
```

- The unit tests use a mock LLM backend and saved responses where useful; those
  are labelled as fixtures, not as model performance. Five tests need
  llama-cpp-python (four of them run the real model) and are skipped when it is
  not installed.
- `run_planner_eval.py` sends test goals of each request type (direct,
  sequential, search, conditional, constrained order, unsupported, replanning)
  to the **real** model and saves the results in `evaluation/results/`.
  Replanning cases start from a recorded failure and take the controller's exact
  path (replan, validation and revision check, one corrected attempt).

**Planner results** (Qwen2.5-3B, CPU only;
`evaluation/results/planner_eval_20261028_113950.json`). 31 cases: four new
phrasings of each request type plus 7 recorded failure situations for
replanning. *Correct* means the plan also does what the goal asked (checked
against the expected plan), not only that it passed the validator. Times
exclude the one-off model load on the first request (about 40 s for the 3B
model).

| Request type | Cases | Valid | Correct | Mean time |
|---|---|---|---|---|
| Direct | 4 | 4 | 4 | 6 s |
| Sequential | 4 | 4 | 4 | 9 s |
| Search | 4 | 4 | 3 | 65 s |
| Conditional | 4 | 4 | 4 | 39 s |
| Constrained order | 4 | 4 | 4 | 36 s |
| Impossible movement | 4 | 4 | 4 | 9 s |
| Replanning | 7 | 7 | 7 | 16 s |
| **Total** | **31** | **31** | **30** | **25 s** |

Comparison on the same cases (first plans correct / replans correct):
Qwen2.5-0.5B 8/24 and 7/7, Qwen2.5-1.5B 13/24 and 7/7, Qwen2.5-3B 23/24
and 7/7; the 3B model without the output schema 18/24 and 0/7. Before the
output schema and the separate replanning prompt, 0 of 5 revised plans were
correct although all 5 passed the validator
(`evaluation/results/planner_eval_20261026_155033.json`).

- **Trial protocol:** press Reset before each trial (new random ball position),
  give one spoken request, and let it finish. Afterwards, save the whole Webots
  console to a text file and run `evaluate_trials.py` on it. The script pairs
  each `[TRIAL]` line with the Supervisor's latest `[SETUP GROUND TRUTH]` line
  and reports search accuracy (correct / uncertain / missed / wrong room / not
  searched), distractor rejection, navigation outcomes, task and planning time
  and replanning success. The ground truth is only used here, after the run.

Logs (not submitted): `logs/llm_<date>.jsonl` holds every model call (prompt,
raw answer, parsed plan, errors, time); `logs/trials_<date>.jsonl` holds the
trial records. `PERCEPTION_SAVE_FRAMES=1` saves search camera frames to
`logs/frames/`.

## 6. Project map

- `worlds/` assessed world (unchanged)
- `controllers/project3_student/` robot controller and its modules
- `controllers/scene_randomizer/` setup Supervisor (random ball placement)
- `config/` locations, waypoint graph, objects, LLM configuration, spawn points
  (Supervisor only)
- `prompts/` planning prompt template
- `tests/` unit tests; `tests/fixtures/frames/` real camera frames
- `evaluation/` planner evaluation, trial evaluation and saved results
- `tools/try_speech.py` microphone and recognition check without Webots
- `models/` place for the downloaded models (not submitted)

The `controller` module comes with Webots; do not install the unrelated PyPI
package named `controller`.
