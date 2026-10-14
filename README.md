# Generative-AI Embodied Robot Assistant — Student Starter Project

This Webots R2025a project is the starting point for the group assessment. It
provides the simulated environment, an E-puck robot, named locations, object
metadata, a structured-plan contract, a deterministic validation gate, and
interfaces for dialogue, planning, navigation, perception, execution and
logging. It is intentionally incomplete: every `TODO` is part of the design
work, and groups may reorganise the controller if they preserve the required
safety and assessment boundaries.

## Open and run

1. Open `worlds/project_genai_embodied_assistant.wbt` in Webots R2025a.
2. Confirm that the robot controller is `project3_student`.
3. Install only the packages your implementation uses; record exact versions
   in `requirements.txt`.
4. Keep Webots running while testing the complete controller.

The `controller` module is supplied by Webots. Do **not** install the unrelated
PyPI package named `controller`.

## Required architecture

```text
speech/text goal -> dialogue/LLM -> structured plan -> deterministic validator
                  -> task executive -> robot skills -> Webots devices
```

The generative model may propose only symbolic actions. It must never write
wheel speeds, execute Python or shell code, or bypass `PlanValidator`. Every
initial or revised plan must pass the same validator before execution.

Webots devices must be accessed only from the simulation/controller thread.
Potentially slow speech recognition and model inference should run in a worker
thread or process, with results returned to the main loop. The supplied main
controller shows the intended boundary; students complete the state machine.

## LLM/model policy

You may use the Lab 5 Qwen GGUF model, another local model, or an online model.
Model brand and size are not marked; reliable goal interpretation, structured
planning and safe execution are marked.

- Do not commit GGUF/model weights or API keys.
- Read credentials from environment variables, never source code.
- Put model-specific code behind `LLMInterface`.
- A mock backend is encouraged for automated tests but is not accepted as the
  assessment demonstration of generative planning.
- Lab 5 Qwen users do not submit its weights or repeat its installation steps.
- Any other backend must already be configured and demonstration-ready.

Copy `config/llm_config.example.json` to `config/llm_config.json`, adapt it to
your backend, and keep the real file out of version control. The example is a
format guide, not a mandatory client implementation.

## Project map

- `worlds/`: assessed Webots world.
- `controllers/project3_student/`: controller modules and TODO interfaces.
- `config/locations.json`: named destinations.
- `config/navigation_graph.json`: suggested collision-free waypoint graph.
- `config/objects.json`: supported object types and colours.
- `prompts/`: planning prompt starter.
- `tests/`: baseline validator tests; add your own tests.
- `models/`: instructions only; do not place weights in the submission.

Run controller-independent tests from the project root:

```bash
python -m unittest discover -s tests -v
```

These tests do not replace testing inside Webots. See the project specification
for the authoritative requirements, submission checklist and marking rubric.
