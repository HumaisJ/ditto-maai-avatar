# CODEX PROJECT GUIDE
## Dialogue Avatar Research Project — Ditto + MaAI

### Purpose of this file

This file is the operating guide for the Codex coding agent working on this project.

The project goal is to build and test a real-time dialogue avatar pipeline in small, controlled stages.

The current chosen core components are:

- **Ditto** — avatar renderer / talking portrait model.
- **MaAI** — lightweight conversational behavior model for listening-related signals such as nodding, VAD, turn-taking, and backchannel timing.
- **Motion Controller** — our own small control layer that converts states and MaAI outputs into motion instructions for Ditto.
- Later: **STT → LLM → TTS** will be added only after Ditto and MaAI work correctly by themselves.

The GPU machine is shared and has an **NVIDIA RTX 5060 Ti 16 GB**. GPU time must be used carefully.

---

# 1. NON-NEGOTIABLE WORKING RULES

Codex must follow these rules for every task.

## Rule 1 — Never code without approval

Codex must never make code changes automatically.

For every new task:

1. Inspect the current project state.
2. Explain what needs to be done.
3. Give a small plan in simple words.
4. State exactly which files would be created or changed.
5. Ask for approval.
6. Only after approval, make the agreed changes.

No approval = no coding.

---

## Rule 2 — Always use this workflow

Every piece of work must follow:

**PLAN → DISCUSS → APPROVAL → WORK → VERIFY → REPORT**

Never skip directly to implementation.

---

## Rule 3 — Work one small step at a time

Do not build an entire phase at once.

Bad example:

> “I will add Ditto, MaAI, GUI, state controller, logging, and streaming.”

Good example:

> “First I will create the experiment manifest and result-recording format. After you approve and we finish that, we will move to the single-file Ditto test.”

Each step must be independently understandable and testable.

---

## Rule 4 — Explain everything simply

Assume the user wants instructions that a beginner can follow.

For every completed task explain:

- What was done.
- Why it was done.
- What files changed.
- How to verify it.
- What the next small step should be.

Avoid unexplained ML/software jargon.

---

## Rule 5 — Never silently change the project direction

The current approved architecture is:

**Ditto + MaAI + Motion Controller**

Do not replace Ditto, MaAI, the folder structure, the evaluation system, or the main architecture without discussing it first.

---

# 2. GPU MACHINE RULES — STRICT

The GPU machine is shared.

Treat it as an **execution and benchmarking machine**, not the main development machine.

## Codex must NOT:

- Edit source code directly on the GPU machine as the normal workflow.
- Install packages globally.
- modify the Conda `base` environment.
- modify system CUDA drivers.
- upgrade/downgrade NVIDIA drivers.
- modify system-wide CUDA libraries.
- use `sudo` for Python/ML dependencies unless the user explicitly approves it.
- delete or modify another user's files.
- kill another user's GPU processes.
- start a long GPU experiment without approval.
- launch all 289 portrait/audio combinations without explicit approval.
- run indefinite background GPU jobs.
- leave GPU processes running after an experiment.
- download very large checkpoints repeatedly.
- change precision, resolution, TensorRT settings, or optimization settings without recording them.
- run multiple heavy Ditto jobs in parallel unless specifically approved.

## Codex must:

- Use a project-specific isolated Conda environment.
- Prefer preparing all source code on the user's PC.
- Commit working source code to Git before important GPU experiments.
- Check GPU status before starting a run.
- Record GPU information for every benchmark.
- Run one controlled experiment first before running a batch.
- Stop and report if the GPU appears heavily occupied.
- Save outputs, metrics, logs, and generated videos.
- Shut down/terminate experiment processes after each run.
- Return to local development for code fixes whenever practical.

Recommended environment names:

```text
avatar-ditto
avatar-maai
```

These may be changed only after discussion.

---

# 3. PROJECT DEVELOPMENT WORKFLOW

The preferred workflow is:

```text
LOCAL PC
    ↓
plan
    ↓
write/edit code
    ↓
lint/test with mocks
    ↓
Git commit
    ↓
copy/pull approved version to GPU machine
    ↓
activate isolated Conda environment
    ↓
run controlled experiment
    ↓
save experiment results
    ↓
copy/analyze results locally
    ↓
decide next experiment
```

If an error only appears on the GPU machine:

```text
GPU error
   ↓
save traceback/log
   ↓
stop experiment
   ↓
return to local source
   ↓
discuss proposed fix
   ↓
approval
   ↓
fix locally
   ↓
commit
   ↓
rerun on GPU
```

Avoid “live coding” on the shared GPU server.

---

# 4. EXISTING PROJECT STATE

The user has already completed:

- Project folder structure.
- Git initialization.
- Asset collection.
- **17 portrait images.**
- **17 audio files.**

The audios have different lengths, from a few seconds to several minutes.

Before changing the structure, Codex must inspect the repository and report what already exists.

Do not recreate folders that are already present.

---

# 5. IMPORTANT TESTING INTERPRETATION

There are:

- 17 portraits.
- 17 audios.

This can mean two different tests:

### Mode A — Paired test

```text
portrait_01 + audio_01
portrait_02 + audio_02
...
portrait_17 + audio_17
```

Total:

**17 Ditto runs**

This is the DEFAULT initial batch.

### Mode B — Full cross-product test

Every portrait is tested with every audio:

```text
17 × 17 = 289 runs
```

This is much more expensive.

**Codex must NEVER start the 289-run matrix automatically.**

Before a 289-run test:

1. Explain expected number of runs.
2. Estimate expected GPU workload using measurements from earlier tests.
3. Explain expected disk usage.
4. Ask for explicit approval.

---

# 6. TARGET REPOSITORY STRUCTURE

Use the user's existing structure where possible.

Recommended logical structure:

```text
dialogue-avatar/
│
├── README.md
├── CODEX_PROJECT_GUIDE.md
│
├── config/
│   ├── base.yaml
│   ├── ditto.yaml
│   ├── maai.yaml
│   └── benchmark.yaml
│
├── assets/
│   ├── portraits/
│   ├── audio/
│   └── manifest.csv
│
├── src/
│   ├── avatar/
│   │   ├── base_renderer.py
│   │   ├── mock_renderer.py
│   │   └── ditto_adapter.py
│   │
│   ├── behavior/
│   │   ├── base_behavior.py
│   │   ├── mock_behavior.py
│   │   └── maai_adapter.py
│   │
│   ├── controller/
│   │   ├── states.py
│   │   ├── motion_controller.py
│   │   └── conversation_controller.py
│   │
│   ├── audio/
│   │   ├── audio_chunks.py
│   │   └── jitter_buffer.py
│   │
│   └── utils/
│       ├── experiment.py
│       ├── logger.py
│       ├── metrics.py
│       └── system_info.py
│
├── scripts/
│   ├── check_project.py
│   ├── check_environment.py
│   ├── run_ditto_single.py
│   ├── run_ditto_batch.py
│   ├── run_maai_audio.py
│   └── benchmark_ditto.py
│
├── tests/
│   ├── test_states.py
│   ├── test_experiment.py
│   └── test_audio_chunks.py
│
├── results/
│   └── experiments/
│
└── logs/
```

Do not create all missing files at once.

Create only the files needed for the currently approved step.

---

# 7. FIRST DEVELOPMENT GOAL

## Goal 1 — Prepare the evaluation system before running Ditto

Before touching the GPU model, prepare the system that will record the experiments.

The first approved coding work should ideally produce:

1. Asset manifest.
2. Experiment ID generator.
3. Result folder creator.
4. Metadata logger.
5. Metrics logger.
6. GPU logging helper or command wrapper.
7. Simple local validation tests.

### Asset manifest

Recommended:

```text
assets/manifest.csv
```

Suggested fields:

```text
pair_id
portrait_path
audio_path
audio_duration_sec
language
notes
```

Example:

```text
P001,assets/portraits/p01.png,assets/audio/a01.wav,8.7,en,
```

Audio duration should be detected automatically where possible.

Do not rename the user's original assets without permission.

---

# 8. STANDARD EXPERIMENT STORAGE FORMAT

Every important run must create a separate directory.

Example:

```text
results/experiments/DITTO-EXP-0001/
```

Recommended contents:

```text
DITTO-EXP-0001/
│
├── experiment.json
├── metrics.json
├── config.json
├── console.log
├── gpu.csv
├── output.mp4
└── notes.md
```

The original portrait/audio do not need to be duplicated if disk space matters.

Their exact relative paths and file hashes can be recorded in `experiment.json`.

Generated videos **must be preserved** unless the user explicitly decides otherwise.

Generated videos are part of the proof/evidence of testing.

---

# 9. EXPERIMENT METADATA STANDARD

Every Ditto run should record at least:

```text
experiment_id
date/time
purpose
Git commit hash
model name
model/checkpoint version
backend
precision
Python version
PyTorch version
CUDA version visible to PyTorch
GPU name
GPU total VRAM
input portrait path
input portrait hash
input audio path
input audio hash
audio duration
requested output path
success/failure
error message if failed
```

Also record relevant Ditto configuration values.

Never compare experiments when important configuration differences are unknown.

---

# 10. DITTO PERFORMANCE METRICS

Record where possible:

```text
model_load_time_sec
inference_time_sec
audio_duration_sec
real_time_factor
output_video_duration_sec
output_frame_count
effective_generation_fps
peak_vram_mb
average_gpu_utilization_percent
maximum_gpu_utilization_percent
```

Useful formula:

```text
real_time_factor = inference_time / audio_duration
```

Interpretation:

```text
RTF < 1.0 → faster than real time
RTF = 1.0 → real time
RTF > 1.0 → slower than real time
```

For later streaming tests also record:

```text
time_to_first_frame_ms
audio_to_motion_latency_ms
frame_interval_jitter
```

For later interruption tests:

```text
speech_detection_to_avatar_stop_ms
```

---

# 11. DITTO VISUAL QUALITY STANDARD

Use a fixed 1–5 scale.

```text
1 = unacceptable
2 = poor
3 = usable
4 = good
5 = excellent
```

Record:

```text
identity_preservation
lip_sync_quality
facial_naturalness
head_motion_naturalness
upper_portrait_motion_naturalness
artifact_level
overall_realism
```

Also allow:

```text
N/A
```

when a metric is not relevant yet.

Do not pretend subjective scores are objective measurements.

They are standardized human observations.

---

# 12. DITTO TEST PLAN

## Stage D0 — Repository and environment inspection

No model run.

Codex must:

1. Inspect project structure.
2. Inspect asset names/formats.
3. Inspect Git status.
4. Identify the operating system used locally.
5. Identify how code will be transferred to the GPU machine.
6. Propose the first small implementation step.

Then ask for approval.

---

## Stage D1 — Local experiment infrastructure

No GPU required.

Prepare only:

- manifest handling.
- experiment directory creation.
- JSON/CSV logging.
- basic configuration.
- local tests.
- mock model output if necessary.

Run:

```text
ruff
pytest
```

Use `mypy` only if type checking has been intentionally introduced.

Goal:

> We can create a fake experiment locally and get a clean result folder without any GPU model.

Do not move to Ditto until this works.

---

## Stage D2 — GPU environment creation

On the GPU machine:

- Create `avatar-ditto` Conda environment.
- Never modify `base`.
- Install only required project dependencies inside this environment.
- Record package versions.
- Record GPU details.
- Verify PyTorch sees the RTX 5060 Ti 16 GB.

Goal:

> The isolated environment can see the GPU.

No batch model tests yet.

---

## Stage D3 — Install and verify Ditto

Install Ditto/checkpoints in the project/user space.

Prefer testing the normal PyTorch path first.

Do not begin with TensorRT optimization unless discussion and approval says otherwise.

Goal:

> Ditto imports successfully and checkpoints are found.

---

## Stage D4 — One single Ditto test

Use exactly:

- 1 portrait.
- 1 short audio.

Prefer a short clean audio first.

Run only one inference.

Save:

- generated video.
- console log.
- experiment metadata.
- GPU log.
- performance metrics.

Goal:

> One portrait + one audio produces one valid video.

If this fails, stop.

Do not run the other 16.

---

## Stage D5 — Second controlled test

Use:

- another portrait.
- a longer audio.

Purpose:

- confirm result is reproducible.
- expose problems hidden by the first input.

Again save everything.

Goal:

> Ditto works on more than one input.

---

## Stage D6 — Small mini-batch

Before all 17:

Run a small approved subset such as:

```text
3 portraits + their 3 paired audios
```

Choose short, medium, and longer audio if possible.

Goal:

> Batch runner and result recorder work correctly.

If the batch runner is broken, fix it before consuming more GPU time.

---

## Stage D7 — Full paired set

Run:

```text
17 paired portrait/audio tests
```

Sequentially by default.

Do not run all jobs in parallel.

Every run must have its own experiment folder and output video.

A batch-level summary should also be created.

Example:

```text
results/batches/DITTO-BATCH-001/
```

Possible summary contents:

```text
batch.json
summary.csv
failures.csv
```

Goal:

> Obtain a complete documented Ditto baseline across the 17 paired inputs.

---

## Stage D8 — Optional 289-run cross-product

NOT part of the default plan.

Only after:

- D7 is complete.
- GPU cost is estimated.
- disk cost is estimated.
- user explicitly approves.

---

## Stage D9 — Ditto streaming test

Only after offline Ditto is stable.

Test chunked audio using Ditto's streaming/online pipeline.

Start with local audio deliberately split into chunks.

Do not introduce network TTS yet.

Goal:

> Local audio chunks can drive continuous Ditto output.

---

# 13. MaAI TEST PLAN

Do not integrate MaAI with Ditto immediately.

Test MaAI independently first.

## Stage M0 — Isolated MaAI environment

Create:

```text
avatar-maai
```

inside Conda.

Do not modify Ditto's environment unless approved.

Goal:

> MaAI imports and runs independently.

---

## Stage M1 — One audio-file test

Use one existing audio file.

Feed it to MaAI.

Record raw behavioral output.

Possible outputs to preserve include:

```text
timestamp
VAD/speaking probability
turn-taking probability
nod signal
nod parameters
backchannel signal
```

Actual fields depend on the selected MaAI model/API.

Save results, for example:

```text
results/experiments/MAAI-EXP-0001/
│
├── experiment.json
├── reaction.csv
├── reaction.json
├── console.log
└── notes.md
```

Goal:

> MaAI listens to an audio file and produces a recorded time-based reaction signal.

No Ditto yet.

---

## Stage M2 — Visualize MaAI reactions

After raw output is correct, discuss a simple visualization.

Examples:

- timeline plot.
- nod markers.
- speech-active region.
- turn-taking probability graph.

Do not build a GUI yet unless approved.

Goal:

> A human can inspect and understand MaAI's reactions.

---

## Stage M3 — More MaAI inputs

Use a very small approved set of different audios.

Do not automatically process all files unless needed.

Goal:

> See whether MaAI behavior is stable across different speech samples.

---

# 14. MOTION CONTROLLER PLAN

Only after Ditto and MaAI both work independently.

The controller owns four states:

```text
IDLE
LISTENING
THINKING
SPEAKING
```

The controller must remain small and explainable.

## IDLE

No premade video is required.

Use subtle procedural motion through Ditto controls where possible.

Examples:

- tiny head pose variation.
- tiny gaze change.
- natural neutral movement.

Do not make large random movements.

## LISTENING

MaAI supplies conversational reaction timing.

Example:

```text
MaAI says nod
    ↓
Motion Controller converts nod parameters
    ↓
Ditto receives motion control
```

## THINKING

This is controlled procedurally.

Examples:

- slight head tilt.
- small gaze change.
- brief quiet motion.

No separate heavy ML model unless later proven necessary.

## SPEAKING

Ditto is driven by avatar/TTS speech audio.

---

# 15. INTERRUPTION PLAN — LATER PHASE

Do not implement interruptions during first Ditto/MaAI testing.

Later architecture:

```text
avatar is SPEAKING
        ↓
real user speech is detected
        ↓
confirm it is not a tiny noise/cough
        ↓
stop TTS playback
        ↓
clear queued TTS audio
        ↓
switch to LISTENING
```

The exact speech-duration threshold must be tested later.

Do not hardcode a “final” interruption threshold before measurement.

---

# 16. NETWORK/JITTER PLAN — LATER PHASE

Do not build network streaming before local streaming works.

Later:

```text
network TTS chunks
        ↓
jitter buffer
        ↓
stable local audio stream
        ↓
Ditto
```

The avatar must continue displaying natural IDLE/LISTENING/THINKING motion while waiting for network audio.

Do not allow network delay to freeze the avatar.

---

# 17. SIMPLE SUPERVISOR GUI PLAN

Build GUI only after core offline tests work.

Do not choose or implement a GUI framework without discussion and approval.

A simple framework such as Gradio may be proposed, but Codex must first explain the tradeoff and ask permission.

## GUI 1 — Ditto Test GUI

Minimum functions:

```text
select portrait
select audio file
run Ditto
show progress/status
play generated video
show main metrics
show experiment ID
```

No advanced design required.

The GUI must call the same tested backend functions used by benchmark scripts.

Do not duplicate model logic inside the GUI.

---

## GUI 2 — MaAI Test GUI

Minimum functions:

```text
select audio file OR microphone input
run MaAI
show reaction timeline/data
show nod events
show speaking/turn signals
show experiment ID
```

Again, use the same backend code as tests.

---

## GUI 3 — Combined Demo

Only after Ditto + MaAI integration works.

Minimum target:

```text
user audio
    ↓
MaAI listening behavior
    ↓
Motion Controller
    ↓
Ditto portrait behavior
```

Later this GUI can add STT/LLM/TTS.

---

# 18. FINAL PIPELINE GOAL

The long-term architecture is:

```text
                    USER AUDIO
                         │
             ┌───────────┴───────────┐
             │                       │
           MaAI                     STT
             │                       │
 listening behavior                 LLM
             │                       │
             │                       TTS
             │                       │
             ▼                       ▼
      Motion Controller        Audio Buffer
             │                       │
             └──────────┬────────────┘
                        │
                      Ditto
                        │
                        ▼
                 Live Avatar Video
```

Final states:

```text
IDLE
LISTENING
THINKING
SPEAKING
```

Later additions:

```text
interruption handling
streaming TTS
network jitter buffer
STT
LLM
live microphone
supervisor demo UI
long-session stability tests
```

Do not implement these before earlier milestones pass.

---

# 19. STANDARD GOAL LADDER

Codex must always know which goal is currently active.

## Goal 1

Local experiment/logger system works.

## Goal 2

GPU environment is isolated and valid.

## Goal 3

One Ditto portrait + one audio generates one saved video.

## Goal 4

Ditto works on multiple controlled inputs.

## Goal 5

All 17 paired Ditto tests are recorded.

## Goal 6

Ditto streaming works with local chunked audio.

## Goal 7

MaAI independently produces recorded reactions from an audio file.

## Goal 8

MaAI reactions are understandable/visualized.

## Goal 9

Motion Controller converts MaAI behavior into Ditto motion.

## Goal 10

IDLE/LISTENING/THINKING/SPEAKING states work.

## Goal 11

Simple supervisor GUIs work.

## Goal 12

Interruptions work.

## Goal 13

STT → LLM → TTS is connected.

## Goal 14

Network audio buffering/jitter handling works.

## Goal 15

Complete live dialogue avatar pipeline works.

Only one goal should be treated as the main active implementation goal at a time.

---

# 20. CODE QUALITY RULES

Before GPU execution, relevant local code should pass appropriate checks.

Recommended:

```text
ruff
pytest
```

Use mocks for GPU components where practical.

Examples:

```text
MockDittoRenderer
MockMaAIBehavior
```

The controller, logging, configuration, manifest, state transitions, and experiment storage should be testable without a GPU.

Do not make the whole project depend on GPU availability.

---

# 21. CONFIGURATION RULES

Avoid hardcoded experiment values scattered through source files.

Prefer configuration files and CLI arguments.

Examples of configurable values:

```text
checkpoint path
backend
precision
portrait path
audio path
output directory
state thresholds
stream chunk size
seed
resolution
FPS
```

Every experiment should save the final effective configuration it used.

---

# 22. REPRODUCIBILITY RULES

For every important experiment:

- Record Git commit.
- Record config.
- Record environment/package versions.
- Record model/checkpoint version.
- Record input file hashes.
- Record GPU.
- Record output video.
- Record logs.
- Record success/failure.
- Record subjective evaluation separately.

Failed experiments must not be silently deleted.

A failure is useful research data.

---

# 23. BATCH SAFETY RULES

Before any batch run Codex must explain:

```text
number of runs
input selection
expected experiment IDs
whether execution is sequential or parallel
estimated runtime based on earlier measurements
estimated output storage
how failure recovery works
```

Default:

**Sequential execution.**

If one run fails:

- Save the failed experiment.
- Record the error.
- Continue only if the approved batch policy says to continue.
- Never silently skip failure.

---

# 24. WHAT CODEX SHOULD SAY BEFORE CODING

For every implementation task, Codex should respond in this style:

> **Current goal:** Prepare Ditto experiment logging.
>
> **What I found:** The project already has X and Y. Z is missing.
>
> **What I propose to do now:** Create only A and B.
>
> **Files I would change:** `...`
>
> **How we will verify it:** Run `...`
>
> **What I will NOT do yet:** Ditto installation, GPU execution, MaAI, GUI.
>
> **Approval needed:** Shall I implement this step?

Only after approval may Codex edit files.

---

# 25. WHAT CODEX SHOULD SAY AFTER WORK

After approved work:

> **Done:** ...
>
> **Files changed:** ...
>
> **Verification:** ...
>
> **Result:** pass/fail.
>
> **What this gives us:** ...
>
> **Next small step:** ...
>
> I will not start the next step until you approve it.

---

# 26. FIRST TASK CODEX SHOULD PERFORM

The first Codex task after reading this file should NOT be model installation.

It should be:

### Project audit + first-step proposal

Codex should inspect:

```text
folder structure
Git status
17 portraits
17 audios
existing Python/config files
```

Then report:

1. What already exists.
2. Any naming/path issues.
3. Whether all 17 portraits and 17 audios are readable.
4. Proposed manifest pairing strategy.
5. Proposed exact files for **Goal 1: experiment recording system**.
6. Local verification plan.
7. Ask for approval.

No coding in this first response.

---

# FINAL PRINCIPLE

This is a research project, not a race to produce one demo.

The priority order is:

```text
correct
↓
reproducible
↓
measured
↓
understood
↓
optimized
↓
integrated
```

Never reverse that order just to make the project appear complete faster.

Every stage must leave behind:

- working code,
- recorded evidence,
- understandable results,
- and a clear next decision.
