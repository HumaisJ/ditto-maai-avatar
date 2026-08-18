# Dialogue Avatar Research Project

This repository is the controlled research workspace for a real-time dialogue avatar built from
Ditto, MaAI, and a small motion controller. Development proceeds in measured stages; local checks
do not run either model, while separately approved GPU scripts guard model setup and experiments.

Current progress, evidence, and the next approved checkpoint are maintained in
[`PROJECT_TIMELINE.md`](PROJECT_TIMELINE.md).

## Local development environment

Create the isolated environment without modifying Conda `base`:

```powershell
conda env create -f environment.dev.yml
conda activate avatar-dev
```

If the environment already exists and the definition changes:

```powershell
conda env update -n avatar-dev -f environment.dev.yml --prune
```

## Validate the project

The manifest contains 17 deterministic portrait/audio pairs. Validate every path, image, WAV, and
recorded duration with:

```powershell
python scripts/check_project.py
ruff check .
python -m pytest -q
```

## Run a local mock experiment

This exercises result recording without Ditto, MaAI, CUDA, or a GPU:

```powershell
python scripts/run_mock_experiment.py --pair-id P001
```

The command creates a numbered directory such as:

```text
results/experiments/MOCK-EXP-0001/
├── experiment.json
├── config.json
├── metrics.json
├── console.log
├── gpu.csv
├── notes.md
└── mock_output.txt
```

Generated experiment directories are retained locally and ignored by Git. Failed runs are also
retained because they are useful research evidence.

## Git LFS and the GPU machine

Portraits and WAV files are tracked using Git LFS. After cloning or pulling on the GPU machine,
retrieve their contents before validation:

```powershell
git lfs install
git lfs pull
python scripts/check_project.py
```

Do not begin model installation or GPU experiments until the relevant stage has been discussed and
approved.

Windows GPU environment preparation is documented in
[`docs/GPU_SETUP_WINDOWS.md`](docs/GPU_SETUP_WINDOWS.md).
