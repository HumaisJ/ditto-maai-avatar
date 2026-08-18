# Dialogue Avatar Project Timeline

This file is the permanent, append-only record of project progress. Update it after every verified
small milestone and before committing that milestone. A failed experiment is recorded as evidence;
it is not silently removed or marked complete.

## Current checkpoint

- **Active goal:** Goal 2 — validate an isolated GPU environment.
- **Current stage:** D2 Windows environment package prepared for the Git-free GPU workflow;
  awaiting execution on GPU 0 of the shared machine.
- **Last completed stage:** D1 — local experiment infrastructure.
- **Next decision:** Review the GPU environment report before installing Ditto or checkpoints.
- **Blocked:** No. Remote validation is awaiting the prepared PowerShell workflow.

## Roadmap

| Area | Checkpoint | Status |
|---|---|---|
| Ditto | D0 — repository and environment inspection | Complete |
| Ditto | Asset normalization prerequisite | Complete |
| Ditto | D1 — local experiment infrastructure | Complete |
| Ditto | D2 — isolated GPU environment | In progress |
| Ditto | D3 — install and verify Ditto/checkpoints | Not started |
| Ditto | D4 — one short portrait/audio inference | Not started |
| Ditto | D5 — second controlled inference | Not started |
| Ditto | D6 — three-pair mini-batch | Not started |
| Ditto | D7 — full 17-pair baseline | Not started |
| Ditto | D8 — optional 289-run cross-product | Not approved |
| Ditto | D9 — local chunked-audio streaming | Not started |
| MaAI | M0 — isolated environment | Not started |
| MaAI | M1 — one recorded audio reaction | Not started |
| MaAI | M2 — reaction visualization | Not started |
| MaAI | M3 — small controlled input set | Not started |
| Integration | Motion Controller and four conversation states | Not started |
| Integration | Supervisor GUIs | Not started |
| Integration | Interruption handling | Not started |
| Integration | STT → LLM → TTS | Not started |
| Integration | Network jitter handling and complete live pipeline | Not started |

## Completed milestone log

### 2026-08-17 — D0 repository and asset audit

- **Result:** Complete.
- **Work:** Inspected the scaffold, Git state, 17 portraits, and 18 initial audio files.
- **Evidence:** Identified one duplicate-format MP3 and malformed size metadata in 11 WAV headers.
- **Decision:** Preserve original media content, normalize names, and use one-to-one paired tests.

### 2026-08-17 — Asset normalization prerequisite

- **Result:** Complete.
- **Work:** Renamed 17 portraits and 17 WAV files to lowercase snake case; retained both Dazai
  identities; repaired RIFF/data-size metadata in 11 WAVs without resampling audio.
- **Verification:** All portraits decoded, all WAVs opened with the standard WAV reader, and all
  measured durations were valid.
- **Decision:** Pair normalized portrait/audio filenames alphabetically as P001–P017.

### 2026-08-17 — D1.1 manifest and controlled assets

- **Result:** Complete.
- **Commit:** `6e94713` (`chore: catalog and track test assets`).
- **Work:** Added the 17-pair manifest, project validator, isolated `avatar-dev` definition, tests,
  and Git LFS tracking for all 34 media assets.
- **Verification:** Ruff passed, 11 tests passed, all 17 pairs validated, and all media appeared in
  `git lfs ls-files`.

### 2026-08-17 — D1.2 local experiment recording

- **Result:** Complete.
- **Commit:** `108aff1` (`feat: add local experiment recording infrastructure`).
- **Work:** Added experiment IDs/directories, atomic JSON output, logs, metrics, hashes, system/GPU
  metadata, failure retention, configuration, and a mock runner.
- **Verification:** Ruff passed, 29 tests passed, and `MOCK-EXP-0001` completed successfully with a
  complete result directory.
- **Evidence:** `results/experiments/MOCK-EXP-0001/`.

### 2026-08-17 — D2 Windows GPU environment preparation

- **Result:** Prepared locally; remote D2 validation pending.
- **Goal/stage:** Goal 2 / Stage D2.
- **Work:** Added the Windows `avatar-ditto` environment definition, guarded PowerShell setup and
  checkpoint-publishing commands, CUDA/Blackwell environment validation, GPU runbook, persistent
  timeline policy, and versioned experiment-result policy.
- **Files/evidence:** `environment.ditto.windows.yml`, `scripts/gpu/`,
  `docs/GPU_SETUP_WINDOWS.md`, and `scripts/check_environment.py`.
- **Verification:** Ruff passed, all 53 tests passed, all 17 asset pairs validated, both PowerShell
  scripts parsed successfully, and generated-video paths resolved to Git LFS.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** D2 is not complete until the shared RTX 5060 Ti produces a passing
  `results/environment/D2-GPU-ENV-*` report. Ditto, checkpoints, and TensorRT remain uninstalled.
- **Next step:** Push the prepared repository, run the documented D2 PowerShell command on the GPU
  machine, and return its preserved report for review.

### 2026-08-17 — D2 Git-free shared-GPU preparation

- **Result:** Complete locally; remote D2 validation pending.
- **Goal/stage:** Goal 2 / Stage D2.
- **Work:** Removed the Git/Git LFS requirement from GPU setup, added explicit GPU selection, scoped
  compute-process safety checks to the selected GPU, and documented PC-to-GPU folder transfer.
- **Files/evidence:** `scripts/gpu/setup_ditto_env.ps1`, `scripts/check_environment.py`,
  `docs/GPU_SETUP_WINDOWS.md`, and focused regression tests.
- **Verification:** Ruff passed and all 55 tests passed, including PowerShell parsing and selected-GPU
  process-filtering checks.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** D2 will use idle GPU 0. GPU 1 is occupied by another user's two Python
  processes and must remain untouched. D2 is not complete until a passing GPU report is returned.
- **Next step:** Copy the materialized project folder to the GPU machine and run the guarded D2 setup
  with `-GpuIndex 0`.

### 2026-08-18 — D2 failure-report numbering correction

- **Result:** Failed GPU attempt preserved; local correction complete.
- **Goal/stage:** Goal 2 / Stage D2.
- **Work:** Diagnosed and corrected a PowerShell formatting error that occurred when allocating the
  report number after `D2-GPU-ENV-0001` already existed.
- **Files/evidence:** GPU terminal traceback at `setup_ditto_env.ps1:33`; corrected setup script and
  regression test.
- **Verification:** The regression reproduces `Measure-Object`'s `System.Double` result and confirms
  that the corrected formatter produces `D2-GPU-ENV-0002`.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** The formatting error masked the original preflight failure. Environment
  creation did not start because the creation message was never reached.
- **Next step:** Copy the corrected setup script to the GPU project and rerun with the same isolated
  Conda paths and GPU 0 selection.

## Entry template

### YYYY-MM-DD — Milestone name

- **Result:** Complete, failed, or blocked.
- **Goal/stage:** The active numbered goal or stage.
- **Work:** What changed or ran.
- **Files/evidence:** Paths to durable outputs.
- **Verification:** Exact checks and their outcomes.
- **Commit:** Commit hash when already known, or “included in this milestone commit.”
- **Decision/limitation:** Important conclusion or unresolved constraint.
- **Next step:** One approved small step only.
