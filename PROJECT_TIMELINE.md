# Dialogue Avatar Project Timeline

This file is the permanent, append-only record of project progress. Update it after every verified
small milestone and before committing that milestone. A failed experiment is recorded as evidence;
it is not silently removed or marked complete.

## Current checkpoint

- **Active goal:** Goal 3 — produce one saved Ditto portrait/audio video.
- **Current stage:** D4 tooling prepared locally; the single GPU inference is pending.
- **Last completed stage:** D3 — install and verify Ditto/checkpoints.
- **Next decision:** Transfer the committed D4 tooling and run it exactly once on GPU 0.
- **Blocked:** No.

## Roadmap

| Area | Checkpoint | Status |
|---|---|---|
| Ditto | D0 — repository and environment inspection | Complete |
| Ditto | Asset normalization prerequisite | Complete |
| Ditto | D1 — local experiment infrastructure | Complete |
| Ditto | D2 — isolated GPU environment | Complete |
| Ditto | D3 — install and verify Ditto/checkpoints | Complete |
| Ditto | D4 — one short portrait/audio inference | Prepared locally |
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

### 2026-08-18 — D2 Windows WDDM process classification correction

- **Result:** Failed GPU attempt preserved; local correction prepared.
- **Goal/stage:** Goal 2 / Stage D2.
- **Work:** Replaced the WDDM-incompatible compute query with selected-GPU process-type filtering,
  retained utilization protection, and added a 12000 MiB free-VRAM threshold.
- **Files/evidence:** GPU report `D2-GPU-ENV-0002`, setup and validation scripts, runbook, and focused
  regression tests.
- **Verification:** Ruff passed and all 60 tests passed, including WDDM desktop-process allowance,
  compute-only process blocking, selected-GPU filtering, and PowerShell parsing.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** Normal Windows `G` and `C+G` desktop processes are observed but allowed;
  compute-only `C`, `M`, and `M+C` processes block D2. No process is stopped or modified.
- **Next step:** Copy the verified correction to the GPU project and rerun on GPU 0, preserving the
  next report as `D2-GPU-ENV-0003`.

### 2026-08-18 — D2 Python 3.10 validation compatibility correction

- **Result:** GPU environment and PyTorch installation succeeded; CUDA validation blocked by a
  local compatibility error.
- **Goal/stage:** Goal 2 / Stage D2.
- **Work:** Replaced the Python 3.11-only `datetime.UTC` usage with `timezone.utc` for the approved
  Python 3.10 environment and made setup verify that a validator report exists before suppressing
  its fallback failure report.
- **Files/evidence:** GPU terminal traceback from `scripts/check_environment.py`; compatibility and
  failure-report regression tests.
- **Verification:** Ruff passed and all 62 tests passed, including the UTC timestamp compatibility
  and validator-report fallback checks.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** `avatar-ditto` and PyTorch 2.8.0+cu128 are installed successfully. They
  will be reused; no environment recreation or package redownload is required.
- **Next step:** Copy the two corrected scripts to the GPU machine and run only the D2 validator in
  the existing `avatar-ditto` environment.

### 2026-08-18 — D2 custom-prefix environment identity correction

- **Result:** GPU/CUDA validation checks passed; environment-name assertion produced one false
  failure.
- **Goal/stage:** Goal 2 / Stage D2.
- **Work:** Updated environment identity validation to accept either `CONDA_DEFAULT_ENV` or the
  basename of `CONDA_PREFIX`, supporting the isolated custom environment directory.
- **Files/evidence:** GPU report `D2-GPU-ENV-0003`, validator, and custom-prefix regression test.
- **Verification:** Ruff passed and all 63 tests passed, including the custom-prefix environment
  identity regression.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** Report `0003` had no CUDA, PyTorch, GPU, FFmpeg, utilization, VRAM, or
  compute-process errors. D2 remains in progress until the corrected validator records a pass.
- **Next step:** Copy the corrected validator to the GPU machine and rerun it, preserving report
  `D2-GPU-ENV-0004`.

### 2026-08-18 — D2 isolated GPU environment validated

- **Result:** Complete.
- **Goal/stage:** Goal 2 / Stage D2.
- **Work:** Created the isolated `avatar-ditto` Conda environment in the student's directory,
  installed PyTorch 2.8.0+cu128, and completed the guarded CUDA validation on GPU 0.
- **Files/evidence:** `results/environment/D2-GPU-ENV-0001` through `D2-GPU-ENV-0004`; report
  `0004` is the passing checkpoint and reports `errors: []`.
- **Verification:** Windows 10.0.26200, Python 3.10.20, FFmpeg, RTX 5060 Ti with 16311 MiB VRAM,
  driver 591.86, 14900 MiB free VRAM, 0% utilization, no compute-only process on GPU 0, PyTorch
  2.8.0+cu128, CUDA 12.8, capability 12.0, compiled `sm_120`, and a successful CUDA tensor result.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** D2 is complete. Failed reports `0001`–`0003` are preserved as diagnostic
  history. Ditto, model checkpoints, inference, TensorRT, MaAI, and batch execution remain outside
  this completed stage.
- **Next step:** Plan D3 Ditto/checkpoint installation and import/path verification; do not install
  or download them until that plan is discussed and approved.

### 2026-08-18 — D3 reproducible installation prepared

- **Result:** Complete locally; GPU installation and model-load validation pending.
- **Goal/stage:** Goal 3 / Stage D3.
- **Work:** Pinned the official Ditto source and PyTorch checkpoint revisions; added a clean
  source-transfer bundle, resumable selective checkpoint download, isolated dependency installation,
  source/checkpoint hash verification, and guarded no-inference model initialization.
- **Files/evidence:** `config/ditto.yaml`, `requirements/ditto.txt`, D3 scripts under `scripts/`, the
  expanded Windows GPU runbook, and the ignored `.transfer/ditto-source-c3e47eee.*` bundle.
- **Verification:** Ruff passed, all 75 tests passed including PowerShell parsing, and the 959061-byte
  source archive matched its manifest SHA-256 with 64 source files and no staging directory left.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** Runtime source, checkpoints, and caches remain untracked under `.runtime/`.
  TensorRT, portrait/audio inference, MaAI, and D4 work remain excluded.
- **Next step:** Prepare the pinned source archive on the PC, copy the updated project and bundle to
  the GPU machine, then run `install_ditto.ps1` on GPU 0 and return its preserved D3 report.

### 2026-08-18 — D3 Windows ONNX Runtime pin correction

- **Result:** Failed GPU attempt preserved; local correction prepared.
- **Goal/stage:** Goal 3 / Stage D3.
- **Work:** Replaced the unavailable ONNX Runtime 1.26.0 Windows/Python 3.10 pin with 1.23.2 and
  added requirements-content and post-install import checks to the guarded installer.
- **Files/evidence:** GPU reports `D3-DITTO-INSTALL-0001` and `0002`, dependency requirements,
  Ditto configuration, installer, and focused regression tests.
- **Verification:** ONNX Runtime 1.23.2 has an official CPython 3.10 Windows wheel and uses the
  CUDA 12.8/cuDNN 9 line required by the validated PyTorch environment; Ruff and all 76 tests passed.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** No checkpoint download or model load occurred. Partial dependency work
  remained confined to `avatar-ditto`; no other environment, GPU process, or system package changed.
- **Next step:** Copy the corrected requirements, config, and installer to the GPU project, verify
  their hashes, and rerun the resumable D3 installer on GPU 0.

### 2026-08-18 — D3 Windows blend compatibility correction

- **Result:** Dependencies and all 12 checkpoints verified; model import failed and a local
  compatibility correction was prepared.
- **Goal/stage:** Goal 3 / Stage D3.
- **Work:** Added a vectorized NumPy equivalent for Ditto's small Cython image-blend kernel and
  inject it under the upstream import name on Windows before `StreamSDK` is imported.
- **Files/evidence:** GPU report `D3-DITTO-INSTALL-0003`, Windows compatibility helper, model-load
  verifier, and numerical equivalence regression tests.
- **Verification:** The fallback preserves the upstream mask blend, clipping, truncation, output
  dtype, in-place result contract, and direct script import path; Ruff and all 79 tests passed.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** No compiler or system package will be installed on the shared GPU machine.
  The pinned upstream source and checkpoint files remain unchanged and hash-verified.
- **Next step:** Copy the verifier and compatibility helper to the GPU project and rerun D3; the
  existing dependencies, source, and verified checkpoint download will be reused.

### 2026-08-18 — D3 MediaPipe dependency correction

- **Result:** Source, dependencies, CUDA preparation, Windows blend import, and checkpoints passed;
  model initialization exposed one missing dependency and a correction was prepared.
- **Goal/stage:** Goal 3 / Stage D3.
- **Work:** Added official MediaPipe 0.10.35 and its pinned Python dependencies while deliberately
  retaining the existing headless OpenCV package instead of installing a conflicting `cv2` wheel.
- **Files/evidence:** GPU report `D3-DITTO-INSTALL-0004`, requirements, Ditto configuration,
  guarded installer, package-report validation, and focused regression tests.
- **Verification:** The official wheel supports Windows x86-64 and contains the required MediaPipe
  Tasks `FaceLandmarker`/`BaseOptions` interfaces; Ruff and all 79 tests passed.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** MediaPipe is installed with `--no-deps`; its needed dependencies are
  explicitly pinned, and `opencv-python-headless` remains the sole provider of `cv2`.
- **Next step:** Copy the corrected requirements, config, installer, and verifier to the GPU project,
  then rerun D3 using the existing source and checkpoint files.

### 2026-08-18 — D3 PyTorch-path import audit

- **Result:** MediaPipe and the Windows compatibility path passed; model initialization exposed
  `einops` as the remaining omitted PyTorch-path dependency.
- **Goal/stage:** Goal 3 / Stage D3.
- **Work:** Audited all top-level imports in the pinned Ditto source and added pinned `einops` to
  installation preflight, dependency imports, configuration, and package-version reporting.
- **Files/evidence:** GPU report `D3-DITTO-INSTALL-0005`, requirements, Ditto configuration,
  installer, verifier, and dependency regression tests.
- **Verification:** Static audit confirms the other uninstalled imports belong only to the excluded
  TensorRT backend; `einops` 0.8.1 supports Python 3.10, NumPy, and PyTorch; Ruff and all 79 tests
  passed.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** No TensorRT, CUDA-Python, compiler, second OpenCV distribution, or system
  dependency will be installed. Existing verified checkpoints remain reusable.
- **Next step:** Copy the four corrected dependency-contract files and rerun D3 on GPU 0.

### 2026-08-18 — D3 Ditto installation and model-load validation

- **Result:** Complete.
- **Goal/stage:** Goal 3 / Stage D3.
- **Work:** Installed the pinned PyTorch Ditto dependency set in `avatar-ditto`, retained headless
  OpenCV, installed official MediaPipe without its conflicting OpenCV dependency, verified pinned
  source/checkpoints, and initialized `stream_pipeline_online.StreamSDK` once on GPU 0.
- **Files/evidence:** `results/environment/D3-DITTO-INSTALL-0001` through `0006`; reports `0001`–
  `0005` preserve diagnostic failures and `0006` is the passing checkpoint with `errors: []`.
- **Verification:** Ditto source `c3e47eee2e626500017a0556b470d6d4182f85e8`; checkpoint revision
  `e4a2f60328ee7c32af585ac4b3cce299e4c8e254`; all 12 files and 2314719638 bytes matched upstream
  hashes; CUDA Execution Provider was available; `StreamSDK` loaded in 5.404 seconds; Ruff and all
  79 local tests passed.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** D3 used the PyTorch backend and the recorded NumPy Windows blend fallback.
  No portrait/audio inference, video generation, TensorRT optimization, MaAI work, or process
  modification occurred.
- **Next step:** Plan D4 as one controlled inference with exactly one portrait and one short WAV;
  do not run it until its input choice, command, output report, and stop conditions are approved.

### 2026-08-18 — D4 single-inference tooling prepared

- **Result:** Complete locally; the one GPU inference is pending.
- **Goal/stage:** Goal 3 / Stage D4.
- **Work:** Added deterministic five-second P007 WAV extraction, the pinned offline PyTorch Ditto
  adapter, bundled-FFmpeg mux and media validation, continuous GPU sampling, a complete experiment
  record, visual-review template, and a one-run Windows GPU wrapper.
- **Files/evidence:** `config/ditto.yaml`, `scripts/test_ditto_file.py`,
  `scripts/gpu/run_ditto_d4.ps1`, the Ditto adapter, experiment utilities, tests, and GPU runbook.
- **Verification:** Ruff passed and all 91 local tests passed, including Python 3.10 compatibility,
  exact PCM extraction,
  single-run enforcement, GPU metric aggregation, output stream validation, and PowerShell parsing.
- **Commit:** Included in this milestone commit.
- **Decision/limitation:** D4 uses only P007 and a reproducible first-five-second excerpt on physical
  GPU 0. No local/GPU inference, second attempt, full audio, batch, TensorRT, or other-process
  modification occurred while preparing the tooling.
- **Next step:** Copy the committed files to the GPU project, run the D4 wrapper exactly once, copy
  `DITTO-EXP-0001` back, and complete its visual review before any D5 work.

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
