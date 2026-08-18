# Windows GPU Environment Checkpoint

This procedure creates and verifies only the isolated `avatar-ditto` environment. It does not
install Ditto, download checkpoints, run inference, or modify Conda `base`.

## Copy the project to the GPU machine

Git is not required on the shared GPU machine. On the development PC, first make sure Git LFS
assets are materialized:

```powershell
git pull --ff-only
git lfs pull
```

Copy the resulting `dialogue-avatar` folder to the user's permitted GPU-machine directory. The
hidden `.git` directory is not required there. Open the copied folder in VS Code and use its
PowerShell terminal for the remaining commands.

## Create and verify the environment

Run one setup command from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu/setup_ditto_env.ps1 -GpuIndex 0
```

The script stops before creating the environment if prerequisites fail, another compute process is
using GPU 0, or `avatar-ditto` already exists. Activity on other GPUs is left untouched. It records
either a passing or failed D2 report under `results/environment/`.

On a Windows WDDM display GPU, ordinary `G` and `C+G` desktop processes are recorded but do not
block D2. Compute-only `C`, `M`, and `M+C` processes do block it. The selected GPU must also remain
at or below 20% utilization with at least 12000 MiB free VRAM. The script never stops processes.

Do not delete or recreate an existing environment automatically. Return its report for review.

## Return and publish the checkpoint report

Copy the report directory printed by the setup script back into `results/environment/` in the Git
checkout on the development PC. From that PC, pass the copied directory to:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu/publish_checkpoint.ps1 `
    -CheckpointPath results/environment/D2-GPU-ENV-0001
```

The publisher refuses unrelated working-tree changes, commits only the named report, and pushes
`main`. A failed report should also be published because it is useful evidence.

Stop after publishing the report. Ditto installation and checkpoints belong to Stage D3 and require
a separate plan and approval.

## D3: install and verify Ditto without inference

D3 reuses the validated `avatar-ditto` environment. It does not recreate the environment, modify
Conda `base`, install TensorRT, process a portrait or WAV file, or touch another user's process.

### 1. Prepare the pinned source on the development PC

From the Git checkout on the PC, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu/prepare_ditto_source.ps1
```

This produces one ZIP and one JSON manifest under `.transfer/`. Copy both files into the GPU copy
of the project's `.transfer/` directory. Neither directory is tracked by Git.

### 2. Restore the isolated Conda paths in GPU VS Code PowerShell

From the GPU copy of the repository, run these session-scoped commands:

```powershell
$condaRoot = "C:\ProgramData\miniconda3"
$studentRoot = "D:\Data of all Students\Humaisa"
$env:Path = "$condaRoot;$condaRoot\Scripts;$condaRoot\Library\bin;$env:Path"
$env:CONDA_ENVS_PATH = "$studentRoot\conda-envs"
$env:CONDA_PKGS_DIRS = "$studentRoot\conda-packages"
$env:PIP_CACHE_DIR = "$studentRoot\pip-cache"
conda env list
```

Confirm that `avatar-ditto` resolves to the student's custom environment directory. No activation
or separate Anaconda Prompt is required.

### 3. Run the resumable D3 installer on GPU 0

```powershell
& .\scripts\gpu\install_ditto.ps1 `
    -SourceArchive .\.transfer\ditto-source-c3e47eee.zip `
    -SourceManifest .\.transfer\ditto-source-c3e47eee.manifest.json `
    -GpuIndex 0
```

The checkpoint download is approximately 2.31 GB and may take time. It is resumable. The final
step initializes the PyTorch and ONNX models once on GPU 0 and then exits; it does not run Ditto
inference. Existing matching runtime files are reused, while conflicting files stop the script.

Every attempt creates or preserves `results/environment/D3-DITTO-INSTALL-NNNN/`. Copy the printed
report back to the development-PC checkout for review before D3 is marked complete. Do not proceed
to D4 until the report says `status: passed`.
