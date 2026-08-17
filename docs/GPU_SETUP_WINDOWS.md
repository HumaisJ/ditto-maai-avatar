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
