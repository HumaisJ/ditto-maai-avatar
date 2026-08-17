# Windows GPU Environment Checkpoint

This procedure creates and verifies only the isolated `avatar-ditto` environment. It does not
install Ditto, download checkpoints, run inference, or modify Conda `base`.

## First checkout on the GPU machine

Run these commands from PowerShell in the directory where the project should live:

```powershell
git clone https://github.com/HumaisJ/ditto-maai-avatar.git dialogue-avatar
Set-Location dialogue-avatar
git lfs pull
```

If the repository already exists, use `git pull --ff-only` followed by `git lfs pull` instead.

## Create and verify the environment

Run one setup command from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu/setup_ditto_env.ps1
```

The script stops before creating the environment if prerequisites fail, another compute process is
using the GPU, or `avatar-ditto` already exists. It records either a passing or failed D2 report
under `results/environment/`.

Do not delete or recreate an existing environment automatically. Return its report for review.

## Publish the checkpoint report

Pass the report directory printed by the setup script to:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/gpu/publish_checkpoint.ps1 `
    -CheckpointPath results/environment/D2-GPU-ENV-0001
```

The publisher refuses unrelated working-tree changes, commits only the named report, and pushes
`main`. A failed report should also be published because it is useful evidence.

Stop after publishing the report. Ditto installation and checkpoints belong to Stage D3 and require
a separate plan and approval.
