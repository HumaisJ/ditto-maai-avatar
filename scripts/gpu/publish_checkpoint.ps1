param(
    [Parameter(Mandatory = $true)]
    [string]$CheckpointPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$allowedRoot = (Resolve-Path (Join-Path $projectRoot "results\environment")).Path
$checkpoint = (Resolve-Path -LiteralPath $CheckpointPath).Path

if (-not $checkpoint.StartsWith($allowedRoot + [System.IO.Path]::DirectorySeparatorChar)) {
    throw "CheckpointPath must be a child of results\environment."
}
if (-not (Test-Path -LiteralPath (Join-Path $checkpoint "environment.json") -PathType Leaf)) {
    throw "The checkpoint does not contain environment.json."
}

Push-Location $projectRoot
try {
    $branch = git branch --show-current
    if ($branch -ne "main") { throw "Checkpoint publishing is allowed only from main." }

    $relative = [System.IO.Path]::GetRelativePath($projectRoot, $checkpoint).Replace('\', '/')
    $statusLines = @(git status --porcelain --untracked-files=all)
    $unrelated = @($statusLines | Where-Object {
        $path = if ($_.Length -gt 3) { $_.Substring(3).Trim('"') } else { "" }
        -not ($path -eq $relative -or $path.StartsWith($relative + "/"))
    })
    if ($unrelated.Count -gt 0) {
        throw "Unrelated working-tree changes exist; publish stopped: $($unrelated -join '; ')"
    }

    git add -- $relative
    if ($LASTEXITCODE -ne 0) { throw "Unable to stage checkpoint report." }
    git commit -m "chore: record D2 GPU environment verification"
    if ($LASTEXITCODE -ne 0) { throw "Unable to commit checkpoint report." }
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw "Unable to push checkpoint report." }
    Write-Host "Published checkpoint: $relative"
} finally {
    Pop-Location
}
