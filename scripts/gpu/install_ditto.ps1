param(
    [Parameter(Mandatory = $true)]
    [string]$SourceArchive,
    [Parameter(Mandatory = $true)]
    [string]$SourceManifest,
    [string]$ProjectRoot = "",
    [ValidateRange(0, 31)]
    [int]$GpuIndex = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
}
$SourceArchive = (Resolve-Path -LiteralPath $SourceArchive).Path
$SourceManifest = (Resolve-Path -LiteralPath $SourceManifest).Path

$environmentName = "avatar-ditto"
$expectedRevision = "c3e47eee2e626500017a0556b470d6d4182f85e8"
$runtimeRoot = Join-Path $ProjectRoot ".runtime\ditto"
$sourceRoot = Join-Path $runtimeRoot "source"
$checkpointRoot = Join-Path $runtimeRoot "checkpoints"
$cacheRoot = Join-Path $runtimeRoot "cache"
$runtimeSourceManifest = Join-Path $runtimeRoot "source-manifest.json"
$checkpointManifest = Join-Path $runtimeRoot "checkpoint-manifest.json"
$requirementsFile = Join-Path $ProjectRoot "requirements\ditto.txt"
$configFile = Join-Path $ProjectRoot "config\ditto.yaml"
$outputRoot = Join-Path $ProjectRoot "results\environment"
$validationReportCreated = $false

function New-FailureReport {
    param([string]$Message)

    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
    $numbers = Get-ChildItem -LiteralPath $outputRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^D3-DITTO-INSTALL-(\d+)$' } |
        ForEach-Object { [int]$Matches[1] }
    $next = if ($numbers) { ($numbers | Measure-Object -Maximum).Maximum + 1 } else { 1 }
    $report = Join-Path $outputRoot ("D3-DITTO-INSTALL-{0:D4}" -f ([int]$next))
    New-Item -ItemType Directory -Path $report | Out-Null
    $payload = [ordered]@{
        checkpoint_id = Split-Path -Leaf $report
        stage = "D3"
        status = "failed"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        backend = "pytorch"
        errors = @($Message)
        model_load = [ordered]@{ status = "not_attempted"; inference_executed = $false }
    }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $report "installation.json") -Encoding UTF8
    "D3 Ditto installation verification: FAILED`nERROR: $Message" |
        Set-Content -LiteralPath (Join-Path $report "verification.log") -Encoding UTF8
    Write-Host "D3 failure report: $report"
    return $report
}

function Get-CondaFingerprint {
    param([string]$Name)
    $raw = (conda list -n $Name --json) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Unable to fingerprint Conda environment '$Name'." }
    return (($raw | ConvertFrom-Json | Sort-Object name | ConvertTo-Json -Depth 5) -join "`n")
}

function Get-TorchFingerprint {
    $records = (conda list -n $environmentName --json) -join "`n" | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "Unable to inspect PyTorch packages." }
    return (($records | Where-Object { $_.name -in @("torch", "torchvision", "torchaudio") } |
        Sort-Object name | Select-Object name, version, channel | ConvertTo-Json -Depth 3) -join "`n")
}

function Test-SourceFiles {
    param([string]$Root, [object]$Manifest)
    $absoluteRoot = [System.IO.Path]::GetFullPath($Root)
    foreach ($entry in $Manifest.files) {
        $candidate = [System.IO.Path]::GetFullPath((Join-Path $absoluteRoot $entry.path))
        if (-not $candidate.StartsWith($absoluteRoot + [System.IO.Path]::DirectorySeparatorChar)) {
            throw "Source manifest contains an unsafe path: $($entry.path)"
        }
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Pinned Ditto source file is missing: $($entry.path)"
        }
        if ((Get-Item -LiteralPath $candidate).Length -ne [long]$entry.size) {
            throw "Pinned Ditto source file size differs: $($entry.path)"
        }
        $actualHash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $entry.sha256) {
            throw "Pinned Ditto source file hash differs: $($entry.path)"
        }
    }
}

try {
    Write-Host "Checking D3 prerequisites without changing other environments or processes..."
    foreach ($command in @("conda", "nvidia-smi")) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "Required command is unavailable: $command"
        }
    }
    foreach ($file in @($requirementsFile, $configFile, $SourceArchive, $SourceManifest)) {
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required file is missing: $file" }
    }
    $requirementsText = (Get-Content -LiteralPath $requirementsFile -Raw).ToLowerInvariant()
    foreach ($requiredPin in @(
        "pyyaml==6.0.2",
        "huggingface-hub==0.36.0",
        "onnxruntime-gpu==1.23.2",
        "einops==0.8.1",
        "matplotlib==3.10.8",
        "sounddevice==0.5.5"
    )) {
        if ($requirementsText -notmatch "(?m)^$([regex]::Escape($requiredPin))\r?$" ) {
            throw "requirements/ditto.txt is empty, outdated, or missing the required pin: $requiredPin"
        }
    }
    $environmentData = conda env list --json | ConvertFrom-Json
    if (-not ($environmentData.envs | Where-Object { (Split-Path -Leaf $_) -eq $environmentName })) {
        throw "The validated Conda environment '$environmentName' was not found."
    }
    $driveName = [System.IO.Path]::GetPathRoot($ProjectRoot).TrimEnd('\').TrimEnd(':')
    if ((Get-PSDrive -Name $driveName).Free -lt 8GB) {
        throw "At least 8 GiB of free project-drive space is required for D3."
    }

    $sourceManifestData = Get-Content -LiteralPath $SourceManifest -Raw | ConvertFrom-Json
    if ($sourceManifestData.revision -ne $expectedRevision) {
        throw "Source manifest revision does not match the pinned Ditto revision."
    }
    $archiveHash = (Get-FileHash -LiteralPath $SourceArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($archiveHash -ne $sourceManifestData.archive_sha256) {
        throw "The transferred Ditto source archive failed SHA-256 verification."
    }

    $baseBefore = Get-CondaFingerprint -Name "base"
    $torchBefore = Get-TorchFingerprint

    Write-Host "Installing only pinned PyTorch-path Ditto dependencies into '$environmentName'..."
    conda run --no-capture-output -n $environmentName python -m pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) { throw "Ditto dependency installation failed." }
    Write-Host "Installing the official MediaPipe wheel without its conflicting OpenCV dependency..."
    conda run --no-capture-output -n $environmentName python -m pip install `
        --no-deps mediapipe==0.10.35
    if ($LASTEXITCODE -ne 0) { throw "MediaPipe installation failed." }
    conda run --no-capture-output -n $environmentName python -c `
        "import torch, yaml, huggingface_hub, onnxruntime, mediapipe, einops; from mediapipe.tasks.python import vision, BaseOptions; assert vision.FaceLandmarker; print('D3 dependency imports passed')"
    if ($LASTEXITCODE -ne 0) { throw "One or more installed Ditto dependencies cannot be imported." }
    $torchAfter = Get-TorchFingerprint
    if ($torchBefore -ne $torchAfter) {
        throw "PyTorch, torchvision, or torchaudio changed during D3; stop and review."
    }
    $baseAfter = Get-CondaFingerprint -Name "base"
    if ($baseBefore -ne $baseAfter) { throw "Conda base changed during D3; stop and review." }

    New-Item -ItemType Directory -Force -Path $runtimeRoot, $cacheRoot | Out-Null
    if (Test-Path -LiteralPath $sourceRoot) {
        Write-Host "Existing Ditto source found; verifying it instead of overwriting it..."
        Test-SourceFiles -Root $sourceRoot -Manifest $sourceManifestData
    } else {
        $stagingRoot = Join-Path $runtimeRoot ("source-staging-" + [guid]::NewGuid().ToString("N"))
        try {
            Expand-Archive -LiteralPath $SourceArchive -DestinationPath $stagingRoot
            Test-SourceFiles -Root $stagingRoot -Manifest $sourceManifestData
            Move-Item -LiteralPath $stagingRoot -Destination $sourceRoot
        } finally {
            $absoluteRuntime = [System.IO.Path]::GetFullPath($runtimeRoot)
            $absoluteStaging = [System.IO.Path]::GetFullPath($stagingRoot)
            if ($absoluteStaging.StartsWith($absoluteRuntime + [System.IO.Path]::DirectorySeparatorChar) -and
                (Test-Path -LiteralPath $absoluteStaging)) {
                Remove-Item -LiteralPath $absoluteStaging -Recurse -Force
            }
        }
    }
    Copy-Item -LiteralPath $SourceManifest -Destination $runtimeSourceManifest -Force

    Write-Host "Downloading/verifying the pinned PyTorch checkpoint subset..."
    conda run --no-capture-output -n $environmentName python `
        (Join-Path $ProjectRoot "scripts\download_ditto_checkpoints.py") `
        --config $configFile --destination $checkpointRoot --cache-dir $cacheRoot
    if ($LASTEXITCODE -ne 0) { throw "Ditto checkpoint download or verification failed." }

    Write-Host "Running the guarded D3 import and model-load validation on GPU $GpuIndex..."
    $reportCountBeforeValidation = @(Get-ChildItem -LiteralPath $outputRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^D3-DITTO-INSTALL-\d+$' }).Count
    $env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
    conda run --no-capture-output -n $environmentName python `
        (Join-Path $ProjectRoot "scripts\check_ditto_install.py") `
        --config $configFile --source-dir $sourceRoot --checkpoint-dir $checkpointRoot `
        --source-manifest $runtimeSourceManifest --checkpoint-manifest $checkpointManifest `
        --gpu-index $GpuIndex
    if ($LASTEXITCODE -ne 0) {
        $reportCountAfterValidation = @(Get-ChildItem -LiteralPath $outputRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^D3-DITTO-INSTALL-\d+$' }).Count
        $validationReportCreated = $reportCountAfterValidation -gt $reportCountBeforeValidation
        throw "D3 model-load validation failed; preserve its report."
    }
    Write-Host "D3 installation and no-inference model-load validation completed successfully."
} catch {
    if (-not $validationReportCreated) {
        New-FailureReport -Message $_.Exception.Message | Out-Null
    }
    Write-Error $_.Exception.Message
    exit 1
}
