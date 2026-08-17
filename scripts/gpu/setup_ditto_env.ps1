param(
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

$outputRoot = Join-Path $ProjectRoot "results\environment"
$environmentFile = Join-Path $ProjectRoot "environment.ditto.windows.yml"
$environmentName = "avatar-ditto"
$minimumDriver = [version]"570.65"
$minimumFreeBytes = 20GB
$minimumVramMb = 15000
$maximumUtilization = 20
$validationReportCreated = $false

function New-FailureReport {
    param([string]$Message)

    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
    $numbers = Get-ChildItem -LiteralPath $outputRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^D2-GPU-ENV-(\d+)$' } |
        ForEach-Object { [int]$Matches[1] }
    $next = if ($numbers) { ($numbers | Measure-Object -Maximum).Maximum + 1 } else { 1 }
    $report = Join-Path $outputRoot ("D2-GPU-ENV-{0:D4}" -f $next)
    New-Item -ItemType Directory -Path $report | Out-Null
    $payload = [ordered]@{
        checkpoint_id = Split-Path -Leaf $report
        stage = "D2"
        status = "failed"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        errors = @($Message)
        environment = [ordered]@{ setup_phase = "powershell_preflight_or_install" }
    }
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $report "environment.json") -Encoding UTF8
    "D2 environment verification: FAILED`nERROR: $Message" |
        Set-Content -LiteralPath (Join-Path $report "verification.log") -Encoding UTF8
    Write-Host "D2 failure report: $report"
    return $report
}

try {
    Write-Host "Checking Windows GPU prerequisites..."
    if ([System.Environment]::OSVersion.Platform -ne [System.PlatformID]::Win32NT) {
        throw "This setup script supports Windows only."
    }
    foreach ($command in @("conda", "nvidia-smi")) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "Required command is unavailable: $command"
        }
    }
    if (-not (Test-Path -LiteralPath $environmentFile -PathType Leaf)) {
        throw "Environment definition is missing: $environmentFile"
    }

    $largestAudio = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "assets\audio") -Filter "*.wav" -File |
        Sort-Object Length -Descending | Select-Object -First 1
    if ($null -eq $largestAudio -or $largestAudio.Length -lt 1MB) {
        throw "Audio assets are missing or were copied as placeholders. Copy the materialized project assets from the development PC."
    }

    $drive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($ProjectRoot).TrimEnd('\').TrimEnd(':'))
    if ($drive.Free -lt $minimumFreeBytes) {
        throw "At least 20 GiB of free disk space is required."
    }

    $gpuLine = nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu --format=csv,noheader,nounits --id=$GpuIndex
    if ($LASTEXITCODE -ne 0 -or -not $gpuLine) { throw "Unable to query the first NVIDIA GPU." }
    $gpuValues = $gpuLine.Split(',') | ForEach-Object { $_.Trim() }
    if ($gpuValues.Count -ne 4) { throw "Unexpected nvidia-smi GPU output: $gpuLine" }
    if ($gpuValues[0] -notlike "*RTX 5060 Ti*") { throw "Unexpected GPU: $($gpuValues[0])" }
    if ([double]$gpuValues[1] -lt $minimumVramMb) { throw "GPU VRAM is below 15000 MiB." }
    if ([version]$gpuValues[2] -lt $minimumDriver) { throw "NVIDIA driver must be at least 570.65." }
    if ([double]$gpuValues[3] -gt $maximumUtilization) {
        throw "GPU utilization exceeds the 20 percent safety threshold."
    }
    $computeProcesses = nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits --id=$GpuIndex
    if ($computeProcesses) { throw "Another compute process is using GPU $GpuIndex`: $computeProcesses" }

    $environmentData = conda env list --json | ConvertFrom-Json
    $existingEnvironment = $environmentData.envs | Where-Object {
        (Split-Path -Leaf $_) -eq $environmentName
    }
    if ($existingEnvironment) {
        throw "The environment '$environmentName' already exists; it will not be overwritten."
    }

    $baseBefore = (conda list -n base --json) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Unable to fingerprint Conda base before setup." }

    Write-Host "Creating isolated Conda environment '$environmentName'..."
    conda env create -f $environmentFile
    if ($LASTEXITCODE -ne 0) { throw "Conda environment creation failed." }

    Write-Host "Installing the CUDA 12.8 PyTorch build into '$environmentName'..."
    conda run -n $environmentName python -m pip install `
        torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 `
        --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "PyTorch installation failed." }

    $baseAfter = (conda list -n base --json) -join "`n"
    if ($LASTEXITCODE -ne 0) { throw "Unable to fingerprint Conda base after setup." }
    if ($baseBefore -ne $baseAfter) { throw "Conda base changed during setup; stop and review." }

    Write-Host "Running D2 CUDA validation..."
    $env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
    conda run -n $environmentName python (Join-Path $ProjectRoot "scripts\check_environment.py") --gpu-index $GpuIndex
    if ($LASTEXITCODE -ne 0) {
        $validationReportCreated = $true
        throw "D2 environment validation failed; preserve its report."
    }

    Write-Host "D2 setup and validation completed successfully."
} catch {
    if (-not $validationReportCreated) {
        New-FailureReport -Message $_.Exception.Message | Out-Null
    }
    Write-Error $_.Exception.Message
    exit 1
}
