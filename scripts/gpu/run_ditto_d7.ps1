param(
    [string]$ProjectRoot = "",
    [ValidateRange(0, 31)]
    [int]$GpuIndex = 0,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
}

$environmentName = "avatar-ditto"
$runtimeRoot = Join-Path $ProjectRoot ".runtime\ditto"
$configFile = Join-Path $ProjectRoot "config\ditto.yaml"
$manifestFile = Join-Path $ProjectRoot "assets\manifest.csv"
$batchRunner = Join-Path $ProjectRoot "scripts\run_ditto_batch.py"
$singleRunner = Join-Path $ProjectRoot "scripts\test_ditto_file.py"
$experimentsRoot = Join-Path $ProjectRoot "results\experiments"
$batchesRoot = Join-Path $ProjectRoot "results\batches"
$environmentReports = Join-Path $ProjectRoot "results\environment"
$batchDirectory = Join-Path $batchesRoot "DITTO-BATCH-0001"
$d4Directory = Join-Path $experimentsRoot "DITTO-EXP-0001"
$d5Directory = Join-Path $experimentsRoot "DITTO-EXP-0002"

try {
    Write-Host "Checking D7 prerequisites without changing any environment or process..."
    if ($GpuIndex -ne 0) { throw "D7 is approved only for physical GPU 0." }
    foreach ($command in @("conda", "nvidia-smi")) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "Required command is unavailable: $command"
        }
    }
    foreach ($path in @(
        $configFile,
        $manifestFile,
        $batchRunner,
        $singleRunner,
        (Join-Path $runtimeRoot "source-manifest.json"),
        (Join-Path $runtimeRoot "checkpoint-manifest.json"),
        (Join-Path $d4Directory "experiment.json"),
        (Join-Path $d5Directory "experiment.json"),
        (Join-Path $d5Directory "notes.md")
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required D7 file is missing: $path"
        }
    }
    foreach ($path in @(
        (Join-Path $runtimeRoot "source"),
        (Join-Path $runtimeRoot "checkpoints")
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Container)) {
            throw "Required D7 directory is missing: $path"
        }
    }

    foreach ($experimentDirectory in @($d4Directory, $d5Directory)) {
        $reportPath = Join-Path $experimentDirectory "experiment.json"
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        if ($report.experiment_id -ne (Split-Path -Leaf $experimentDirectory) -or
            $report.status -ne "succeeded") {
            throw "D7 requires successful retained D4 and D5 experiments."
        }
    }
    $d5Notes = Get-Content -LiteralPath (Join-Path $d5Directory "notes.md") -Raw
    if ($d5Notes -match '(?im)^-\s+[a-z_]+:\s*pending\s*$') {
        throw "D7 requires the completed D5 visual review."
    }
    if ($d5Notes -match '(?im)^-\s+[a-z_]+:\s*1(?:\.0+)?\s*$') {
        throw "A D5 visual score of 1 blocks D7."
    }

    if ($Resume) {
        if (-not (Test-Path -LiteralPath $batchDirectory -PathType Container)) {
            throw "Cannot resume because DITTO-BATCH-0001 does not exist."
        }
        $batchReportPath = Join-Path $batchDirectory "batch.json"
        if (-not (Test-Path -LiteralPath $batchReportPath -PathType Leaf)) {
            throw "Cannot resume because DITTO-BATCH-0001 is missing batch.json."
        }
        $batchReport = Get-Content -LiteralPath $batchReportPath -Raw | ConvertFrom-Json
        if ($batchReport.batch_id -ne "DITTO-BATCH-0001" -or
            $batchReport.status -ne "interrupted") {
            throw "Only a cleanly interrupted DITTO-BATCH-0001 may be resumed."
        }
    } else {
        if (Test-Path -LiteralPath $batchDirectory) {
            throw "DITTO-BATCH-0001 already exists. Preserve it; do not start a fresh D7 batch."
        }
        $unexpectedRuns = @(Get-ChildItem -LiteralPath $experimentsRoot -Directory `
            -ErrorAction SilentlyContinue | Where-Object {
                $_.Name -match '^DITTO-EXP-\d+$' -and
                $_.Name -notin @("DITTO-EXP-0001", "DITTO-EXP-0002")
            })
        if ($unexpectedRuns.Count -gt 0) {
            throw "Later Ditto experiment evidence already exists; stop and review it before D7."
        }
    }

    $passingD3 = Get-ChildItem -LiteralPath $environmentReports -Directory `
        -ErrorAction SilentlyContinue | Where-Object {
            $_.Name -match '^D3-DITTO-INSTALL-\d+$'
        } | Sort-Object Name -Descending | ForEach-Object {
            $reportPath = Join-Path $_.FullName "installation.json"
            if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
                $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
                if ($report.stage -eq "D3" -and $report.status -eq "passed") { $_ }
            }
        } | Select-Object -First 1
    if (-not $passingD3) { throw "No passing D3 installation report was found." }

    $environmentData = conda env list --json | ConvertFrom-Json
    if (-not ($environmentData.envs | Where-Object {
        (Split-Path -Leaf $_) -eq $environmentName
    })) {
        throw "The validated Conda environment '$environmentName' was not found."
    }

    $driveRoot = [System.IO.Path]::GetPathRoot($ProjectRoot)
    $freeDiskMb = ([System.IO.DriveInfo]::new($driveRoot)).AvailableFreeSpace / 1MB
    if ($freeDiskMb -lt 5120) {
        throw "D7 requires at least 5120 MiB free on the project drive."
    }

    $gpuLine = nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu `
        --format=csv,noheader,nounits --id=$GpuIndex
    if ($LASTEXITCODE -ne 0 -or -not $gpuLine) { throw "Unable to query GPU $GpuIndex." }
    $gpuValues = $gpuLine.Split(',') | ForEach-Object { $_.Trim() }
    if ($gpuValues.Count -ne 4) { throw "Unexpected nvidia-smi GPU output: $gpuLine" }
    if ($gpuValues[0] -notlike "*RTX 5060 Ti*") { throw "Unexpected GPU: $($gpuValues[0])" }
    if ([double]$gpuValues[1] -lt 15000) { throw "GPU VRAM is below 15000 MiB." }
    if ([double]$gpuValues[2] -lt 12000) { throw "GPU free VRAM is below 12000 MiB." }
    if ([double]$gpuValues[3] -gt 20) {
        throw "GPU utilization exceeds the 20 percent safety threshold."
    }
    $processTable = nvidia-smi --id=$GpuIndex
    if ($LASTEXITCODE -ne 0) { throw "Unable to inspect processes on GPU $GpuIndex." }
    $blockingProcessPattern = "^\|\s*$GpuIndex\s+(?:N/A|\d+)\s+(?:N/A|\d+)\s+\d+\s+(?:C|M|M\+C)\s+"
    $computeProcesses = @($processTable | Where-Object { $_ -match $blockingProcessPattern })
    if ($computeProcesses.Count -gt 0) {
        throw "Another compute-only process is using GPU $GpuIndex`: $($computeProcesses -join '; ')"
    }

    $resumeArgument = @()
    if ($Resume) { $resumeArgument = @("--resume") }
    Write-Host "Starting the approved 17-pair sequential D7 batch on GPU $GpuIndex..."
    $env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
    conda run --no-capture-output -n $environmentName python $batchRunner `
        --config $configFile --manifest $manifestFile --runtime-root $runtimeRoot `
        --experiments-root $experimentsRoot --batches-root $batchesRoot `
        --gpu-index $GpuIndex @resumeArgument
    if ($LASTEXITCODE -ne 0) {
        throw "D7 stopped or failed. Preserve all batch and experiment evidence; do not rerun blindly."
    }
    Write-Host "D7 technical execution finished. Copy the batch and 17 experiment directories to the PC."
    Write-Host "Do not start another model stage before reviewing all 17 generated videos."
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
