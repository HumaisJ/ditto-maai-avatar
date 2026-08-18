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

$environmentName = "avatar-ditto"
$runtimeRoot = Join-Path $ProjectRoot ".runtime\ditto"
$configFile = Join-Path $ProjectRoot "config\ditto.yaml"
$runner = Join-Path $ProjectRoot "scripts\test_ditto_file.py"
$resultsRoot = Join-Path $ProjectRoot "results\experiments"
$environmentReports = Join-Path $ProjectRoot "results\environment"
$d4Directory = Join-Path $resultsRoot "DITTO-EXP-0001"
$d5Directory = Join-Path $resultsRoot "DITTO-EXP-0002"

try {
    Write-Host "Checking D5 prerequisites without changing any environment or process..."
    if ($GpuIndex -ne 0) { throw "D5 is approved only for physical GPU 0." }
    foreach ($command in @("conda", "nvidia-smi")) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "Required command is unavailable: $command"
        }
    }
    foreach ($path in @(
        $configFile,
        $runner,
        (Join-Path $runtimeRoot "source-manifest.json"),
        (Join-Path $runtimeRoot "checkpoint-manifest.json"),
        (Join-Path $ProjectRoot "assets\portraits\queen_elizabeth.jpeg"),
        (Join-Path $ProjectRoot "assets\audio\zia_mohyeddin_hamlet_act_2_scene_2.wav")
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required D5 file is missing: $path"
        }
    }
    foreach ($path in @(
        (Join-Path $runtimeRoot "source"),
        (Join-Path $runtimeRoot "checkpoints")
    )) {
        if (-not (Test-Path -LiteralPath $path -PathType Container)) {
            throw "Required D5 directory is missing: $path"
        }
    }

    if (Test-Path -LiteralPath $d5Directory) {
        throw "DITTO-EXP-0002 already exists. Preserve and review it; do not run D5 again."
    }
    if (-not (Test-Path -LiteralPath $d4Directory -PathType Container)) {
        throw "D5 requires the retained DITTO-EXP-0001 result."
    }
    $d4ReportPath = Join-Path $d4Directory "experiment.json"
    if (-not (Test-Path -LiteralPath $d4ReportPath -PathType Leaf)) {
        throw "DITTO-EXP-0001 is missing experiment.json."
    }
    $d4Report = Get-Content -LiteralPath $d4ReportPath -Raw | ConvertFrom-Json
    if ($d4Report.experiment_id -ne "DITTO-EXP-0001" -or $d4Report.status -ne "succeeded") {
        throw "D5 requires a successful DITTO-EXP-0001 result."
    }
    $unexpectedRuns = @(Get-ChildItem -LiteralPath $resultsRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^DITTO-EXP-\d+$' -and $_.Name -ne "DITTO-EXP-0001" })
    if ($unexpectedRuns.Count -gt 0) {
        throw "An unexpected later DITTO experiment exists; stop and review it before D5."
    }

    $passingD3 = Get-ChildItem -LiteralPath $environmentReports -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^D3-DITTO-INSTALL-\d+$' } |
        Sort-Object Name -Descending |
        ForEach-Object {
            $reportPath = Join-Path $_.FullName "installation.json"
            if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
                $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
                if ($report.stage -eq "D3" -and $report.status -eq "passed") { $_ }
            }
        } | Select-Object -First 1
    if (-not $passingD3) { throw "No passing D3 installation report was found." }

    $environmentData = conda env list --json | ConvertFrom-Json
    if (-not ($environmentData.envs | Where-Object { (Split-Path -Leaf $_) -eq $environmentName })) {
        throw "The validated Conda environment '$environmentName' was not found."
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

    Write-Host "Starting the one approved D5 inference in '$environmentName' on GPU $GpuIndex..."
    $env:CUDA_VISIBLE_DEVICES = "$GpuIndex"
    conda run --no-capture-output -n $environmentName python $runner `
        --stage D5 --config $configFile --runtime-root $runtimeRoot `
        --results-root $resultsRoot --gpu-index $GpuIndex
    if ($LASTEXITCODE -ne 0) {
        throw "D5 failed. Preserve DITTO-EXP-0002 and do not rerun."
    }
    Write-Host "D5 finished. Do not run another inference before the result is copied and reviewed."
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
