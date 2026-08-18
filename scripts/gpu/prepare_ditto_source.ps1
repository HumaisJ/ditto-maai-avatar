param(
    [string]$ProjectRoot = "",
    [string]$OutputDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $ProjectRoot ".transfer"
}

$repository = "https://github.com/antgroup/ditto-talkinghead.git"
$revision = "c3e47eee2e626500017a0556b470d6d4182f85e8"
$shortRevision = $revision.Substring(0, 8)
$outputDirectoryAbsolute = [System.IO.Path]::GetFullPath($OutputDirectory)
$archivePath = Join-Path $outputDirectoryAbsolute "ditto-source-$shortRevision.zip"
$manifestPath = Join-Path $outputDirectoryAbsolute "ditto-source-$shortRevision.manifest.json"
$workRoot = Join-Path $outputDirectoryAbsolute ("source-work-" + [guid]::NewGuid().ToString("N"))
$cloneRoot = Join-Path $workRoot "repository"
$extractRoot = Join-Path $workRoot "extracted"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required on the development PC for source preparation."
}
New-Item -ItemType Directory -Force -Path $outputDirectoryAbsolute | Out-Null
$archiveExists = Test-Path -LiteralPath $archivePath
$manifestExists = Test-Path -LiteralPath $manifestPath
if ($archiveExists -and $manifestExists) {
    throw "The pinned transfer bundle already exists; it will not be overwritten: $archivePath"
}
if ($archiveExists -xor $manifestExists) {
    Write-Host "Removing an incomplete transfer bundle from a previous failed attempt..."
    foreach ($partial in @($archivePath, $manifestPath)) {
        if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
    }
}

try {
    New-Item -ItemType Directory -Path $workRoot | Out-Null
    git clone --filter=blob:none --no-checkout $repository $cloneRoot
    if ($LASTEXITCODE -ne 0) { throw "Ditto source clone failed." }
    git -C $cloneRoot fetch --depth 1 origin $revision
    if ($LASTEXITCODE -ne 0) { throw "Unable to fetch the pinned Ditto revision." }
    $resolvedRevision = (git -C $cloneRoot rev-parse FETCH_HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $resolvedRevision -ne $revision) {
        throw "Fetched Ditto revision does not match the configured revision."
    }
    git -C $cloneRoot archive --format=zip --output=$archivePath $revision
    if ($LASTEXITCODE -ne 0) { throw "Unable to create the clean Ditto source archive." }

    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractRoot
    $extractPrefix = $extractRoot.TrimEnd('\') + '\'
    $files = @(Get-ChildItem -LiteralPath $extractRoot -Recurse -File | Sort-Object FullName | ForEach-Object {
        [ordered]@{
            path = $_.FullName.Substring($extractPrefix.Length).Replace('\', '/')
            size = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
    $manifest = [ordered]@{
        schema_version = 1
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        repository = $repository
        revision = $revision
        archive_sha256 = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        files = $files
    }
    $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
    Write-Host "Prepared pinned Ditto source archive: $archivePath"
    Write-Host "Prepared source manifest: $manifestPath"
} finally {
    $resolvedWorkRoot = [System.IO.Path]::GetFullPath($workRoot)
    if ($resolvedWorkRoot.StartsWith($outputDirectoryAbsolute + [System.IO.Path]::DirectorySeparatorChar) -and
        (Test-Path -LiteralPath $resolvedWorkRoot)) {
        Remove-Item -LiteralPath $resolvedWorkRoot -Recurse -Force
    }
}
