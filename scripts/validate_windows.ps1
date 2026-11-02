# Validate the YOLO26 ROCm Day 1 labs on Windows native (AMD gfx1100).
# Entrypoint: powershell -ExecutionPolicy Bypass -File scripts\validate_windows.ps1
#             [-CourseHome <dir>] [-SkipNotebooks] [-KeepGoing]
param(
    [string]$CourseHome = "",
    [switch]$SkipNotebooks,
    [switch]$KeepGoing
)

$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
Set-Location $Project
$VenvPython = Join-Path $Project '.venv\Scripts\python.exe'

if (-not (Test-Path $VenvPython)) {
    Write-Host "FATAL: $VenvPython not found. Create the uv venv first (docs/ROCM_WINDOWS_SETUP.md)." -ForegroundColor Red
    exit 2
}

Write-Host "== YOLO26 ROCm Day1 Windows validation ==" -ForegroundColor Cyan
Write-Host "project : $Project"
Write-Host "venv    : $VenvPython"

# 1. Dependency sanity: exact ROCm trio, no broken requirements.
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) { Write-Host 'pip check FAILED' -ForegroundColor Red; exit 1 }
$installed = (& $VenvPython -m pip show torch torchvision torchaudio | Select-String '^Version') -join ' '
Write-Host "torch trio: $installed"
if ($installed -notmatch '2\.12\.0\+rocm7\.14\.1' -or
    $installed -notmatch '0\.27\.0\+rocm7\.14\.1' -or
    $installed -notmatch '2\.11\.0\+rocm7\.14\.1') {
    Write-Host 'FATAL: torch trio drifted off AMD ROCm 7.14.1 wheels' -ForegroundColor Red
    exit 1
}

# 2. Physical GPU admission gate.
& $VenvPython 'scripts\gpu_smoke.py'
if ($LASTEXITCODE -ne 0) { Write-Host 'GPU smoke FAILED - GPU training validation must not proceed (no CPU fallback).' -ForegroundColor Red; exit 1 }

# 3. Offline regression tests.
& $VenvPython -m pytest -q
if ($LASTEXITCODE -ne 0) { Write-Host 'pytest FAILED' -ForegroundColor Red; exit 1 }

# 4. Full notebook end-to-end run (order enforced by the runner).
if (-not $SkipNotebooks) {
    $runnerArgs = @('scripts\run_all_labs.py', '--timeout', '7200')
    if ($CourseHome -ne '') { $runnerArgs += @('--course-home', $CourseHome) }
    if ($KeepGoing) { $runnerArgs += '--keep-going' }
    & $VenvPython @runnerArgs
    if ($LASTEXITCODE -ne 0) { Write-Host 'Notebook run FAILED' -ForegroundColor Red; exit 1 }
}

Write-Host 'ALL VALIDATION STAGES PASSED' -ForegroundColor Green
exit 0
