param(
    [Parameter(Mandatory = $true)]
    [string]$CorpusCache,
    [Parameter(Mandatory = $true)]
    [string]$EvidenceRegistry,
    [Parameter(Mandatory = $true)]
    [string]$Tariffs,
    [Parameter(Mandatory = $true)]
    [string]$BenchmarkSourceRun,
    [Parameter(Mandatory = $true)]
    [string]$ExpectedCorpusFingerprint,
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$BenchmarkRoot = "artifacts\resource_benchmark\fixed_configuration_split7",
    [int]$BenchmarkRepetitions = 3
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
foreach ($required in @($CorpusCache, $EvidenceRegistry, $Tariffs, $BenchmarkSourceRun)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required input not found: $required"
    }
}

$validationRoot = Join-Path "artifacts" "validation"
$releaseRoot = Join-Path "artifacts" "releases\thesis-evidence"
$testReport = Join-Path $validationRoot "test_report.json"
$engineeringCosts = Join-Path $BenchmarkRoot "resource_benchmark_raw.csv"
$benchmarkAudit = Join-Path $BenchmarkRoot "resource_benchmark_audit.json"
$costReport = Join-Path $BenchmarkRoot "cost_scenarios.json"
$contextPath = Join-Path $validationRoot "validation-context.json"
$derivedRegistry = Join-Path $validationRoot "evidence-runs-with-benchmark.json"
$matrixRoot = Join-Path $validationRoot "requirements"

if (Test-Path -LiteralPath $BenchmarkRoot) {
    throw "Benchmark output already exists; preserve it and choose a fresh BenchmarkRoot: $BenchmarkRoot"
}
if (Test-Path -LiteralPath $releaseRoot) {
    throw "Evidence output already exists; preserve it or choose a fresh workspace: $releaseRoot"
}
if ($BenchmarkRepetitions -lt 3) {
    throw "BenchmarkRepetitions must be at least 3 for requirement R5."
}
if ($ExpectedCorpusFingerprint -notmatch "^[A-Fa-f0-9]{64}$") {
    throw "ExpectedCorpusFingerprint must be a 64-character SHA-256 value."
}

Invoke-CheckedPython -Arguments @(
    "scripts\run_validation_tests.py",
    "--output", $testReport
)

Invoke-CheckedPython -Arguments @(
    "scripts\run_resource_benchmark.py",
    "--source-run", $BenchmarkSourceRun,
    "--fraction", "1.0",
    "--split-seed", "7",
    "--repetitions", "$BenchmarkRepetitions",
    "--output-dir", $BenchmarkRoot,
    "--corpus-cache", $CorpusCache,
    "--expected-corpus-fingerprint", $ExpectedCorpusFingerprint,
    "--measurement-condition", "isolated"
)

Invoke-CheckedPython -Arguments @(
    "-m", "Comparacion.cli",
    "--cost-input", $engineeringCosts,
    "--tariffs", $Tariffs,
    "--cost-output", $costReport
)

$registry = Get-Content -LiteralPath $EvidenceRegistry -Raw | ConvertFrom-Json
$registryBase = Split-Path -Parent ([System.IO.Path]::GetFullPath($EvidenceRegistry))
foreach ($run in @($registry.runs)) {
    if (-not [System.IO.Path]::IsPathRooted([string]$run.path)) {
        $run.path = [System.IO.Path]::GetFullPath((Join-Path $registryBase ([string]$run.path)))
    }
}
$supplemental = @(
    [ordered]@{
        name = "resource-benchmark"
        path = [System.IO.Path]::GetFullPath($BenchmarkRoot)
        audit = "resource_benchmark_audit.json"
    }
)
if ($registry.PSObject.Properties.Name -contains "supplemental_artifacts") {
    $existingSupplemental = @(
        $registry.supplemental_artifacts | Where-Object { $_.name -ne "resource-benchmark" }
    )
    foreach ($item in $existingSupplemental) {
        if (-not [System.IO.Path]::IsPathRooted([string]$item.path)) {
            $item.path = [System.IO.Path]::GetFullPath((Join-Path $registryBase ([string]$item.path)))
        }
    }
    $supplemental = $existingSupplemental + $supplemental
}
$registry | Add-Member -NotePropertyName supplemental_artifacts -NotePropertyValue $supplemental -Force
$registryDirectory = Split-Path -Parent $derivedRegistry
New-Item -ItemType Directory -Path $registryDirectory -Force | Out-Null
$registry | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $derivedRegistry -Encoding utf8
Invoke-CheckedPython -Arguments @(
    "-m", "Comparacion.cli",
    "--export-evidence", $derivedRegistry,
    "--evidence-output", $releaseRoot
)

$context = [ordered]@{
    evidence_package = [System.IO.Path]::GetFullPath($releaseRoot)
    resource_costs = [System.IO.Path]::GetFullPath($engineeringCosts)
    resource_benchmark_audit = [System.IO.Path]::GetFullPath($benchmarkAudit)
    cost_scenarios = [System.IO.Path]::GetFullPath($costReport)
    test_report = [System.IO.Path]::GetFullPath($testReport)
    design_spec = [System.IO.Path]::GetFullPath("docs\superpowers\specs\2026-09-01-cierre-criterios-tesis-design.md")
    engineering_practices = [System.IO.Path]::GetFullPath("docs\engineering-practices.md")
    social_protocol = [System.IO.Path]::GetFullPath("docs\social-viability-protocol.md")
}
$contextDirectory = Split-Path -Parent $contextPath
New-Item -ItemType Directory -Path $contextDirectory -Force | Out-Null
$context | ConvertTo-Json | Set-Content -LiteralPath $contextPath -Encoding utf8

Invoke-CheckedPython -Arguments @(
    "-m", "Comparacion.cli",
    "--validate-requirements", "docs\engineering-requirements.json",
    "--validation-context", $contextPath,
    "--validation-output", $matrixRoot
)

Write-Host "Closure artifacts ready under artifacts/validation and artifacts/releases."
