param(
    [Parameter(Mandatory = $true)]
    [string]$CorpusRoot,
    [Parameter(Mandatory = $true)]
    [string]$CorpusCache,
    [Parameter(Mandatory = $true)]
    [string]$EvidenceRegistry,
    [Parameter(Mandatory = $true)]
    [string]$Tariffs,
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$ResultsRoot = "artifacts\Comparacion",
    [string]$RunName = "resource_benchmark_isolated",
    [int]$Workers = 1
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE: $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
foreach ($required in @($CorpusRoot, $CorpusCache, $EvidenceRegistry, $Tariffs)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required input not found: $required"
    }
}

$validationRoot = Join-Path "artifacts" "validation"
$releaseRoot = Join-Path "artifacts" "releases\thesis-evidence"
$testReport = Join-Path $validationRoot "test_report.json"
$runRoot = Join-Path $ResultsRoot $RunName
$engineeringCosts = Join-Path $runRoot "engineering_costs.csv"
$costReport = Join-Path $runRoot "cost_scenarios.json"
$contextPath = Join-Path $validationRoot "validation-context.json"
$matrixRoot = Join-Path $validationRoot "requirements"

Invoke-CheckedPython -Arguments @(
    "scripts\run_validation_tests.py",
    "--output", $testReport
)

$runArguments = @(
    "-m", "Comparacion.cli",
    "--run-name", $RunName,
    "--results-root", $ResultsRoot,
    "--corpus-root", $CorpusRoot,
    "--corpus-cache", $CorpusCache,
    "--max-files", "3000",
    "--corpus-sample-seed", "7",
    "--fractions", "1.0",
    "--data-seeds", "1",
    "--model-seeds", "1",
    "--n-workers", "$Workers",
    "--transformer-device", "cuda",
    "--resource-measurement-condition", "isolated"
)
if (Test-Path -LiteralPath $runRoot) {
    $runArguments += "--resume"
}
Invoke-CheckedPython -Arguments $runArguments

Invoke-CheckedPython -Arguments @(
    "-m", "Comparacion.cli",
    "--cost-input", $engineeringCosts,
    "--tariffs", $Tariffs,
    "--cost-output", $costReport
)

if (Test-Path -LiteralPath $releaseRoot) {
    throw "Evidence output already exists; choose a fresh workspace or archive it first: $releaseRoot"
}
Invoke-CheckedPython -Arguments @(
    "-m", "Comparacion.cli",
    "--export-evidence", $EvidenceRegistry,
    "--evidence-output", $releaseRoot
)

$context = [ordered]@{
    evidence_package = [System.IO.Path]::GetFullPath($releaseRoot)
    resource_costs = [System.IO.Path]::GetFullPath($engineeringCosts)
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
