$ErrorActionPreference = "Stop"

$DataRoot = "logs\supervised_data"
$OutputRoot = "logs\supervised_runs_map_specialists_20260505"
$Maps = @(
    "single_surface_flat",
    "multi_surface_flat",
    "single_surface_height"
)

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Get-LatestDataset {
    param([string]$MapName)

    $pattern = "*_map_${MapName}_v3d_surface_target_dataset"
    $datasets = Get-ChildItem -Path $DataRoot -Directory -Filter $pattern |
        Sort-Object LastWriteTime -Descending
    if (-not $datasets -or $datasets.Count -eq 0) {
        throw "No v3d surface supervised dataset found for map '$MapName' under '$DataRoot'."
    }
    return $datasets[0].FullName
}

foreach ($MapName in $Maps) {
    $Dataset = Get-LatestDataset -MapName $MapName
    $MapOutputRoot = Join-Path $OutputRoot $MapName
    New-Item -ItemType Directory -Force -Path $MapOutputRoot | Out-Null

    Write-Host ""
    Write-Host "=== Training supervised specialist for $MapName ==="
    Write-Host "Dataset: $Dataset"
    Write-Host "Output : $MapOutputRoot"

    python SupervisedTraining.py `
        --data-root $Dataset `
        --output-root $MapOutputRoot `
        --vertical-mode true `
        --multi-surface-mode true `
        --hidden-dim "48,24" `
        --hidden-activation "relu,tanh" `
        --epochs 150 `
        --disable-boring-filter

    if ($LASTEXITCODE -ne 0) {
        throw "Supervised specialist training failed for $MapName with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Supervised map specialist training finished."
Write-Host "Results root: $OutputRoot"
