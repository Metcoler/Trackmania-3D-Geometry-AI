$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Hardcoded run recipe lives in GeneticTrainer.py:
# - map multi_surface_flat
# - vertical=false
# - multi_surface=true
# - 48x24 relu,tanh
# - GA profile pop=48, parents=14, elites=2
# - supervised 48x24 multi_surface_flat model used as dense-noise seed
$SeedRoot = "logs\supervised_runs_multi_surface_flat_pretrain_48x24_20260507"
$SeedModel = Get-ChildItem -Path $SeedRoot -Recurse -Filter "best_model.pt" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $SeedModel) {
    throw "No multi_surface_flat seed model found under '$SeedRoot'. Run .\Experiments\run_supervised_pretrain_multi_surface_flat_48x24_20260507.ps1 first."
}

Write-Host "Multi-surface-flat live GA seeded run"
Write-Host "Seed model: $($SeedModel.FullName)"
Write-Host "Map/layout: multi_surface_flat, vertical=false, multi_surface=true"
Write-Host "GA profile: pop=48, parents=14, elites=2, generations=300"
Write-Host "Ranking: (finished, progress, -time, -crashes)"
Write-Host ""
Write-Host "python GeneticTrainer.py"

& python GeneticTrainer.py
if ($LASTEXITCODE -ne 0) {
    throw "Multi-surface-flat GA run failed with exit code $LASTEXITCODE"
}
