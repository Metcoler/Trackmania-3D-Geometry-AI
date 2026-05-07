$ErrorActionPreference = "Stop"

# Sequential supervised pretrain -> TM2D GA fine-tuning experiment.
# Dense variant: every weight in each seeded tier receives Gaussian noise.
# Run from repository root with:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_ga_supervised_seeded_dense_pretrain_20260505.ps1

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Seed = 2026050407
$MapName = "AI Training #5"
$DataRoot = "logs\supervised_data"
$DatasetPattern = "*_map_AI Training #5_v3d_surface_target_dataset"
$SupervisedRoot = "logs\supervised_runs_ai5_pretrain_20260505"
$GaLogRoot = "Experiments\runs_ga_training_improvements\seed_$Seed\exp06b_supervised_seeded_dense_fixed100"
$RankingKey = "(finished, progress, -time, -crashes)"

$Datasets = Get-ChildItem -Path $DataRoot -Directory -Filter $DatasetPattern |
    Sort-Object LastWriteTime -Descending
if (-not $Datasets -or $Datasets.Count -eq 0) {
    throw "No AI Training #5 v3d/surface dataset found under '$DataRoot'."
}
$Dataset = $Datasets[0].FullName
$AttemptCount = @(Get-ChildItem -Path (Join-Path $Dataset "attempts") -Filter "attempt_*.npz").Count
if ($AttemptCount -lt 10) {
    throw "Dataset '$Dataset' has only $AttemptCount attempts; expected at least 10."
}

Write-Host "Supervised-seeded dense-noise TM2D GA experiment"
Write-Host "Dataset: $Dataset"
Write-Host "Attempts: $AttemptCount"
Write-Host "Supervised output root: $SupervisedRoot"
Write-Host "GA output root: $GaLogRoot"

New-Item -ItemType Directory -Force -Path $SupervisedRoot | Out-Null

$ExistingModel = Get-ChildItem -Path $SupervisedRoot -Recurse -Filter "best_model.pt" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($ExistingModel) {
    $BestModel = $ExistingModel
    Write-Host ""
    Write-Host "Reusing existing supervised model: $($BestModel.FullName)"
} else {
    $SupervisedArgs = @(
        "SupervisedTraining.py",
        "--data-root", $Dataset,
        "--output-root", $SupervisedRoot,
        "--vertical-mode", "false",
        "--multi-surface-mode", "false",
        "--hidden-dim", "32,16",
        "--hidden-activation", "relu,tanh",
        "--epochs", "150",
        "--disable-boring-filter",
        "--random-seed", "$Seed"
    )

    Write-Host ""
    Write-Host "=== Training supervised v2d/asphalt seed model ==="
    Write-Host "python $($SupervisedArgs -join ' ')"
    & python @SupervisedArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Supervised pretraining failed with exit code $LASTEXITCODE"
    }

    $BestModel = Get-ChildItem -Path $SupervisedRoot -Recurse -Filter "best_model.pt" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $BestModel) {
        throw "No best_model.pt found under '$SupervisedRoot' after supervised training."
    }
}

Write-Host ""
Write-Host "Best supervised model: $($BestModel.FullName)"

$GaArgs = @(
    "Experiments\train_ga.py",
    "--map-name", $MapName,
    "--generations", "200",
    "--population-size", "48",
    "--elite-count", "2",
    "--parent-count", "14",
    "--max-time", "30",
    "--hidden-dim", "32,16",
    "--hidden-activation", "relu,tanh",
    "--mutation-prob", "0.10",
    "--mutation-sigma", "0.25",
    "--fitness-mode", "ranking",
    "--ranking-mode", "lexicographic",
    "--ranking-key", $RankingKey,
    "--ranking-progress-source", "progress",
    "--reward-mode", "terminal_fitness",
    "--collision-mode", "lidar",
    "--physics-tick-profile", "fixed100",
    "--seed", "$Seed",
    "--num-workers", "4",
    "--disable-elite-cache",
    "--initial-population-model", $BestModel.FullName,
    "--initial-model-exact-copies", "1",
    "--initial-model-noise-mode", "dense",
    "--initial-model-mutation-probs", "1,1,1,1",
    "--initial-model-mutation-sigmas", "0.01,0.03,0.06,0.10",
    "--initial-model-tier-counts", "8,14,14,11",
    "--log-dir", $GaLogRoot
)

Write-Host ""
Write-Host "=== Running dense supervised-seeded TM2D GA fine-tuning ==="
Write-Host "python $($GaArgs -join ' ')"
& python @GaArgs
if ($LASTEXITCODE -ne 0) {
    throw "Dense supervised-seeded GA failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Dense supervised-seeded GA experiment finished."
Write-Host "Supervised model: $($BestModel.FullName)"
Write-Host "GA root: $GaLogRoot"
