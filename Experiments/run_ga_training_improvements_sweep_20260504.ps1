$ErrorActionPreference = "Stop"

# Sequential TM2D GA training-improvements sweep.
# Run from the repository root with:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_ga_training_improvements_sweep_20260504.ps1

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Seed = 2026050407
$SweepRoot = "Experiments\runs_ga_training_improvements\seed_$Seed"
$RankingKey = "(finished, progress, -time, -crashes)"

$CommonArgs = @(
    "Experiments\train_ga.py",
    "--map-name", "AI Training #5",
    "--generations", "200",
    "--population-size", "48",
    "--elite-count", "2",
    "--parent-count", "14",
    "--max-time", "30",
    "--hidden-dim", "32,16",
    "--hidden-activation", "relu,tanh",
    "--fitness-mode", "ranking",
    "--ranking-mode", "lexicographic",
    "--ranking-key", $RankingKey,
    "--ranking-progress-source", "dense_progress",
    "--reward-mode", "terminal_fitness",
    "--collision-mode", "lidar",
    "--seed", "$Seed",
    "--num-workers", "4"
)

function Invoke-GaRun {
    param(
        [Parameter(Mandatory = $true)][string]$Tag,
        [Parameter(Mandatory = $true)][string[]]$ExtraArgs
    )

    $LogDir = Join-Path $SweepRoot $Tag
    $RunArgs = @($CommonArgs + $ExtraArgs + @("--log-dir", $LogDir))

    Write-Host ""
    Write-Host "=== Running $Tag ==="
    Write-Host "Log dir: $LogDir"
    Write-Host "python $($RunArgs -join ' ')"

    & python @RunArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Run '$Tag' failed with exit code $LASTEXITCODE"
    }
}

Write-Host "TM2D GA training-improvements sweep"
Write-Host "Seed: $Seed"
Write-Host "Root: $SweepRoot"
Write-Host "Ranking key: $RankingKey"
Write-Host "Base: pop=48, parents=14, elites=2, hidden=32,16 relu,tanh, max_time=30, workers=4"

Invoke-GaRun -Tag "exp00_base_fixed100" -ExtraArgs @(
    "--mutation-prob", "0.10",
    "--mutation-sigma", "0.25",
    "--physics-tick-profile", "fixed100",
    "--disable-elite-cache"
)

Invoke-GaRun -Tag "exp01_mutation_decay_fixed100" -ExtraArgs @(
    "--mutation-prob", "0.15",
    "--mutation-prob-min", "0.05",
    "--mutation-prob-decay", "0.9944938",
    "--mutation-sigma", "0.30",
    "--mutation-sigma-min", "0.20",
    "--mutation-sigma-decay", "0.9979648",
    "--physics-tick-profile", "fixed100",
    "--disable-elite-cache"
)

Invoke-GaRun -Tag "exp02_both_mirrors_fixed100" -ExtraArgs @(
    "--mutation-prob", "0.10",
    "--mutation-sigma", "0.25",
    "--physics-tick-profile", "fixed100",
    "--evaluate-both-mirrors",
    "--disable-elite-cache"
)

Invoke-GaRun -Tag "exp03_mirror_prob_050_fixed100" -ExtraArgs @(
    "--mutation-prob", "0.10",
    "--mutation-sigma", "0.25",
    "--physics-tick-profile", "fixed100",
    "--mirror-episode-prob", "0.5",
    "--disable-elite-cache"
)

Invoke-GaRun -Tag "exp04_max_touches_3_fixed100" -ExtraArgs @(
    "--mutation-prob", "0.10",
    "--mutation-sigma", "0.25",
    "--physics-tick-profile", "fixed100",
    "--max-touches", "3",
    "--collision-bounce-speed-retention", "0.40",
    "--disable-elite-cache"
)

Invoke-GaRun -Tag "exp05a_variable_tick_no_cache_control" -ExtraArgs @(
    "--mutation-prob", "0.10",
    "--mutation-sigma", "0.25",
    "--physics-tick-profile", "supervised_v2d",
    "--disable-elite-cache"
)

Invoke-GaRun -Tag "exp05b_variable_tick_elite_cache" -ExtraArgs @(
    "--mutation-prob", "0.10",
    "--mutation-sigma", "0.25",
    "--physics-tick-profile", "supervised_v2d",
    "--enable-elite-cache"
)

Write-Host ""
Write-Host "Training-improvements sweep finished. Root: $SweepRoot"
