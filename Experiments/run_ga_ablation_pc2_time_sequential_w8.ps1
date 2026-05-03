$ErrorActionPreference = "Stop"

# Internal diagnostic GA ablation sweep.
# Sequential memory-safer PC2 variant.
# Ranking key: (finished, progress, -time)
#
# Run from repository root:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_ga_ablation_pc2_time_sequential_w8.ps1
#
# This runs one ablation after another, but each run uses --num-workers 8.

$Seed = 2026050203
$RankingKey = "(finished, progress, -time)"
$SweepRoot = "Experiments\runs_ga_ablation\pc2_finished_progress_time_seed_$Seed"

function Start-AblationRun {
    param(
        [Parameter(Mandatory = $true)][string]$Tag,
        [switch]$UseCornersCollision,
        [switch]$UseVariableFps,
        [switch]$UseMaxTime45,
        [switch]$UseEliteCache,
        [switch]$UseContinuousGasBrake
    )

    $LogDir = Join-Path $SweepRoot $Tag
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    $MaxTime = if ($UseMaxTime45) { "45" } else { "30" }
    $CollisionMode = if ($UseCornersCollision) { "corners" } else { "laser" }

    $ArgsList = @(
        "Experiments\train_ga.py",
        "--map-name", "AI Training #5",
        "--generations", "300",
        "--population-size", "64",
        "--elite-count", "8",
        "--parent-count", "32",
        "--max-time", $MaxTime,
        "--hidden-dim", "32,16",
        "--hidden-activation", "relu,tanh",
        "--mutation-prob", "0.2",
        "--mutation-sigma", "0.2",
        "--fitness-mode", "ranking",
        "--ranking-mode", "lexicographic",
        "--ranking-key", $RankingKey,
        "--ranking-progress-source", "dense_progress",
        "--reward-mode", "terminal_fitness",
        "--collision-mode", $CollisionMode,
        "--collision-distance-threshold", "2.0",
        "--seed", "$Seed",
        "--num-workers", "8",
        "--log-dir", $LogDir
    )

    if (-not $UseVariableFps) {
        $ArgsList += @("--fixed-fps", "60")
    }
    if ($UseEliteCache) {
        $ArgsList += "--enable-elite-cache"
    }
    if ($UseContinuousGasBrake) {
        $ArgsList += "--continuous-gas-brake"
    }

    Write-Host ""
    Write-Host "=== Running $Tag -> $LogDir ==="
    python @ArgsList
}

Start-AblationRun -Tag "collision_corners" -UseCornersCollision
Start-AblationRun -Tag "variable_fps" -UseVariableFps
Start-AblationRun -Tag "max_time_45" -UseMaxTime45
Start-AblationRun -Tag "elite_cache" -UseEliteCache
Start-AblationRun -Tag "continuous_gas_brake" -UseContinuousGasBrake

Write-Host ""
Write-Host "Sequential PC2 ablation sweep finished."
Write-Host "Root: $SweepRoot"
