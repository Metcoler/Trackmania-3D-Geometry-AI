$ErrorActionPreference = "Stop"

# Internal diagnostic GA ablation sweep.
# PC1 ranking key: (finished, progress, -crashes, -time)
#
# Run from repository root:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_ga_ablation_pc1_crash_time.ps1
#
# This intentionally launches all ablations in parallel. Each run also uses
# --num-workers 4, so this is a heavy CPU workload.

$Seed = 2026050203
$RankingKey = "(finished, progress, -crashes, -time)"
$SweepRoot = "Experiments\runs_ga_ablation\pc1_finished_progress_crashes_time_seed_$Seed"

function Quote-Arg {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value -match '^[A-Za-z0-9_./\\:=#-]+$') {
        return $Value
    }
    return "'" + $Value.Replace("'", "''") + "'"
}

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
        "--num-workers", "4",
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

    $QuotedArgs = ($ArgsList | ForEach-Object { Quote-Arg $_ }) -join " "
    $Command = "`$env:PYTHONUNBUFFERED='1'; python $QuotedArgs; Write-Host ''; Write-Host 'Finished ablation: $Tag'; Read-Host 'Press Enter to close'"
    Write-Host "Launching $Tag -> $LogDir"
    Start-Process powershell.exe -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $Command)
}

Start-AblationRun -Tag "collision_corners" -UseCornersCollision
Start-AblationRun -Tag "variable_fps" -UseVariableFps
Start-AblationRun -Tag "max_time_45" -UseMaxTime45
Start-AblationRun -Tag "elite_cache" -UseEliteCache
Start-AblationRun -Tag "continuous_gas_brake" -UseContinuousGasBrake

Write-Host ""
Write-Host "Launched all PC1 ablation runs."
Write-Host "Root: $SweepRoot"
