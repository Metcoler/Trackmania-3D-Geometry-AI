$ErrorActionPreference = "Stop"

# Diagnostic GA sweep for the intermediate TM2D physics profile.
# Baseline inside this script:
#   variable FPS, center collision, continuous gas/brake, max_time=45
#   population 48, elite 4, parents 16, 200 generations
#
# Each run below changes exactly one setting:
#   1. fixed_fps_60
#   2. collision_lidar
#   3. binary_gas_brake
#   4. max_time_30
#
# Run from repository root:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_mid_physics_ga_metric_ablation_seq_w8.ps1

$Seed = 2026050301
$RankingKey = "(finished, progress, -time)"
$SweepRoot = "Experiments\runs_ga_diagnostic\mid_physics_metric_ablation_seed_$Seed"

function Start-MidPhysicsRun {
    param(
        [Parameter(Mandatory = $true)][string]$Tag,
        [switch]$UseFixedFps,
        [switch]$UseLidarCollision,
        [switch]$UseBinaryGasBrake,
        [switch]$UseMaxTime30
    )

    $LogDir = Join-Path $SweepRoot $Tag
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

    $MaxTime = if ($UseMaxTime30) { "30" } else { "45" }
    $CollisionMode = if ($UseLidarCollision) { "lidar" } else { "center" }

    $ArgsList = @(
        "Experiments\train_ga.py",
        "--map-name", "AI Training #5",
        "--generations", "200",
        "--population-size", "48",
        "--elite-count", "4",
        "--parent-count", "16",
        "--max-time", $MaxTime,
        "--hidden-dim", "32,16",
        "--hidden-activation", "relu,tanh",
        "--mutation-prob", "0.18",
        "--mutation-sigma", "0.2",
        "--fitness-mode", "ranking",
        "--ranking-mode", "lexicographic",
        "--ranking-key", $RankingKey,
        "--ranking-progress-source", "dense_progress",
        "--reward-mode", "progress_primary_delta",
        "--collision-mode", $CollisionMode,
        "--collision-distance-threshold", "2.0",
        "--seed", "$Seed",
        "--num-workers", "8",
        "--log-dir", $LogDir
    )

    if ($UseFixedFps) {
        $ArgsList += @("--fixed-fps", "60")
    }
    if (-not $UseBinaryGasBrake) {
        $ArgsList += "--continuous-gas-brake"
    }

    Write-Host ""
    Write-Host "=== Running $Tag -> $LogDir ==="
    python @ArgsList
}

Start-MidPhysicsRun -Tag "fixed_fps_60" -UseFixedFps
Start-MidPhysicsRun -Tag "collision_lidar" -UseLidarCollision
Start-MidPhysicsRun -Tag "binary_gas_brake" -UseBinaryGasBrake
Start-MidPhysicsRun -Tag "max_time_30" -UseMaxTime30

Write-Host ""
Write-Host "Intermediate-physics sequential ablation sweep finished."
Write-Host "Root: $SweepRoot"
