$ErrorActionPreference = "Stop"

# Second-seed replication of the TM2D lexicographic reward sweep after the
# AABB-clearance lidar patch.
# Run from any directory:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_lexicographic_sweep_aabb_lidar_fixed100_2.ps1

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Seed = 2026050307
$SweepRoot = "Experiments\runs_ga\lex_sweep_aabb_lidar_fixed100_seed_2026050306_2"

$Variants = @(
    @{
        Tag = "finished_progress"
        Key = "(finished, progress)"
    },
    @{
        Tag = "finished_progress_time"
        Key = "(finished, progress, -time)"
    },
    @{
        Tag = "finished_progress_time_crashes"
        Key = "(finished, progress, -time, -crashes)"
    },
    @{
        Tag = "finished_progress_crashes_time"
        Key = "(finished, progress, -crashes, -time)"
    }
)

Write-Host "TM2D lexicographic sweep replication"
Write-Host "Seed: $Seed"
Write-Host "Root: $SweepRoot"
Write-Host "Config: fixed_fps=100, collision=lidar(AABB clearance), max_time=30, binary gas/brake, no elite cache"
Write-Host "Evolution: population=48, elite=4, parents=16, mutation_prob=0.2, mutation_sigma=0.2"

foreach ($Variant in $Variants) {
    $Tag = $Variant["Tag"]
    $Key = $Variant["Key"]
    $LogDir = Join-Path $SweepRoot $Tag

    Write-Host ""
    Write-Host "=== Running $Tag ==="
    Write-Host "Ranking key: $Key"
    Write-Host "Log dir: $LogDir"

    python Experiments\train_ga.py `
        --map-name "AI Training #5" `
        --generations 300 `
        --population-size 48 `
        --elite-count 4 `
        --parent-count 16 `
        --max-time 30 `
        --hidden-dim "32,16" `
        --hidden-activation "relu,tanh" `
        --mutation-prob 0.2 `
        --mutation-sigma 0.2 `
        --fixed-fps 100 `
        --fitness-mode ranking `
        --ranking-mode lexicographic `
        --ranking-key $Key `
        --ranking-progress-source dense_progress `
        --reward-mode terminal_fitness `
        --collision-mode lidar `
        --seed $Seed `
        --num-workers 8 `
        --log-dir $LogDir `
        --disable-elite-cache
}

Write-Host ""
Write-Host "Sweep finished. Root: $SweepRoot"
