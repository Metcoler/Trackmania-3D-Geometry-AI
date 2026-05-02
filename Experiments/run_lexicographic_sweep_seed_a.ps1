$ErrorActionPreference = "Stop"

# Fair lexicographic GA reward sweep for thesis experiments.
# Run from repository root:
#   .\Experiments\run_lexicographic_sweep_seed_a.ps1

$Seed = 2026050201
$SweepRoot = "Experiments\runs_ga\lex_sweep_seed_$Seed"

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

foreach ($Variant in $Variants) {
    $Tag = $Variant["Tag"]
    $Key = $Variant["Key"]
    $LogDir = Join-Path $SweepRoot $Tag
    Write-Host ""
    Write-Host "=== Running $Tag with seed $Seed ==="

    python Experiments\train_ga.py `
        --map-name "AI Training #5" `
        --generations 300 `
        --population-size 64 `
        --elite-count 8 `
        --parent-count 32 `
        --max-time 30 `
        --hidden-dim "32,16" `
        --hidden-activation "relu,tanh" `
        --mutation-prob 0.2 `
        --mutation-sigma 0.2 `
        --fixed-fps 60 `
        --fitness-mode ranking `
        --ranking-mode lexicographic `
        --ranking-key $Key `
        --ranking-progress-source dense_progress `
        --reward-mode terminal_fitness `
        --collision-mode laser `
        --collision-distance-threshold 2.0 `
        --seed $Seed `
        --num-workers 4 `
        --log-dir $LogDir `
        --disable-elite-cache
}

Write-Host ""
Write-Host "Sweep finished. Root: $SweepRoot"
