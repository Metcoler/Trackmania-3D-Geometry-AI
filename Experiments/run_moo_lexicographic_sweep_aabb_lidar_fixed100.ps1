$ErrorActionPreference = "Stop"

# Sequential TM2D Pareto/NSGA-II sweep over the same primitive objective sets
# used in the fixed100 lexicographic reward sweep.
#
# Run from any directory:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_moo_lexicographic_sweep_aabb_lidar_fixed100.ps1

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Seed = 2026050406
$SweepRoot = "Experiments\runs_ga_moo\moo_lex_sweep_aabb_lidar_fixed100_seed_$Seed"

$Variants = @(
    @{
        Tag = "finished_progress"
        Subset = "finished,progress"
        Priority = "finished,progress"
    },
    @{
        Tag = "finished_progress_time"
        Subset = "finished,progress,neg_time"
        Priority = "finished,progress,neg_time"
    },
    @{
        Tag = "finished_progress_time_crashes"
        Subset = "finished,progress,neg_time,neg_crashes"
        Priority = "finished,progress,neg_time,neg_crashes"
    },
    @{
        Tag = "finished_progress_crashes_time"
        Subset = "finished,progress,neg_crashes,neg_time"
        Priority = "finished,progress,neg_crashes,neg_time"
    }
)

Write-Host "TM2D MOO Pareto/NSGA-II objective sweep"
Write-Host "Seed: $Seed"
Write-Host "Root: $SweepRoot"
Write-Host "Config: fixed100 physics tick, collision=lidar(AABB clearance), max_time=30, binary gas/brake, no elite cache"
Write-Host "Evolution: population=48, elite=4, parents=16, mutation_prob=0.2, mutation_sigma=0.2, workers=4"

foreach ($Variant in $Variants) {
    $Tag = $Variant["Tag"]
    $Subset = $Variant["Subset"]
    $Priority = $Variant["Priority"]
    $LogDir = Join-Path $SweepRoot $Tag

    Write-Host ""
    Write-Host "=== Running $Tag ==="
    Write-Host "Objective subset: $Subset"
    Write-Host "Priority tiebreak: $Priority"
    Write-Host "Log dir: $LogDir"

    python Experiments\train_ga_moo.py `
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
        --physics-tick-profile fixed100 `
        --objective-mode lexicographic_primitives `
        --objective-subset $Subset `
        --objective-priority $Priority `
        --pareto-tiebreak priority `
        --reward-mode terminal_fitness `
        --collision-mode lidar `
        --seed $Seed `
        --num-workers 4 `
        --worker-backend auto `
        --log-dir $LogDir

    if ($LASTEXITCODE -ne 0) {
        throw "MOO run '$Tag' failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "MOO sweep finished. Root: $SweepRoot"
