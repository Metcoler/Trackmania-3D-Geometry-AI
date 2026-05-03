$ErrorActionPreference = "Stop"

# Sequential TM2D GA hyperparameter refinement sweep for exact selection counts.
# Run from the repository root with:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_ga_hyperparam_selection_count_grid_pc2.ps1

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Seed = 2026050312
$PopulationSize = 48
$SweepRoot = "Experiments\runs_ga_hyperparam\pc2_selection_count_grid_seed_$Seed"
$RankingKey = "(finished, progress, -time, -crashes)"
$ParentCounts = @(8, 10, 12, 14, 16, 18)
$EliteCounts = @(1, 2, 3, 4)

Write-Host "TM2D GA hyperparameter refinement sweep - exact selection counts"
Write-Host "Seed: $Seed"
Write-Host "Root: $SweepRoot"
Write-Host "Ranking key: $RankingKey"
Write-Host "Config: fixed_fps=100, collision=lidar(AABB clearance), max_time=30, binary gas/brake, no elite cache"
Write-Host "Evolution base: population=$PopulationSize, mutation_prob=0.2, mutation_sigma=0.2, generations=200"
Write-Host "Parent counts: $($ParentCounts -join ', ')"
Write-Host "Elite counts: $($EliteCounts -join ', ')"

foreach ($ParentCount in $ParentCounts) {
    foreach ($EliteCount in $EliteCounts) {
        if ($EliteCount -ge $ParentCount) {
            Write-Host "Skipping invalid config: parent_count=$ParentCount elite_count=$EliteCount"
            continue
        }

        $ParentTag = $ParentCount.ToString("000")
        $EliteTag = $EliteCount.ToString("000")
        $Tag = "parent_count_$ParentTag`_elite_count_$EliteTag"
        $LogDir = Join-Path $SweepRoot $Tag

        Write-Host ""
        Write-Host "=== Running $Tag ==="
        Write-Host "parent_count=$ParentCount elite_count=$EliteCount"
        Write-Host "Log dir: $LogDir"

        python Experiments\train_ga.py `
            --map-name "AI Training #5" `
            --generations 200 `
            --population-size $PopulationSize `
            --elite-count $EliteCount `
            --parent-count $ParentCount `
            --max-time 30 `
            --hidden-dim "32,16" `
            --hidden-activation "relu,tanh" `
            --mutation-prob 0.2 `
            --mutation-sigma 0.2 `
            --fixed-fps 100 `
            --fitness-mode ranking `
            --ranking-mode lexicographic `
            --ranking-key $RankingKey `
            --ranking-progress-source dense_progress `
            --reward-mode terminal_fitness `
            --collision-mode lidar `
            --seed $Seed `
            --num-workers 8 `
            --log-dir $LogDir `
            --disable-elite-cache
    }
}

Write-Host ""
Write-Host "Selection count grid finished. Root: $SweepRoot"
