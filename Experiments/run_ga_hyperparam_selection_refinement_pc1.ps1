$ErrorActionPreference = "Stop"

# Sequential TM2D GA selection refinement sweep.
# Run from the repository root with:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_ga_hyperparam_selection_refinement_pc1.ps1

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Seed = 2026050401
$PopulationSize = 48
$MutationProb = "0.2"
$MutationSigma = "0.2"
$SweepRoot = "Experiments\runs_ga_hyperparam\pc1_selection_refinement_seed_$Seed"
$RankingKey = "(finished, progress, -time, -crashes)"
$ParentCounts = @(8, 10, 12, 14, 16, 18)
$EliteCounts = @(1, 2, 3, 4)

Write-Host "TM2D GA hyperparameter refinement - selection counts"
Write-Host "Seed: $Seed"
Write-Host "Root: $SweepRoot"
Write-Host "Ranking key: $RankingKey"
Write-Host "Config: fixed_fps=100, collision=lidar(AABB clearance), max_time=30, binary gas/brake, no elite cache"
Write-Host "Evolution base: population=$PopulationSize, mutation_prob=$MutationProb, mutation_sigma=$MutationSigma, generations=200, workers=4"
Write-Host "Parent counts: $($ParentCounts -join ', ')"
Write-Host "Elite counts: $($EliteCounts -join ', ')"

foreach ($ParentCount in $ParentCounts) {
    foreach ($EliteCount in $EliteCounts) {
        if ($EliteCount -gt $ParentCount) {
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
            --mutation-prob $MutationProb `
            --mutation-sigma $MutationSigma `
            --fixed-fps 100 `
            --fitness-mode ranking `
            --ranking-mode lexicographic `
            --ranking-key $RankingKey `
            --ranking-progress-source dense_progress `
            --reward-mode terminal_fitness `
            --collision-mode lidar `
            --seed $Seed `
            --num-workers 4 `
            --log-dir $LogDir `
            --disable-elite-cache
    }
}

Write-Host ""
Write-Host "Selection refinement finished. Root: $SweepRoot"
