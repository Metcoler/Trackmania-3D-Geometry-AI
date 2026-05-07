$ErrorActionPreference = "Stop"

# Sequential TM2D GA mutation refinement sweep.
# Run from the repository root with:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_ga_hyperparam_mutation_refinement_pc2.ps1

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Seed = 2026050401
$SweepRoot = "Experiments\runs_ga_hyperparam\pc2_mutation_refinement_seed_$Seed"
$RankingKey = "(finished, progress, -time, -crashes)"
$MutationProbs = @(0.05, 0.075, 0.10, 0.125, 0.15)
$MutationSigmas = @(0.20, 0.225, 0.25, 0.275, 0.30, 0.325)
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

Write-Host "TM2D GA hyperparameter refinement - mutation grid"
Write-Host "Seed: $Seed"
Write-Host "Root: $SweepRoot"
Write-Host "Ranking key: $RankingKey"
Write-Host "Config: fixed_fps=100, collision=lidar(AABB clearance), max_time=30, binary gas/brake, no elite cache"
Write-Host "Evolution base: population=48, parents=16, elites=4, generations=200, workers=4"
Write-Host "Mutation probs: $($MutationProbs -join ', ')"
Write-Host "Mutation sigmas: $($MutationSigmas -join ', ')"

foreach ($MutationProb in $MutationProbs) {
    foreach ($MutationSigma in $MutationSigmas) {
        $MutationProbArg = $MutationProb.ToString("0.000", $InvariantCulture)
        $MutationSigmaArg = $MutationSigma.ToString("0.000", $InvariantCulture)
        $Tag = "prob_$($MutationProbArg.Replace('.', ''))_sigma_$($MutationSigmaArg.Replace('.', ''))"
        $LogDir = Join-Path $SweepRoot $Tag

        Write-Host ""
        Write-Host "=== Running $Tag ==="
        Write-Host "mutation_prob=$MutationProbArg mutation_sigma=$MutationSigmaArg"
        Write-Host "Log dir: $LogDir"

        python Experiments\train_ga.py `
            --map-name "AI Training #5" `
            --generations 200 `
            --population-size 48 `
            --elite-count 4 `
            --parent-count 16 `
            --max-time 30 `
            --hidden-dim "32,16" `
            --hidden-activation "relu,tanh" `
            --mutation-prob $MutationProbArg `
            --mutation-sigma $MutationSigmaArg `
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
Write-Host "Mutation refinement finished. Root: $SweepRoot"
