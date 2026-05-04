$ErrorActionPreference = "Stop"

# Sequential TM2D GA hyperparameter sweep for mutation settings.
# Run from the repository root with:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_ga_hyperparam_mutation_grid_pc1.ps1

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Seed = 2026050311
$SweepRoot = "Experiments\runs_ga_hyperparam\pc1_mutation_grid_seed_$Seed"
$RankingKey = "(finished, progress, -time, -crashes)"
$Values = @(0.10, 0.15, 0.20, 0.25, 0.30)
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

Write-Host "TM2D GA hyperparameter sweep - PC1 mutation grid"
Write-Host "Seed: $Seed"
Write-Host "Root: $SweepRoot"
Write-Host "Ranking key: $RankingKey"
Write-Host "Config: fixed_fps=100, collision=lidar(AABB clearance), max_time=30, binary gas/brake, no elite cache"
Write-Host "Evolution base: population=48, elite=4, parents=16, generations=200"

foreach ($MutationProb in $Values) {
    foreach ($MutationSigma in $Values) {
        $MutationProbArg = $MutationProb.ToString("0.00", $InvariantCulture)
        $MutationSigmaArg = $MutationSigma.ToString("0.00", $InvariantCulture)
        $ProbTag = $MutationProbArg
        $SigmaTag = $MutationSigmaArg
        $Tag = "prob_$($ProbTag.Replace('.', ''))_sigma_$($SigmaTag.Replace('.', ''))"
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
            --num-workers 8 `
            --log-dir $LogDir `
            --disable-elite-cache
    }
}

Write-Host ""
Write-Host "Mutation grid finished. Root: $SweepRoot"
