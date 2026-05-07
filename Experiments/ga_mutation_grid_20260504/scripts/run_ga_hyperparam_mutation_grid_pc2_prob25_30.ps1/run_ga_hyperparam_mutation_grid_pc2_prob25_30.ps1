$ErrorActionPreference = "Stop"

# Partial sequential TM2D GA hyperparameter sweep for mutation settings.
# Intended for offloading the unfinished high-probability part of PC1's mutation grid to PC2.
# Run from the repository root with:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_ga_hyperparam_mutation_grid_pc2_prob25_30.ps1
#
# Important: stop the original PC1 mutation grid before it reaches prob_025_* to avoid duplicate runs.

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Seed = 2026050311
$SweepRoot = "Experiments\runs_ga_hyperparam\pc1_mutation_grid_seed_$Seed"
$RankingKey = "(finished, progress, -time, -crashes)"
$MutationProbs = @(0.25, 0.30)
$MutationSigmas = @(0.10, 0.15, 0.20, 0.25, 0.30)
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

function Test-CompleteRun {
    param(
        [Parameter(Mandatory = $true)]
        [string] $LogDir,
        [Parameter(Mandatory = $true)]
        [int] $ExpectedGenerations
    )

    if (-not (Test-Path $LogDir)) {
        return $false
    }

    $MetricsFiles = Get-ChildItem -Path $LogDir -Recurse -Filter "generation_metrics.csv" -ErrorAction SilentlyContinue
    foreach ($MetricsFile in $MetricsFiles) {
        $LastRow = Import-Csv $MetricsFile.FullName | Select-Object -Last 1
        if ($null -ne $LastRow -and [int]$LastRow.generation -ge $ExpectedGenerations) {
            return $true
        }
    }
    return $false
}

Write-Host "TM2D GA hyperparameter sweep - PC2 partial mutation grid"
Write-Host "Seed: $Seed"
Write-Host "Root: $SweepRoot"
Write-Host "Ranking key: $RankingKey"
Write-Host "Running only mutation_prob = 0.25 and 0.30"
Write-Host "Config: fixed_fps=100, collision=lidar(AABB clearance), max_time=30, binary gas/brake, no elite cache"
Write-Host "Evolution base: population=48, elite=4, parents=16, generations=200"

foreach ($MutationProb in $MutationProbs) {
    foreach ($MutationSigma in $MutationSigmas) {
        $MutationProbArg = $MutationProb.ToString("0.00", $InvariantCulture)
        $MutationSigmaArg = $MutationSigma.ToString("0.00", $InvariantCulture)
        $ProbTag = $MutationProbArg
        $SigmaTag = $MutationSigmaArg
        $Tag = "prob_$($ProbTag.Replace('.', ''))_sigma_$($SigmaTag.Replace('.', ''))"
        $LogDir = Join-Path $SweepRoot $Tag

        if (Test-CompleteRun -LogDir $LogDir -ExpectedGenerations 200) {
            Write-Host ""
            Write-Host "=== Skipping ${Tag}: complete run already exists ==="
            continue
        }

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
Write-Host "Partial mutation grid finished. Root: $SweepRoot"
