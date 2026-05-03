$ErrorActionPreference = "Stop"

# Sequential TM2D GA hyperparameter sweep for selection pressure settings.
# Run from the repository root with:
#   powershell -ExecutionPolicy Bypass -File .\Experiments\run_ga_hyperparam_selection_grid_pc2.ps1

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

$Seed = 2026050311
$PopulationSize = 48
$SweepRoot = "Experiments\runs_ga_hyperparam\pc2_selection_grid_seed_$Seed"
$RankingKey = "(finished, progress, -time, -crashes)"
$Ratios = @(0.10, 0.20, 0.30, 0.40, 0.50)
$InvariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

function Get-NearestEvenCount {
    param(
        [double]$Value,
        [int]$Minimum
    )
    $Rounded = [int]([Math]::Round($Value / 2.0, 0, [MidpointRounding]::AwayFromZero) * 2)
    if ($Rounded -lt $Minimum) {
        return $Minimum
    }
    return $Rounded
}

function Get-EliteCount {
    param(
        [int]$ParentCount,
        [double]$EliteParentRatio
    )
    $Rounded = [int][Math]::Round($ParentCount * $EliteParentRatio, 0, [MidpointRounding]::AwayFromZero)
    if ($Rounded -lt 1) {
        $Rounded = 1
    }
    $Maximum = [Math]::Max(1, $ParentCount - 1)
    if ($Rounded -gt $Maximum) {
        $Rounded = $Maximum
    }
    return $Rounded
}

Write-Host "TM2D GA hyperparameter sweep - PC2 selection grid"
Write-Host "Seed: $Seed"
Write-Host "Root: $SweepRoot"
Write-Host "Ranking key: $RankingKey"
Write-Host "Config: fixed_fps=100, collision=lidar(AABB clearance), max_time=30, binary gas/brake, no elite cache"
Write-Host "Evolution base: population=$PopulationSize, mutation_prob=0.2, mutation_sigma=0.2, generations=200"

foreach ($ParentRatio in $Ratios) {
    $ParentCount = Get-NearestEvenCount -Value ($PopulationSize * $ParentRatio) -Minimum 2
    if ($ParentCount -ge $PopulationSize) {
        $ParentCount = $PopulationSize - 2
    }

    foreach ($EliteParentRatio in $Ratios) {
        $EliteCount = Get-EliteCount -ParentCount $ParentCount -EliteParentRatio $EliteParentRatio
        $ParentRatioTag = $ParentRatio.ToString("0.00", $InvariantCulture)
        $EliteRatioTag = $EliteParentRatio.ToString("0.00", $InvariantCulture)
        $Tag = "parents_ratio_$($ParentRatioTag.Replace('.', ''))_elites_ratio_$($EliteRatioTag.Replace('.', ''))_p$ParentCount`_e$EliteCount"
        $LogDir = Join-Path $SweepRoot $Tag

        Write-Host ""
        Write-Host "=== Running $Tag ==="
        Write-Host "parents_ratio=$ParentRatio parent_count=$ParentCount elite_parent_ratio=$EliteParentRatio elite_count=$EliteCount"
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
Write-Host "Selection grid finished. Root: $SweepRoot"
