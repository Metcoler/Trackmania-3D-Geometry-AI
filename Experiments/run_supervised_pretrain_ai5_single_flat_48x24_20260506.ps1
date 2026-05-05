param(
    [string]$AiTrainingDataset = "logs\supervised_data\20260505_195744_map_AI Training #5_v3d_surface_target_dataset",
    [string]$SingleFlatDataset = "logs\supervised_data\20260505_132438_map_single_surface_flat_v3d_surface_target_dataset",
    [string]$CombinedDataset = "logs\supervised_data\20260506_map_AI_Training_5_plus_single_surface_flat_v3d_surface_target_dataset",
    [string]$OutputRoot = "logs\supervised_runs_ai5_single_flat_pretrain_48x24_20260506",
    [int]$Epochs = 250,
    [switch]$RebuildCombinedDataset
)

$ErrorActionPreference = "Stop"

function Resolve-RequiredDirectory {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function New-LinkOrCopy {
    param(
        [Parameter(Mandatory=$true)][string]$Source,
        [Parameter(Mandatory=$true)][string]$Target
    )
    try {
        New-Item -ItemType HardLink -Path $Target -Target $Source | Out-Null
    }
    catch {
        Copy-Item -LiteralPath $Source -Destination $Target
    }
}

function Add-Attempts {
    param(
        [Parameter(Mandatory=$true)][string]$SourceRoot,
        [Parameter(Mandatory=$true)][string]$Prefix,
        [Parameter(Mandatory=$true)][string]$TargetAttemptsDir
    )
    $sourceAttemptsDir = Join-Path $SourceRoot "attempts"
    if (-not (Test-Path -LiteralPath $sourceAttemptsDir -PathType Container)) {
        throw "Dataset is missing attempts directory: $sourceAttemptsDir"
    }
    $attempts = @(Get-ChildItem -LiteralPath $sourceAttemptsDir -Filter "attempt_*.npz" | Sort-Object Name)
    if ($attempts.Count -eq 0) {
        throw "Dataset has no attempt_*.npz files: $sourceAttemptsDir"
    }

    $rows = @()
    $index = 1
    foreach ($attempt in $attempts) {
        $targetName = "attempt_{0}_{1:D4}.npz" -f $Prefix, $index
        $targetPath = Join-Path $TargetAttemptsDir $targetName
        New-LinkOrCopy -Source $attempt.FullName -Target $targetPath
        $rows += [PSCustomObject]@{
            combined_attempt = $targetName
            source_label = $Prefix
            source_path = $attempt.FullName
        }
        $index += 1
    }
    return $rows
}

$aiTrainingRoot = Resolve-RequiredDirectory -Path $AiTrainingDataset -Label "AI Training #5 dataset"
$singleFlatRoot = Resolve-RequiredDirectory -Path $SingleFlatDataset -Label "single_surface_flat dataset"

if (Test-Path -LiteralPath $CombinedDataset) {
    if (-not $RebuildCombinedDataset) {
        throw "Combined dataset already exists: $CombinedDataset. Use -RebuildCombinedDataset to recreate it."
    }
    $combinedFull = (Resolve-Path -LiteralPath $CombinedDataset).Path
    $safeBase = (Resolve-Path -LiteralPath "logs\supervised_data").Path
    if (-not $combinedFull.StartsWith($safeBase)) {
        throw "Refusing to rebuild outside logs\supervised_data: $combinedFull"
    }
    Remove-Item -LiteralPath $CombinedDataset -Recurse -Force
}

$combinedAttemptsDir = Join-Path $CombinedDataset "attempts"
New-Item -ItemType Directory -Force -Path $combinedAttemptsDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$attemptRows = @()
$attemptRows += Add-Attempts -SourceRoot $aiTrainingRoot -Prefix "ai5" -TargetAttemptsDir $combinedAttemptsDir
$attemptRows += Add-Attempts -SourceRoot $singleFlatRoot -Prefix "flat" -TargetAttemptsDir $combinedAttemptsDir

$attemptRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $CombinedDataset "attempts.csv")

$config = [ordered]@{
    created_by = "Experiments/run_supervised_pretrain_ai5_single_flat_48x24_20260506.ps1"
    purpose = "Combined behavior-cloning pretrain for real TM 2D asphalt GA dense-noise seeding."
    sources = @(
        [ordered]@{ label = "ai5"; path = $aiTrainingRoot },
        [ordered]@{ label = "flat"; path = $singleFlatRoot }
    )
    target_vertical_mode = $false
    target_multi_surface_mode = $false
    target_observation_dim = 34
    hidden_dim = @(48, 24)
    hidden_activation = @("relu", "tanh")
    epochs = $Epochs
    mirror_augmentation = $true
    boring_filter = $false
    attempt_count = $attemptRows.Count
}
$config | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path (Join-Path $CombinedDataset "config.json")

Write-Host "Combined dataset: $CombinedDataset"
Write-Host "Attempts linked/copied: $($attemptRows.Count)"
Write-Host "Output root: $OutputRoot"

$trainArgs = @(
    "SupervisedTraining.py",
    "--data-root", $CombinedDataset,
    "--output-root", $OutputRoot,
    "--vertical-mode", "false",
    "--multi-surface-mode", "false",
    "--hidden-dim", "48,24",
    "--hidden-activation", "relu,tanh",
    "--epochs", "$Epochs",
    "--disable-boring-filter"
)

& python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Supervised pretrain failed with exit code $LASTEXITCODE"
}

$latestRun = Get-ChildItem -LiteralPath $OutputRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $latestRun) {
    throw "Training finished but no run directory was found under $OutputRoot"
}
$bestModel = Join-Path $latestRun.FullName "best_model.pt"
if (-not (Test-Path -LiteralPath $bestModel)) {
    throw "Training finished but best_model.pt was not found: $bestModel"
}

Write-Host ""
Write-Host "Best model ready:"
Write-Host $bestModel
Write-Host ""
Write-Host "Set GeneticTrainer.py initial_population_source to this path for the real TM dense-seeded run."
