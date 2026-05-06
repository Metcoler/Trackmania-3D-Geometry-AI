param(
    [string]$SingleHeightDataset = "logs\supervised_data\20260505_135345_map_single_surface_height_v3d_surface_target_dataset",
    [string]$OutputRoot = "logs\supervised_runs_single_height_pretrain_48x24_20260506",
    [int]$Epochs = 250
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

$heightRoot = Resolve-RequiredDirectory -Path $SingleHeightDataset -Label "single_surface_height dataset"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

Write-Host "Single-surface-height supervised pretrain"
Write-Host "Dataset: $heightRoot"
Write-Host "Output root: $OutputRoot"
Write-Host "Layout: v3d/asphalt (vertical=true, multi_surface=false)"
Write-Host "Architecture: 48,24 relu,tanh"
Write-Host "Epochs: $Epochs"

$trainArgs = @(
    "SupervisedTraining.py",
    "--data-root", $heightRoot,
    "--output-root", $OutputRoot,
    "--vertical-mode", "true",
    "--multi-surface-mode", "false",
    "--hidden-dim", "48,24",
    "--hidden-activation", "relu,tanh",
    "--epochs", "$Epochs",
    "--disable-boring-filter"
)

Write-Host ""
Write-Host "python $($trainArgs -join ' ')"
& python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "single_surface_height supervised pretrain failed with exit code $LASTEXITCODE"
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
Write-Host "Use this model as a v3d/asphalt seed for height GA experiments."
