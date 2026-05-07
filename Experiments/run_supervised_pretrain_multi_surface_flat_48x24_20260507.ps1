param(
    [string]$MultiSurfaceFlatDataset = "logs\supervised_data\20260505_134201_map_multi_surface_flat_v3d_surface_target_dataset",
    [string]$OutputRoot = "logs\supervised_runs_multi_surface_flat_pretrain_48x24_20260507",
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

$surfaceRoot = Resolve-RequiredDirectory -Path $MultiSurfaceFlatDataset -Label "multi_surface_flat dataset"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

Write-Host "Multi-surface-flat supervised pretrain"
Write-Host "Dataset: $surfaceRoot"
Write-Host "Output root: $OutputRoot"
Write-Host "Layout: v2d/surface (vertical=false, multi_surface=true)"
Write-Host "Architecture: 48,24 relu,tanh"
Write-Host "Epochs: $Epochs"

$trainArgs = @(
    "SupervisedTraining.py",
    "--data-root", $surfaceRoot,
    "--output-root", $OutputRoot,
    "--vertical-mode", "false",
    "--multi-surface-mode", "true",
    "--hidden-dim", "48,24",
    "--hidden-activation", "relu,tanh",
    "--epochs", "$Epochs",
    "--disable-boring-filter"
)

Write-Host ""
Write-Host "python $($trainArgs -join ' ')"
& python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "multi_surface_flat supervised pretrain failed with exit code $LASTEXITCODE"
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
Write-Host "Use this model as a v2d/surface seed for multi_surface_flat GA experiments."
