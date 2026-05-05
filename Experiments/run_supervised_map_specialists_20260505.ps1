param(
    [string]$DataRoot = "logs\supervised_data",
    [string]$OutputRoot = "",
    [string[]]$Maps = @(
        "single_surface_flat",
        "multi_surface_flat",
        "single_surface_height"
    ),
    [ValidateSet("v2d_asphalt", "v2d_surface", "v3d_asphalt", "v3d_surface")]
    [string]$Layout = "v2d_asphalt",
    [ValidateSet("any", "v2d_asphalt", "v2d_surface", "v3d_asphalt", "v3d_surface")]
    [string]$SourceLayout = "any",
    [string]$HiddenDim = "48,24",
    [string]$HiddenActivation = "relu,tanh",
    [int]$Epochs = 150,
    [int]$RandomSeed = 2026050505,
    [switch]$UseBoringFilter,
    [switch]$DisableMirror
)

$ErrorActionPreference = "Stop"

function Get-LayoutModes {
    param([string]$LayoutName)

    switch ($LayoutName) {
        "v2d_asphalt" { return @{ Vertical = "false"; MultiSurface = "false"; ObsDim = 34 } }
        "v2d_surface" { return @{ Vertical = "false"; MultiSurface = "true"; ObsDim = 40 } }
        "v3d_asphalt" { return @{ Vertical = "true"; MultiSurface = "false"; ObsDim = 43 } }
        "v3d_surface" { return @{ Vertical = "true"; MultiSurface = "true"; ObsDim = 53 } }
        default { throw "Unsupported layout '$LayoutName'." }
    }
}

$TargetModes = Get-LayoutModes -LayoutName $Layout
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = "logs\supervised_runs_map_specialists_${Layout}_20260505"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Get-LatestDataset {
    param([string]$MapName)

    $datasets = @()
    foreach ($Config in Get-ChildItem -Path $DataRoot -Recurse -Filter "config.json") {
        try {
            $Json = Get-Content -Path $Config.FullName -Raw | ConvertFrom-Json
        }
        catch {
            continue
        }
        if ([string]$Json.map_name -ne $MapName) {
            continue
        }
        if ($SourceLayout -ne "any") {
            $SourceModes = Get-LayoutModes -LayoutName $SourceLayout
            if ([bool]$Json.vertical_mode -ne [System.Convert]::ToBoolean($SourceModes.Vertical)) {
                continue
            }
            if ([bool]$Json.multi_surface_mode -ne [System.Convert]::ToBoolean($SourceModes.MultiSurface)) {
                continue
            }
        }
        $datasets += $Config.Directory
    }
    $datasets = $datasets | Sort-Object LastWriteTime -Descending
    if (-not $datasets -or $datasets.Count -eq 0) {
        throw "No supervised dataset found for map '$MapName' under '$DataRoot' with source layout '$SourceLayout'."
    }
    return $datasets[0].FullName
}

foreach ($MapName in $Maps) {
    $Dataset = Get-LatestDataset -MapName $MapName
    $MapOutputRoot = Join-Path $OutputRoot $MapName
    New-Item -ItemType Directory -Force -Path $MapOutputRoot | Out-Null

    Write-Host ""
    Write-Host "=== Training supervised specialist for $MapName ==="
    Write-Host "Dataset: $Dataset"
    Write-Host "Output : $MapOutputRoot"
    Write-Host "Layout : $Layout (vertical=$($TargetModes.Vertical), multi_surface=$($TargetModes.MultiSurface), obs_dim=$($TargetModes.ObsDim))"

    $Command = @(
        "SupervisedTraining.py",
        "--data-root", $Dataset,
        "--output-root", $MapOutputRoot,
        "--vertical-mode", $TargetModes.Vertical,
        "--multi-surface-mode", $TargetModes.MultiSurface,
        "--hidden-dim", $HiddenDim,
        "--hidden-activation", $HiddenActivation,
        "--epochs", "$Epochs",
        "--random-seed", "$RandomSeed"
    )
    if (-not $UseBoringFilter) {
        $Command += "--disable-boring-filter"
    }
    if ($DisableMirror) {
        $Command += "--disable-mirror"
    }

    python @Command

    <#
    Equivalent expanded call:
    python SupervisedTraining.py `
        --data-root $Dataset `
        --output-root $MapOutputRoot `
        --vertical-mode $($TargetModes.Vertical) `
        --multi-surface-mode $($TargetModes.MultiSurface) `
        --hidden-dim $HiddenDim `
        --hidden-activation $HiddenActivation `
        --epochs $Epochs `
        --disable-boring-filter
    #>

    if ($LASTEXITCODE -ne 0) {
        throw "Supervised specialist training failed for $MapName with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Supervised map specialist training finished."
Write-Host "Results root: $OutputRoot"
Write-Host "Target layout: $Layout"
