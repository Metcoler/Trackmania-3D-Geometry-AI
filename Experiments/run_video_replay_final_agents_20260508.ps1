param(
    [ValidateSet("single_surface_flat", "single_surface_height", "multi_surface_flat", "all")]
    [string]$Map = "all",

    [double]$EnvMaxTime = 120
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$presets = @{
    single_surface_flat = @{
        Title = "Rovinna trat s jednym povrchom"
        MapName = "single_surface_flat"
        ModelFile = "logs\tm_finetune_runs\20260506_004011_tm_seed_map_single_surface_flat_v2d_asphalt_h48x24_p48_src_best_model\global_best.pt"
        VerticalMode = "false"
        MultiSurfaceMode = "false"
    }
    single_surface_height = @{
        Title = "Trat s vyskovymi zmenami"
        MapName = "single_surface_height"
        ModelFile = "logs\tm_finetune_runs\20260506_160030_tm_seed_map_single_surface_height_v3d_asphalt_h48x24_p48_src_best_model\global_best.pt"
        VerticalMode = "true"
        MultiSurfaceMode = "false"
    }
    multi_surface_flat = @{
        Title = "Rovinna trat s roznymi povrchmi"
        MapName = "multi_surface_flat"
        ModelFile = "logs\tm_finetune_runs\20260507_090226_tm_seed_map_multi_surface_flat_v2d_surface_h48x24_p48_src_best_model\global_best.pt"
        VerticalMode = "false"
        MultiSurfaceMode = "true"
    }
}

if ($Map -eq "all") {
    $selectedMaps = @("single_surface_flat", "single_surface_height", "multi_surface_flat")
} else {
    $selectedMaps = @($Map)
}

foreach ($mapKey in $selectedMaps) {
    $preset = $presets[$mapKey]

    if (-not (Test-Path $preset.ModelFile)) {
        throw "Model file does not exist: $($preset.ModelFile)"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Video replay: $($preset.Title)"
    Write-Host "Map: $($preset.MapName)"
    Write-Host "Model: $($preset.ModelFile)"
    Write-Host "============================================================"
    Write-Host ""
    Read-Host "Open this map in Trackmania, start video recording, then press Enter"

    python .\Driver.py `
        --map-name $preset.MapName `
        --model-file $preset.ModelFile `
        --action-mode target `
        --vertical-mode $preset.VerticalMode `
        --multi-surface-mode $preset.MultiSurfaceMode `
        --never-quit true `
        --max-touches inf `
        --env-max-time $EnvMaxTime `
        --target-steer-deadzone 0.0 `
        --wait-for-positive-time false `
        --invert-steer false

    Write-Host ""
    Write-Host "Replay finished for $($preset.MapName)."
    if ($mapKey -ne $selectedMaps[-1]) {
        Read-Host "Stop/save the recording if needed, then press Enter for the next map"
    }
}
