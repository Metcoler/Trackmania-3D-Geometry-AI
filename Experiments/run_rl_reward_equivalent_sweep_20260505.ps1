$ErrorActionPreference = "Stop"

$Root = "Experiments\runs_rl\reward_equivalent_aabb_tick_sweep_seed_2026050508"
$Seed = 2026050508

New-Item -ItemType Directory -Force -Path $Root | Out-Null

$CommonArgs = @(
    "Experiments\train_sac.py",
    "--map-name", "AI Training #5",
    "--reward-mode", "delta_finished_progress_time_crashes",
    "--action-layout", "gas_brake_steer",
    "--strict-full-action-layout",
    "--env-max-time", "30",
    "--episodes", "1000",
    "--total-timesteps", "3500000",
    "--max-runtime-minutes", "0",
    "--net-arch", "32,16",
    "--activation-fn", "relu",
    "--collision-mode", "lidar",
    "--checkpoint-every-episodes", "100",
    "--seed", "$Seed"
)

function Run-RLExperiment {
    param(
        [string]$Name,
        [string]$Algorithm,
        [string]$PhysicsTickProfile
    )

    $RunDir = Join-Path $Root $Name
    Write-Host ""
    Write-Host "=== Running $Name ==="
    python @CommonArgs `
        --algorithm $Algorithm `
        --physics-tick-profile $PhysicsTickProfile `
        --run-dir $RunDir

    if ($LASTEXITCODE -ne 0) {
        throw "Run $Name failed with exit code $LASTEXITCODE"
    }
}

Run-RLExperiment -Name "ppo_fixed100" -Algorithm "PPO" -PhysicsTickProfile "fixed100"
Run-RLExperiment -Name "sac_fixed100" -Algorithm "SAC" -PhysicsTickProfile "fixed100"
Run-RLExperiment -Name "td3_fixed100" -Algorithm "TD3" -PhysicsTickProfile "fixed100"
Run-RLExperiment -Name "ppo_supervised_v2d" -Algorithm "PPO" -PhysicsTickProfile "supervised_v2d"
Run-RLExperiment -Name "sac_supervised_v2d" -Algorithm "SAC" -PhysicsTickProfile "supervised_v2d"
Run-RLExperiment -Name "td3_supervised_v2d" -Algorithm "TD3" -PhysicsTickProfile "supervised_v2d"

Write-Host ""
Write-Host "RL reward-equivalent sweep finished."
Write-Host "Results root: $Root"
