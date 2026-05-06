$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Hardcoded run recipe:
# - same real-TM GA profile as the overnight flat run
# - new map single_surface_height
# - 3D observation layout enabled
# - surface channels disabled
# - supervised 48x24 height model used as dense-noise seed
$env:TM_GA_MAP_NAME = "single_surface_height"
$env:TM_GA_ENV_MAX_TIME = "45"
$env:TM_GA_VERTICAL_MODE = "true"
$env:TM_GA_MULTI_SURFACE_MODE = "false"
$env:TM_GA_ACTION_MODE = "target"
$env:TM_GA_HIDDEN_DIMS = "48,24"
$env:TM_GA_HIDDEN_ACTIVATIONS = "relu,tanh"

$env:TM_GA_POP_SIZE = "48"
$env:TM_GA_PARENT_COUNT = "14"
$env:TM_GA_ELITE_COUNT = "2"
$env:TM_GA_GENERATIONS = "300"
$env:TM_GA_CHECKPOINT_EVERY = "10"

$env:TM_GA_SELECTION_MODE = "lexicographic"
$env:TM_GA_SELECTION_FITNESS_MODE = "ranking"
$env:TM_GA_RANKING_MODE = "lexicographic"
$env:TM_GA_RANKING_KEY = "(finished, progress, -time, -crashes)"
$env:TM_GA_RANKING_PROGRESS_SOURCE = "progress"

$env:TM_GA_MUTATION_PROB = "0.15"
$env:TM_GA_MUTATION_PROB_DECAY = "1.0"
$env:TM_GA_MUTATION_PROB_MIN = "0.05"
$env:TM_GA_MUTATION_SIGMA = "0.30"
$env:TM_GA_MUTATION_SIGMA_DECAY = "1.0"
$env:TM_GA_MUTATION_SIGMA_MIN = "0.20"
$env:TM_GA_MUTATION_DECAY_TRIGGER = "first_finish"

$env:TM_GA_REUSE_ELITE_EVALUATIONS = "true"
$env:TM_GA_MIRROR_EPISODE_PROB = "0.0"
$env:TM_GA_EVALUATE_BOTH_MIRRORS = "false"
$env:TM_GA_MAX_TOUCHES = "1"
$env:TM_GA_START_IDLE_MAX_TIME = "2.0"
$env:TM_GA_TARGET_STEER_DEADZONE = "0.0"
$env:TM_GA_TRAJECTORY_LOG_MODE = "top"
$env:TM_GA_TRAJECTORY_TOP_K = "2"
$env:TM_GA_TRAJECTORY_LOG_ACTIONS = "true"

$env:TM_GA_SEED_MODEL_ROOT = "logs\supervised_runs_single_height_pretrain_48x24_20260506"
$env:TM_GA_SEED_MODEL_EXACT_COPIES = "1"
$env:TM_GA_SEED_MODEL_NOISE_MODE = "dense"
$env:TM_GA_SEED_MODEL_MUTATION_PROBS = "0.02,0.05,0.10"
$env:TM_GA_SEED_MODEL_MUTATION_SIGMAS = "0.01,0.025,0.05"
$env:TM_GA_SEED_MODEL_TIER_COUNTS = "16,16,15"

$SeedRoot = $env:TM_GA_SEED_MODEL_ROOT
$SeedModel = Get-ChildItem -Path $SeedRoot -Recurse -Filter "best_model.pt" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $SeedModel) {
    throw "No height seed model found under '$SeedRoot'. Run .\Experiments\run_supervised_pretrain_single_height_48x24_20260506.ps1 first."
}

Write-Host "Height-map live GA seeded run"
Write-Host "Seed model: $($SeedModel.FullName)"
Write-Host "Map/layout: single_surface_height, vertical=true, multi_surface=false"
Write-Host "GA profile: pop=48, parents=14, elites=2, generations=300"
Write-Host "Ranking: (finished, progress, -time, -crashes)"
Write-Host ""
Write-Host "python GeneticTrainer.py"

& python GeneticTrainer.py
if ($LASTEXITCODE -ne 0) {
    throw "Height-map GA run failed with exit code $LASTEXITCODE"
}
