from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Preferred names for the cleaned-up project layout.
ASSETS_DIR = PROJECT_ROOT / "Assets"
BLOCK_MESHES_DIR = ASSETS_DIR / "BlockMeshes"
MAPS_DIR = PROJECT_ROOT / "Maps"
MAP_BLOCK_LAYOUTS_DIR = MAPS_DIR / "BlockLayouts"
MAP_GBX_DIR = MAPS_DIR / "Gbx"
MAP_TRACK_MESHES_DIR = MAPS_DIR / "TrackMeshes"

# Historical names kept as fallback until the physical directory move is done.
LEGACY_BLOCK_MESHES_DIR = PROJECT_ROOT / "Meshes"
LEGACY_MAP_BLOCK_LAYOUTS_DIR = MAPS_DIR / "ExportedBlocks"
LEGACY_MAP_GBX_DIR = MAPS_DIR / "GameFiles"
LEGACY_MAP_TRACK_MESHES_DIR = MAPS_DIR / "Meshes"


def _preferred_or_legacy(preferred: Path, legacy: Path) -> Path:
    if preferred.exists():
        return preferred
    return legacy


def block_meshes_dir() -> Path:
    return _preferred_or_legacy(BLOCK_MESHES_DIR, LEGACY_BLOCK_MESHES_DIR)


def map_block_layouts_dir() -> Path:
    return _preferred_or_legacy(MAP_BLOCK_LAYOUTS_DIR, LEGACY_MAP_BLOCK_LAYOUTS_DIR)


def map_gbx_dir() -> Path:
    return _preferred_or_legacy(MAP_GBX_DIR, LEGACY_MAP_GBX_DIR)


def map_track_meshes_dir() -> Path:
    return _preferred_or_legacy(MAP_TRACK_MESHES_DIR, LEGACY_MAP_TRACK_MESHES_DIR)


def block_mesh_path(mesh_name: str) -> Path:
    return block_meshes_dir() / f"{mesh_name}.obj"


def map_block_layout_path(map_name: str) -> Path:
    return map_block_layouts_dir() / f"{map_name}.txt"


def map_track_mesh_path(map_name: str) -> Path:
    return map_track_meshes_dir() / f"{map_name}.obj"
