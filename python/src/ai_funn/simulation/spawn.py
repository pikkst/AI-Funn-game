from __future__ import annotations

from typing import Any

from .terrain import Terrain


def _in_bounds(x: int, y: int, width: int, height: int) -> bool:
    return 0 <= x < width and 0 <= y < height


def _passable(terrain: list[list[int]], x: int, y: int) -> bool:
    if not _in_bounds(x, y, len(terrain[0]), len(terrain)):
        return False
    return terrain[y][x] in (int(Terrain.GROUND), int(Terrain.FOREST), int(Terrain.FOOD))


def viable_surroundings(terrain: list[list[int]], x: int, y: int) -> bool:
    if not _passable(terrain, x, y):
        return False
    width = len(terrain[0])
    height = len(terrain)
    adjacent_passable = 0
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        nx, ny = x + dx, y + dy
        if _in_bounds(nx, ny, width, height) and _passable(terrain, nx, ny):
            adjacent_passable += 1
    return adjacent_passable >= 1


def find_valid_spawn(
    rng: Any, terrain: list[list[int]], max_attempts: int = 1000
) -> tuple[int, int]:
    width = len(terrain[0])
    height = len(terrain)
    for _ in range(max_attempts):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        if viable_surroundings(terrain, x, y):
            return x, y
    raise ValueError("no valid spawn position found")
