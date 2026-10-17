from __future__ import annotations

from enum import IntEnum


class Terrain(IntEnum):
    GROUND = 0
    WATER = 1
    FOREST = 2
    FOOD = 3


def is_passable(terrain: Terrain | int) -> bool:
    value = int(terrain)
    return value in (Terrain.GROUND, Terrain.FOREST, Terrain.FOOD)


def is_navigable(terrain: Terrain | int) -> bool:
    return is_passable(terrain)
