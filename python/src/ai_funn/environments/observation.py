"""Observation encoder for survival-v1 environment.

Converts a WorldState into the fixed float32[199] observation vector
documented in docs/SIMULATION.md: four 7x7 spatial channels laid out
row-major (passability, food, vegetation, visibility) followed by three
normalized internal values (energy, health, hunger).

Cells outside the world are masked: visibility=0, passability=0, food=0,
vegetation=0.  Masked cells are never confused with traversable empty
ground.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..simulation.terrain import Terrain, is_passable
from ..simulation.world import WorldState

ENERGY_MAX = 100.0
HEALTH_MAX = 100.0
HUNGER_MAX = 1.0
WINDOW_RADIUS = 3
WINDOW_SIZE = 2 * WINDOW_RADIUS + 1  # 7
CHANNELS = 4
SPATIAL_DIM = WINDOW_SIZE * WINDOW_SIZE * CHANNELS  # 196
INTERNAL_DIM = 3
TOTAL_DIM = SPATIAL_DIM + INTERNAL_DIM  # 199


def _normalize(value: float, maximum: float) -> float:
    if maximum <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(value) / float(maximum)))


def encode_observation(state: WorldState) -> npt.NDArray[np.float32]:
    """Encode WorldState to a float32[199] observation vector."""
    terrain = state.terrain
    food = state.food
    cx = state.organism_x
    cy = state.organism_y
    width = state.width
    height = state.height

    passability = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
    food_channel = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
    vegetation = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)
    visibility = np.zeros((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float32)

    for dy in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
        for dx in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
            wy = cy + dy
            wx = cx + dx
            row = dy + WINDOW_RADIUS
            col = dx + WINDOW_RADIUS
            if 0 <= wx < width and 0 <= wy < height:
                visibility[row, col] = 1.0
                tile = int(terrain[wy][wx])
                if is_passable(tile):
                    passability[row, col] = 1.0
                else:
                    passability[row, col] = 0.0
                food_channel[row, col] = float(food[wy][wx])
                vegetation[row, col] = 1.0 if tile == int(Terrain.FOREST) else 0.0
            else:
                visibility[row, col] = 0.0
                passability[row, col] = 0.0
                food_channel[row, col] = 0.0
                vegetation[row, col] = 0.0

    spatial = np.concatenate(
        [
            passability.flatten(),
            food_channel.flatten(),
            vegetation.flatten(),
            visibility.flatten(),
        ],
        axis=0,
    ).astype(np.float32)

    internal = np.array(
        [
            _normalize(state.energy, ENERGY_MAX),
            _normalize(state.health, HEALTH_MAX),
            _normalize(state.hunger, HUNGER_MAX),
        ],
        dtype=np.float32,
    )

    result: npt.NDArray[np.float32] = np.concatenate([spatial, internal], axis=0)
    return result
